"""Collation, packing, prompt datasets, reward rules and the reward server."""

from __future__ import annotations

import json

import pytest
import torch

from kiln.config.schema import DataConfig, RewardConfig, RewardFnConfig
from kiln.data.packing import first_fit_decreasing, pack, packing_efficiency, unpack
from kiln.data.prompts import ByteTokenizer, PreferenceDataset, PromptDataset
from kiln.reward import (
    RewardScorer,
    answers_match,
    extract_boxed,
    extract_final_answer,
    normalize_answer,
    rules,
)
from kiln.rollout.types import Sample, collate_samples, concat_batches


# --------------------------------------------------------------- collation
def test_collate_alignment_of_behavior_logprobs():
    s = Sample(
        prompt_ids=[10, 11, 12],
        response_ids=[20, 21],
        behavior_logprobs=[-0.5, -0.7],
        group_id=0,
        policy_version=3,
        reward=1.0,
    )
    b = collate_samples([s], pad_id=0)
    assert b.input_ids.tolist() == [[10, 11, 12, 20, 21]]
    assert b.response_mask.tolist() == [[0, 0, 0, 1, 1]]
    # response token k is at index p+k, predicted at logit index p+k-1 -> behavior_logp[p-1+k]
    assert torch.allclose(b.behavior_logp[0], torch.tensor([0, 0, -0.5, -0.7]))
    assert b.token_mask().tolist() == [[0, 0, 1, 1]]
    assert int(b.policy_version[0]) == 3 and b.staleness(5).tolist() == [2]


def test_collate_pads_and_truncates():
    a = Sample([1], [2, 3, 4], [-1, -1, -1], 0, 0)
    c = Sample([1, 2, 3], [4], [-1], 1, 0)
    b = collate_samples([a, c], pad_id=99, max_len=3)
    assert b.seq_len == 3
    assert b.input_ids.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert b.attention_mask.sum(-1).tolist() == [3, 3]
    assert b.response_mask.tolist() == [[0, 1, 1], [0, 0, 0]]


def test_slice_microbatch_and_concat_roundtrip():
    samples = [Sample([1, 2], [3] * (i + 1), [-1.0] * (i + 1), i, 0, reward=float(i)) for i in range(5)]
    b = collate_samples(samples)
    mbs = list(b.micro_batches(2))
    assert [m.batch_size for m in mbs] == [2, 2, 1]
    again = concat_batches(mbs)
    assert torch.equal(again.input_ids, b.input_ids) and torch.equal(again.rewards, b.rewards)
    b.extra["x"] = torch.arange(5)
    assert b.slice(1, 3).extra["x"].tolist() == [1, 2]


# ----------------------------------------------------------------- packing
def test_first_fit_decreasing_bins():
    bins = first_fit_decreasing([5, 3, 4, 2, 6], capacity=8)
    assert all(sum([5, 3, 4, 2, 6][i] for i in b) <= 8 for b in bins)
    assert sorted(i for b in bins for i in b) == [0, 1, 2, 3, 4]
    assert len(bins) == 3
    with pytest.raises(ValueError):
        first_fit_decreasing([9], 8)


def test_pack_unpack_roundtrip_and_block_mask():
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]])
    am = (ids != 0).long()
    packed = pack(ids, am, capacity=6, pad_id=0)
    assert packed.input_ids.shape[1] == 6
    # per-token values in packed layout = the token ids themselves; unpacking must return them
    per_tok = packed.input_ids.float()
    back = unpack(packed, per_tok, T=5, shift=False)
    assert torch.equal(back.long(), ids)
    # shifted convention: value at packed c corresponds to predicted token c+1
    shifted = per_tok[:, 1:]
    back_s = unpack(packed, shifted, T=5, shift=True)
    assert torch.equal(back_s.long(), ids[:, 1:])
    mask = packed.block_diag_mask()
    P, _, C, _ = mask.shape
    for p in range(P):
        for i in range(C):
            for j in range(C):
                si, sj = int(packed.seq_index[p, i]), int(packed.seq_index[p, j])
                expect = (si == sj and si >= 0 and j <= i) or i == j
                assert bool(mask[p, 0, i, j]) == expect
    assert 0 < packing_efficiency(packed) <= 1


def test_packed_forward_matches_padded_forward():
    """The whole point of packing: logits must be identical to the padded computation."""
    from kiln.kernels import sequence_logprobs
    from kiln.model import TinyConfig, TinyLM

    torch.manual_seed(0)
    m = TinyLM(TinyConfig(hidden=32, layers=2, heads=2, vocab_size=64)).eval()
    ids = torch.randint(1, 64, (3, 7))
    am = torch.ones(3, 7, dtype=torch.long)
    am[0, 4:] = 0
    am[1, 6:] = 0
    ids = ids * am
    resp = am.clone()
    resp[:, :2] = 0
    pos = torch.arange(7)[None].expand(3, 7) * am
    with torch.no_grad():
        logits_pad = m(ids, attention_mask=am, position_ids=pos)
        lp_pad, _ = sequence_logprobs(logits_pad, ids, resp)
        packed = pack(ids, am, capacity=14, pad_id=0)
        logits_packed = m(
            packed.input_ids,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            block_mask=packed.block_diag_mask(),
        )
        resp_p = torch.zeros_like(packed.attention_mask)
        valid = packed.seq_index >= 0
        resp_p[valid] = resp[packed.seq_index[valid], packed.position_ids[valid]]
        lp_p, _ = sequence_logprobs(logits_packed, packed.input_ids, resp_p)
        lp_unpacked = unpack(packed, lp_p, T=7)
    assert torch.allclose(lp_pad, lp_unpacked, atol=1e-4)


# ---------------------------------------------------------------- dataset
def test_prompt_dataset_jsonl_and_sharded_sampler(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"prompt": f"question {i}", "answer": str(i), "difficulty": i % 2}) for i in range(10)
        ),
        encoding="utf-8",
    )
    ds = PromptDataset(DataConfig(train_path=str(p), max_prompt_len=4), ByteTokenizer())
    assert len(ds) == 10 and ds[3].answer == "3" and ds[3].meta == {"difficulty": 1}
    assert len(ds[0].ids) == 4  # truncated to max_prompt_len (keeps the tail)
    seen0 = {pr.uid for pr, _ in zip(ds.sampler(seed=1, rank=0, world_size=2), range(5))}
    seen1 = {pr.uid for pr, _ in zip(ds.sampler(seed=1, rank=1, world_size=2), range(5))}
    assert seen0.isdisjoint(seen1) and len(seen0 | seen1) == 10
    a = [pr.uid for pr, _ in zip(ds.sampler(seed=7), range(10))]
    b = [pr.uid for pr, _ in zip(ds.sampler(seed=7), range(10))]
    assert a == b and sorted(a) == list(range(10))


def test_chat_messages_prompt_and_preference_dataset(tmp_path):
    rows = [{"prompt": [{"role": "user", "content": "hi"}], "answer": "x"}]
    ds = PromptDataset(DataConfig(), ByteTokenizer(), rows=rows)
    assert ds[0].text == "hi"
    p = tmp_path / "pref.jsonl"
    p.write_text(json.dumps({"prompt": "p", "chosen": "good", "rejected": "bad"}) + "\n", encoding="utf-8")
    pd = PreferenceDataset(p, ByteTokenizer())
    assert len(pd) == 1 and pd[0].chosen_ids[-1] == ByteTokenizer.EOS


# ----------------------------------------------------------------- rewards
@pytest.mark.parametrize(
    "text,expected",
    [
        (r"so \boxed{42}.", "42"),
        (r"\boxed{\frac{1}{2}} then \boxed{x^{2}+1}", "x^{2}+1"),
        ("no box", None),
        (r"\boxed{unbalanced", None),
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


def test_extract_final_answer_fallbacks():
    assert extract_final_answer("The final answer is: 17.") == "17"
    assert extract_final_answer("blah #### 1,234") == "1,234"


@pytest.mark.parametrize(
    "a,b,ok",
    [
        ("42", "42.0", True),
        ("1/2", "0.5", True),
        (r"\frac{3}{4}", "0.75", True),
        ("1,000", "1000", True),
        (r"\text{m}", "m", True),
        ("$12$", "12", True),
        ("-0.333", "-1/3", False),
        ("41", "42", False),
    ],
)
def test_answers_match(a, b, ok):
    assert answers_match(a, b) == ok


def test_normalize_answer_idempotent():
    s = normalize_answer(r"\boxed{ 1,000.00 }")
    assert normalize_answer(s) == s


def test_rule_functions_and_composition():
    assert rules.get("math_boxed")(r"\boxed{7}", "7") == 1.0
    assert rules.get("math_boxed")(r"\boxed{8}", "7") == 0.0
    assert rules.get("format_think")("<think>x</think>y", None) == 1.0
    assert rules.get("no_repetition")("a b c d e f g h", None) == 0.0
    assert rules.get("no_repetition")("a b c d " * 20, None) < 0.0
    assert rules.get("length_penalty")("short", None, max_tokens=100) == 0.0
    with pytest.raises(KeyError):
        rules.get("nope")
    scorer = RewardScorer(
        RewardConfig(
            functions=[RewardFnConfig(name="math_boxed"), RewardFnConfig(name="format_think", weight=0.5)]
        )
    )
    r, bd = scorer.score_batch(
        [{"response": r"<think>a</think>\boxed{7}", "answer": "7"}, {"response": "nope", "answer": "7"}]
    )
    assert r == [1.5, 0.0] and bd[0]["math_boxed"] == 1.0


def test_python_tests_reward_runs_sandboxed_code():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    tests = "assert add(1, 2) == 3"
    assert rules.get("python_tests")(good, tests, timeout_s=10) == 1.0
    assert (
        rules.get("python_tests")("```python\ndef add(a, b):\n    return a - b\n```", tests, timeout_s=10)
        == 0.0
    )
    assert rules.get("python_tests")("```python\nwhile True: pass\n```", tests, timeout_s=1) == 0.0


def test_reward_server_roundtrip():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kiln.reward.server import create_app

    app = create_app(RewardConfig(functions=[RewardFnConfig(name="exact_match")]))
    with TestClient(app) as c:
        assert c.get("/healthz").json()["ok"]
        r = c.post(
            "/score",
            json={"items": [{"response": "yes", "answer": "yes"}, {"response": "no", "answer": "yes"}]},
        )
        assert r.json()["rewards"] == [1.0, 0.0]
        assert "kiln_reward_items_total" in c.get("/metrics").text
