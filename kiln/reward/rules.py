"""Verifiable (rule-based) reward functions.

Each function has signature ``fn(response: str, answer: str | None, **params) -> float``
and is registered by name so a YAML can compose them with weights::

    reward:
      mode: rules
      functions:
        - {name: math_boxed, weight: 1.0}
        - {name: format_think, weight: 0.2}
        - {name: length_penalty, weight: 0.1, params: {max_tokens: 2048}}

The math verifier does the usual normalisation dance (strip ``\\boxed{}``,
``\\text{}``, units, thousands separators, trailing periods; compare as
fractions when both sides parse as numbers; fall back to symbolic
equivalence via sympy when available; finally exact-match on the normalised
string). It is intentionally conservative: when in doubt, zero reward.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from fractions import Fraction

RewardFn = Callable[..., float]
_REGISTRY: dict[str, RewardFn] = {}


def register(name: str):
    def deco(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> RewardFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward function {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


# ------------------------------------------------------------------ math

_BOXED = re.compile(r"\\boxed\s*\{")


def extract_boxed(text: str) -> str | None:
    """Return the content of the *last* ``\\boxed{...}`` with balanced braces."""
    last = None
    for m in _BOXED.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
    return last


def extract_final_answer(text: str) -> str | None:
    b = extract_boxed(text)
    if b is not None:
        return b
    m = re.findall(r"(?:final answer|answer)\s*(?:is\s*:?|:)\s*\**\s*([^\n]+)", text, flags=re.I)
    if m:
        return m[-1].strip().rstrip(".")
    m = re.findall(r"####\s*([^\n]+)", text)  # GSM8K style
    if m:
        return m[-1].strip()
    return None


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|mathbf|left|right|,|;|!)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("$", "").replace("\\%", "%").replace("%", "")
    s = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"(\1)/(\2)", s)
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = re.sub(r"\b(dollars?|units?|cm|km|m|kg|g|degrees?|°)\b\.?", "", s, flags=re.I)
    s = s.replace(" ", "").rstrip(".")
    if s.startswith("(") and s.endswith(")") and s.count("(") == 1:
        s = s[1:-1]
    return s


def _to_number(s: str) -> Fraction | None:
    s = s.strip()
    try:
        if "/" in s and not re.search(r"[a-zA-Z]", s):
            num, den = s.replace("(", "").replace(")", "").split("/", 1)
            return Fraction(Fraction(num), Fraction(den))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _sympy_equal(a: str, b: str) -> bool | None:
    try:
        import sympy
        from sympy.parsing.latex import parse_latex  # noqa: F401
    except Exception:
        return None
    try:
        ea = sympy.sympify(a.replace("^", "**"))
        eb = sympy.sympify(b.replace("^", "**"))
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return None


def answers_match(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        return pn == gn or math.isclose(float(pn), float(gn), rel_tol=1e-6, abs_tol=1e-9)
    eq = _sympy_equal(p, g)
    return bool(eq) if eq is not None else False


@register("math_boxed")
def math_boxed(response: str, answer: str | None, **_) -> float:
    """1.0 if the final answer matches ``answer``, else 0.0."""
    if answer is None:
        return 0.0
    pred = extract_final_answer(response)
    return 1.0 if pred is not None and answers_match(pred, answer) else 0.0


@register("exact_match")
def exact_match(response: str, answer: str | None, **_) -> float:
    return 1.0 if answer is not None and response.strip() == answer.strip() else 0.0


# ---------------------------------------------------------------- format


@register("format_think")
def format_think(response: str, answer: str | None, **_) -> float:
    """Reward a single well-formed ``<think>...</think>`` block followed by content."""
    m = re.fullmatch(r"\s*<think>(.*?)</think>\s*(.+)", response, flags=re.S)
    if not m:
        return 0.0
    think, rest = m.group(1).strip(), m.group(2).strip()
    return 1.0 if think and rest and "<think>" not in rest else 0.0


@register("length_penalty")
def length_penalty(
    response: str, answer: str | None, max_tokens: int = 2048, soft_start: float = 0.8, **_
) -> float:
    """0 up to ``soft_start*max_tokens`` characters-as-tokens, linearly to -1 at ``max_tokens`` (DAPO-style)."""
    n = len(response.split())
    start = soft_start * max_tokens
    if n <= start:
        return 0.0
    return -min((n - start) / max(max_tokens - start, 1), 1.0)


@register("no_repetition")
def no_repetition(response: str, answer: str | None, ngram: int = 4, threshold: float = 0.3, **_) -> float:
    """Penalise degenerate repetition: fraction of repeated n-grams above ``threshold`` costs up to -1."""
    toks = response.split()
    if len(toks) < ngram * 2:
        return 0.0
    grams = [tuple(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)]
    repeated = 1.0 - len(set(grams)) / len(grams)
    return -max(0.0, (repeated - threshold) / (1 - threshold)) if repeated > threshold else 0.0


# ------------------------------------------------------------------ code

_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


@register("python_tests")
def python_tests(response: str, answer: str | None, timeout_s: float = 5.0, **_) -> float:
    """Run the last python code block against ``answer`` (a test snippet using ``assert``).

    Executes in a subprocess with ``-I`` (isolated), a wall-clock timeout and no
    network expectations. This is **not** a security sandbox; run the reward
    server in a locked-down container for untrusted model output.
    """
    if answer is None:
        return 0.0
    blocks = _CODE_BLOCK.findall(response)
    if not blocks:
        return 0.0
    src = blocks[-1] + "\n\n" + answer + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout_s)
        return 1.0 if r.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0


def score(
    response: str, answer: str | None, functions: list[tuple[str, float, dict]]
) -> tuple[float, dict[str, float]]:
    """Weighted sum of registered functions. Returns ``(total, breakdown)``."""
    total, breakdown = 0.0, {}
    for name, weight, params in functions:
        v = float(get(name)(response, answer, **params))
        breakdown[name] = v
        total += weight * v
    return total, breakdown
