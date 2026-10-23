# Contributing

```bash
pip install -e ".[dev]" && pre-commit install
pytest -q            # must pass on CPU
ruff check . && ruff format --check .
```

Rules of the house:

* **Every kernel has a reference implementation and a parity test** (forward and backward). A Triton change without a test diff is not reviewable.
* **Nothing in `kiln.algorithms` touches a model, optimizer or device.** It takes tensors, returns a loss and metrics. Keep it that way so the math stays unit-testable in milliseconds.
* **Alignment conventions** live in `kiln/rollout/types.py`; if you need an off-by-one, put it there and add a test in `tests/test_data_and_rewards.py`.
* **New config fields** go in `kiln/config/schema.py` with a docstring, and `kiln validate configs/*.yaml` must still pass. The schema is `extra="forbid"` on purpose.
* **Benchmarks report measured numbers only.** Do not commit numbers into docs that `kiln bench` did not print.
* Design-level changes get an ADR in `docs/design/`.

PR checklist: tests added, `kiln validate configs/*.yaml` passes, perf gate (`scripts/check_perf_regression.py`) not regressed, ADR if the design changed.
