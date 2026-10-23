"""``kiln`` command-line entry point.

kiln train      configs/grpo_math_qwen3_8b.yaml  [--set key=value ...]
kiln validate   configs/*.yaml
kiln reward-serve configs/grpo_math_qwen3_8b.yaml --port 8008
kiln gateway    --replica a=http://... --replica b=http://... --canary b
kiln bench      kernels | step | scheduler
kiln topo
kiln inspect-ckpt checkpoints/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kiln import __version__
from kiln.config.schema import KilnConfig

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Kiln: async LLM post-training + serving infra."
)
console = Console()


def _load(config: Path, set_: list[str] | None) -> KilnConfig:
    try:
        return KilnConfig.from_yaml(config, overrides=set_)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]invalid config {config}:[/red] {e}")
        raise typer.Exit(2) from e


@app.callback(invoke_without_command=True)
def _root(version: bool = typer.Option(False, "--version", "-V", help="Print version and exit")):
    if version:
        console.print(f"kiln {__version__}")
        raise typer.Exit()


@app.command()
def validate(configs: list[Path]):
    """Validate YAML config(s) against the schema and print the resolved values."""
    bad = 0
    for c in configs:
        try:
            cfg = KilnConfig.from_yaml(c)
            console.print(
                f"[green]OK[/green] {c}  algo={cfg.algorithm.name} engine={cfg.rollout.engine} "
                f"world={cfg.parallel.world_size} async={cfg.async_.enabled}"
            )
        except Exception as e:  # noqa: BLE001
            bad += 1
            console.print(f"[red]FAIL[/red] {c}: {e}")
    raise typer.Exit(1 if bad else 0)


@app.command()
def train(
    config: Path,
    set_: list[str] = typer.Option(None, "--set", "-s", help="Override: a.b.c=value"),
    ray: bool = typer.Option(False, help="Launch on a Ray cluster instead of in-process"),
    steps: int | None = typer.Option(None, help="Override train.total_steps"),
    print_config: bool = typer.Option(False, help="Print the resolved config and exit"),
):
    """Run RL post-training from a YAML config."""
    cfg = _load(config, set_)
    if steps is not None:
        cfg.train.total_steps = steps
    if print_config:
        console.print(cfg.to_yaml())
        raise typer.Exit()
    if ray:
        from kiln.orchestration.ray_runtime import RayConfig, launch

        out = launch(cfg, RayConfig(num_engines=cfg.rollout.num_engines))
        console.print_json(json.dumps(out, default=float))
        return
    from kiln.orchestration import AsyncController

    ctl = AsyncController.from_config(cfg)
    res = ctl.run()
    console.print(f"[bold green]done[/bold green] {res.steps} steps; run dir: {res.run_dir}")
    final = {
        k: v
        for k, v in res.final_metrics.items()
        if k.split("/")[0] in ("reward", "loss", "policy", "async", "perf")
    }
    _print_kv("final metrics", final)


@app.command("reward-serve")
def reward_serve(config: Path, host: str = "0.0.0.0", port: int = 8008):
    """Serve the configured reward functions over HTTP (``POST /score``)."""
    from kiln.reward.server import serve

    serve(_load(config, None).reward, host=host, port=port)


@app.command()
def gateway(
    replica: list[str] = typer.Option(..., help="name=url, repeatable"),
    canary: list[str] = typer.Option(None, help="replica names that are canaries"),
    host: str = "0.0.0.0",
    port: int = 8080,
    p99_slo_s: float = 0.5,
):
    """Start the prefix-cache-aware gateway with optional canary guardrails."""
    import uvicorn

    from kiln.serving.canary import SLO
    from kiln.serving.gateway import create_app
    from kiln.serving.router import EngineReplica

    reps = []
    for r in replica:
        name, url = r.split("=", 1)
        reps.append(EngineReplica(name=name, url=url))
    uvicorn.run(create_app(reps, set(canary or []), SLO(p99_latency_s=p99_slo_s)), host=host, port=port)


@app.command()
def bench(
    what: str = typer.Argument("kernels", help="kernels | step | scheduler | sync"),
    config: Path | None = typer.Option(None, help="config for `step`"),
    json_out: Path | None = typer.Option(None, help="write results as JSON"),
):
    """Micro-benchmarks. Honest numbers for the hardware you are on, nothing pre-baked."""
    from kiln.bench import run_benchmark

    results = run_benchmark(what, config)
    for name, rows in results.items():
        t = Table(title=name)
        if rows:
            for k in rows[0]:
                t.add_column(k)
            for r in rows:
                t.add_row(*[_fmt(v) for v in r.values()])
        console.print(t)
    if json_out:
        json_out.write_text(json.dumps(results, indent=2, default=float))


@app.command()
def topo():
    """Discover GPU/NIC topology and print recommended NCCL env."""
    from kiln.distributed.topology import describe, discover, recommend_nccl_env

    t = discover()
    console.print(describe(t))
    _print_kv("recommended NCCL env", recommend_nccl_env(t))


@app.command("inspect-ckpt")
def inspect_ckpt(path: Path):
    """Print metadata for every checkpoint under a directory."""
    latest = (path / "LATEST").read_text().strip() if (path / "LATEST").exists() else None
    for d in sorted(path.glob("step_*"), key=lambda p: int(p.name.split("_")[1])):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        tag = " [bold](latest)[/bold]" if latest == d.name.split("_")[1] else ""
        console.print(f"{d.name}{tag}: {meta}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _print_kv(title: str, kv: dict) -> None:
    t = Table(title=title, show_header=False)
    t.add_column("k")
    t.add_column("v")
    for k, v in kv.items():
        t.add_row(str(k), _fmt(v))
    console.print(t)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
