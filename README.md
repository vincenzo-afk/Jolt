# JOLT — Judgment On Low-latency Thresholds

A tiny (≤100M hard cap, ~3.4M default) reflex model that sits in front of an LLM agent.
It picks tools, classifies failures, chooses retry/switch/stop policy, scores loops and goal
drift, gates risk, and **escalates to the LLM only when uncertain**.

## Success contract (Config C)
| Metric | Target |
|---|---|
| Task success | ≥ 0.90 × LLM-only |
| Cost per task | ≤ 0.25 × LLM-only |
| p50 / p99 step latency | < 5 ms / < 10 ms |
| LLM calls per task | ≤ 0.30 × baseline |
| Escalation rate | 10–25% |
| Safety violations (Config D) | 0 |
| ECE per head | < 0.05 |
| Judge-vs-human agreement | ≥ 85% |

## Architecture
TinyEncoder (reads only the current observation) → fuse [goal; obs; memory; budget] →
GRUCell (only carrier of episode context) → 6 heads + StatePred aux.
Per-head temperature scaling + conformal/threshold abstention → escalation OR-rule →
hard code-level safety guard (the model can never loosen it).

See [docs/architecture.md](docs/architecture.md) and [docs/PLAN.md](docs/PLAN.md).

## Scaling ladder
T0 ~3.4M → T1 ~10M → T2 ~25M → T3 ≤100M. Promote only if per-head val accuracy rises
AND int8 p99 stays < 10 ms on the target CPU.

## Quickstart
```bash
pip install -e ".[dev]"
make test
python -m jolt.bench.latency --random-weights   # prove <10ms before training anything
```

## Pipeline
1. Collect & log: `scripts/collect.sh` (with failure injection, ≥15% non-success steps)
2. Label: `scripts/label.sh` (one judge prompt per head, 2-judge agreement filter)
3. Train T0: `make train-t0`
4. Calibrate → ONNX int8 export
5. Tune thresholds (cost grid search) → run Configs A–D
6. Weekly retrain from the escalation log

## Safety
Hard guard in `jolt/runtime/guard.py`: tools flagged `irreversible` never execute on model
confidence alone. The risk head can only tighten.

## Status
Roadmap: see Milestones. Contributions welcome — read CONTRIBUTING.md.

## License
Apache-2.0
