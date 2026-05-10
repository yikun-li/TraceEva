# TraceEval — Replication Package

This directory packages the artifacts needed to reproduce the results in the
TraceEval paper:

- the held-out **test split** (2,129 programs across Python, JavaScript, Java)
- the **train split** (8,454 programs)
- the **scripts** that build and evaluate against the splits

## Layout

```text
TraceEva/
├── README.md                            # this file
├── scripts/
│   ├── run_llm.py                       # zero-shot evaluation (frontier + open-weight)
│   ├── compute_metrics.py               # edge-level P/R/F1 by language and overall
│   ├── normalize_edges.py               # caller->callees JSON schema canonicalization
│   ├── run_tracers.py                   # cross-language tracer orchestrator
│   └── tracers/
│       ├── python_tracer.py             # Python tracer
│       ├── java_tracer.py               # Java tracer
│       └── js_tracer.js                 # JavaScript tracer
└── data/
    ├── traceeval.zip                    # full corpus (test + train programs with source + GT)
    └── traceeval_split/                 # canonical split definitions
        ├── {python,javascript,java}_repo_assignment.json  # per-language repo->split mapping
        ├── test_ids.json                # 2,129-program test IDs
        └── train_ids.json               # 8,454-program train IDs
```

## Reproducing the paper's results

The corpus archive `data/traceeval.zip` extracts to
`data/benchmark/{python,javascript,java}/<program_id>/`; each program directory
contains the program source plus a `callgraph.json` that holds the
tracer-witnessed ground-truth edge set. The split definitions in
`data/traceeval_split/{test,train}_ids.json` enumerate the program IDs that
belong to each split.

1. **Zero-shot evaluation table**:

   ```bash
   unzip data/traceeval.zip -d data/
   python scripts/run_llm.py data/benchmark --model <model_name>
   # compute_metrics.py applies edge normalization by default.
   python scripts/compute_metrics.py --results-dir scripts/data/results --gt-dir data/benchmark
   ```

2. **Deterministic-replay tracer check**:

   ```bash
   python scripts/run_tracers.py --split test --check
   ```

   `run_tracers.py` extracts `traceeval.zip` on first invocation, re-runs the
   language-specific tracer on every program in the requested split, and with
   `--check` reports per-program agreement against the shipped `callgraph.json`.
   This is the regression test backing the deterministic-tracer claim in the
   paper.

## License

- Code (this directory): Apache-2.0
- Data (`traceeval.zip` and split JSONs): CC-BY 4.0
- Each program retains the license of its upstream GitHub repository in the
  per-instance provenance metadata.
