#!/usr/bin/env python3
"""
Re-trace \\bench programs and compare against the released ground truth.

Iterates the programs in the test (or train) split, invokes the appropriate
language tracer on each one, and writes the per-program tracer output as JSON.
With ``--check``, also compares the freshly traced edge set against the
released ``callgraph.json`` and reports per-program agreement; this is the
deterministic-replay regression test referenced in the paper's Validation
section.

Layout assumptions:
    <package_root>/data/traceeval.zip                # full corpus archive
    <package_root>/data/traceeval_split/test_ids.json
    <package_root>/data/traceeval_split/train_ids.json

If the corpus has not been extracted yet, the script will extract
``traceeval.zip`` to ``<package_root>/data/benchmark/`` on first use.

Usage:
    python scripts/run_tracers.py                    # trace all test programs
    python scripts/run_tracers.py --split train      # trace train split
    python scripts/run_tracers.py --lang python      # restrict to one language
    python scripts/run_tracers.py --check            # compare against shipped GT
    python scripts/run_tracers.py --limit 50         # cap programs per language
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PKG_ROOT, "data")
CORPUS_ZIP = os.path.join(DATA_DIR, "traceeval.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "benchmark")
SPLIT_DIR = os.path.join(DATA_DIR, "traceeval_split")
OUTPUT_DIR = os.path.join(DATA_DIR, "tracer_output")

LANGUAGE_TRACERS = {
    "python":     [sys.executable, os.path.join(SCRIPT_DIR, "tracers", "python_tracer.py")],
    "javascript": ["node",         os.path.join(SCRIPT_DIR, "tracers", "js_tracer.js")],
    "java":       [sys.executable, os.path.join(SCRIPT_DIR, "tracers", "java_tracer.py")],
}


def ensure_corpus_extracted():
    """Extract traceeval.zip on first run; subsequent calls are no-ops."""
    if os.path.isdir(EXTRACT_DIR) and any(
        os.path.isdir(os.path.join(EXTRACT_DIR, lang)) for lang in LANGUAGE_TRACERS
    ):
        return
    if not os.path.isfile(CORPUS_ZIP):
        sys.exit(f"Error: corpus archive not found at {CORPUS_ZIP}")
    print(f"Extracting {CORPUS_ZIP} to {DATA_DIR} ...")
    with zipfile.ZipFile(CORPUS_ZIP) as zf:
        zf.extractall(DATA_DIR)
    # The archive root directory inside the zip is `benchmark/`; if it landed
    # at a different name, surface a clear error rather than silently failing.
    if not os.path.isdir(EXTRACT_DIR):
        sys.exit(f"Error: expected {EXTRACT_DIR} after extraction, not found")


def load_split_ids(split):
    """Return {language: [program_id, ...]} for the requested split."""
    path = os.path.join(SPLIT_DIR, f"{split}_ids.json")
    if not os.path.isfile(path):
        sys.exit(f"Error: split file not found at {path}")
    with open(path) as f:
        return json.load(f)


def edge_set_from_callgraph(cg):
    """Convert a caller->callees mapping into a set of (caller, callee) tuples."""
    edges = set()
    for caller, callees in cg.items():
        for callee in callees:
            edges.add((caller, callee))
    return edges


def run_tracer(language, program_dir, timeout=30):
    """Invoke the language-specific tracer; return parsed call graph dict or None."""
    cmd = LANGUAGE_TRACERS[language] + [program_dir]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    output = result.stdout
    marker = output.find("===TRACE===")
    if marker < 0:
        return None
    try:
        return json.loads(output[marker + len("===TRACE===") :].strip())
    except json.JSONDecodeError:
        return None


def trace_program(language, program_id, check):
    """Trace a single program; optionally compare against the shipped GT."""
    program_dir = os.path.join(EXTRACT_DIR, language, program_id)
    if not os.path.isdir(program_dir):
        return {"program_id": program_id, "status": "missing"}

    cg = run_tracer(language, program_dir)
    if cg is None:
        return {"program_id": program_id, "status": "tracer_failed"}

    out_dir = os.path.join(OUTPUT_DIR, language)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{program_id}.json"), "w") as f:
        json.dump(cg, f, indent=2)

    record = {"program_id": program_id, "status": "ok", "edges": sum(len(v) for v in cg.values())}

    if check:
        gt_path = os.path.join(program_dir, "callgraph.json")
        if not os.path.isfile(gt_path):
            record["check"] = "no_gt"
        else:
            with open(gt_path) as f:
                gt_cg = json.load(f)
            traced_edges = edge_set_from_callgraph(cg)
            gt_edges = edge_set_from_callgraph(gt_cg)
            record["check"] = "match" if traced_edges == gt_edges else "differ"
            record["gt_edges"] = len(gt_edges)
            record["traced_edges"] = len(traced_edges)
            record["only_in_trace"] = len(traced_edges - gt_edges)
            record["only_in_gt"] = len(gt_edges - traced_edges)
    return record


def run_language(language, program_ids, check, limit):
    """Trace all programs in one language; print a progress line and a summary."""
    if limit:
        program_ids = program_ids[:limit]
    print(f"\n{'='*60}\n  {language.upper()}: {len(program_ids)} programs\n{'='*60}")

    rows = []
    counts = {"ok": 0, "tracer_failed": 0, "missing": 0, "match": 0, "differ": 0}
    for i, pid in enumerate(program_ids):
        rec = trace_program(language, pid, check)
        rows.append(rec)
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        if check and "check" in rec:
            counts[rec["check"]] = counts.get(rec["check"], 0) + 1
        suffix = f"  [{rec.get('check', '')}]" if check else ""
        print(f"  [{i+1:4d}/{len(program_ids)}] {pid:40s} {rec['status']}{suffix}")

    print(f"\n  Traced ok: {counts['ok']}  Failed: {counts['tracer_failed']}  Missing: {counts['missing']}")
    if check:
        print(f"  Match: {counts.get('match', 0)}  Differ: {counts.get('differ', 0)}")

    csv_path = os.path.join(DATA_DIR, f"tracer_results_{language}.csv")
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Saved per-program log to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test", "train"], default="test",
                        help="which split to trace (default: test)")
    parser.add_argument("--lang", choices=list(LANGUAGE_TRACERS) + ["all"], default="all",
                        help="restrict to one language (default: all three)")
    parser.add_argument("--check", action="store_true",
                        help="compare each tracer output against the shipped callgraph.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap programs per language (useful for smoke-testing)")
    args = parser.parse_args()

    ensure_corpus_extracted()
    split_ids = load_split_ids(args.split)
    languages = list(LANGUAGE_TRACERS) if args.lang == "all" else [args.lang]
    for language in languages:
        run_language(language, split_ids.get(language, []), args.check, args.limit)


if __name__ == "__main__":
    main()
