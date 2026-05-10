#!/usr/bin/env python3
"""Recompute edge-level P/R/F1 from raw GT + predictions and print a summary table.

Compares model predictions in ``data/results/<model>_<lang>_t0.0_r0.json``
against ground-truth call graphs in ``<gt-dir>/<lang>/<benchmark_id>/callgraph.json``.

Edges are treated as ``(caller, callee)`` tuples; metrics are micro-averaged
over all edges in the test split (TP/FP/FN summed across benchmarks, then
P, R, F1 computed once at the end).

Usage
-----
    # Default: all 10 paper models, paper layout
    python compute_metrics.py

    # Different GT or results dir
    python compute_metrics.py --gt-dir data/test --results-dir data/results

    # Only some models / languages
    python compute_metrics.py --models claude-opus-4-6 gpt-5_4
    python compute_metrics.py --languages python java

    # Don't restrict to the canonical 10; show every model file we find
    python compute_metrics.py --all-models
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import normalize_edges as normalization

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "results"
DEFAULT_GT_DIR = REPO_ROOT / "data" / "extracted"
DEFAULT_LANGS = ["python", "javascript", "java"]

# Canonical model ordering for the summary table. Each entry: (slug-on-disk, display name, block).
# block: "frontier" (upper) or "open" (lower).
PAPER_MODELS: list[tuple[str, str, str]] = [
    ("claude-opus-4-6",              "Claude Opus 4.6",              "frontier"),
    ("claude-sonnet-4_6",            "Claude Sonnet 4.6",            "frontier"),
    ("gpt-5_4",                      "GPT-5.4",                      "frontier"),
    ("gpt-5_4-mini",                 "GPT-5.4-mini",                 "frontier"),
    ("gemini-3_1-pro-preview",       "Gemini 3.1 Pro Preview",       "frontier"),
    ("deepseek-v3_2",                "DeepSeek v3.2",                "open"),
    ("llama-3_3-70b-instruct",       "Llama-3.3-70B-Instruct",       "open"),
    ("qwen2_5-coder-32b-instruct",   "Qwen2.5-Coder-32B-Instruct",   "open"),
    ("qwen2_5-coder-7b-instruct",    "Qwen2.5-Coder-7B-Instruct",    "open"),
    ("qwen2_5-coder-1_5b-instruct",  "Qwen2.5-Coder-1.5B-Instruct",  "open"),
]

RESULT_PATTERN = re.compile(r"^(?P<model>.+?)_(?P<lang>python|javascript|java)_t0\.0_r0\.json$")


def edges_from_graph(
    graph: dict,
    language: str | None = None,
    norm_modes: list[str] | None = None,
) -> tuple[set[tuple[str, str]], int]:
    """Flatten a {caller: [callees]} dict into a set of (caller, callee) edges.

    Active normalization modes are applied to each endpoint: rewrite
    transforms run first (e.g., canonicalize constructor edges), then
    drop predicates. Returns ``(edges, n_dropped)`` where ``n_dropped``
    counts edges removed by drop predicates.
    """
    out: set[tuple[str, str]] = set()
    n_dropped = 0
    if not isinstance(graph, dict):
        return out, 0
    modes = norm_modes or []
    use_norm = bool(modes) and language is not None
    for caller, callees in graph.items():
        if not isinstance(callees, list):
            continue
        new_caller = normalization.apply_transforms(caller, language, modes) if use_norm else caller
        caller_drop = use_norm and normalization.should_drop(new_caller, language, modes)
        for cee in callees:
            if caller_drop:
                n_dropped += 1
                continue
            new_cee = normalization.apply_transforms(cee, language, modes) if use_norm else cee
            if use_norm and normalization.should_drop(new_cee, language, modes):
                n_dropped += 1
                continue
            out.add((new_caller, new_cee))
    return out, n_dropped


def score_one(
    gt_dir: Path,
    results_path: Path,
    language: str,
    norm_modes: list[str] | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    """Score one (model, language) result file.

    Returns ``(tp, fp, fn, n_scored, n_missing, gt_dropped, pred_dropped)``.
    """
    with open(results_path) as f:
        data = json.load(f)

    tp = fp = fn = 0
    scored = 0
    missing = 0
    gt_dropped_total = 0
    pred_dropped_total = 0
    for bid, rec in data["results"].items():
        gt_path = gt_dir / language / bid / "callgraph.json"
        if not gt_path.is_file():
            missing += 1
            continue
        with open(gt_path) as f:
            gt_graph = json.load(f)
        gt_edges, gt_drop = edges_from_graph(gt_graph, language, norm_modes)
        pred_edges, pred_drop = edges_from_graph(rec.get("prediction") or {}, language, norm_modes)

        tp += len(gt_edges & pred_edges)
        fp += len(pred_edges - gt_edges)
        fn += len(gt_edges - pred_edges)
        scored += 1
        gt_dropped_total += gt_drop
        pred_dropped_total += pred_drop

    return tp, fp, fn, scored, missing, gt_dropped_total, pred_dropped_total


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def discover_models(results_dir: Path) -> list[str]:
    """Find every distinct model slug under results_dir/."""
    seen: set[str] = set()
    for p in sorted(results_dir.glob("*_t0.0_r0.json")):
        m = RESULT_PATTERN.match(p.name)
        if m:
            seen.add(m.group("model"))
    return sorted(seen)


def display_name_for(slug: str) -> str:
    for s, name, _ in PAPER_MODELS:
        if s == slug:
            return name
    # Fallback: humanise the slug.
    return slug.replace("_", ".").replace("-", " ")


def block_for(slug: str) -> str:
    for s, _, block in PAPER_MODELS:
        if s == slug:
            return block
    return "open"  # default ordering bucket for unknown models


def fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}"


def render_text_table(rows: list[dict], counts: dict[str, int]) -> str:
    langs = list(counts.keys())
    # column widths
    name_w = max(len(r["display"]) for r in rows + [{"display": "Model"}])
    headers = ["Model"]
    for lang in langs:
        n = counts[lang]
        label = lang.capitalize() if lang != "javascript" else "JavaScript"
        headers.append(f"{label} (n={n})")
    headers.append("Average")

    # find best F1 per language and for Avg, for bolding (mark with *)
    best_f1: dict[str, float] = {}
    for r in rows:
        for lang in langs + ["avg"]:
            f1 = r["per_lang"][lang]["f1"]
            if lang not in best_f1 or f1 > best_f1[lang]:
                best_f1[lang] = f1

    # build lines
    lines: list[str] = []

    def col_label_row() -> str:
        # 4 column groups (3 langs + avg), 3 sub-cols each (P R F1)
        parts = [f"{'':<{name_w}}"]
        for lang in langs:
            label = lang.capitalize() if lang != "javascript" else "JavaScript"
            n = counts[lang]
            parts.append(f"  {label} (n={n})".ljust(20))
        parts.append("  Average".ljust(20))
        return "  ".join(parts)

    def subhead_row() -> str:
        parts = [f"{'Model':<{name_w}}"]
        for _ in langs + ["avg"]:
            parts.append("    P     R    F1 ")
        return "  ".join(parts)

    sep_w = name_w + 2 + (4 * (4 * 3 + 6))
    lines.append(col_label_row())
    lines.append(subhead_row())
    lines.append("-" * (name_w + len(langs) * 22 + 22 + 4))

    cur_block = None
    for r in rows:
        if r["block"] != cur_block:
            label = "Frontier proprietary" if r["block"] == "frontier" else "Open-weight"
            lines.append(f"-- {label} " + "-" * 30)
            cur_block = r["block"]
        parts = [f"{r['display']:<{name_w}}"]
        for lang in langs + ["avg"]:
            m = r["per_lang"][lang]
            star_p = "*" if abs(m["f1"] - best_f1[lang]) < 1e-9 and lang != "avg_marker" else " "
            # Bold marker only on F1
            f1_str = fmt_pct(m["f1"])
            if abs(m["f1"] - best_f1[lang]) < 1e-9:
                f1_str = f"*{f1_str.strip()}*".rjust(6)
            else:
                f1_str = f1_str.rjust(6)
            parts.append(f" {fmt_pct(m['p'])} {fmt_pct(m['r'])} {f1_str}")
        lines.append("  ".join(parts))
    lines.append("")
    lines.append("(F1 marked with *...* is best in that column.)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR),
                    help="Directory containing <lang>/<benchmark_id>/callgraph.json")
    ap.add_argument("--languages", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Only score these model slugs (default: the canonical model set listed in PAPER_MODELS).")
    ap.add_argument("--all-models", action="store_true",
                    help="Score every model file found, not just the canonical set.")
    args = ap.parse_args()

    # Edge normalization is always applied.
    norm_modes = list(normalization.ALL_MODES)

    results_dir = Path(args.results_dir)
    gt_dir = Path(args.gt_dir)
    if not results_dir.is_dir():
        sys.exit(f"results-dir not found: {results_dir}")
    if not gt_dir.is_dir():
        sys.exit(f"gt-dir not found: {gt_dir}")

    # Pick model slugs to score.
    if args.models:
        slugs = args.models
    elif args.all_models:
        slugs = discover_models(results_dir)
    else:
        slugs = [s for s, _, _ in PAPER_MODELS]

    # Track per-language test sizes from the first model's result file.
    lang_counts: dict[str, int] = {lang: 0 for lang in args.languages}

    rows: list[dict] = []
    drop_summary: dict[str, dict[str, int]] = {l: {"gt": 0, "pred": 0} for l in args.languages}
    for slug in slugs:
        per_lang: dict[str, dict] = {}
        agg_tp = agg_fp = agg_fn = 0
        for lang in args.languages:
            path = results_dir / f"{slug}_{lang}_t0.0_r0.json"
            if not path.is_file():
                print(f"[skip] missing: {path.name}", file=sys.stderr)
                per_lang[lang] = {"p": 0.0, "r": 0.0, "f1": 0.0,
                                  "tp": 0, "fp": 0, "fn": 0, "n": 0}
                continue
            tp, fp, fn, n_scored, n_missing, gt_drop, pred_drop = score_one(
                gt_dir, path, lang, norm_modes=norm_modes)
            drop_summary[lang]["gt"] += gt_drop
            drop_summary[lang]["pred"] += pred_drop
            if n_missing:
                print(f"[warn] {slug}/{lang}: {n_missing} benchmarks had no GT in {gt_dir}",
                      file=sys.stderr)
            p, r, f = prf1(tp, fp, fn)
            per_lang[lang] = {"p": p, "r": r, "f1": f,
                              "tp": tp, "fp": fp, "fn": fn, "n": n_scored}
            agg_tp += tp
            agg_fp += fp
            agg_fn += fn
            if n_scored > lang_counts[lang]:
                lang_counts[lang] = n_scored
        # Macro-average across the 3 languages (matches paper "Average" column).
        ps = [per_lang[l]["p"] for l in args.languages]
        rs = [per_lang[l]["r"] for l in args.languages]
        fs = [per_lang[l]["f1"] for l in args.languages]
        n_langs = len(args.languages) or 1
        per_lang["avg"] = {
            "p": sum(ps) / n_langs,
            "r": sum(rs) / n_langs,
            "f1": sum(fs) / n_langs,
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "n": -1,
        }
        rows.append({
            "slug": slug,
            "display": display_name_for(slug),
            "block": block_for(slug),
            "per_lang": per_lang,
        })

    # Sort: frontier first then open, then preserve PAPER_MODELS order within block,
    # falling back to alphabetical for unknown models.
    paper_order = {s: i for i, (s, _, _) in enumerate(PAPER_MODELS)}
    rows.sort(key=lambda r: (
        0 if r["block"] == "frontier" else 1,
        paper_order.get(r["slug"], 1_000 + ord(r["slug"][0])),
    ))

    counts_for_render = {l: lang_counts[l] for l in args.languages}
    if norm_modes:
        print(f"[normalize] modes active: {', '.join(norm_modes)}")
        for lang in args.languages:
            d = drop_summary[lang]
            n_models = sum(
                1 for slug in slugs
                if (results_dir / f"{slug}_{lang}_t0.0_r0.json").is_file()
            ) or 1
            print(f"[normalize] {lang}: dropped {d['gt'] // n_models:,} GT edges and "
                  f"{d['pred']:,} pred edges (summed across {n_models} models)")
        print()
    print(render_text_table(rows, counts_for_render))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
