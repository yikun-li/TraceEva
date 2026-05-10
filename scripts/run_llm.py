#!/usr/bin/env python3
"""
Zero-shot LLM evaluation runner for the \\bench corpus via OpenRouter (or any
OpenAI-compatible chat-completion endpoint).

For each program, the runner reads the ground-truth callers and asks the LLM
to identify the callees of each function. The output is a per-program
JSON edge-list that compute_metrics.py scores against the tracer-witnessed
ground truth.

Requires: OPENROUTER_API_KEY environment variable (or a .env file).

Usage:
    python run_llm.py <program_or_split_dir> --model <model_id> [--summary]

Supported model IDs (via OpenRouter):
    anthropic/claude-opus-4-6
    anthropic/claude-sonnet-4-6
    openai/gpt-5.4
    openai/gpt-5.4-mini
    google/gemini-3.1-pro
    deepseek/deepseek-v3.2
    meta-llama/llama-3.3-70b-instruct
    (any OpenRouter model ID)

Local vLLM / Ollama server:
    python run_llm.py <program_or_split_dir> \\
        --model Qwen/Qwen2.5-Coder-32B-Instruct \\
        --api-url http://localhost:8001/v1/chat/completions \\
        --summary
"""

import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def get_api_key(required=True):
    """Get OpenRouter API key from environment or .env file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    # Try .env file
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")

    if required:
        print("Error: OPENROUTER_API_KEY not set.")
        print("Set it via: export OPENROUTER_API_KEY=sk-or-...")
        print("Or add to .env file: OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)
    return None


def call_llm_api(messages, model, api_key=None, api_url=None, temperature=0.0, max_tokens=4096):
    """Call an OpenAI-compatible API (OpenRouter, vLLM, Ollama, etc.)."""
    if api_url:
        url = api_url
    else:
        url = "http://localhost:8000/api/llm-proxy/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        # Tag requests in the proxy/OpenRouter dashboard so usage from this
        # script is easy to filter and bill against.
        "X-App-Name": "TraceEval-Eval",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if not api_url:
        headers["HTTP-Referer"] = "https://anonymous.4open.science/r/traceeval"

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # Send both spellings; some providers/proxies key on one or the other.
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
    }
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429 and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    API error {e.code}: {body[:200]}")
            return ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"    Request failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# Language detection & benchmark discovery
# ---------------------------------------------------------------------------

def detect_language(benchmark_dir):
    for root, dirs, files in os.walk(benchmark_dir):
        for f in files:
            if f.endswith(".py"):
                return "python"
            if f.endswith(".js"):
                return "javascript"
            if f.endswith(".java"):
                return "java"
    return "unknown"


def get_language_extension(language):
    return {"python": "py", "javascript": "js", "java": "java"}.get(language, "py")


def is_benchmark_dir(path):
    if not os.path.isfile(os.path.join(path, "callgraph.json")):
        return False
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith((".py", ".js", ".java")):
                return True
    return False


def find_benchmarks(target):
    benchmarks = []
    if is_benchmark_dir(target):
        return [target]
    for entry in sorted(os.listdir(target)):
        full_path = os.path.join(target, entry)
        if os.path.isdir(full_path):
            benchmarks.extend(find_benchmarks(full_path))
    return benchmarks


def load_ground_truth(benchmark_dir):
    with open(os.path.join(benchmark_dir, "callgraph.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt generation: caller-conditioned callee identification
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in {language} programming. You will examine and identify "
    "the function calls in the given code. You must examine the code in detail by "
    "resolving aliases, tracking variable assignments, following return values, and "
    "understanding inheritance/method resolution."
)

PYTHON_EXAMPLE = """**Example Python Code**:
```main.py
def return_func():
    func()

def func():
    a = return_func
    return a

a = func
a()()
```

**Example Questions**:
1. What are the module-level function calls in the file "main.py"?
2. What are the function calls inside the "main.return_func" function in the file "main.py"?
3. What are the function calls inside the "main.func" function in the file "main.py"?

**Example Answers**:
1. main.func, main.return_func
2. main.func
3."""

JS_EXAMPLE = """**Example JavaScript Code**:
```main.js
function returnFunc() {
    func();
}

function func() {
    a = returnFunc;
    return a;
}

a = func;
a()();
```

**Example Questions**:
1. What are the module-level function calls in the file "main.js"?
2. What are the function calls inside the "main.returnFunc" function in the file "main.js"?
3. What are the function calls inside the "main.func" function in the file "main.js"?

**Example Answers**:
1. main.func, main.returnFunc
2. main.func
3."""

JAVA_EXAMPLE = """**Example Java Code**:
```java
// vc/Class.java
package vc;

class Class {
    public void target(){ }
    public static void main(String[] args){
        Class cls = new Class();
        cls.target();
    }
}
```

**Example Questions**:
1. What are the target functions invoked by vc.Class:main(java.lang.String[]) in the vc.Class class?

**Example Answers**:
1. vc.Class:target()"""

USER_PROMPT = """## Task Description

**Objective**: Examine the given {language} code and identify the function calls that occur when this program is executed, then answer the questions.

**Instructions**:
1. For each question, list the function calls as a comma-separated list.
2. Do not include additional explanations or commentary.
3. Include both explicit and implicit function calls (e.g., __init__ when an object is created).
4. If a function is called through an alias or variable, resolve it to the actual function being called.
5. If a passed argument is not invoked within the function, do not include it.
6. If there are no function calls, leave the answer empty.
7. **IMPORTANT**: Always use fully qualified names with the module prefix. For example, use "main.MyClass.func" not "MyClass.func". The module name is the filename without extension (e.g., "main.py" → "main", "to_import.py" → "to_import").

**Format for Answers**:
- Provide your answer next to each question number.
- Do not include the questions in your answer.
- Example:
    1. module.func1, module.func2
    2. module.func3
    3.

{example}

**{language} Code Provided**:

{code}

**Questions**:
{questions}

**Answers**:"""


def gather_code(benchmark_dir, language):
    """Collect all source files as formatted code blocks."""
    ext = get_language_extension(language)
    code = ""
    for root, dirs, files in os.walk(benchmark_dir):
        for f in sorted(files):
            if f.endswith(f".{ext}"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, benchmark_dir)
                with open(full_path) as fh:
                    content = fh.read()
                code += f"```{rel_path}\n{content}\n```\n\n"
    return code


def normalize_path(path):
    normalized = path.replace("\\", ".").replace("/", ".")
    return normalized.rsplit(".", 1)[0]


def generate_questions(gt, benchmark_dir, language):
    """Generate one prompt per ground-truth caller, asking the model to fill in callees."""
    questions = []

    if language == "java":
        for caller in gt:
            if ":" in caller:
                class_name = caller.split(":")[0]
                questions.append(
                    f"What are the target functions invoked by {caller} in the {class_name} class?"
                )
    else:
        # Python / JavaScript
        file_map = {}
        ext = get_language_extension(language)
        for root, dirs, files in os.walk(benchmark_dir):
            for f in files:
                if f.endswith(f".{ext}"):
                    rel = os.path.relpath(os.path.join(root, f), benchmark_dir)
                    normalized = normalize_path(rel)
                    file_map[normalized] = rel

        init_file = "__init__" if language == "python" else "index"
        default_file = file_map.get("main", list(file_map.values())[0] if file_map else "main.py")

        for key in gt:
            file_name = None
            # Try to match key to a file
            if key in file_map:
                file_name = file_map[key]
            else:
                # Check if key is a prefix of any file
                for fk, fv in file_map.items():
                    if key.startswith(fk):
                        file_name = fv
                        break
                # Check init files
                if not file_name:
                    init_key = f"{key}.{init_file}"
                    if init_key in file_map:
                        file_name = file_map[init_key]

            if not file_name:
                file_name = default_file

            normalized_file = normalize_path(file_name)
            if key == normalized_file or f"{key}.{init_file}" == normalized_file:
                questions.append(
                    f"What are the module-level function calls in the file \"{file_name}\"?"
                )
            else:
                questions.append(
                    f"What are the function calls inside the \"{key}\" function in the file \"{file_name}\"?"
                )

    # Number them
    return [f"{i}. {q}" for i, q in enumerate(questions, 1)]


def build_prompt(benchmark_dir, gt, language):
    """Build the full prompt messages for the LLM."""
    code = gather_code(benchmark_dir, language)
    questions = generate_questions(gt, benchmark_dir, language)

    if language == "python":
        example = PYTHON_EXAMPLE
    elif language == "javascript":
        example = JS_EXAMPLE
    elif language == "java":
        example = JAVA_EXAMPLE
    else:
        example = PYTHON_EXAMPLE

    lang_name = language.capitalize()

    system_msg = SYSTEM_PROMPT.format(language=lang_name)
    user_msg = USER_PROMPT.format(
        language=lang_name,
        example=example,
        code=code,
        questions="\n".join(questions),
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _split_callees(answer):
    """Split a comma-separated callee list into individual callees.

    Splits on commas at paren / angle-bracket depth 0 only, so that
    Java/Python signatures with multi-arg parens like ``Foo:bar(int,int)``
    or generic types like ``Map<String, Integer>`` survive intact.

    The previous implementation used a naive ``answer.split(",")`` which
    broke every signature containing commas: e.g. ``Foo:bar(int,int)``
    became two FP fragments ``Foo:bar(int`` and ``int)``.
    """
    out = []
    depth = 0
    cur = []
    for ch in answer:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def parse_llm_response(response, gt):
    """Parse numbered answers back into call graph format."""
    cg = {}
    pattern = re.compile(r"(\d+)\.\s*(.*)")

    parsed = {}
    for line in response.split("\n"):
        match = pattern.match(line.strip())
        if match:
            num, answer = int(match.group(1)), match.group(2).strip()
            parsed[num] = answer

    gt_keys = list(gt.keys())
    for i, key in enumerate(gt_keys, 1):
        if i in parsed:
            answer = parsed[i]
            if answer == "" or answer.lower() in ("none", "n/a", "no function calls", "no calls"):
                cg[key] = []
            else:
                callees = [c.strip() for c in _split_callees(answer) if c.strip()]
                cg[key] = callees
        else:
            cg[key] = []

    return cg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def to_edges(cg):
    edges = set()
    for caller, callees in cg.items():
        for callee in callees:
            edges.add((caller, callee))
    return edges


def compute_metrics(ground_truth, result):
    gt_edges = to_edges(ground_truth)
    result_edges = to_edges(result)

    tp = gt_edges & result_edges
    fp = result_edges - gt_edges
    fn = gt_edges - result_edges

    precision = len(tp) / len(result_edges) if result_edges else 1.0
    recall = len(tp) / len(gt_edges) if gt_edges else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
    }


def measure_exact_matches(actual, expected):
    num_all = 0
    num_exact = 0
    for node in expected:
        expected_items = expected[node]
        actual_items = actual.get(node, None)
        if not expected_items:
            num_all += 1
            if actual_items is not None and actual_items == []:
                num_exact += 1
            continue
        num_all += len(expected_items)
        for item in expected_items:
            if actual_items and item in actual_items:
                num_exact += 1
    return num_exact, num_all


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def parse_readme_provenance(readme_path):
    """Extract source_repo + original_file from a benchmark's README.md."""
    if not os.path.isfile(readme_path):
        return None, None
    try:
        with open(readme_path) as f:
            content = f.read()
    except Exception:
        return None, None
    src = re.search(r"^Source:\s*(.+)$", content, re.MULTILINE)
    orig = re.search(r"^Original file:\s*(.+)$", content, re.MULTILINE)
    return (
        src.group(1).strip() if src else None,
        orig.group(1).strip() if orig else None,
    )


def run_single_benchmark(benchmark_dir, language, model, api_key=None, api_url=None, verbose=True, temperature=0.0, run_id=0):
    name = os.path.relpath(benchmark_dir)
    gt = load_ground_truth(benchmark_dir)

    # No per-benchmark cache file is written into the benchmark folder.
    # The canonical output is the aggregate JSON at data/results/<model>_<lang>_t<T>_r<R>.json
    # written by main() after the run loop.
    #
    # If the API fails (returns empty) or raises, we re-raise so the
    # worker pool's `except as_completed` skips saving this benchmark.
    # That way, resume will naturally retry it on the next run instead of
    # treating an API failure as a legitimate "model predicted nothing".
    messages = build_prompt(benchmark_dir, gt, language)
    response = call_llm_api(messages, model, api_key=api_key, api_url=api_url, temperature=temperature)
    if not response:
        raise RuntimeError(f"LLM API returned empty response for {name}")
    result = parse_llm_response(response, gt)

    metrics = compute_metrics(gt, result)
    exact, total = measure_exact_matches(result, gt)

    # Build rich per-item record for the aggregate JSON output. We also
    # store the raw response text so any future parser bug is fixable
    # without re-running the LLM (the original parser silently mangled
    # multi-arg signatures via a naive comma split: see _split_callees).
    bench_id = os.path.basename(os.path.normpath(benchmark_dir))
    src_repo, orig_file = parse_readme_provenance(os.path.join(benchmark_dir, "README.md"))
    num_gt_edges = sum(len(callees) for callees in gt.values())
    record = {
        "benchmark_id": bench_id,
        "benchmark_dir": name,
        "category": Path(benchmark_dir).parent.name if is_benchmark_dir(benchmark_dir) else None,
        "input": {
            "source_repo": src_repo,
            "original_file": orig_file,
            "num_functions": len(gt),
            "num_gt_edges": num_gt_edges,
        },
        "raw_response": response,
        "ground_truth": gt,
        "prediction": result,
        "metrics": {
            "tp": len(metrics["true_positives"]),
            "fp": len(metrics["false_positives"]),
            "fn": len(metrics["false_negatives"]),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "edge_diff": {
            "true_positives": [list(e) for e in metrics["true_positives"]],
            "false_positives": [list(e) for e in metrics["false_positives"]],
            "false_negatives": [list(e) for e in metrics["false_negatives"]],
        },
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")

        print(f"\n--- Ground Truth ---")
        for caller, callees in gt.items():
            if callees:
                for callee in callees:
                    print(f"  {caller}  -->  {callee}")
            else:
                print(f"  {caller}  -->  (none)")

        print(f"\n--- LLM Result ({model}) ---")
        if result:
            for caller, callees in result.items():
                if callees:
                    for callee in callees:
                        print(f"  {caller}  -->  {callee}")
                else:
                    print(f"  {caller}  -->  (none)")
        else:
            print(f"  (empty)")

        print(f"\n--- Evaluation ---")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1 Score:  {metrics['f1']:.2%}")
        print(f"  TP={len(metrics['true_positives'])}  FP={len(metrics['false_positives'])}  FN={len(metrics['false_negatives'])}")
        print(f"  Edges:     {exact}/{total}")

        if metrics["false_positives"]:
            print(f"  FP: {metrics['false_positives'][:5]}{'...' if len(metrics['false_positives']) > 5 else ''}")
        if metrics["false_negatives"]:
            print(f"  FN: {metrics['false_negatives'][:5]}{'...' if len(metrics['false_negatives']) > 5 else ''}")
        if metrics["f1"] == 1.0:
            print(f"  PERFECT MATCH!")

    return metrics, exact, total, record


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-shot LLM evaluation on the \\bench corpus via OpenRouter"
    )
    parser.add_argument("target", help="Benchmark directory or category path")
    parser.add_argument(
        "--model", default="anthropic/claude-opus-4-6",
        help="OpenRouter model ID (default: anthropic/claude-opus-4-6)"
    )
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument(
        "--api-url", default=None,
        help="Custom API endpoint URL (e.g., http://localhost:8001/v1/chat/completions for vLLM)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs for variance estimation (default: 1). Uses different cache per run."
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel LLM workers per category (default: 16). Set to 1 for sequential."
    )
    args = parser.parse_args()

    # API key is required for OpenRouter, optional for local vLLM/Ollama
    api_key = get_api_key(required=(args.api_url is None))
    verbose = not args.summary

    benchmarks = find_benchmarks(args.target)
    if not benchmarks:
        print(f"No benchmarks found in {args.target}")
        sys.exit(1)

    language = detect_language(benchmarks[0])
    model_short = args.model.split("/")[-1]
    print(f"Found {len(benchmarks)} benchmarks (language: {language}, model: {args.model}, temp: {args.temperature}, runs: {args.runs})")

    # Group by category
    categories = defaultdict(list)
    for b in benchmarks:
        if is_benchmark_dir(b):
            cat = Path(b).parent.name
        else:
            cat = "root"
        categories[cat].append(b)

    # Collect results across runs
    all_run_results = []

    from datetime import datetime, timezone

    def aggregate_payload(records, cat_results, run_id):
        """Build the full aggregate JSON payload from current state."""
        gt = sum(r["metrics"]["tp"] for r in records.values())
        gf = sum(r["metrics"]["fp"] for r in records.values())
        gn = sum(r["metrics"]["fn"] for r in records.values())
        p = gt / (gt + gf) if (gt + gf) > 0 else 0
        r = gt / (gt + gn) if (gt + gn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {
            "metadata": {
                "model": args.model,
                "language": language,
                "temperature": args.temperature,
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "num_benchmarks": len(records),
                "aggregate_metrics": {
                    "tp": gt, "fp": gf, "fn": gn,
                    "precision": p, "recall": r, "f1": f1,
                },
                "cat_results": cat_results,
            },
            "results": records,
        }

    def write_aggregate(agg_path, payload):
        """Atomic-ish write: tmp file + rename, so partial writes don't corrupt."""
        tmp_path = agg_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, agg_path)

    for run_id in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {run_id + 1}/{args.runs} ---")

        results_dir = os.path.join(SCRIPT_DIR, "data", "results")
        os.makedirs(results_dir, exist_ok=True)
        agg_path = os.path.join(
            results_dir,
            f"{model_short.replace('.', '_')}_{language}_t{args.temperature}_r{run_id}.json",
        )

        # Resume support: load any existing partial results so we can skip them.
        run_records = {}
        if os.path.isfile(agg_path):
            try:
                with open(agg_path) as f:
                    prior = json.load(f)
                run_records = prior.get("results", {})
                if run_records:
                    print(f"[resume] {len(run_records)} benchmarks already in {agg_path}, will skip")
            except Exception as e:
                print(f"[resume] couldn't parse existing {agg_path} ({e}); starting fresh")
                run_records = {}

        cat_results = {}
        grand_exact, grand_total = 0, 0
        grand_tp, grand_fp, grand_fn = 0, 0, 0

        # Lock guarding run_records updates and the atomic JSON write.
        # The thread pool only parallelizes the LLM API call; everything that
        # touches shared state goes through this lock.
        state_lock = threading.Lock()

        def _evaluate(bench_dir):
            return bench_dir, run_single_benchmark(
                bench_dir, language, args.model, api_key=api_key,
                api_url=args.api_url, verbose=verbose,
                temperature=args.temperature, run_id=run_id,
            )

        for cat in sorted(categories):
            cat_exact, cat_total, cat_perfect, cat_count = 0, 0, 0, 0
            cat_tp, cat_fp, cat_fn = 0, 0, 0

            # Partition into already-done (resume) vs to-evaluate
            todo = []
            for bench_dir in sorted(categories[cat]):
                bench_id = os.path.basename(os.path.normpath(bench_dir))
                if bench_id in run_records:
                    rec = run_records[bench_id]
                    cat_exact += rec["metrics"]["tp"]
                    cat_total += rec["input"]["num_gt_edges"]
                    cat_count += 1
                    cat_tp += rec["metrics"]["tp"]
                    cat_fp += rec["metrics"]["fp"]
                    cat_fn += rec["metrics"]["fn"]
                    if rec["metrics"]["f1"] == 1.0:
                        cat_perfect += 1
                else:
                    todo.append(bench_dir)

            if todo:
                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                    futures = {pool.submit(_evaluate, b): b for b in todo}
                    for fut in as_completed(futures):
                        try:
                            _, (metrics, exact, total, record) = fut.result()
                        except Exception as e:
                            print(f"\n[!] worker exception: {e}")
                            continue
                        with state_lock:
                            cat_exact += exact
                            cat_total += total
                            cat_count += 1
                            cat_tp += len(metrics["true_positives"])
                            cat_fp += len(metrics["false_positives"])
                            cat_fn += len(metrics["false_negatives"])
                            if metrics["f1"] == 1.0:
                                cat_perfect += 1
                            run_records[record["benchmark_id"]] = record
                            # Incremental save (atomic). Hold the lock so writes
                            # don't race with each other.
                            write_aggregate(
                                agg_path,
                                aggregate_payload(run_records, cat_results, run_id),
                            )

            cat_results[cat] = {
                "perfect": cat_perfect, "count": cat_count,
                "exact": cat_exact, "total": cat_total,
                "tp": cat_tp, "fp": cat_fp, "fn": cat_fn,
            }
            grand_exact += cat_exact
            grand_total += cat_total
            grand_tp += cat_tp
            grand_fp += cat_fp
            grand_fn += cat_fn

        grand_p = grand_tp / (grand_tp + grand_fp) if (grand_tp + grand_fp) > 0 else 0
        grand_r = grand_tp / (grand_tp + grand_fn) if (grand_tp + grand_fn) > 0 else 0
        grand_f1 = 2 * grand_p * grand_r / (grand_p + grand_r) if (grand_p + grand_r) > 0 else 0

        all_run_results.append({
            "cat_results": cat_results,
            "grand_tp": grand_tp, "grand_fp": grand_fp, "grand_fn": grand_fn,
            "grand_exact": grand_exact, "grand_total": grand_total,
            "soundness": grand_r, "completeness": grand_p,
        })

        # Final write with the full cat_results populated
        write_aggregate(agg_path, aggregate_payload(run_records, cat_results, run_id))
        print(f"\n[+] Aggregate results: {agg_path} ({len(run_records)} benchmarks)")

        # Print per-run summary
        print(f"\n{'='*100}")
        print(f"LLM RESULTS {'RUN ' + str(run_id+1) + ' ' if args.runs > 1 else ''}SUMMARY ({model_short.upper()}, {language.upper()}, temp={args.temperature})")
        print(f"{'='*100}")
        print(f"{'Category':<25} {'Perfect':>10} {'Edges':>12} {'TP':>5} {'FP':>5} {'FN':>5} {'Sound':>7} {'Comp':>7}")
        print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*7}")

        total_perfect, total_count = 0, 0
        for cat in sorted(cat_results):
            r = cat_results[cat]
            p = r['tp'] / (r['tp'] + r['fp']) if (r['tp'] + r['fp']) > 0 else 0
            rc = r['tp'] / (r['tp'] + r['fn']) if (r['tp'] + r['fn']) > 0 else 0
            print(f"{cat:<25} {r['perfect']:>4}/{r['count']:<5} {r['exact']:>5}/{r['total']:<5} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} {rc:>6.1%} {p:>6.1%}")
            total_perfect += r['perfect']
            total_count += r['count']

        print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*7}")
        print(f"{'TOTAL':<25} {total_perfect:>4}/{total_count:<5} {grand_exact:>5}/{grand_total:<5} {grand_tp:>5} {grand_fp:>5} {grand_fn:>5} {grand_r:>6.1%} {grand_p:>6.1%}")

    # Multi-run summary
    if args.runs > 1:
        from statistics import mean, stdev
        soundness_vals = [r["soundness"] for r in all_run_results]
        completeness_vals = [r["completeness"] for r in all_run_results]

        def fmt_pm(vals):
            m = mean(vals)
            sd = stdev(vals) if len(vals) >= 2 else 0
            return f"{m*100:.1f}+-{sd*100:.1f}%"

        print(f"\n{'='*100}")
        print(f"AGGREGATE ACROSS {args.runs} RUNS ({model_short.upper()}, {language.upper()}, temp={args.temperature})")
        print(f"{'='*100}")
        print(f"  Soundness:    {fmt_pm(soundness_vals)}")
        print(f"  Completeness: {fmt_pm(completeness_vals)}")
        print()

        # Per-run breakdown
        print(f"  {'Run':<8} {'Soundness':>12} {'Completeness':>14} {'TP':>6} {'FP':>6} {'FN':>6}")
        print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*6} {'-'*6} {'-'*6}")
        for i, r in enumerate(all_run_results):
            print(f"  Run {i+1:<3} {r['soundness']*100:>11.1f}% {r['completeness']*100:>13.1f}% {r['grand_tp']:>6} {r['grand_fp']:>6} {r['grand_fn']:>6}")
        print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*6} {'-'*6} {'-'*6}")
        print(f"  {'Mean':<8} {fmt_pm(soundness_vals):>12} {fmt_pm(completeness_vals):>14}")


if __name__ == "__main__":
    main()
