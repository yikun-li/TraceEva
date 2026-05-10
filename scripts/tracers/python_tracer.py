#!/usr/bin/env python3
"""
Dynamic call tracer for Python programs.

Attaches to the interpreter via sys.settrace, executes the program's
main.py end-to-end, and emits the observed caller->callee edge set
under a unified JSON schema. Handles multi-file imports, classes,
nested functions, lambdas, decorators, closures, __init__, and MRO.

Usage:
    python scripts/tracers/python_tracer.py <program_dir>
    (pass the program DIRECTORY containing main.py, not a single file)

Output: JSON call graph after ===TRACE=== marker.
"""

import ast
import json
import os
import sys
import types


def trace_benchmark(benchmark_dir):
    """
    Trace a Python benchmark directory.
    Sets up sys.path so multi-file imports work, then traces execution.
    """
    main_py = os.path.join(benchmark_dir, "main.py")
    if not os.path.isfile(main_py):
        return {}

    with open(main_py) as f:
        main_code = f.read()

    # Collect all .py module files for name mapping
    module_files = {}
    for root, dirs, files in os.walk(benchmark_dir):
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), benchmark_dir)
                mod_name = rel.replace(os.sep, ".").replace("/__init__", "").replace(".py", "")
                # Handle __init__.py in subdirectories
                if mod_name.endswith(".__init__"):
                    mod_name = mod_name[:-9]  # Remove .__init__
                module_files[mod_name] = os.path.join(root, f)

    # Build qualified name map from AST for ALL source files
    name_map = build_name_map(benchmark_dir, module_files)

    # Trace execution
    call_edges = run_with_trace(main_py, benchmark_dir, name_map)

    # Build call graph
    return build_callgraph(call_edges, name_map)


def build_name_map(benchmark_dir, module_files):
    """
    Build a mapping from (filename, code_object_name) -> fully qualified name
    by parsing ASTs of all source files.
    """
    name_map = {}

    for mod_name, filepath in module_files.items():
        with open(filepath) as f:
            try:
                source = f.read()
                tree = ast.parse(source, filename=filepath)
            except SyntaxError:
                continue

        abs_path = os.path.abspath(filepath)

        # Module-level: add the module itself as a node.
        name_map[(abs_path, "<module>")] = mod_name
        # Also add as a "leaf" so modules with no calls still appear
        name_map[("__module__", mod_name)] = mod_name

        # Walk AST to find all function/method definitions
        _walk_ast(tree, mod_name, abs_path, name_map)

    return name_map


def _walk_ast(node, prefix, filepath, name_map):
    """Recursively walk AST and build qualified name mappings."""
    lambda_count = [0]  # mutable counter for lambdas

    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_prefix = f"{prefix}.{child.name}"
            for item in child.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual_name = f"{class_prefix}.{item.name}"
                    name_map[(filepath, item.name, item.lineno)] = qual_name
                    _walk_nested(item, qual_name, filepath, name_map)

        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qual_name = f"{prefix}.{child.name}"
            name_map[(filepath, child.name, child.lineno)] = qual_name
            _walk_nested(child, qual_name, filepath, name_map)

    # Also find module-level lambdas (not inside functions)
    lambda_count = [0]
    _find_lambdas(node, prefix, filepath, name_map, lambda_count, skip_nested=True)


def _find_lambdas(node, prefix, filepath, name_map, lambda_count, skip_nested=False):
    """Find lambda nodes and assign them scoped names."""
    for child in ast.iter_child_nodes(node):
        if skip_nested and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # These are handled by _walk_nested with their own prefix
        if isinstance(child, ast.Lambda):
            lambda_count[0] += 1
            qual_name = f"{prefix}.<lambda{lambda_count[0]}>"
            name_map[(filepath, "<lambda>", child.lineno)] = qual_name
        else:
            _find_lambdas(child, prefix, filepath, name_map, lambda_count, skip_nested=skip_nested)


def _walk_nested(func_node, prefix, filepath, name_map):
    """Walk inside a function to find nested functions/classes and lambdas."""
    lambda_count = [0]
    for child in ast.walk(func_node):
        if child is func_node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qual_name = f"{prefix}.{child.name}"
            name_map[(filepath, child.name, child.lineno)] = qual_name
        elif isinstance(child, ast.ClassDef):
            class_prefix = f"{prefix}.{child.name}"
            for item in child.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name_map[(filepath, item.name, item.lineno)] = f"{class_prefix}.{item.name}"
        elif isinstance(child, ast.Lambda):
            lambda_count[0] += 1
            qual_name = f"{prefix}.<lambda{lambda_count[0]}>"
            name_map[(filepath, "<lambda>", child.lineno)] = qual_name


def resolve_qualname(frame, name_map, benchmark_dir):
    """
    Resolve a frame to its fully qualified name using multiple strategies.
    """
    code = frame.f_code
    filename = os.path.abspath(code.co_filename)
    func_name = code.co_name
    lineno = code.co_firstlineno

    # Strategy 1: Exact match (filename, name, lineno)
    key = (filename, func_name, lineno)
    if key in name_map:
        return name_map[key]

    # Strategy 2: Use code.co_qualname (Python 3.11+)
    qualname = getattr(code, 'co_qualname', None)
    if qualname:
        # Convert Python's co_qualname into the unified module.class.method form.
        rel_path = os.path.relpath(filename, benchmark_dir)
        mod_name = rel_path.replace(os.sep, ".").replace("/__init__", "").replace(".py", "")
        cleaned = qualname.replace(".<locals>.", ".").replace(".<locals>", "")
        if cleaned == "<module>":
            return mod_name

        # Handle lambdas: number them by the scope prefix + call order
        if "<lambda>" in cleaned:
            if not hasattr(resolve_qualname, '_lambda_counter'):
                resolve_qualname._lambda_counter = {}
                resolve_qualname._lambda_call_order = 0
            counter = resolve_qualname._lambda_counter

            # Use (filename, lineno, code_id) as unique key for each lambda
            # code_id distinguishes multiple lambdas on the same line
            lambda_id = (filename, lineno, id(code))
            if lambda_id not in counter:
                resolve_qualname._lambda_call_order += 1
                # Scope prefix: everything before <lambda>
                # e.g., "func.<lambda>" -> scope is "func."
                scope_prefix = cleaned.rsplit("<lambda>", 1)[0]  # e.g., "" or "func."
                counter[lambda_id] = (scope_prefix, resolve_qualname._lambda_call_order)

            scope_prefix, num = counter[lambda_id]
            cleaned = f"{scope_prefix}<lambda{num}>"

        return f"{mod_name}.{cleaned}"

    # Strategy 3: Derive from filename + function name
    if filename.startswith(os.path.abspath(benchmark_dir)):
        rel_path = os.path.relpath(filename, benchmark_dir)
        mod_name = rel_path.replace(os.sep, ".").replace("/__init__", "").replace(".py", "")

        if func_name == "<module>":
            return mod_name

        # Try to find class context from locals
        if 'self' in frame.f_locals:
            obj = frame.f_locals['self']
            class_name = type(obj).__name__
            return f"{mod_name}.{class_name}.{func_name}"
        elif 'cls' in frame.f_locals:
            cls = frame.f_locals['cls']
            class_name = cls.__name__
            return f"{mod_name}.{class_name}.{func_name}"

        return f"{mod_name}.{func_name}"

    # Strategy 4: Lambda
    if func_name == "<lambda>":
        rel_path = os.path.relpath(filename, benchmark_dir) if filename.startswith(os.path.abspath(benchmark_dir)) else None
        if rel_path:
            mod_name = rel_path.replace(os.sep, ".").replace(".py", "")
            return f"{mod_name}.<lambda>"

    return None


def run_with_trace(main_py, benchmark_dir, name_map):
    """
    Execute main.py with sys.path set up for imports, tracing all calls.
    """
    abs_benchmark = os.path.abspath(benchmark_dir)
    abs_main = os.path.abspath(main_py)
    call_edges = []
    lambda_counter = {}  # Track lambda numbering per file

    def tracer(frame, event, arg):
        if event != "call":
            return tracer

        callee_file = os.path.abspath(frame.f_code.co_filename)
        caller_frame = frame.f_back
        is_eval_callee = callee_file == "<string>"
        is_benchmark_callee = callee_file.startswith(abs_benchmark)

        # Only track calls within our benchmark or from eval/exec
        if not is_benchmark_callee and not is_eval_callee:
            return tracer

        callee_name = resolve_qualname(frame, name_map, benchmark_dir)
        if not callee_name:
            return tracer

        if caller_frame:
            caller_file = os.path.abspath(caller_frame.f_code.co_filename)

            if caller_file.startswith(abs_benchmark):
                caller_name = resolve_qualname(caller_frame, name_map, benchmark_dir)
            elif caller_frame.f_code.co_filename in ("<string>", "<stdin>"):
                # eval/exec: walk up the stack to find the real benchmark caller
                caller_name = None
                f = caller_frame.f_back
                while f:
                    ff = os.path.abspath(f.f_code.co_filename)
                    if ff.startswith(abs_benchmark):
                        caller_name = resolve_qualname(f, name_map, benchmark_dir)
                        break
                    f = f.f_back
            else:
                caller_name = None
        else:
            caller_name = None

        if caller_name and callee_name and caller_name != callee_name:
            # Filter out class body execution: when a module "calls" a class
            # definition in the same module, it is class creation, not a real call.
            callee_co_name = frame.f_code.co_name
            is_class_body = (callee_co_name != "<module>"
                             and not callee_co_name.startswith("<")
                             and frame.f_code.co_qualname
                             and "." not in getattr(frame.f_code, 'co_qualname', callee_co_name)
                             and callee_co_name[0].isupper()
                             and caller_name == callee_name.rsplit(".", 1)[0])
            if not is_class_body:
                call_edges.append((caller_name, callee_name))

        return tracer

    # Set up sys.path for multi-file imports
    old_path = sys.path[:]
    old_modules = dict(sys.modules)

    # Add benchmark dir and its parent to path (for package imports)
    sys.path.insert(0, abs_benchmark)
    parent = os.path.dirname(abs_benchmark)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Also add any subdirectories that have __init__.py
    for root, dirs, files in os.walk(abs_benchmark):
        if "__init__.py" in files:
            pkg_parent = os.path.dirname(root)
            if pkg_parent not in sys.path:
                sys.path.insert(0, pkg_parent)

    old_trace = sys.gettrace()
    old_cwd = os.getcwd()
    sys.settrace(tracer)

    try:
        # chdir into the benchmark dir so any relative-path file writes
        # made by the traced code stay contained in tmp_dir, never leaking
        # into the project root (which would clobber .env, etc.).
        os.chdir(abs_benchmark)
        with open(main_py) as f:
            code = f.read()
        compiled = compile(code, abs_main, "exec")
        exec(compiled, {"__name__": "__main__", "__file__": abs_main})
    except Exception:
        # Programs that crash mid-execution still yield a partial trace from
        # the edges captured before the exception; the validation pass rejects
        # programs whose final edge count falls below the acceptance threshold.
        pass
    finally:
        sys.settrace(old_trace)
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
        # Restore sys.path and clean up loaded modules
        sys.path[:] = old_path
        # Remove modules that were loaded from benchmark dir
        for mod_name in list(sys.modules.keys()):
            if mod_name not in old_modules:
                del sys.modules[mod_name]

    return call_edges


def build_callgraph(call_edges, name_map):
    """Convert traced edges into the unified caller->callees JSON schema."""
    cg = {}
    all_funcs = set()

    for caller, callee in call_edges:
        all_funcs.add(caller)
        all_funcs.add(callee)

        if caller not in cg:
            cg[caller] = []
        if callee not in cg[caller]:
            cg[caller].append(callee)

    # Ensure all functions have entries (from traced calls)
    for func in all_funcs:
        if func not in cg:
            cg[func] = []

    # Also add ALL defined functions from name_map as leaf nodes
    # This ensures functions that exist but make no calls show up with []
    for key, qual_name in name_map.items():
        if qual_name not in cg:
            cg[qual_name] = []

    return cg


def normalize_cg(cg):
    """Normalize call graph names by stripping <locals> qualifiers."""
    import re
    normalized = {}
    for caller, callees in cg.items():
        norm_caller = re.sub(r'\.<locals>\.', '.', caller)
        norm_callees = []
        for callee in callees:
            norm_callee = re.sub(r'\.<locals>\.', '.', callee)
            if norm_callee not in norm_callees:
                norm_callees.append(norm_callee)
        normalized[norm_caller] = norm_callees
    return normalized


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/tracers/python_tracer.py <program_dir>")
        sys.exit(1)

    cg = trace_benchmark(sys.argv[1])
    cg = normalize_cg(cg)
    print("===TRACE===")
    print(json.dumps(cg, indent=2))
