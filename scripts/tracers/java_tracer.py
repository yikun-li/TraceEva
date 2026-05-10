#!/usr/bin/env python3
"""
Dynamic call tracer for Java programs.

Instruments each method in the source by injecting an entry-tracking
call, compiles the result with javac, runs the main class, and emits
the observed caller->callee edge set under the unified JSON schema.

Usage:
    python scripts/tracers/java_tracer.py <program_dir>
    (the directory should contain .java files including a class with main())

Output: JSON call graph after ===TRACE=== marker.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import shutil

JAVA_CMD = os.environ.get("JAVA_HOME", "/opt/homebrew/opt/openjdk") + "/bin/java"
JAVAC_CMD = os.environ.get("JAVA_HOME", "/opt/homebrew/opt/openjdk") + "/bin/javac"

# Fallback: try system java
if not os.path.isfile(JAVA_CMD):
    JAVA_CMD = "java"
    JAVAC_CMD = "javac"


def find_java_files(benchmark_dir):
    """Find all .java files in the benchmark directory."""
    java_files = []
    for root, dirs, files in os.walk(benchmark_dir):
        for f in files:
            if f.endswith(".java"):
                java_files.append(os.path.join(root, f))
    return java_files


def extract_package(source):
    """Extract package name from Java source."""
    match = re.search(r'package\s+([\w.]+)\s*;', source)
    return match.group(1) if match else ""


def extract_class_name(source):
    """Extract the public class name (or first class)."""
    # Prefer public class (the one with main)
    match = re.search(r'public\s+class\s+(\w+)', source)
    if match:
        return match.group(1)
    match = re.search(r'class\s+(\w+)', source)
    return match.group(1) if match else None


def extract_methods(source):
    """Extract method signatures from Java source."""
    methods = []
    # Match: modifiers returnType methodName(params)
    pattern = re.compile(
        r'(?:public|private|protected|static|\s)*\s+'
        r'[\w<>\[\].]+\s+'
        r'(\w+)\s*\(([^)]*)\)',
        re.MULTILINE
    )
    for m in pattern.finditer(source):
        method_name = m.group(1)
        params = m.group(2).strip()
        if method_name in ('if', 'for', 'while', 'switch', 'catch', 'return'):
            continue
        methods.append((method_name, params))
    return methods


def _get_local_classes(source):
    """Find all class/interface names defined in this source file."""
    return set(m.group(1) for m in re.finditer(r'(?:class|interface)\s+(\w+)', source))


def add_simple_trace(match, full_source, package_name):
    """Add a simple trace call at method entry."""
    full = match.group(0)
    method_name = match.group(3)

    if method_name in ('if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'try'):
        return full

    # Find enclosing class/interface by tracking brace depth
    before = full_source[:match.start()]
    class_name = "Main"
    # Find all class/interface declarations with their positions
    class_decls = [(m.start(), m.group(1)) for m in re.finditer(r'(?:class|interface)\s+(\w+)', before)]
    if class_decls:
        # Track brace depth to find which class/interface encloses this method
        depth = 0
        enclosing_stack = []
        for i, ch in enumerate(before):
            if ch == '{':
                depth += 1
                # Check if this brace opens a class/interface
                for pos, name in class_decls:
                    brace_after = before.find('{', pos)
                    if brace_after == i:
                        enclosing_stack.append((name, depth))
                        break
            if ch == '}':
                depth -= 1
                # Pop classes that have been closed
                while enclosing_stack and enclosing_stack[-1][1] > depth:
                    enclosing_stack.pop()
        class_name = enclosing_stack[-1][0] if enclosing_stack else (class_decls[-1][1] if class_decls else "Main")

    # Build param signature
    params = match.group(4).strip() if match.group(4) else ""
    param_types = []
    if params:
        for p in params.split(','):
            p = p.strip()
            parts = p.split()
            if parts:
                ptype = parts[0]
                type_map = {
                    "String[]": "java.lang.String[]",
                    "String": "java.lang.String",
                    "Object": "java.lang.Object",
                    "Throwable": "java.lang.Throwable",
                    "int": "int", "boolean": "boolean",
                    "long": "long", "double": "double",
                    "float": "float", "char": "char",
                    "byte": "byte", "short": "short", "void": "void",
                }
                local_classes = _get_local_classes(full_source)
                if ptype in type_map:
                    param_types.append(type_map[ptype])
                elif ptype[0].isupper() and package_name and ptype in local_classes:
                    # Locally defined type: prepend the source-file package.
                    param_types.append(f"{package_name}.{ptype}")
                elif ptype[0].isupper():
                    # External type: map common java.lang aliases, else keep as written.
                    java_lang = {"Class": "java.lang.Class", "Thread": "java.lang.Thread",
                                 "Runnable": "java.lang.Runnable"}
                    param_types.append(java_lang.get(ptype, ptype))
                else:
                    param_types.append(ptype)
    param_sig = ",".join(param_types)

    # Map constructor names to <init>
    if method_name == class_name:
        method_name = "<init>"

    qual = f"{package_name}.{class_name}:{method_name}({param_sig})" if package_name else f"{class_name}:{method_name}({param_sig})"

    fq_class = f"{package_name}.{class_name}" if package_name else class_name
    return full + f'\n__CallTracer.enter("{qual}");'


def create_tracer_class(package_name):
    """Create the __CallTracer helper class."""
    pkg_line = f"package {package_name};\n" if package_name else ""
    return f"""{pkg_line}
import java.util.*;

public class __CallTracer {{
    private static final List<String[]> edges = new ArrayList<>();
    // Map className.methodName → qualified name for caller resolution
    private static final Map<String, String> qualNames = new HashMap<>();

    public static void enter(String method) {{
        // Register this method's qualified name for future caller lookups
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        // stack[2] is the instrumented method that called enter()
        if (stack.length > 2) {{
            String key = stack[2].getClassName() + "." + stack[2].getMethodName();
            qualNames.put(key, method);
        }}

        // Find the caller by walking up the real JVM stack
        String caller = "__module__";
        for (int i = 3; i < stack.length; i++) {{
            String key = stack[i].getClassName() + "." + stack[i].getMethodName();
            if (qualNames.containsKey(key)) {{
                caller = qualNames.get(key);
                break;
            }}
        }}
        edges.add(new String[]{{caller, method}});
    }}

    public static void exit() {{
    }}

    public static void printTrace() {{
        System.out.println("===TRACE===");
        Map<String, Set<String>> cg = new LinkedHashMap<>();
        for (String[] edge : edges) {{
            cg.computeIfAbsent(edge[0], k -> new LinkedHashSet<>()).add(edge[1]);
        }}
        // Ensure all callees also have entries
        Set<String> allFuncs = new LinkedHashSet<>();
        for (Map.Entry<String, Set<String>> e : cg.entrySet()) {{
            allFuncs.add(e.getKey());
            allFuncs.addAll(e.getValue());
        }}
        for (String f : allFuncs) {{
            cg.putIfAbsent(f, new LinkedHashSet<>());
        }}

        // Output as JSON
        StringBuilder sb = new StringBuilder();
        sb.append("{{\\n");
        int i = 0;
        for (Map.Entry<String, Set<String>> e : cg.entrySet()) {{
            sb.append("  \\"").append(e.getKey()).append("\\": [");
            int j = 0;
            for (String callee : e.getValue()) {{
                if (j > 0) sb.append(", ");
                sb.append("\\"").append(callee).append("\\"");
                j++;
            }}
            sb.append("]");
            if (i < cg.size() - 1) sb.append(",");
            sb.append("\\n");
            i++;
        }}
        sb.append("}}");
        System.out.println(sb.toString());
    }}
}}
"""


def trace_java(benchmark_dir):
    """Compile and run Java code with tracing, return call graph."""
    java_files = find_java_files(benchmark_dir)
    if not java_files:
        return {}

    # Read all source files
    sources = {}
    package_name = ""
    main_class = None
    for jf in java_files:
        with open(jf) as f:
            src = f.read()
        sources[jf] = src
        pkg = extract_package(src)
        if pkg:
            package_name = pkg
        if 'public static void main' in src:
            main_class = extract_class_name(src)

    if not main_class:
        return {}

    # Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix="java_trace_")

    try:
        # Write instrumented sources
        for jf, src in sources.items():
            instrumented = src
            # Add trace entry to each method
            instrumented = re.sub(
                r'((?:public|private|protected|static|final|synchronized)\s+)*'
                r'([\w<>\[\].]+\s+)?'
                r'(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w.,\s]+)?\s*\{',
                lambda m: add_simple_trace(m, src, package_name),
                instrumented
            )

            # Add printTrace call before end of main method
            instrumented = re.sub(
                r'(public\s+static\s+void\s+main\s*\([^)]*\)\s*(?:throws\s+[\w.,\s]+)?\s*\{)',
                r'\1\nRuntime.getRuntime().addShutdownHook(new Thread(() -> __CallTracer.printTrace()));',
                instrumented
            )

            # Write to temp dir with package structure
            if package_name:
                pkg_dir = os.path.join(tmp_dir, package_name.replace(".", os.sep))
            else:
                pkg_dir = tmp_dir
            os.makedirs(pkg_dir, exist_ok=True)

            basename = os.path.basename(jf)
            with open(os.path.join(pkg_dir, basename), "w") as f:
                f.write(instrumented)

        # Write tracer class
        tracer_src = create_tracer_class(package_name)
        if package_name:
            tracer_dir = os.path.join(tmp_dir, package_name.replace(".", os.sep))
        else:
            tracer_dir = tmp_dir
        with open(os.path.join(tracer_dir, "__CallTracer.java"), "w") as f:
            f.write(tracer_src)

        # Compile
        all_java = []
        for root, dirs, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith(".java"):
                    all_java.append(os.path.join(root, f))

        # Run javac and java with CWD=tmp_dir so any relative-path file
        # writes by the rewritten code (e.g. new FileWriter(".env"))
        # stay contained in tmp_dir and can't touch the project root.
        compile_result = subprocess.run(
            [JAVAC_CMD] + all_java,
            capture_output=True, text=True, timeout=15, cwd=tmp_dir,
        )
        if compile_result.returncode != 0:
            return {}

        # Run
        fq_main = f"{package_name}.{main_class}" if package_name else main_class
        run_result = subprocess.run(
            [JAVA_CMD, "-cp", tmp_dir, fq_main],
            capture_output=True, text=True, timeout=15, cwd=tmp_dir,
        )

        output = run_result.stdout
        marker = output.find("===TRACE===")
        if marker >= 0:
            json_str = output[marker + len("===TRACE===\n"):]
            try:
                cg = json.loads(json_str)
                # Rename __module__ to the main caller format
                if "__module__" in cg:
                    cg_fixed = {}
                    for k, v in cg.items():
                        if k == "__module__":
                            continue
                        cg_fixed[k] = v
                    cg = cg_fixed

                # Add all defined methods as leaf nodes
                for jf, src in sources.items():
                    pkg = extract_package(src)
                    # Find all class/interface names and their methods
                    for cls_match in re.finditer(r'(?:class|interface)\s+(\w+)', src):
                        cls_name = cls_match.group(1)
                        fq_class = f"{pkg}.{cls_name}" if pkg else cls_name

                        # Find methods in this class
                        cls_start = src.find('{', cls_match.end())
                        if cls_start == -1:
                            continue
                        depth = 0
                        cls_end = cls_start
                        for ci in range(cls_start, len(src)):
                            if src[ci] == '{':
                                depth += 1
                            if src[ci] == '}':
                                depth -= 1
                                if depth == 0:
                                    cls_end = ci
                                    break
                        cls_body = src[cls_start:cls_end + 1]

                        method_re = re.compile(
                            r'(?:public|private|protected|static|final|synchronized|\s)+'
                            r'[\w<>\[\].]+\s+'
                            r'(\w+)\s*\(([^)]*)\)',
                            re.MULTILINE
                        )
                        for mm in method_re.finditer(cls_body):
                            mname = mm.group(1)
                            if mname in ('if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'try'):
                                continue
                            # Map constructor name to <init>
                            if mname == cls_name:
                                mname = "<init>"
                            params = mm.group(2).strip()
                            param_types = []
                            if params:
                                for p in params.split(','):
                                    p = p.strip()
                                    parts = p.split()
                                    if parts:
                                        ptype = parts[0]
                                        type_map = {
                                            "String[]": "java.lang.String[]",
                                            "String": "java.lang.String",
                                            "Object": "java.lang.Object",
                                            "Class": "java.lang.Class",
                                            "Throwable": "java.lang.Throwable",
                                            "Thread": "java.lang.Thread",
                                            "int": "int", "boolean": "boolean",
                                            "long": "long", "double": "double",
                                            "float": "float", "char": "char",
                                            "byte": "byte", "short": "short",
                                        }
                                        if ptype in type_map:
                                            param_types.append(type_map[ptype])
                                        elif ptype[0].isupper() and pkg:
                                            param_types.append(f"{pkg}.{ptype}")
                                        else:
                                            param_types.append(ptype)
                            param_sig = ",".join(param_types)
                            qual = f"{fq_class}:{mname}({param_sig})"
                            if qual not in cg:
                                cg[qual] = []

                return cg
            except json.JSONDecodeError:
                return {}
        return {}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/tracers/java_tracer.py <program_dir>")
        sys.exit(1)

    cg = trace_java(sys.argv[1])
    print("===TRACE===")
    print(json.dumps(cg, indent=2))
