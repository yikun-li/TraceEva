"""Edge-set normalization for compute_metrics.py.

Two kinds of normalization modes are supported, both applied symmetrically
to ground truth and predictions before computing P/R/F1:

1. **Edge-drop predicates**: ``(name, language) -> bool``. Returns
   ``True`` when an edge endpoint should be DROPPED. Edges are dropped
   when either the caller or the callee matches. Registered in
   :data:`MODE_PREDICATES`.
2. **Edge-rewrite transforms**: ``(name, language) -> name``. Rewrite
   an endpoint to a canonical form so multiple surface forms collapse to
   the same edge. Applied BEFORE predicates. Registered in
   :data:`MODE_TRANSFORMS`.

Currently implemented modes
---------------------------
- ``stdlib`` (predicate): drop calls into language standard libraries
  - Python: derived from ``sys.stdlib_module_names`` + ``dir(builtins)`` +
    ``dir(<builtin types>)``: exhaustive against the live runtime.
  - JavaScript: ECMAScript built-in objects + global functions +
    Node.js built-in modules.
  - Java: ``java.``/``javax.``/``jdk.``/``sun.``/``com.sun.``/SAX/DOM
    package prefixes, plus a list of bare stdlib class names for when the
    LLM drops the package prefix (e.g. ``String:length`` instead of
    ``java.lang.String:length``).
- ``init`` (transform): collapse class-name vs.\\ constructor surface
  forms to a canonical bare-class endpoint. The LLM often emits
  ``main.Foo`` while the trace records ``main.Foo.__init__`` (Python),
  ``main.Foo.constructor`` (JS), or ``Foo:<init>(int,int)`` (Java); both
  refer to the same construction event. Strips the constructor suffix
  on both sides so the two forms match. Strictly less aggressive than
  dropping all constructor edges: explicit-constructor calls still
  count, only the surface-form disagreement is resolved.
- ``nested_class`` (transform, Java only): strip outer-class qualifiers
  from Java type paths so ``Outer.Inner:method(args)`` and
  ``Outer$Inner:method(args)`` collapse to bare ``Inner:method(args)``,
  which is what the JVM trace records. Detects outer classes by
  finding the trailing run of capitalized segments in the type path
  (Java packages are conventionally lowercase, classes capitalized).

Add new modes by writing a predicate or transform and registering it in
:data:`MODE_PREDICATES` or :data:`MODE_TRANSFORMS`.
"""

from __future__ import annotations

import builtins
import re
import sys

# =============================================================================
# Python: runtime-introspected (no hand-curated list to drift)
# =============================================================================

_PY_STDLIB_MODULES: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", set()))
_PY_BUILTINS: frozenset[str] = frozenset(dir(builtins))

# Names of every builtin type whose method calls show up in traces.
_PY_BUILTIN_TYPES = {
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "bytes": bytes, "bytearray": bytearray,
    "frozenset": frozenset, "int": int, "float": float, "complex": complex,
    "bool": bool, "type": type, "object": object,
    "memoryview": memoryview, "range": range, "slice": slice,
}
_PY_TYPE_METHODS: dict[str, frozenset[str]] = {
    name: frozenset(dir(t)) for name, t in _PY_BUILTIN_TYPES.items()
}


def _strip_call_args(name: str) -> str:
    """Drop everything from the first '(' onward.

    Defensive: accepts anything castable to ``str`` so a malformed prediction
    (e.g. a ``None`` or numeric leaf) cannot crash the pipeline.
    """
    if not isinstance(name, str):
        return ""
    return name.split("(", 1)[0].strip()


def is_python_stdlib(name: str) -> bool:
    """True if ``name`` resolves to a Python stdlib symbol.

    Catches: ``builtins.<X>``, ``<stdlib_module>.<...>``,
    ``<builtin_type>.<method>`` (e.g. ``list.append``), and bare
    builtins (e.g. ``print``).
    """
    name = _strip_call_args(name)
    if not name:
        return False

    parts = name.split(".")
    head = parts[0]

    if head == "builtins":
        return True
    if head in _PY_STDLIB_MODULES:
        return True
    if len(parts) >= 2 and head in _PY_TYPE_METHODS:
        method = parts[1]
        if method in _PY_TYPE_METHODS[head]:
            return True
    if name in _PY_BUILTINS:
        return True
    return False


# =============================================================================
# JavaScript: ECMAScript globals + Node built-ins
# =============================================================================

# ECMAScript built-in objects (TC39 spec: closed set, rarely changes)
_JS_BUILTIN_GLOBALS: frozenset[str] = frozenset({
    # Fundamental
    "Object", "Function", "Boolean", "Symbol",
    # Numbers & math
    "Number", "BigInt", "Math",
    # Date, strings, regexp
    "Date", "String", "RegExp",
    # Errors
    "Error", "AggregateError", "EvalError", "RangeError", "ReferenceError",
    "SyntaxError", "TypeError", "URIError", "InternalError",
    # Collections
    "Array", "Map", "Set", "WeakMap", "WeakSet",
    # Typed arrays / buffers
    "ArrayBuffer", "SharedArrayBuffer", "Atomics", "DataView",
    "Int8Array", "Uint8Array", "Uint8ClampedArray",
    "Int16Array", "Uint16Array", "Int32Array", "Uint32Array",
    "Float32Array", "Float64Array", "BigInt64Array", "BigUint64Array",
    # Reflection / control
    "Reflect", "Proxy",
    # Async
    "Promise", "Generator", "GeneratorFunction",
    "AsyncFunction", "AsyncGenerator", "AsyncGeneratorFunction",
    # Other ES
    "JSON", "WeakRef", "FinalizationRegistry",
    "Iterator", "AsyncIterator", "Intl",
    # Browser-ish but commonly seen even in Node code
    "console", "URL", "URLSearchParams", "TextEncoder", "TextDecoder",
    "globalThis", "Buffer",
})

_JS_GLOBAL_FUNCTIONS: frozenset[str] = frozenset({
    "parseInt", "parseFloat", "isNaN", "isFinite",
    "encodeURI", "decodeURI", "encodeURIComponent", "decodeURIComponent",
    "eval", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "setImmediate", "clearImmediate", "queueMicrotask",
    "structuredClone", "atob", "btoa", "fetch",
    "require",  # CommonJS: never user code
})

# Node.js built-in modules (https://nodejs.org/api/)
_NODE_BUILTIN_MODULES: frozenset[str] = frozenset({
    "assert", "async_hooks", "buffer", "child_process", "cluster",
    "console", "constants", "crypto", "dgram", "diagnostics_channel",
    "dns", "domain", "events", "fs", "http", "http2", "https",
    "inspector", "module", "net", "os", "path", "perf_hooks",
    "process", "punycode", "querystring", "readline", "repl",
    "stream", "string_decoder", "sys", "timers", "tls",
    "trace_events", "tty", "url", "util", "v8", "vm", "wasi",
    "worker_threads", "zlib",
})


def is_js_stdlib(name: str) -> bool:
    """True if ``name`` is an ECMAScript or Node.js built-in."""
    name = _strip_call_args(name)
    if not name:
        return False
    parts = name.split(".")
    head = parts[0]

    if head in _JS_BUILTIN_GLOBALS:
        return True
    # Prototype methods on any class that includes "prototype" in its path:
    # Map.prototype.set, Array.prototype.map, String.prototype.split, ...
    if "prototype" in parts:
        return True
    if name in _JS_GLOBAL_FUNCTIONS:
        return True
    if head in _NODE_BUILTIN_MODULES or head.startswith("node:"):
        return True

    # Module-prefixed JS-built-in patterns. The LLM frequently emits
    # ``main.Object.keys``, ``main.Array.map``, ``main.console.log``,
    # ``main.Date``, ``main.fs.readFile``, ``main.path.join``, etc.,
    # where a user-code module/file segment qualifies the JS built-in or
    # Node.js built-in module. The head-based checks above miss these
    # because ``head == "main"`` (or some other user-side prefix).
    if len(parts) >= 2:
        # 1. ``<prefix>.<BuiltinClass>.<method>(...)``: covers
        #    ``main.Array.map``, ``main.Object.keys``, ``main.console.log``,
        #    ``main.String.replace``, ``main.RegExp.test``, etc.
        if parts[-2] in _JS_BUILTIN_GLOBALS:
            return True
        # 2. ``<prefix>.<NodeModule>.<method>(...)``: covers
        #    ``main.fs.readFile``, ``main.path.join``,
        #    ``main.crypto.randomBytes``, ``main.util.promisify``,
        #    ``main.stream.<method>``, etc. Same recording-convention
        #    rationale as rule 1: the trace records 0 stdlib calls.
        if parts[-2] in _NODE_BUILTIN_MODULES:
            return True
        # 3. ``<prefix>.<BuiltinClass>`` (no method, constructor-style),
        #    covers ``main.Date`` (calling ``new Date()``), ``main.Map``,
        #    ``main.Promise``, etc., once any ``init`` transform has
        #    already canonicalized the constructor suffix away.
        if parts[-1] in _JS_BUILTIN_GLOBALS:
            return True
        # 4. ``<prefix>.<global_function>(...)``: covers
        #    ``main.parseInt``, ``main.setTimeout``, ``main.fetch``, etc.
        if parts[-1] in _JS_GLOBAL_FUNCTIONS:
            return True
    return False


# =============================================================================
# Java: JDK package prefixes + bare class allowlist
# =============================================================================

_JAVA_STDLIB_PACKAGE_PREFIXES: tuple[str, ...] = (
    "java.", "javax.", "jdk.", "sun.", "com.sun.",
    "org.w3c.dom.", "org.xml.sax.", "org.ietf.jgss.", "org.omg.",
)

# Bare class names that nearly always refer to JDK types when the LLM
# drops the package prefix. Conservative: include only well-known names
# that user code rarely shadows.
_JAVA_BARE_STDLIB_CLASSES: frozenset[str] = frozenset({
    # java.lang
    "String", "Object", "Integer", "Long", "Double", "Float", "Short",
    "Byte", "Boolean", "Character", "Number", "Math", "StrictMath", "System",
    "Thread", "ThreadGroup", "Runnable", "Throwable", "Exception",
    "RuntimeException", "IllegalArgumentException", "IllegalStateException",
    "NullPointerException", "IndexOutOfBoundsException",
    "ArrayIndexOutOfBoundsException", "ClassNotFoundException",
    "ClassCastException", "UnsupportedOperationException",
    "NumberFormatException", "Error", "AssertionError", "OutOfMemoryError",
    "StackOverflowError",
    "StringBuilder", "StringBuffer", "CharSequence", "Class", "ClassLoader",
    "Enum", "Iterable", "Comparable", "Cloneable", "AutoCloseable",
    "Process", "ProcessBuilder", "Package", "Void", "Record",
    # java.util
    "ArrayList", "LinkedList", "Vector", "Stack",
    "HashMap", "LinkedHashMap", "TreeMap", "Hashtable", "WeakHashMap",
    "IdentityHashMap", "ConcurrentHashMap", "ConcurrentSkipListMap",
    "HashSet", "LinkedHashSet", "TreeSet", "ConcurrentSkipListSet",
    "PriorityQueue", "ArrayDeque", "ArrayBlockingQueue",
    "LinkedBlockingQueue", "Deque", "Queue", "BlockingQueue",
    "List", "Set", "Map", "Collection", "SortedMap", "SortedSet",
    "NavigableMap", "NavigableSet",
    "Iterator", "ListIterator", "Spliterator", "Enumeration",
    "Comparator", "Optional", "OptionalInt", "OptionalLong", "OptionalDouble",
    "Random", "SecureRandom", "ThreadLocalRandom", "Scanner",
    "Date", "Calendar", "GregorianCalendar", "TimeZone", "SimpleTimeZone",
    "Locale", "Properties", "ResourceBundle",
    "Collections", "Arrays", "Objects", "EnumSet", "EnumMap",
    "Timer", "TimerTask", "UUID", "Base64", "BitSet", "Currency",
    "StringJoiner", "StringTokenizer",
    # java.util.stream
    "Stream", "IntStream", "LongStream", "DoubleStream",
    "Collectors", "Collector",
    # java.util.concurrent
    "CompletableFuture", "Future", "Executor", "Executors",
    "ExecutorService", "ScheduledExecutorService", "ThreadPoolExecutor",
    "ForkJoinPool", "AtomicInteger", "AtomicLong", "AtomicBoolean",
    "AtomicReference", "CountDownLatch", "CyclicBarrier", "Semaphore",
    "Phaser", "ReentrantLock", "ReentrantReadWriteLock",
    "Lock", "ReadWriteLock", "Condition",
    # java.util.function
    "Function", "BiFunction", "Consumer", "BiConsumer", "Supplier",
    "Predicate", "BiPredicate", "UnaryOperator", "BinaryOperator",
    # java.io
    "File", "FileReader", "FileWriter", "FileInputStream", "FileOutputStream",
    "BufferedReader", "BufferedWriter", "BufferedInputStream",
    "BufferedOutputStream", "PrintWriter", "PrintStream",
    "InputStream", "OutputStream", "Reader", "Writer",
    "InputStreamReader", "OutputStreamWriter", "DataInputStream",
    "DataOutputStream", "ObjectInputStream", "ObjectOutputStream",
    "Serializable", "Externalizable", "IOException",
    "FileNotFoundException", "EOFException",
    "ByteArrayInputStream", "ByteArrayOutputStream",
    "StringWriter", "StringReader",
    "PipedInputStream", "PipedOutputStream",
    # java.nio
    "Path", "Paths", "Files", "FileSystems", "FileSystem",
    "ByteBuffer", "CharBuffer", "IntBuffer", "LongBuffer",
    "FloatBuffer", "DoubleBuffer",
    "Channel", "Channels", "FileChannel",
    "SocketChannel", "ServerSocketChannel",
    "Charset", "StandardCharsets", "CharsetEncoder", "CharsetDecoder",
    "StandardOpenOption", "OpenOption",
    # java.math
    "BigInteger", "BigDecimal", "MathContext", "RoundingMode",
    # java.security
    "MessageDigest", "MessageDigestSpi", "KeyPair", "KeyPairGenerator",
    "Signature", "Key", "PrivateKey", "PublicKey", "Provider",
    "Security", "SignatureException", "NoSuchAlgorithmException",
    "KeyStore", "Certificate",
    # java.text
    "SimpleDateFormat", "DateFormat", "DecimalFormat", "NumberFormat",
    "MessageFormat", "ChoiceFormat", "Format", "FieldPosition",
    "ParsePosition", "ParseException",
    # java.time
    "LocalDate", "LocalTime", "LocalDateTime", "ZonedDateTime",
    "OffsetDateTime", "OffsetTime", "Instant", "Duration", "Period",
    "Year", "YearMonth", "MonthDay", "DayOfWeek", "Month", "Clock",
    "ZoneId", "ZoneOffset", "DateTimeFormatter",
    # java.net
    "URL", "URI", "URLConnection", "HttpURLConnection",
    "Socket", "ServerSocket", "InetAddress", "Inet4Address",
    "Inet6Address", "InetSocketAddress",
    "MalformedURLException", "URISyntaxException", "UnknownHostException",
    # java.sql
    "Connection", "Statement", "PreparedStatement", "CallableStatement",
    "ResultSet", "ResultSetMetaData", "DatabaseMetaData", "DriverManager",
    "SQLException", "Driver", "Time", "Timestamp",
    # java.util.regex
    "Pattern", "Matcher", "MatchResult",
    # java.util.logging
    "Logger", "Level", "Handler", "Formatter", "LogRecord",
    # javax.* common
    "Servlet", "ServletRequest", "ServletResponse",
    "HttpServlet", "HttpServletRequest", "HttpServletResponse",
})

# Iteratively strip generic type parameters: handles "Map<String, List<Integer>>"
_JAVA_GENERIC_RE = re.compile(r"<[^<>]*>")


def is_java_stdlib(name: str) -> bool:
    """True if ``name`` is a JDK / javax / sun.* symbol, fully-qualified or bare."""
    if not isinstance(name, str) or not name:
        return False
    # Strip generic args repeatedly to handle nested generics.
    prev = None
    cleaned = name
    while prev != cleaned:
        prev = cleaned
        cleaned = _JAVA_GENERIC_RE.sub("", cleaned)
    head = re.split(r"[:(]", cleaned, 1)[0].rstrip(".").strip()
    if not head:
        return False
    for prefix in _JAVA_STDLIB_PACKAGE_PREFIXES:
        if head.startswith(prefix):
            return True
    # Bare-class fallback: only fire when the LLM dropped the package
    # entirely (head has no dots). Otherwise ``io.grpc.Key`` would wrongly
    # match because ``Key`` collides with ``java.security.Key``.
    if "." not in head and head in _JAVA_BARE_STDLIB_CLASSES:
        return True
    return False


# =============================================================================
# Constructor / init edges (cross-language): canonicalization transform
# =============================================================================

def canonicalize_init(name: str, language: str) -> str:
    """Collapse a constructor edge to the bare class form.

    Pattern (a) from the failure-mode audit: when the LLM emits
    ``main.Foo`` (bare class) and the trace records
    ``main.Foo.__init__`` (Python) / ``main.Foo.constructor`` (JS) /
    ``Foo:<init>(int,int)`` (Java) for the same call, both refer to
    the same construction event. Strip the constructor suffix on both
    sides so the two forms match.

    Strictly less aggressive than dropping all constructor edges:
    explicit-constructor calls still count, only the surface-form
    disagreement on bare-class-vs-init is resolved.

    Side effect: in Java this collapses overloaded constructors of the
    same class to a single edge (``Bar:<init>(int,int)`` and
    ``Bar:<init>(String)`` both become ``Bar``); the loss is symmetric
    between GT and predictions.
    """
    if not isinstance(name, str) or not name:
        return name
    if language == "python":
        if name.endswith(".__init__"):
            return name[: -len(".__init__")]
        if name.endswith(".__new__"):
            return name[: -len(".__new__")]
        return name
    if language == "javascript":
        if name.endswith(".constructor"):
            return name[: -len(".constructor")]
        return name
    if language == "java":
        for marker in (":<init>", ":<clinit>"):
            idx = name.find(marker)
            if idx != -1:
                return name[:idx]
        return name
    return name


# =============================================================================
# Java nested-class qualifier: canonicalization transform
# =============================================================================

_JAVA_NESTED_PATH_SPLIT = re.compile(r"[.$]")


def canonicalize_nested_class(name: str, language: str) -> str:
    """Strip outer-class qualifiers from a Java callee.

    Java nested classes can be written either ``Outer.Inner:method(args)``
    (source form) or ``Outer$Inner:method(args)`` (JVM bytecode form),
    while our dynamic tracer records the bare innermost class
    ``Inner:method(args)``. This transform finds the trailing run of
    capitalized segments in the type path before the ``:`` separator
    and keeps only the innermost one, treating any preceding consecutive
    capitalized segments as outer-class qualifiers to drop.

    Heuristic boundary: Java packages are conventionally lowercase and
    classes capitalized, so the transition from lowercase to capitalized
    in the type path marks the package/class boundary. False positives
    on rare uppercase-package conventions (e.g. ``Apple.Foo``) are
    accepted as a known limitation of this opt-in mode.
    """
    if language != "java":
        return name
    if not isinstance(name, str) or ":" not in name:
        return name
    type_path, rest = name.split(":", 1)
    segs = _JAVA_NESTED_PATH_SPLIT.split(type_path)
    if len(segs) < 2:
        return name
    # Walk backward from the last segment; collect consecutive capitalized
    # segments at the end. Anything earlier is the package.
    end = len(segs)
    start = end - 1
    while start > 0 and segs[start - 1] and segs[start - 1][0].isupper():
        start -= 1
    if end - start <= 1:
        # Only one capitalized class segment at the end: no nesting to strip.
        return name
    inner = segs[end - 1]
    package = ".".join(segs[:start])
    new_path = f"{package}.{inner}" if package else inner
    return f"{new_path}:{rest}"


# =============================================================================
# Dispatch
# =============================================================================

_STDLIB_DISPATCH = {
    "python": is_python_stdlib,
    "javascript": is_js_stdlib,
    "java": is_java_stdlib,
}


def is_stdlib(name: str, language: str) -> bool:
    fn = _STDLIB_DISPATCH.get(language)
    return fn(name) if fn else False


# Maps a normalization mode name to a predicate (name, language) -> drop?
MODE_PREDICATES: dict[str, callable] = {
    "stdlib": is_stdlib,
}

# Maps a normalization mode name to a transform (name, language) -> name.
# Transforms run BEFORE predicates and may rewrite endpoints to a canonical
# form so multiple surface forms collapse to the same edge.
MODE_TRANSFORMS: dict[str, callable] = {
    "init": canonicalize_init,
    "nested_class": canonicalize_nested_class,
}

ALL_MODES: tuple[str, ...] = tuple(sorted(set(MODE_PREDICATES) | set(MODE_TRANSFORMS)))


def apply_transforms(name: str, language: str, modes: list[str]) -> str:
    """Apply all active rewrite transforms to ``name`` in order."""
    for mode in modes:
        fn = MODE_TRANSFORMS.get(mode)
        if fn:
            name = fn(name, language)
    return name


def should_drop(name: str, language: str, modes: list[str]) -> bool:
    """Return True if ``name`` matches any active drop predicate."""
    for mode in modes:
        pred = MODE_PREDICATES.get(mode)
        if pred and pred(name, language):
            return True
    return False
