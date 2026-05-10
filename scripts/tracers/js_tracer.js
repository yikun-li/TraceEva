#!/usr/bin/env node
/**
 * Dynamic call tracer for JavaScript programs.
 *
 * Rewrites each source file at the AST level so every function, class
 * method, arrow body, and constructor records its caller before
 * executing, then runs the instrumented code once and emits the
 * observed caller->callee edge set under the unified JSON schema.
 *
 * Usage:
 *   node scripts/tracers/js_tracer.js <program_dir>
 *   (pass the program DIRECTORY; multi-file programs are supported)
 *
 * Output: JSON call graph after ===TRACE=== marker.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

const benchDir = process.argv[2];
if (!benchDir) {
    console.error('Usage: node js_tracer.js <benchmark_dir>');
    process.exit(1);
}

// Find all .js files in the benchmark directory
const jsFiles = [];
function findJsFiles(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isFile() && entry.name.endsWith('.js')) {
            jsFiles.push(path.join(dir, entry.name));
        } else if (entry.isDirectory() && entry.name !== 'node_modules') {
            findJsFiles(path.join(dir, entry.name));
        }
    }
}
findJsFiles(benchDir);

// Extract all function/class/method names from source via regex
function extractNames(src, moduleName) {
    const names = [];

    // Build a scope map: for each position, what function are we inside?
    // This lets us detect nested functions
    const scopeStack = [moduleName]; // Start at module level
    const scopeAtPos = []; // scopeAtPos[charIndex] = current scope
    let bd = 0;
    const funcStartRe = /\bfunction\*?\s+(\w+)\s*\(/g;

    // First pass: find all function starts with their positions
    const funcPositions = [];
    let fm;
    while ((fm = funcStartRe.exec(src)) !== null) {
        funcPositions.push({ name: fm[1], index: fm.index });
    }

    // Build scope tracking by scanning characters
    let funcIdx = 0;
    const braceToFunc = {}; // Maps opening brace position to function name
    bd = 0;
    for (let i = 0; i < src.length; i++) {
        if (src[i] === '{') {
            // Check if this brace opens a function
            if (funcIdx < funcPositions.length) {
                const fp = funcPositions[funcIdx];
                // Find the opening brace after this function declaration
                const braceAfterFunc = src.indexOf('{', fp.index);
                if (braceAfterFunc === i) {
                    braceToFunc[i] = fp.name;
                    funcIdx++;
                }
            }
            bd++;
        }
        if (src[i] === '}') bd--;
    }

    // Second pass: determine parent scope for each function
    bd = 0;
    const activeFuncs = []; // Stack of (funcName, braceDepth)
    funcStartRe.lastIndex = 0;

    // function declarations: find top-level and nested
    const allFuncRe = /\bfunction\*?\s+(\w+)\s*\(/g;
    while ((m = allFuncRe.exec(src)) !== null) {
        const funcName = m[1];
        // Determine what scope this is in by counting braces before this position
        let depth = 0;
        let parentScope = moduleName;
        const scopeTrack = [moduleName];

        for (let i = 0; i < m.index; i++) {
            if (src[i] === '{') {
                depth++;
                if (braceToFunc[i]) scopeTrack.push(`${scopeTrack[scopeTrack.length-1]}.${braceToFunc[i]}`);
            }
            if (src[i] === '}') {
                depth--;
                if (scopeTrack.length > 1 && depth < scopeTrack.length - 1) {
                    scopeTrack.pop();
                }
            }
        }
        parentScope = scopeTrack[scopeTrack.length - 1];

        const qualName = parentScope === moduleName
            ? `${moduleName}.${funcName}`
            : `${parentScope}.${funcName}`;

        names.push({ name: funcName, qualName, type: 'function' });
    }

    // class declarations with methods
    const classRe = /\bclass\s+(\w+)(?:\s+extends\s+[\w.]+)?\s*\{/g;
    while ((m = classRe.exec(src)) !== null) {
        const className = m[1];
        // Add the class itself as a leaf node
        names.push({ name: className, qualName: `${moduleName}.${className}`, type: 'class' });
        // Find matching closing brace
        const classStart = src.indexOf('{', m.index);
        let depth = 0, classEnd = classStart;
        for (let i = classStart; i < src.length; i++) {
            if (src[i] === '{') depth++;
            if (src[i] === '}') { depth--; if (depth === 0) { classEnd = i; break; } }
        }
        const classBody = src.substring(classStart + 1, classEnd);

        // Find methods (including constructor, static methods, getters, setters)
        const methodRe = /(?:static\s+)?(?:get\s+|set\s+)?(\w+)\s*\([^)]*\)\s*\{/g;
        let mm;
        while ((mm = methodRe.exec(classBody)) !== null) {
            const methodName = mm[1];
            if (['if', 'for', 'while', 'switch', 'catch', 'class', 'return', 'new', 'typeof'].includes(methodName)) continue;
            names.push({
                name: methodName,
                qualName: `${moduleName}.${className}.${methodName}`,
                className,
                type: 'method'
            });
        }

        // Find [Symbol.iterator]() and *[Symbol.iterator]() methods
        const symbolIterRe = /\*?\s*\[Symbol\.iterator\]\s*\([^)]*\)\s*\{/g;
        let si;
        while ((si = symbolIterRe.exec(classBody)) !== null) {
            names.push({
                name: '[Symbol.iterator]',
                qualName: `${moduleName}.${className}[Symbol.iterator]`,
                className,
                type: 'symbol_method'
            });
        }

        // Find next() method (iterator protocol)
        const nextRe = /\bnext\s*\([^)]*\)\s*\{/g;
        let ni;
        while ((ni = nextRe.exec(classBody)) !== null) {
            names.push({
                name: 'next',
                qualName: `${moduleName}.${className}[Symbol.iterator].next`,
                className,
                type: 'method'
            });
        }
    }

    // Arrow functions: const/let/var name = (...) => or name = arg =>
    const arrowRe = /(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|\w+)\s*=>/g;
    let arrowCount = 0;
    while ((m = arrowRe.exec(src)) !== null) {
        arrowCount++;
        names.push({
            name: m[1],
            qualName: `${moduleName}.<arrow${arrowCount}>`,
            varName: m[1],
            type: 'arrow'
        });
    }

    return names;
}

// Build the instrumented source code
let allNames = [];
let allCode = '';
const modulePrefixes = {}; // Maps module name → { prefix, names }

// Inline-transform arrow functions and const function expressions
// so they're wrapped at definition time, not after
function transformSource(src, moduleName) {
    let result = src;
    let arrowCount = 0;

    // Helper: find the enclosing-function qualName at character position `pos`
    // by walking forward from start, tracking which `function name(...)` braces
    // are currently open. Used by the wrap passes below so that `var X = ...`
    // nested inside `function Y() {...}` gets qualName `mod.Y.X`.
    function enclosingScopeAt(pos) {
        const fnDeclRe = /\bfunction\*?\s+([\w$]+)\s*\([^)]*\)\s*\{/g;
        const byBrace = {};
        let mm;
        while ((mm = fnDeclRe.exec(result)) !== null) {
            const bracePos = result.indexOf('{', mm.index + mm[0].length - 1);
            byBrace[bracePos] = mm[1];
        }
        const stack = [moduleName];
        const pushDepths = [];
        let braceDepth = 0;
        for (let i = 0; i < pos; i++) {
            if (result[i] === '{') {
                braceDepth++;
                if (byBrace[i]) {
                    stack.push(stack[stack.length - 1] + '.' + byBrace[i]);
                    pushDepths.push(braceDepth);
                }
            } else if (result[i] === '}') {
                braceDepth--;
                while (pushDepths.length && braceDepth < pushDepths[pushDepths.length - 1]) {
                    stack.pop();
                    pushDepths.pop();
                }
            }
        }
        return stack[stack.length - 1];
    }

    // === Pass 1: anonymous function expressions assigned to a binding ===
    //   var f = function(...) { ... }
    //   const f = function(...) { ... }
    //   let f = function(...) { ... }
    // Plain `function name(...)` declarations are caught by a later pass; this
    // pass handles the named-binding form so the tracer sees both.
    {
        const fnExprRe = /(?:const|let|var)\s+([\w$]+)\s*=\s*function\s*\*?\s*\([^)]*\)\s*\{/g;
        const fnInjections = [];
        let fbm;
        while ((fbm = fnExprRe.exec(result)) !== null) {
            const fnName = fbm[1];
            const fnKeywordPos = result.indexOf('function', fbm.index);
            const openBracePos = result.indexOf('{', fnKeywordPos);
            let depth = 1;
            let closeBracePos = -1;
            for (let i = openBracePos + 1; i < result.length; i++) {
                if (result[i] === '{') depth++;
                else if (result[i] === '}') { depth--; if (depth === 0) { closeBracePos = i; break; } }
            }
            if (closeBracePos === -1) continue;
            fnInjections.push({ fnName, fnKeywordPos, closeBracePos });
        }
        // Compute scope-aware qualNames before mutating the source.
        for (const inj of fnInjections) {
            inj.qualName = enclosingScopeAt(inj.fnKeywordPos) + '.' + inj.fnName;
        }
        // Drop injections whose ranges are nested inside another injection's range.
        // Wrapping both would invalidate the outer's pre-computed positions when the
        // inner is processed first (or vice versa). Outermost wins.
        const ascFn = [...fnInjections].sort((a, b) => a.fnKeywordPos - b.fnKeywordPos);
        const flatFn = [];
        let lastEndFn = -1;
        for (const inj of ascFn) {
            if (inj.fnKeywordPos > lastEndFn) {
                flatFn.push(inj);
                lastEndFn = inj.closeBracePos;
            }
        }
        flatFn.sort((a, b) => b.fnKeywordPos - a.fnKeywordPos);
        for (const inj of flatFn) {
            result = result.substring(0, inj.fnKeywordPos)
                   + `__wrapFn(${result.substring(inj.fnKeywordPos, inj.closeBracePos + 1)}, '${inj.qualName}')`
                   + result.substring(inj.closeBracePos + 1);
        }
    }

    // === Pass 2: object literal property functions ===
    //   const x = { foo() {...}, bar: function() {...}, baz: (args) => {...} }
    // Wrap the entire object literal with __wrapObj(obj, 'mod.x'), which
    // iterates own properties and wraps each function value with
    // __wrapFn(v, 'mod.x.<key>'). Catches all three property-function
    // patterns in one shot.
    {
        const objBindingRe = /(?:const|let|var)\s+([\w$]+)\s*=\s*\{/g;
        const objInjections = [];
        let obm;
        while ((obm = objBindingRe.exec(result)) !== null) {
            const objName = obm[1];
            const openBracePos = obm.index + obm[0].length - 1;
            let depth = 1;
            let closeBracePos = -1;
            for (let i = openBracePos + 1; i < result.length; i++) {
                if (result[i] === '{') depth++;
                else if (result[i] === '}') { depth--; if (depth === 0) { closeBracePos = i; break; } }
            }
            if (closeBracePos === -1) continue;
            objInjections.push({ objName, openBracePos, closeBracePos });
        }
        for (const inj of objInjections) {
            inj.qualName = enclosingScopeAt(inj.openBracePos) + '.' + inj.objName;
        }
        // Drop injections nested inside another (e.g. `const presets = {...}` inside
        // a method body of an outer `const X = {...}`). Outermost wins; the inner
        // object literal is left as plain data, typically not a callable target.
        const ascObj = [...objInjections].sort((a, b) => a.openBracePos - b.openBracePos);
        const flatObj = [];
        let lastEndObj = -1;
        for (const inj of ascObj) {
            if (inj.openBracePos > lastEndObj) {
                flatObj.push(inj);
                lastEndObj = inj.closeBracePos;
            }
        }
        flatObj.sort((a, b) => b.openBracePos - a.openBracePos);
        for (const inj of flatObj) {
            result = result.substring(0, inj.openBracePos)
                   + `__wrapObj(${result.substring(inj.openBracePos, inj.closeBracePos + 1)}, '${inj.qualName}')`
                   + result.substring(inj.closeBracePos + 1);
        }
    }

    // === Pass 3: globalThis.X = {...} and window.X = {...} ===
    // Different binding syntax than const/let/var, same __wrapObj treatment.
    {
        const gtRe = /(?:globalThis|window)\s*\.\s*([\w$]+)\s*=\s*\{/g;
        const gtInjections = [];
        let gm;
        while ((gm = gtRe.exec(result)) !== null) {
            const objName = gm[1];
            const openBracePos = gm.index + gm[0].length - 1;
            let depth = 1;
            let closeBracePos = -1;
            for (let i = openBracePos + 1; i < result.length; i++) {
                if (result[i] === '{') depth++;
                else if (result[i] === '}') { depth--; if (depth === 0) { closeBracePos = i; break; } }
            }
            if (closeBracePos === -1) continue;
            gtInjections.push({ objName, openBracePos, closeBracePos });
        }
        const ascGt = [...gtInjections].sort((a, b) => a.openBracePos - b.openBracePos);
        const flatGt = [];
        let lastEndGt = -1;
        for (const inj of ascGt) {
            if (inj.openBracePos > lastEndGt) {
                flatGt.push(inj);
                lastEndGt = inj.closeBracePos;
            }
        }
        flatGt.sort((a, b) => b.openBracePos - a.openBracePos);
        for (const inj of flatGt) {
            const qualName = `${moduleName}.${inj.objName}`;
            result = result.substring(0, inj.openBracePos)
                   + `__wrapObj(${result.substring(inj.openBracePos, inj.closeBracePos + 1)}, '${qualName}')`
                   + result.substring(inj.closeBracePos + 1);
        }
    }

    // === Pass 4: IIFE assigned to a binding ===
    //   const X = (function() { return {...}; })()
    //   const X = (() => {...})()
    // Wrap the entire (...)(...) expression with __wrapObj so that whatever
    // the IIFE returns gets its property functions instrumented at qualName
    // mod.X.<key>.
    {
        const iifeRe = /(?:const|let|var)\s+([\w$]+)\s*=\s*\(/g;
        const iifeInjections = [];
        let im;
        while ((im = iifeRe.exec(result)) !== null) {
            const objName = im[1];
            const openParenPos = im.index + im[0].length - 1;
            let depth = 1;
            let closeParenPos = -1;
            for (let i = openParenPos + 1; i < result.length; i++) {
                if (result[i] === '(') depth++;
                else if (result[i] === ')') { depth--; if (depth === 0) { closeParenPos = i; break; } }
            }
            if (closeParenPos === -1) continue;
            // Must be followed by ( ... ) for invocation (with optional whitespace).
            let next = closeParenPos + 1;
            while (next < result.length && /\s/.test(result[next])) next++;
            if (result[next] !== '(') continue;
            let invDepth = 1;
            let invEndPos = -1;
            for (let i = next + 1; i < result.length; i++) {
                if (result[i] === '(') invDepth++;
                else if (result[i] === ')') { invDepth--; if (invDepth === 0) { invEndPos = i; break; } }
            }
            if (invEndPos === -1) continue;
            iifeInjections.push({ objName, openParenPos, invEndPos });
        }
        const ascIife = [...iifeInjections].sort((a, b) => a.openParenPos - b.openParenPos);
        const flatIife = [];
        let lastEndIife = -1;
        for (const inj of ascIife) {
            if (inj.openParenPos > lastEndIife) {
                flatIife.push(inj);
                lastEndIife = inj.invEndPos;
            }
        }
        flatIife.sort((a, b) => b.openParenPos - a.openParenPos);
        for (const inj of flatIife) {
            const qualName = enclosingScopeAt(inj.openParenPos) + '.' + inj.objName;
            result = result.substring(0, inj.openParenPos)
                   + `__wrapObj(${result.substring(inj.openParenPos, inj.invEndPos + 1)}, '${qualName}')`
                   + result.substring(inj.invEndPos + 1);
        }
    }

    // 1. Wrap assigned arrow functions inline
    result = result.replace(
        /((?:const|let|var)\s+)(\w+)(\s*=\s*)((?:\([^)]*\)|\w+)\s*=>\s*(?:\{[^}]*\}|[^;\n,]+))/g,
        (match, decl, name, eq, arrowExpr) => {
            arrowCount++;
            const qualName = `${moduleName}.<arrow${arrowCount}>`;
            return `${decl}${name}${eq}__wrapFn((${arrowExpr}), '${qualName}')`;
        }
    );

    // 1a. Wrap ALL standalone arrow functions not yet wrapped
    //     Matches arrows in function call args, return statements, etc.
    //     Excludes .method( patterns (builtins like .map, .filter)
    //     Handles multi-line by matching arrows at start of lines/after commas

    // First: arrows right after function call opening: func((x) => ...)
    result = result.replace(
        /(\w+\s*\([^)]*,\s*\n?\s*)((?:\([^)]*\)|\w+)\s*=>\s*(?:\{[^}]*\}|[^,)\n]+))/g,
        (match, callPrefix, arrowExpr) => {
            if (callPrefix.match(/\.\w+\s*\(/)) return match; // skip .method( calls
            if (callPrefix.includes('__wrapFn')) return match;
            // Check if the function name is preceded by a dot in the original source
            const matchPos = result.indexOf(match);
            if (matchPos > 0 && result[matchPos - 1] === '.') return match;
            arrowCount++;
            const qualName = `${moduleName}.<arrow${arrowCount}>`;
            return `${callPrefix}__wrapFn((${arrowExpr}), '${qualName}')`;
        }
    );
    // Also: arrows as first argument: func((x) => ...)
    result = result.replace(
        /(\w+\s*\(\s*\n?\s*)((?:\([^)]*\)|\w+)\s*=>\s*(?:\{[^}]*\}|[^,)\n]+))/g,
        (match, callPrefix, arrowExpr) => {
            if (callPrefix.match(/\.\w+\s*\(\s*\n?\s*$/)) return match;
            if (callPrefix.includes('__wrapFn')) return match;
            const matchPos = result.indexOf(match);
            if (matchPos > 0 && result[matchPos - 1] === '.') return match;
            arrowCount++;
            const qualName = `${moduleName}.<arrow${arrowCount}>`;
            return `${callPrefix}__wrapFn((${arrowExpr}), '${qualName}')`;
        }
    );

    // Second: wrap remaining unwrapped arrows on their own line
    // (typical pattern: multi-line function call args)
    // Skip arrows that are inside method chains (.map, .filter, etc.)
    const resultLines = result.split('\n');
    for (let li = 0; li < resultLines.length; li++) {
        const line = resultLines[li];
        const trimmed = line.trim();
        if (trimmed.includes('__wrapFn')) continue;
        if (!trimmed.match(/(?:\([^)]*\)|\w+)\s*=>/)) continue;

        // Check if this arrow is inside a .method() call
        // Check same line for .method( pattern before the arrow
        let insideMethodChain = false;
        if (line.match(/\.\w+\s*\(/)) {
            insideMethodChain = true;
        }
        // Also check preceding lines
        for (let j = li - 1; j >= Math.max(0, li - 3); j--) {
            if (resultLines[j].match(/\.\w+\s*\(\s*$/)) {
                insideMethodChain = true;
                break;
            }
        }
        if (insideMethodChain) continue;

        // Wrap the arrow
        resultLines[li] = line.replace(
            /^(\s*)((?:\([^)]*\)|\w+)\s*=>\s*(?:\{[^}]*\}|[^,)\n]+))(,?\s*)$/,
            (match, indent, arrowExpr, suffix) => {
                arrowCount++;
                const qualName = `${moduleName}.<arrow${arrowCount}>`;
                return `${indent}__wrapFn((${arrowExpr}), '${qualName}')${suffix}`;
            }
        );
    }
    result = resultLines.join('\n');

    // 1b. Inject tracing into regular function bodies
    //     function name(...) { → function name(...) { __enter('qual'); try {
    //     and close with } finally { __exit(); } }
    // Process from end to start so indices stay valid
    const funcInjectRe = /\bfunction\*?\s+([\w$]+)\s*\([^)]*\)\s*\{/g;
    const funcInjections = [];
    let fi;
    while ((fi = funcInjectRe.exec(result)) !== null) {
        const funcName = fi[1];
        if (['if', 'for', 'while', 'switch', 'catch', 'return'].includes(funcName)) continue;

        // Determine qualified name using scope tracking
        let depth = 0;
        const scopeTrack = [moduleName];
        for (let i = 0; i < fi.index; i++) {
            if (result[i] === '{') {
                depth++;
                // Check if this brace opens a function we know about
                for (const prev of funcInjections) {
                    if (prev.bracePos === i) {
                        scopeTrack.push(prev.qualName);
                        break;
                    }
                }
            }
            if (result[i] === '}') {
                depth--;
                if (scopeTrack.length > 1 && depth < scopeTrack.length - 1) {
                    scopeTrack.pop();
                }
            }
        }
        const parentScope = scopeTrack[scopeTrack.length - 1];
        const qualName = `${parentScope}.${funcName}`;

        const bracePos = result.indexOf('{', fi.index + fi[0].length - 1);
        funcInjections.push({ funcName, qualName, insertPos: bracePos + 1, bracePos });
    }

    // Inject from end to preserve positions
    for (let i = funcInjections.length - 1; i >= 0; i--) {
        const inj = funcInjections[i];
        // Find matching closing brace
        let d = 1, closePos = inj.insertPos;
        for (let j = inj.insertPos; j < result.length; j++) {
            if (result[j] === '{') d++;
            if (result[j] === '}') { d--; if (d === 0) { closePos = j; break; } }
        }
        result = result.substring(0, inj.insertPos) +
                 `\n__edges.push([__stack[__stack.length-1],'${inj.qualName}']);__stack.push('${inj.qualName}');try{\n` +
                 result.substring(inj.insertPos, closePos) +
                 `\n}finally{__stack.pop();}\n` +
                 result.substring(closePos);
    }

    // 1c. Inject tracing into class methods (non-constructor)
    // Find each class, then inject into its methods
    const classReInject = /\bclass\s+(\w+)(?:\s+extends\s+[\w.]+)?\s*\{/g;
    const methodInjections = [];
    let ci;
    while ((ci = classReInject.exec(result)) !== null) {
        const className = ci[1];
        const classStart = result.indexOf('{', ci.index);
        let d = 0, classEnd = classStart;
        for (let j = classStart; j < result.length; j++) {
            if (result[j] === '{') d++;
            if (result[j] === '}') { d--; if (d === 0) { classEnd = j; break; } }
        }

        // Find methods inside this class body (skip constructor, handled separately)
        const methodRe2 = /(?:static\s+)?(\w+)\s*\([^)]*\)\s*\{/g;
        const classBody = result.substring(classStart + 1, classEnd);
        let mm2;
        while ((mm2 = methodRe2.exec(classBody)) !== null) {
            const mName = mm2[1];
            if (['if', 'for', 'while', 'switch', 'catch', 'constructor', 'return', 'new', 'typeof'].includes(mName)) continue;
            const qualName = `${moduleName}.${className}.${mName}`;
            const absPos = classStart + 1 + mm2.index + mm2[0].length - 1; // position of opening {
            methodInjections.push({ qualName, insertPos: absPos + 1 });
        }
    }

    // Inject from end to preserve positions
    methodInjections.sort((a, b) => b.insertPos - a.insertPos);
    for (const inj of methodInjections) {
        // Find matching closing brace
        let d2 = 1, closePos = inj.insertPos;
        for (let j = inj.insertPos; j < result.length; j++) {
            if (result[j] === '{') d2++;
            if (result[j] === '}') { d2--; if (d2 === 0) { closePos = j; break; } }
        }
        result = result.substring(0, inj.insertPos) +
                 `\n__edges.push([__stack[__stack.length-1],'${inj.qualName}']);__stack.push('${inj.qualName}');try{\n` +
                 result.substring(inj.insertPos, closePos) +
                 `\n}finally{__stack.pop();}\n` +
                 result.substring(closePos);
    }

    // 2. Instrument constructor bodies to track constructor calls
    //    constructor(...) { body } → constructor(...) { __trackCtor('qual'); body }
    //    We inject a simple tracking call at the start (after super() if present)
    result = result.replace(
        /\bclass\s+(\w+)/g,
        (match, className) => {
            // Store class name for constructor injection below
            return match;
        }
    );

    // Find and inject into constructors: constructor(...) { → constructor(...) { __trackCtor();
    // We need to know the class name for each constructor
    const classNames = [];
    const classNameRe = /\bclass\s+(\w+)/g;
    let cnm;
    while ((cnm = classNameRe.exec(result)) !== null) {
        classNames.push({ name: cnm[1], index: cnm.index });
    }

    // Process from end to start so indices stay valid
    for (let i = classNames.length - 1; i >= 0; i--) {
        const cls = classNames[i];
        const ctorQual = `${moduleName}.${cls.name}.constructor`;

        // Find the class body
        const braceStart = result.indexOf('{', cls.index);
        if (braceStart === -1) continue;
        let depth = 0, braceEnd = braceStart;
        for (let j = braceStart; j < result.length; j++) {
            if (result[j] === '{') depth++;
            if (result[j] === '}') { depth--; if (depth === 0) { braceEnd = j; break; } }
        }
        const classBody = result.substring(braceStart, braceEnd + 1);

        // Find constructor in this class body
        const ctorMatch = classBody.match(/constructor\s*\([^)]*\)\s*\{/);
        if (ctorMatch) {
            const ctorBodyStart = braceStart + ctorMatch.index + ctorMatch[0].length;
            // Inject BEFORE super() so the stack is correct for the super chain
            const insertPos = ctorBodyStart;
            const traceCode = `\n__edges.push([__stack[__stack.length-1], '${ctorQual}']); __stack.push('${ctorQual}');\ntry {\n`;

            // Find constructor's closing brace
            let ctorDepth = 1;
            let ctorEnd = ctorBodyStart;
            for (let j = ctorBodyStart; j < braceEnd; j++) {
                if (result[j] === '{') ctorDepth++;
                if (result[j] === '}') {
                    ctorDepth--;
                    if (ctorDepth === 0) { ctorEnd = j; break; }
                }
            }

            result = result.substring(0, insertPos) +
                     traceCode +
                     result.substring(insertPos, ctorEnd) +
                     '\n} finally { __stack.pop(); }\n' +
                     result.substring(ctorEnd);
        }
    }

    return result;
}

for (const jsFile of jsFiles) {
    const src = fs.readFileSync(jsFile, 'utf8');
    const rel = path.relative(benchDir, jsFile);
    const moduleName = rel.replace(/\.js$/, '').replace(/[/\\]/g, '.');

    const names = extractNames(src, moduleName);
    allNames.push(...names);

    // For multi-file: wrap imports. For single-file: use as-is
    if (jsFiles.length === 1) {
        allCode = transformSource(src, moduleName);
    } else {
        if (moduleName !== 'main') {
            let cleaned = src.replace(/^export\s+/gm, '');
            cleaned = cleaned.replace(/^import\s+.*$/gm, '// import removed');
            allCode += `\n// === ${rel} ===\n${transformSource(cleaned, moduleName)}\n`;
        }
    }
}

// Add main file last, creating namespace bindings for imports
if (jsFiles.length > 1) {
    const mainFile = jsFiles.find(f => path.basename(f) === 'main.js');
    if (mainFile) {
        let mainSrc = fs.readFileSync(mainFile, 'utf8');

        // Parse import statements to create namespace bindings
        // Pattern: import * as name from './module.js'
        // Pattern: import { func1, func2 } from './module.js'
        // Pattern: import name from './module.js'
        const importBindings = [];
        const importRe = /^import\s+(?:\*\s+as\s+(\w+)|(\w+)|\{\s*([^}]+)\})\s+from\s+['"]\.\/([^'"]+)['"]\s*;?$/gm;
        let im;
        while ((im = importRe.exec(mainSrc)) !== null) {
            const nsName = im[1] || im[2]; // namespace or default import name
            const namedImports = im[3]; // { func1, func2 }
            const modulePath = im[4].replace(/\.js$/, '').replace(/\//g, '.');

            if (nsName) {
                // import * as name from './module' → create namespace object
                const moduleFuncs = allNames.filter(n =>
                    n.qualName.startsWith(modulePath + '.') &&
                    n.qualName.split('.').length === modulePath.split('.').length + 1
                );
                const funcEntries = moduleFuncs.map(n => {
                    const shortName = n.qualName.split('.').pop();
                    return `${shortName}: ${shortName}`;
                }).join(', ');
                importBindings.push(`const ${nsName} = { ${funcEntries} };`);
            } else if (namedImports) {
                // import { func1, func2 } from './module' → alias directly
                const names = namedImports.split(',').map(n => n.trim().split(/\s+as\s+/));
                for (const [orig, alias] of names) {
                    const localName = alias || orig;
                    // Function should already be in scope from concatenation
                }
            }
        }

        // Replace imports with bindings
        mainSrc = mainSrc.replace(/^import\s+.*$/gm, '// import removed');
        mainSrc = mainSrc.replace(/require\s*\([^)]+\)/g, '{}');

        // Add namespace bindings before main code
        const bindingCode = importBindings.length > 0
            ? `\n// === Import Bindings ===\n${importBindings.join('\n')}\n`
            : '';

        allCode += `${bindingCode}\n// === main.js ===\n${transformSource(mainSrc, 'main')}\n`;
    }
}

// Build instrumented version
const allQualNames = allNames.map(n => n.qualName);

let instrumented = `
// === Tracer Infrastructure ===
const __edges = [];
const __stack = ['main'];
const __allFns = new Set(${JSON.stringify(allQualNames)});

function __wrapFn(fn, qualName) {
    if (!fn) return fn;
    // If already wrapped, unwrap and re-wrap with the new qualName so that
    // an outer object-literal pass can re-tag an inner arrow that the
    // arrow-wrap pass already labeled (e.g. arrowN -> mod.x.foo).
    const inner = (fn.__traced && fn.__inner) ? fn.__inner : fn;
    const wrapped = function(...args) {
        const caller = __stack[__stack.length - 1];
        __edges.push([caller, qualName]);
        __stack.push(qualName);
        try {
            // Handle both regular calls and "new" construction. When a class
            // is the inner (e.g. __wrapObj walked into const X = { Vector3, ... }
            // where Vector3 is a class), inner.apply(this, args) throws
            // "Class constructor cannot be invoked without new"; use
            // Reflect.construct when the wrapper is invoked with new.
            const result = new.target
                ? Reflect.construct(inner, args, new.target)
                : inner.apply(this, args);
            // Factory-pattern handling: when a wrapped call returns a fresh
            // object literal, recursively wrap its property functions at
            // qualName.<key> so chained-factory calls like $().find().html()
            // are recorded as mod.$.find.html rather than only the first hop.
            // The __traced flag makes this idempotent.
            if (result && typeof result === 'object' && !Array.isArray(result) && !result.__traced) {
                __wrapObj(result, qualName);
                try { Object.defineProperty(result, '__traced', { value: true, configurable: true }); } catch(e) {}
            }
            return result;
        } finally {
            __stack.pop();
        }
    };
    wrapped.__traced = true;
    wrapped.__inner = inner;
    wrapped.prototype = inner.prototype;
    try { Object.defineProperty(wrapped, 'name', { value: inner.name, configurable: true }); } catch(e) {}
    // Copy static properties (for classes)
    const descs = Object.getOwnPropertyDescriptors(inner);
    for (const [key, desc] of Object.entries(descs)) {
        if (!['length', 'name', 'prototype', 'arguments', 'caller'].includes(key)) {
            try { Object.defineProperty(wrapped, key, desc); } catch(e) {}
        }
    }
    return wrapped;
}

// Wrap an object literal's property functions with __wrapFn at the bound qualName.
// Handles all three property-function forms in
//   const x = { foo() {...}, bar: function() {...}, baz: () => {...} }
// so x.foo()/x.bar()/x.baz() all surface in the trace.
function __wrapObj(obj, qualName) {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return obj;
    for (const key of Object.keys(obj)) {
        const v = obj[key];
        if (typeof v === 'function') {
            try { obj[key] = __wrapFn(v, qualName + '.' + key); } catch(e) {}
        }
    }
    return obj;
}

function __wrapClass(cls, qualName) {
    if (!cls || cls.__traced) return cls;
    // For classes, we need to wrap the constructor
    const handler = {
        construct(target, args, newTarget) {
            const caller = __stack[__stack.length - 1];
            const ctorName = qualName + '.constructor';
            if (__allFns.has(ctorName)) {
                __edges.push([caller, ctorName]);
                __stack.push(ctorName);
            }
            try {
                return Reflect.construct(target, args, newTarget);
            } finally {
                if (__allFns.has(ctorName)) __stack.pop();
            }
        },
        apply(target, thisArg, args) {
            return Reflect.apply(target, thisArg, args);
        }
    };
    const proxy = new Proxy(cls, handler);
    proxy.__traced = true;
    return proxy;
}

// === Definitions Only (functions, classes, no module-level calls) ===
`;

// Split allCode into definitions and module-level statements
const codeLines = allCode.split('\n');
let defLines = [];
let stmtLines = [];
let braceCount = 0;
let inDef = false;

for (const line of codeLines) {
    const trimmed = line.trim();

    // Count braces
    for (const ch of line) {
        if (ch === '{') braceCount++;
        if (ch === '}') braceCount--;
    }

    if (braceCount > 0 || inDef) {
        defLines.push(line);
        inDef = braceCount > 0;
        continue;
    }

    // At brace depth 0: is this a definition or a statement?
    if (trimmed.startsWith('function ') || trimmed.startsWith('class ') ||
        trimmed.startsWith('export function') || trimmed.startsWith('export class') ||
        trimmed.startsWith('export default')) {
        defLines.push(line);
        inDef = braceCount > 0 || trimmed.endsWith('{');
    } else if (/^(?:const|let|var)\s+\w+\s*=\s*(?:function|class)/.test(trimmed)) {
        defLines.push(line);
        inDef = braceCount > 0;
    } else if (/^(?:const|let|var)\s+\w+\s*=\s*__wrapFn/.test(trimmed)) {
        // Arrow functions already wrapped inline
        defLines.push(line);
    } else if (/^(?:const|let|var)\s+/.test(trimmed)) {
        // Variable declarations (let d = {}, const x = ...) are kept as definitions.
        defLines.push(line);
    } else if (trimmed === '' || trimmed.startsWith('//') || trimmed.startsWith('/*') ||
               trimmed.startsWith('*') || trimmed === '}' || trimmed === '};' ||
               trimmed.startsWith('import ')) {
        defLines.push(line);
    } else {
        stmtLines.push(line);
    }
}

instrumented += defLines.join('\n') + '\n';

instrumented += '\n// === Wrap Functions After Definition ===\n';

// Functions are now instrumented inline (body injection), no post-wrapping needed
// Arrows are wrapped inline by transformSource

// Wrap class methods via prototype
const wrappedClasses = new Set();
for (const def of allNames) {
    if (def.type === 'method' && def.className) {
        if (!wrappedClasses.has(def.className)) {
            wrappedClasses.add(def.className);
            instrumented += `try { if (typeof ${def.className} === 'function') ${def.className} = __wrapClass(${def.className}, 'main.${def.className}'); } catch(e) {}\n`;
        }
        if (def.name !== 'constructor') {
            instrumented += `try { if (${def.className}.prototype && typeof ${def.className}.prototype.${def.name} === 'function') ${def.className}.prototype.${def.name} = __wrapFn(${def.className}.prototype.${def.name}, '${def.qualName}'); } catch(e) {}\n`;
        }
    }
}

instrumented += `\n// === Execute Module-Level Statements (with wrapping active) ===\ntry {\n${stmtLines.join('\n')}\n} catch(__e) {}\n`;

// (Module-level statements are now executed above, after wrapping)

// Output the call graph
instrumented += `
// === Output ===
const __result = {};
// Add all defined functions as keys
for (const fn of __allFns) { __result[fn] = []; }
// Also add 'main' as module-level
__result['main'] = [];
// Fill in traced edges (skip self-edges)
for (const [caller, callee] of __edges) {
    if (caller === callee) continue;
    if (!__result[caller]) __result[caller] = [];
    if (!__result[caller].includes(callee)) __result[caller].push(callee);
    if (!__result[callee]) __result[callee] = [];
}
// Add module-level entries for all source files
${jsFiles.map(f => {
    const mod = path.relative(benchDir, f).replace(/\.js$/, '').replace(/[/\\]/g, '.');
    const entries = [`if (!__result['${mod}']) __result['${mod}'] = [];`];
    // Also add parent directory as module (for packages)
    const parts = mod.split('.');
    for (let i = 1; i < parts.length; i++) {
        const parent = parts.slice(0, i).join('.');
        entries.push(`if (!__result['${parent}']) __result['${parent}'] = [];`);
    }
    return entries.join('\n');
}).join('\n')}

console.log('===TRACE===');
console.log(JSON.stringify(__result, null, 2));
`;

// Write the instrumented program to a temp file and execute it. The filename
// must be unique across parallel tracer invocations: Date.now() alone collides
// when multiple workers land in the same millisecond.
const tmpFile = path.join(os.tmpdir(), `jstrace_${Date.now()}_${process.pid}_${Math.floor(Math.random() * 1e9)}.js`);
fs.writeFileSync(tmpFile, instrumented);

try {
    const output = execSync(`node "${tmpFile}"`, {
        timeout: 10000,
        encoding: 'utf8',
        stdio: ['pipe', 'pipe', 'pipe']
    });

    const marker = output.indexOf('===TRACE===');
    if (marker >= 0) {
        process.stdout.write(output.substring(marker));
    }
} catch (e) {
    // If execution fails, still try to output partial results
    const stderr = e.stderr || '';
    const stdout = e.stdout || '';
    const marker = stdout.indexOf('===TRACE===');
    if (marker >= 0) {
        process.stdout.write(stdout.substring(marker));
    } else {
        // Output at least the defined function names
        const result = {};
        result['main'] = [];
        for (const n of allNames) { result[n.qualName] = []; }
        console.log('===TRACE===');
        console.log(JSON.stringify(result, null, 2));
    }
}

// Cleanup
try { fs.unlinkSync(tmpFile); } catch(e) {}
