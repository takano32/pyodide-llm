// tests/names.mjs (T350; a file of its own from T368.8): the names of a module, read off its syntax tree.
//
//   namesOf(source)   source: JavaScript (a .ts file with its types blanked: tests/untyped.mjs)
//
// What reads it: the rule of tests/types.mjs (a name a file of the runtime uses and neither declares nor imports is one
// of the worker's ambient names), and the checks of what the modules keep to their owners (worker-modules-check.mjs,
// page-modules-check.mjs) and of the benchmark's tables (bench-modules-check.mjs).
// The parser is @babel/parser, which Astro's packages bring (npm ci installs it; this repository does not name it).
import assert from "node:assert/strict";
import { parse } from "@babel/parser";

/** { free: the names the file uses and does not declare, imported: the names it takes from other files, inner: the
 * names it declares below its top level, kinds: how each top-level name is declared ("let", "const", "function",
 * "class", "import") } */
export function namesOf(source) {
  const program = parse(source, { sourceType: "module" }).program;
  const scopes = [], free = new Set(), inner = new Set();
  const known = (name) => scopes.some((scope) => scope.has(name));
  // the names a pattern binds, and the expressions inside it (defaults, computed keys), which are read
  const bound = (node, names = [], inside = []) => {
    if (!node) return names;
    if (node.type === "Identifier") names.push(node.name);
    else if (node.type === "ObjectPattern") {
      for (const property of node.properties) {
        if (property.type !== "RestElement" && property.computed) inside.push(property.key);
        bound(property.type === "RestElement" ? property.argument : property.value, names, inside);
      }
    } else if (node.type === "ArrayPattern") node.elements.forEach((element) => bound(element, names, inside));
    else if (node.type === "AssignmentPattern") {
      inside.push(node.right);
      bound(node.left, names, inside);
    } else if (node.type === "RestElement") bound(node.argument, names, inside);
    else if (node.type === "MemberExpression") inside.push(node);  // (an assignment's pattern: [a.b, c] = ...)
    else throw new Error(`a pattern this check does not know: ${node.type}`);
    return names;
  };
  const declarations = (statements) => statements.flatMap((statement) => {
    const s = statement.type === "ExportNamedDeclaration" && statement.declaration ? statement.declaration : statement;
    if (s.type === "VariableDeclaration") {
      assert.notEqual(s.kind, "var", "var: this check knows let and const");
      return s.declarations.flatMap((d) => bound(d.id));
    }
    if (s.type === "ImportDeclaration") return s.specifiers.map((specifier) => specifier.local.name);
    return s.type === "FunctionDeclaration" || s.type === "ClassDeclaration" ? [s.id.name] : [];
  });
  const scoped = (names, inside) => {
    if (scopes.length) names.forEach((name) => inner.add(name));
    scopes.push(new Set(names));
    inside();
    scopes.pop();
  };
  const block = (statements) => scoped(declarations(statements), () => statements.forEach((statement) => walk(statement)));
  const pattern = (node) => {
    const inside = [];
    const names = bound(node, [], inside);
    return { names, read: () => inside.forEach((expression) => walk(expression)) };
  };
  const callable = (node) => {
    const parameters = node.params.map(pattern);
    scoped([...parameters.flatMap((p) => p.names), "arguments", ...(node.type === "FunctionExpression" && node.id ? [node.id.name] : [])], () => {
      parameters.forEach((p) => p.read());
      if (node.body.type === "BlockStatement") block(node.body.body);
      else walk(node.body);
    });
  };
  function walk(node) {
    if (!node || typeof node.type !== "string") return;
    switch (node.type) {
      case "Identifier":
        if (!known(node.name)) free.add(node.name);
        return;
      case "AssignmentExpression":
        if (node.left.type === "Identifier" || node.left.type.endsWith("Pattern")) {
          const target = pattern(node.left);
          target.names.forEach((name) => !known(name) && free.add(name));
          target.read();
        } else walk(node.left);
        return walk(node.right);
      case "MemberExpression": case "OptionalMemberExpression":
        walk(node.object);
        if (node.computed) walk(node.property);
        return;
      case "ObjectProperty": case "ClassProperty":
        if (node.computed) walk(node.key);
        return walk(node.value);
      case "ObjectMethod": case "ClassMethod": case "ClassPrivateMethod":
        if (node.computed) walk(node.key);
        return callable(node);
      case "FunctionDeclaration": case "FunctionExpression": case "ArrowFunctionExpression":
        return callable(node);
      case "ClassDeclaration": case "ClassExpression":
        return scoped(node.id ? [node.id.name] : [], () => {
          walk(node.superClass);
          node.body.body.forEach((member) => walk(member));
        });
      case "BlockStatement": case "StaticBlock":
        return block(node.body);
      case "VariableDeclaration":
        for (const d of node.declarations) {
          pattern(d.id).read();
          walk(d.init);
        }
        return;
      case "ForStatement": case "ForInStatement": case "ForOfStatement": {
        const head = node.init ?? node.left;
        return scoped(head?.type === "VariableDeclaration" ? head.declarations.flatMap((d) => bound(d.id)) : [],
          () => [node.init, node.left, node.right, node.test, node.update, node.body].forEach((part) => walk(part)));
      }
      case "CatchClause": {
        const caught = node.param ? pattern(node.param) : { names: [], read() {} };
        return scoped(caught.names, () => {
          caught.read();
          block(node.body.body);
        });
      }
      case "SwitchStatement":
        walk(node.discriminant);
        return scoped(declarations(node.cases.flatMap((c) => c.consequent)), () => node.cases.forEach((c) => {
          walk(c.test);
          c.consequent.forEach((statement) => walk(statement));
        }));
      case "LabeledStatement":
        return walk(node.body);
      case "BreakStatement": case "ContinueStatement": case "MetaProperty": case "ImportDeclaration":
        return;
      case "ExportNamedDeclaration":
        // (T349: `export { a, b };` uses the file's own a and b, tests/forward-modules-check.mjs)
        if (!node.declaration && !node.source) return node.specifiers.forEach((specifier) => walk(specifier.local));
        assert.ok(node.declaration, "an export from another file: this check knows `export const`, `export function` and `export { a }`");
        return walk(node.declaration);
      default:
        for (const [key, value] of Object.entries(node)) {
          if (key === "loc" || key.endsWith("Comments") || !value || typeof value !== "object") continue;
          (Array.isArray(value) ? value : [value]).forEach((child) => walk(child));
        }
    }
  }
  const declared = declarations(program.body);
  scoped(declared, () => program.body.forEach((statement) => walk(statement)));
  const kinds = new Map(program.body.flatMap((statement) => {
    const s = statement.type === "ExportNamedDeclaration" && statement.declaration ? statement.declaration : statement;
    const kind = s.type === "VariableDeclaration" ? s.kind : s.type === "ImportDeclaration" ? "import" : s.type === "FunctionDeclaration" ? "function" : "class";
    return declarations([statement]).map((name) => [name, kind]);
  }));
  const imported = [...kinds].filter(([, kind]) => kind === "import").map(([name]) => name);
  return { free, imported, inner, kinds };
}
