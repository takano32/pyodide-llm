// tests/contracts.ts (T.368.2): what the types of src/runtime/contracts/ say, read for the checks.
//
// A contract is a shape that crosses a line as a plain object: the plan Python hands forward.js, the messages of a
// page and its worker, of forward.js and the GPU's worker, the form of a checkpoint, the requests of a conversion's
// conduct. tsc holds a contract to whatever TypeScript reads it. It cannot hold it to Python, nor to a .js file (which
// is not checked), so the checks that already hold the two ends of a line together hold the type to them as well:
// a key that is added on one side and not in the type, or the other way, fails the light suite.
//
//   const c = contract("plan.ts")        a file of src/runtime/contracts/ (of this tree, or of treeOf(...)'s)
//   c.keysOf("Plan")                     the keys of an object type, through the aliases of the same file, its
//                                        intersections, and `| null`; a key that can only be undefined (`load?:
//                                        undefined`, which says "never has it") is none; an index by a template
//                                        (`signs.${number}`) is the key "signs.<number>"
//   c.literalsOf("TensorName")           the texts of a union of string literals
//   c.membersOf("FromWorker")            of a union of object types told apart by `type`: Map of the tag -> its keys
//                                        (members with one tag together), the tag itself not among them
//   c.tuplesOf("Request")                of a union of tuples that begin with a text: Map of that text -> how long
//
// And what the JavaScript and the TypeScript of the runtime write and read of a message, by their syntax trees:
//   posted(trees, to)                    Map of the type -> { keys, spreads } of every postMessage({ type: "...", ... })
//   told(trees)                          the texts `data.type` is compared with
//   differing(what, declared, written, more)   both ways; more: { key: why it is not written there }. The lines.
//
// Only what these checks need of TypeScript's types is read: anything else in the place of a type asked for throws.
import fs from "node:fs";
import { parse } from "@babel/parser";
import { runtime } from "./tree.ts";
import { untyped } from "./untyped.ts";

export function contract(name, find = runtime) {
  const file = find(`contracts/${name}`);
  return contractOf(fs.readFileSync(file, "utf8"), name);
}

export function contractOf(text, name = "a contract") {
  const { program } = parse(text, { sourceType: "module", plugins: ["typescript"] });
  const aliases = new Map(), imported = new Set();
  for (const statement of program.body) {
    const declaration = statement.type === "ExportNamedDeclaration" ? statement.declaration : statement;
    if (declaration?.type === "TSTypeAliasDeclaration") aliases.set(declaration.id.name, declaration.typeAnnotation);
    else if (statement.type === "ImportDeclaration") statement.specifiers.forEach((specifier) => imported.add(specifier.local.name));
    else if (statement.type !== "ExportNamedDeclaration" || declaration) {
      // (a contract is types alone: a value here would be in what a browser gets)
      throw new Error(`${name}: a ${declaration?.type ?? statement.type} is no type: a contract holds types alone`);
    }
  }
  const named = (alias) => {
    if (!aliases.has(alias)) throw new Error(`${name} has no type ${alias}${imported.has(alias) ? " of its own (it imports one)" : ""}`);
    return aliases.get(alias);
  };
  const nothing = (node) => ["TSNullKeyword", "TSUndefinedKeyword"].includes(node.type);
  const keyOf = (member) => member.key.name ?? member.key.value;
  // the object types a type is made of: itself, what an alias of this file names, the parts of an intersection, and
  // of a union what is not null or undefined
  function shapes(node, where) {
    if (node.type === "TSParenthesizedType") return shapes(node.typeAnnotation, where);
    if (node.type === "TSTypeLiteral") return [node];
    if (node.type === "TSTypeReference" && node.typeName.type === "Identifier" && !node.typeParameters && aliases.has(node.typeName.name)) return shapes(named(node.typeName.name), where);
    if (node.type === "TSIntersectionType") return node.types.flatMap((part) => shapes(part, where));
    if (node.type === "TSUnionType") return node.types.filter((part) => !nothing(part)).flatMap((part) => shapes(part, where));
    throw new Error(`${name}: ${where} is not an object type that tests/contracts.ts reads (a ${node.type}${node.typeName?.name ? ` ${node.typeName.name}` : ""})`);
  }
  function keysIn(literals, where) {
    const keys = new Set();
    for (const literal of literals) {
      for (const member of literal.members) {
        if (member.type === "TSPropertySignature" && !member.computed) {
          if (!nothing(member.typeAnnotation.typeAnnotation)) keys.add(keyOf(member));
        } else if (member.type === "TSMethodSignature" && !member.computed) {
          keys.add(keyOf(member));
        } else if (member.type === "TSIndexSignature" && member.parameters[0].typeAnnotation.typeAnnotation.literal?.type === "TemplateLiteral") {
          const template = member.parameters[0].typeAnnotation.typeAnnotation.literal;
          keys.add(template.quasis.map((quasi) => quasi.value.cooked).join("<number>"));
        } else throw new Error(`${name}: ${where} has a member tests/contracts.ts does not read (a ${member.type})`);
      }
    }
    return keys;
  }
  const keysOf = (alias) => keysIn(shapes(named(alias), alias), alias);
  function literalsOf(alias) {
    const node = named(alias), parts = node.type === "TSUnionType" ? node.types : [node];
    return new Set(parts.map((part) => {
      if (part.type !== "TSLiteralType" || part.literal.type !== "StringLiteral") throw new Error(`${name}: ${alias} is not a union of texts (a ${part.type})`);
      return part.literal.value;
    }));
  }
  function membersOf(alias, tag = "type") {
    const node = named(alias), members = new Map();
    for (const part of node.type === "TSUnionType" ? node.types : [node]) {
      const literals = shapes(part, alias);
      const tags = literals.flatMap((literal) => literal.members.filter((member) => member.type === "TSPropertySignature" && keyOf(member) === tag));
      const said = tags[0]?.typeAnnotation.typeAnnotation;
      if (tags.length !== 1 || said.type !== "TSLiteralType" || said.literal.type !== "StringLiteral") {
        throw new Error(`${name}: a member of ${alias} does not say its ${tag} as one text`);
      }
      const keys = members.get(said.literal.value) ?? new Set();
      for (const key of keysIn(literals, alias)) if (key !== tag) keys.add(key);
      members.set(said.literal.value, keys);
    }
    return members;
  }
  function tuplesOf(alias) {
    const node = named(alias), tuples = new Map();
    for (let part of node.type === "TSUnionType" ? node.types : [node]) {
      if (part.type === "TSTypeReference" && aliases.has(part.typeName.name)) part = named(part.typeName.name);
      const first = part.elementTypes?.[0], head = first?.type === "TSNamedTupleMember" ? first.elementType : first;
      if (part.type !== "TSTupleType" || head?.type !== "TSLiteralType" || head.literal.type !== "StringLiteral") {
        throw new Error(`${name}: a member of ${alias} is not a tuple that begins with a text`);
      }
      if (tuples.has(head.literal.value)) throw new Error(`${name}: two tuples of ${alias} begin with ${head.literal.value}`);
      tuples.set(head.literal.value, part.elementTypes.length);
    }
    return tuples;
  }
  return { keysOf, literalsOf, membersOf, tuplesOf, names: [...aliases.keys()] };
}

// ---- the syntax trees of the runtime's files (a .ts file with its types blanked)
// (T.368.3: by the name the tree has the file under: one asked for as x.js may be x.ts, and its types are blanked then)
export const treeOf = (file, find = runtime) => ({ file, program: parse(untyped(fs.readFileSync(find(file), "utf8"), find(file)), { sourceType: "module" }).program });
export function walk(node, visit, parent = null) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) { node.forEach((child) => walk(child, visit, parent)); return; }
  if (node.type) visit(node, parent);
  for (const key of Object.keys(node)) if (!["loc", "start", "end", "extra", "leadingComments", "trailingComments", "innerComments"].includes(key)) walk(node[key], visit, node.type ? node : parent);
}

const keyName = (property) => property.key.name ?? property.key.value;
/** The keys of an object literal, through the spreads that are literals themselves (the branches of `a ? {...} :
 * {...}`, the right of `a && {...}`, both sides of `a ?? {...}`); unknown: how many of its spreads are something else. */
export function literalKeys(node, into = { keys: new Set(), unknown: 0 }) {
  if (node?.type === "ObjectExpression") {
    for (const property of node.properties) {
      if (property.type === "SpreadElement") literalKeys(property.argument, into);
      else if (!property.computed) into.keys.add(keyName(property));
    }
  } else if (node?.type === "ConditionalExpression") { literalKeys(node.consequent, into); literalKeys(node.alternate, into); }
  else if (node?.type === "LogicalExpression" && node.operator === "&&") literalKeys(node.right, into);
  else if (node?.type === "LogicalExpression") { literalKeys(node.left, into); literalKeys(node.right, into); }
  else if (node?.type === "ParenthesizedExpression") literalKeys(node.expression, into);
  else into.unknown += 1;
  return into;
}

/** every message a file posts as a literal, postMessage({ type: "x", a, b: c, ...more }): to: "owner", the bare
 * postMessage() of a worker (to whoever made it); "worker", <something>.postMessage() (to a worker it made); else both.
 * Map of the type -> { keys: the literal's own (not type; a spread of a literal is the literal's), spreads: how many
 * of its `...` are of something else, files } */
export function posted(trees, to) {
  const messages = new Map();
  for (const { file, program } of trees) {
    walk(program, (node) => {
      if (node.type !== "CallExpression" && node.type !== "OptionalCallExpression") return;
      const callee = node.callee, called = callee.type === "Identifier" ? callee.name : callee.property?.name;
      const literal = node.arguments[0];
      if (called !== "postMessage" || literal?.type !== "ObjectExpression") return;
      if (to && (to === "owner") !== (callee.type === "Identifier")) return;
      const type = literal.properties.find((property) => property.type === "ObjectProperty" && keyName(property) === "type")?.value;
      if (type?.type !== "StringLiteral") return;
      const message = messages.get(type.value) ?? { keys: new Set(), spreads: 0, files: new Set() };
      const { keys, unknown } = literalKeys(literal);
      for (const key of keys) if (key !== "type") message.keys.add(key);
      message.spreads += unknown;
      message.files.add(file);
      messages.set(type.value, message);
    });
  }
  return messages;
}

/** the keys of every object literal the function of that name returns (its own returns, not those of a function inside it) */
export function returnedBy(trees, name) {
  const keys = new Set();
  let found = 0;
  // (down a function's body, and into no function inside it)
  const returns = (node) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) return void node.forEach(returns);
    if (/Function/.test(node.type ?? "")) return;
    if (node.type === "ReturnStatement" && node.argument) for (const key of literalKeys(node.argument).keys) keys.add(key);
    for (const key of Object.keys(node)) if (!["loc", "start", "end", "extra", "leadingComments", "trailingComments", "innerComments"].includes(key)) returns(node[key]);
  };
  for (const { program } of trees) {
    walk(program, (node) => {
      if (node.type === "FunctionDeclaration" && node.id?.name === name) { found += 1; returns(node.body); }
    });
  }
  if (found !== 1) throw new Error(`${found} functions named ${name}(): tests/contracts.ts reads what the one returns`);
  return keys;
}

/** the keys of the object literals handed to a call of that name (`onCompared?.({ best, ... })`) */
export function handedTo(trees, name) {
  const keys = new Set();
  for (const { program } of trees) {
    walk(program, (node) => {
      if ((node.type === "CallExpression" || node.type === "OptionalCallExpression") && node.callee.type === "Identifier" && node.callee.name === name) {
        for (const argument of node.arguments) if (argument.type === "ObjectExpression") for (const key of literalKeys(argument).keys) keys.add(key);
      }
    });
  }
  return keys;
}

/** the keys a variable of that name is given: those of the literals assigned to it, and of `<name>.<key> = ...` */
export function givenTo(trees, name) {
  const keys = new Set();
  for (const { program } of trees) {
    walk(program, (node) => {
      if (node.type !== "AssignmentExpression") return;
      if (node.left.type === "Identifier" && node.left.name === name && node.right.type === "ObjectExpression") for (const key of literalKeys(node.right).keys) keys.add(key);
      if (node.left.type === "MemberExpression" && !node.left.computed && node.left.object.type === "Identifier" && node.left.object.name === name) keys.add(node.left.property.name);
    });
  }
  return keys;
}

/** the properties read off a name (`data.load`, not `data.type`'s comparisons alone: every `<name>.<key>`), and the
 * keys a parameter pattern of the function of that name takes ({ a, b, ...rest } takes a and b) */
export function readOff(trees, name) {
  const keys = new Set();
  for (const { program } of trees) {
    walk(program, (node) => {
      if ((node.type === "MemberExpression" || node.type === "OptionalMemberExpression") && !node.computed && node.object.type === "Identifier" && node.object.name === name) keys.add(node.property.name);
    });
  }
  return keys;
}
export function takenBy(trees, name) {
  const keys = new Set();
  let found = 0;
  for (const { program } of trees) {
    walk(program, (node) => {
      if (node.type !== "FunctionDeclaration" || node.id?.name !== name) return;
      found += 1;
      for (const parameter of node.params) {
        if (parameter.type === "ObjectPattern") for (const property of parameter.properties) if (property.type === "ObjectProperty" && !property.computed) keys.add(keyName(property));
      }
    });
  }
  if (found !== 1) throw new Error(`${found} functions named ${name}(): tests/contracts.ts reads what the one takes`);
  return keys;
}

/** every text a file compares `<name>.type` with (`data.type === "load"`): what a receiver tells its messages apart by */
export function told(trees, name = "data") {
  const types = new Set();
  for (const { program } of trees) {
    walk(program, (node) => {
      if (node.type !== "BinaryExpression" || !["===", "=="].includes(node.operator)) return;
      const [side, text] = node.left.type === "StringLiteral" ? [node.right, node.left] : [node.left, node.right];
      if (text.type === "StringLiteral" && side.type === "MemberExpression" && !side.computed && side.property.name === "type" &&
          side.object.type === "Identifier" && side.object.name === name) types.add(text.value);
    });
  }
  return types;
}

const sorted = (names) => [...names].sort();
/** A set of names a type declares against the same set as something else writes it; more: { name: why it is declared
 * and not written there }. Returns the lines of what is wrong (none: the two are one). */
export function differing(what, declared, written, more = {}) {
  const lines = [];
  const undeclared = sorted(written).filter((key) => !declared.has(key));
  const unwritten = sorted(declared).filter((key) => !written.has(key) && !(key in more));
  const stale = Object.keys(more).filter((key) => written.has(key) || !declared.has(key));
  if (undeclared.length) lines.push(`${what}: written and not in the type: ${undeclared.join(", ")}`);
  if (unwritten.length) lines.push(`${what}: in the type and written nowhere: ${unwritten.join(", ")}`);
  if (stale.length) lines.push(`${what}: listed as written elsewhere, and it is written here now or in the type no more: ${stale.join(", ")}`);
  return lines;
}
