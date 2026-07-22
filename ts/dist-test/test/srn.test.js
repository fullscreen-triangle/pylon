import { test } from "node:test";
import assert from "node:assert/strict";
import { makeCoords, isCoordError } from "../src/coords.js";
import { glyph, evalExpr } from "../src/srn/expr.js";
import { parseSrn, isParseError } from "../src/srn/parse.js";
function coords(n, l, m, s) {
    const c = makeCoords(n, l, m, s);
    if (isCoordError(c))
        throw new Error(c.message);
    return c;
}
function frame(c, M = 0) {
    return { coords: c, trajectoryCount: M };
}
const EMPTY = new Map();
test("receiver-relativity: same glyph, different frames -> different value (SRN Thm 4.2)", () => {
    const g = glyph({
        name: "ping",
        target: coords(2, 1, 0, 1),
        notGuard: "blocked",
        body: "echo self",
        toTarget: "return",
    });
    const r1 = evalExpr(g, frame(coords(1, 0, 0, 1)), EMPTY);
    const r2 = evalExpr(g, frame(coords(2, 1, 0, 1)), EMPTY);
    assert.equal(r1.kind, "value");
    assert.equal(r2.kind, "value");
    const v1 = r1.value;
    const v2 = r2.value;
    assert.notEqual(v1.self_n, v2.self_n);
});
test("determinism: same glyph, same frame -> identical value (SRN Thm 4.1)", () => {
    const g = glyph({
        name: "det",
        target: coords(2, 1, 0, 1),
        notGuard: "blocked",
        body: "b",
        toTarget: "return",
    });
    const f = frame(coords(2, 1, 0, 1), 5);
    const a = evalExpr(g, f, EMPTY);
    const b = evalExpr(g, f, EMPTY);
    assert.deepEqual(a, b);
});
test("not-guard fires when the boundary key is present", () => {
    const g = glyph({
        name: "blocked-glyph",
        target: coords(1, 0, 0, 1),
        notGuard: "forbidden-key",
        body: "body",
        toTarget: "nowhere",
    });
    const env = new Map([["forbidden-key", true]]);
    const r = evalExpr(g, frame(coords(1, 0, 0, 1)), env);
    assert.equal(r.kind, "rejected");
});
test("parseSrn rejects a glyph with no `not` clause (SRN Cor 2.1)", () => {
    const src = `|noboundary : (2,1,0,+)| do { emit self.n } to { n = 2 }`;
    const r = parseSrn(src);
    assert.ok(isParseError(r));
    assert.equal(r.kind, "no-negation-boundary");
});
test("parseSrn parses a well-formed glyph with all clauses", () => {
    const src = `|identity : (2,1,0,+)|
      not { n != 2 }
      do  { emit self.n }
      to  { n = 2, * }
      as  { id }`;
    const r = parseSrn(src);
    assert.ok(!isParseError(r));
    const g = r;
    assert.equal(g.name, "identity");
    assert.equal(g.target.n, 2);
    assert.equal(g.target.s, 1);
    assert.equal(g.notGuard, "n != 2");
    assert.equal(g.alias, "id");
});
test("parseSrn reports malformed header as malformed-srn", () => {
    const r = parseSrn(`not a glyph at all`);
    assert.ok(isParseError(r));
    assert.equal(r.kind, "malformed-srn");
});
test("parsed glyph evaluates receiver-relatively", () => {
    const src = `|p : (2,1,0,-)| not { blocked } do { self } to { * }`;
    const g = parseSrn(src);
    assert.ok(!isParseError(g));
    const r = evalExpr(g, frame(coords(3, 2, 1, 1), 9), EMPTY);
    assert.equal(r.kind, "value");
    assert.equal(r.value.self_n, 3);
    assert.equal(r.value.self_M, 9);
});
//# sourceMappingURL=srn.test.js.map