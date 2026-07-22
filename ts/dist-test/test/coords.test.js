import { test } from "node:test";
import assert from "node:assert/strict";
import { makeCoords, isCoordError, shell, shellCapacity, referenceCoords, coordKey, coordsEqual, coordsFromTuple, coordTuple, } from "../src/coords.js";
test("shell capacity formula: |shell(n)| == 2n^2 for n=1..10", () => {
    for (let n = 1; n <= 10; n++) {
        assert.equal(shell(n).length, shellCapacity(n), `C(${n}) mismatch`);
        assert.equal(shellCapacity(n), 2 * n * n);
    }
});
test("reference node (1,0,0,+1) is valid", () => {
    const c = makeCoords(1, 0, 0, 1);
    assert.ok(!isCoordError(c));
    assert.ok(coordsEqual(c, referenceCoords()));
});
test("invalid coords rejected with typed error (never thrown)", () => {
    assert.ok(isCoordError(makeCoords(1, 1, 0, 1))); // l >= n
    assert.ok(isCoordError(makeCoords(2, 1, 2, 1))); // |m| > l
    assert.ok(isCoordError(makeCoords(2, 1, 0, 0))); // s not in {+-1}
    assert.ok(isCoordError(makeCoords(0, 0, 0, 1))); // n < 1
});
test("every enumerated shell coordinate validates", () => {
    for (let n = 1; n <= 6; n++) {
        for (const c of shell(n)) {
            const round = makeCoords(c.n, c.l, c.m, c.s);
            assert.ok(!isCoordError(round), `shell coord ${coordKey(c)} should validate`);
        }
    }
});
test("coordKey formats parity as +/-", () => {
    assert.equal(coordKey({ n: 2, l: 1, m: -1, s: 1 }), "(2,1,-1,+)");
    assert.equal(coordKey({ n: 3, l: 2, m: 0, s: -1 }), "(3,2,0,-)");
});
test("tuple round-trip", () => {
    const c = makeCoords(4, 2, -1, -1);
    assert.ok(!isCoordError(c));
    const t = coordTuple(c);
    assert.deepEqual(t, [4, 2, -1, -1]);
    const back = coordsFromTuple(t);
    assert.ok(!isCoordError(back));
    assert.ok(coordsEqual(back, c));
});
//# sourceMappingURL=coords.test.js.map