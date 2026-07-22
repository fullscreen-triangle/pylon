import { test } from "node:test";
import assert from "node:assert/strict";
import { makeCoords, isCoordError, coordsEqual, shell, type PartitionCoords } from "../src/coords.js";
import {
  encodeCoords,
  decodeCoords,
  encodeTrajectory,
  decodeTrajectory,
} from "../src/srn/trajectory.js";
import { glyph } from "../src/srn/expr.js";
import { SrnNode } from "../src/node.js";

function coords(n: number, l: number, m: number, s: number): PartitionCoords {
  const c = makeCoords(n, l, m, s);
  if (isCoordError(c)) throw new Error(c.message);
  return c;
}

test("coord round-trip: decode(encode(coords)) == coords for all shells n<=6", () => {
  for (let n = 1; n <= 6; n++) {
    for (const c of shell(n)) {
      const t = encodeCoords(c, "00000000");
      const d = decodeCoords(t);
      assert.ok(!("error" in d), `decode failed for ${JSON.stringify(c)}`);
      assert.ok(coordsEqual((d as any).coords, c));
    }
  }
});

test("encoding injectivity: distinct glyph bodies -> distinct trajectories", () => {
  const target = coords(2, 1, 0, 1);
  const g1 = glyph({ name: "a", target, notGuard: "x", body: "body-one", toTarget: "*" });
  const g2 = glyph({ name: "a", target, notGuard: "x", body: "body-two", toTarget: "*" });
  const t1 = encodeTrajectory(g1);
  const t2 = encodeTrajectory(g2);
  assert.notDeepEqual(t1.deltas, t2.deltas, "different bodies must encode differently");
});

test("decodeTrajectory recovers the exact target address", () => {
  const target = coords(3, 2, -1, -1);
  const g = glyph({ name: "p", target, notGuard: "x", body: "b", toTarget: "*" });
  const decoded = decodeTrajectory(encodeTrajectory(g));
  assert.ok(!("error" in decoded));
  assert.ok(coordsEqual((decoded as any).target, target));
});

test("replay immunity: re-submitting a glyph gives a different frame (self.M advanced)", () => {
  const node = SrnNode.reference(coords(2, 1, 0, 1));
  const g = glyph({ name: "replay", target: coords(2, 1, 0, 1), notGuard: "x", body: "b", toTarget: "*" });
  const r1 = node.evaluate(g);
  const m1 = node.trajectoryCount();
  const r2 = node.evaluate(g);
  const m2 = node.trajectoryCount();
  assert.ok(m2 > m1, "committed count advances on replay");
  // the receiver-relative value embeds self_M, so the two evaluations differ
  assert.equal((r1 as any).value.self_M !== (r2 as any).value.self_M, true);
});

test("malformed trajectory decodes to a typed error, never throws", () => {
  const d = decodeCoords({ deltas: [1e-6], channels: 4 });
  assert.ok("error" in d);
});
