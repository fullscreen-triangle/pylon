/**
 * API-surface test: import every name the contract §6.5 locks at 1.0 and assert
 * the shapes typecheck and the headline flows work through the public entry
 * point only (no deep imports). If this compiles and passes, the locked surface
 * is intact.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { 
// constants
TICK, 
// coords
makeCoords, isCoordError, shell, shellCapacity, coordTuple, 
// SRN
parseSrn, isParseError, glyph, evalExpr, encodeTrajectory, decodeTrajectory, 
// agents
ProcessAgent, agentId, 
// market
yieldOf, separationCost, clearMarket, forcedUtilisation, 
// node
SrnNode, 
// cluster
Cluster, 
// kuramoto
orderParameter, } from "../src/index.js";
test("constants: TICK is a positive number", () => {
    assert.equal(typeof TICK, "number");
    assert.ok(TICK > 0);
});
test("coords surface: make/validate/shell/tuple all reachable from index", () => {
    const c = makeCoords(2, 1, 0, 1);
    assert.ok(!isCoordError(c));
    const t = coordTuple(c);
    assert.deepEqual(t, [2, 1, 0, 1]);
    assert.equal(shell(3).length, shellCapacity(3));
});
test("SRN surface: parse, eval, trajectory codec reachable from index", () => {
    const g = parseSrn(`|t : (2,1,0,+)| not { x } do { emit self.n } to { * }`);
    assert.ok(!isParseError(g));
    const r = evalExpr(g, { coords: makeCoords(2, 1, 0, 1), trajectoryCount: 0 }, new Map());
    assert.equal(r.kind, "value");
    const decoded = decodeTrajectory(encodeTrajectory(g));
    assert.ok(!("error" in decoded));
});
test("agent surface: ProcessAgent satisfies the Agent interface", () => {
    const a = new ProcessAgent({
        id: agentId("api"),
        frame: makeCoords(1, 0, 0, 1),
        loc: [0, 0],
        goal: [0.05, 0],
    });
    assert.equal(typeof a.residual(), "number");
    assert.equal(typeof a.committedStep(), "number");
    assert.equal(typeof a.monitoringCell(), "string");
});
test("market surface: yieldOf / separationCost / clearMarket / forcedUtilisation reachable", () => {
    const slots = [
        { id: "e0", capacity: 1, maxRate: 1 },
        { id: "e1", capacity: 1, maxRate: 1 },
    ];
    const ags = [agentId("x0"), agentId("x1")];
    const payoff = (a, s) => (a === "x0" ? (s === "e0" ? 5 : 1) : s === "e1" ? 5 : 1);
    const { assignment, prices } = clearMarket(ags, slots, payoff, TICK);
    assert.equal(assignment.get(agentId("x0")), "e0");
    const p = prices.get("e0");
    assert.ok(p >= 0);
    assert.equal(typeof yieldOf(assignment, ags, slots, payoff, TICK), "number");
    assert.equal(typeof separationCost("e0", assignment, ags, slots, payoff, TICK), "number");
    assert.ok(forcedUtilisation({ pressure: 3, capacity: 1, tick: 1, maxRate: 1 }) > 0);
});
test("node surface: SrnNode eval reachable from index", () => {
    const node = SrnNode.reference(makeCoords(1, 0, 0, 1));
    const g = glyph({ name: "n", target: makeCoords(1, 0, 0, 1), notGuard: "x", body: "b", toTarget: "*" });
    node.evaluate(g);
    assert.equal(node.trajectoryCount(), 1);
});
test("cluster surface: Cluster.submit returns a Yield union", () => {
    const config = {
        nodes: [{ id: "n0", frame: [2, 1, 0, 1], capacity: 1, taskDuration: 1, maxRate: 1 }],
    };
    const c = new Cluster(config);
    const y = c.submit(`|t : (2,1,0,+)| not { z } do { emit self.n } to { * }`);
    assert.equal(y.ok, true);
    assert.equal(typeof orderParameter([0, 0]).R, "number");
});
//# sourceMappingURL=api-surface.test.js.map