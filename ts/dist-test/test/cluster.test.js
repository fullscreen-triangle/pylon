import { test } from "node:test";
import assert from "node:assert/strict";
import { Cluster } from "../src/cluster.js";
import { orderParameter, criticalCoupling, KuramotoBank } from "../src/kuramoto.js";
function threeNodes() {
    return [
        { id: "node0", frame: [1, 0, 0, 1], capacity: 1, taskDuration: 1, maxRate: 1.0 },
        { id: "node1", frame: [2, 1, 0, 1], capacity: 2, taskDuration: 1, maxRate: 1.5 },
        { id: "node2", frame: [2, 1, 1, -1], capacity: 1, taskDuration: 1, maxRate: 2.0 },
    ];
}
const NOT_GUARD = "unreachable-key";
function srn(targetRegion = "*") {
    return `|task : (2,1,0,+)| not { ${NOT_GUARD} } do { emit self.n } to { ${targetRegion} }`;
}
test("submit rejects an expression with no `not` clause (SRN Cor 2.1)", () => {
    const c = new Cluster({ nodes: threeNodes() });
    const r = c.submit(`|bad : (2,1,0,+)| do { emit self.n } to { * }`);
    assert.equal(r.ok, false);
    assert.equal(r.error.kind, "no-negation-boundary");
});
test("submit lands an agent on a node and reports allocation + price", () => {
    const c = new Cluster({ nodes: threeNodes() });
    const r = c.submit(srn());
    assert.equal(r.ok, true);
    const y = r;
    assert.ok(threeNodes().some((n) => n.id === y.allocation.node));
    assert.ok(typeof y.allocation.price === "number");
    assert.equal(c.liveAgents().length, 1);
});
test("committed step advances (occupied, not waiting — Thm occupancy)", () => {
    const c = new Cluster({ nodes: threeNodes() });
    const r = c.submit(srn());
    assert.equal(r.ok, true);
    assert.ok(r.committedStep >= 1, "agent has taken at least one committed step");
});
test("persistence: an agent with a succession map keeps a live goal (Thm persistence)", () => {
    const c = new Cluster({ nodes: threeNodes() });
    const ring = [[0.02, 0.02], [0.03, 0.02], [0.02, 0.03]];
    let i = 0;
    const succession = () => {
        i = (i + 1) % ring.length;
        return ring[i];
    };
    const r = c.submit(srn(), { succession, goal: [0.02, 0.02] });
    assert.equal(r.ok, true);
    const agent = c.agent(r.agent);
    // drive it well past first completion; it must not retire (goal succession)
    for (let k = 0; k < 400; k++)
        agent.step?.();
    assert.notEqual(agent.currentState?.(), "retired");
});
test("forest openness: a node added after construction is a peer immediately (SRN Thm 8.1)", () => {
    const c = new Cluster({ nodes: threeNodes() });
    assert.equal(c.nodes().length, 3);
    c.addNode({ id: "node3", frame: [3, 2, -1, 1], capacity: 1, taskDuration: 1, maxRate: 1.2 });
    assert.equal(c.nodes().length, 4);
    assert.ok(c.nodes().some((n) => n.id === "node3"), "new node joined without enrolment");
});
test("snapshot / fromSnapshot preserves monotone committed step (incorruptibility)", () => {
    const c = new Cluster({ nodes: threeNodes() });
    c.submit(srn());
    c.submit(srn());
    const snap = c.snapshot();
    const restored = Cluster.fromSnapshot(snap);
    assert.equal(restored.liveAgents().length, snap.agents.length);
    for (const a of restored.liveAgents()) {
        const orig = snap.agents.find((x) => x.id === a.id);
        assert.ok(a.committedStep() >= orig.committedStep, "M does not regress across restart");
    }
});
// ---- Kuramoto phase-lock (network-yield §swarm) ----
test("order parameter is 1 for identical phases, ~0 for uniform spread", () => {
    assert.ok(Math.abs(orderParameter([1, 1, 1, 1]).R - 1) < 1e-9);
    const spread = [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2];
    assert.ok(orderParameter(spread).R < 1e-9);
});
test("phase-lock: coupling above K_c* drives R >= 0.95 (Thm phase-lock)", () => {
    const sigma = 0.05;
    const kc = criticalCoupling(sigma);
    // couple well above critical; identical-ish frequencies lock quickly
    const freqs = [1.0, 1.0 + sigma, 1.0 - sigma, 1.0 + sigma / 2, 1.0 - sigma / 2];
    const bank = new KuramotoBank(freqs, 4 * kc + 2.0);
    bank.run(2000, 0.02);
    assert.ok(bank.order().R >= 0.95, `expected lock, got R=${bank.order().R}`);
});
test("cluster reports phase-lock through its own API", () => {
    const c = new Cluster({ nodes: threeNodes(), coupling: 6.0 });
    const { R } = c.orderParameter();
    assert.ok(R >= 0 && R <= 1);
    assert.equal(typeof c.isPhaseLocked(), "boolean");
});
//# sourceMappingURL=cluster.test.js.map