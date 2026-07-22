import { test } from "node:test";
import assert from "node:assert/strict";
import { makeCoords, isCoordError } from "../src/coords.js";
import { ProcessAgent, agentId, TICK, } from "../src/agent.js";
function coords(n, l, m, s) {
    const c = makeCoords(n, l, m, s);
    if (isCoordError(c))
        throw new Error(c.message);
    return c;
}
// A deterministic occupation that cycles through a fixed ring of goals.
function ringOccupation(goals) {
    let i = 0;
    return () => {
        i = (i + 1) % goals.length;
        return goals[i];
    };
}
test("occupancy: every live (non-attaining) step descends residual by >= tau_0 (Thm 9.1)", () => {
    const a = new ProcessAgent({
        id: agentId("occ"),
        frame: coords(1, 0, 0, 1),
        loc: [0, 0],
        goal: [5, 5], // far -> many live steps before attainment
        cellWidth: 0.1,
    });
    a.run(200);
    // minLiveStepSeen is the smallest per-tick decrement over live steps; must be >= tau_0.
    assert.ok(a.minLiveStepSeen() >= TICK - 1e-9, `min live step ${a.minLiveStepSeen()} should be >= tau_0 ${TICK}`);
});
test("monotone committed step M never decreases (invariant I2 / incorruptibility)", () => {
    const a = new ProcessAgent({
        id: agentId("mono"),
        frame: coords(2, 1, 0, 1),
        loc: [0, 0],
        goal: [1, 0],
        occupation: ringOccupation([[1, 0], [2, 0], [0, 0]]),
    });
    let prev = a.committedStep();
    for (let i = 0; i < 300; i++) {
        a.step();
        const now = a.committedStep();
        assert.ok(now >= prev, `M decreased: ${prev} -> ${now}`);
        prev = now;
    }
});
test("persistence: with an occupation, the agent succeeds past r<=beta repeatedly (Thm 9.x)", () => {
    // Goals a few dozen ticks apart so several successions fit the tick budget:
    // attainment takes ceil(r0/tau_0) ticks (Cor. settling), r0 ~ 0.05 -> ~50 ticks.
    const a = new ProcessAgent({
        id: agentId("persist"),
        frame: coords(2, 1, 0, 1),
        loc: [0, 0],
        goal: [0.05, 0],
        occupation: ringOccupation([[0.05, 0], [0.1, 0.05], [0, 0.08]]),
    });
    a.run(1000);
    assert.ok(a.goalsAttained() >= 2, `expected multiple successions, got ${a.goalsAttained()}`);
    assert.notEqual(a.currentState(), "retired");
});
test("no succession map: agent retires at first completion", () => {
    const a = new ProcessAgent({
        id: agentId("mortal"),
        frame: coords(1, 0, 0, 1),
        loc: [0, 0],
        goal: [0.3, 0],
    });
    a.run(500);
    assert.equal(a.currentState(), "retired");
    assert.equal(a.goalsAttained(), 1);
});
test("ever-fresh response: cell crosses and response changes across interactions (Prop 9.x)", () => {
    const a = new ProcessAgent({
        id: agentId("fresh"),
        frame: coords(2, 1, 0, 1),
        loc: [0, 0],
        goal: [0.5, 0], // ~500 ticks away; cell width 0.1 -> crossings every ~100 ticks
        cellWidth: 0.1,
    });
    const before = a.respond((cell, M) => `${cell}#${M}`);
    a.run(300);
    const after = a.respond((cell, M) => `${cell}#${M}`);
    assert.ok(a.cellCrossingCount() >= 1, "agent should cross >= 1 monitoring cell");
    assert.notEqual(before, after, "response should differ once the inner state advances");
});
test("exclusive phases (I4): after a step the agent is in a defined phase", () => {
    const a = new ProcessAgent({
        id: agentId("phase"),
        frame: coords(1, 0, 0, 1),
        loc: [0, 0],
        goal: [2, 0],
    });
    a.step();
    assert.ok(a.currentPhase() === "observe" || a.currentPhase() === "commit");
});
test("residual is the felt drive: descends monotonically and reaches the floor", () => {
    const a = new ProcessAgent({
        id: agentId("drive"),
        frame: coords(1, 0, 0, 1),
        loc: [0, 0],
        goal: [0.2, 0], // ~200 ticks -> attains within budget
    });
    const r0 = a.residual();
    // descent is monotone tick to tick
    let prev = r0;
    for (let i = 0; i < 250; i++) {
        a.step();
        const now = a.residual();
        assert.ok(now <= prev + 1e-9, `residual rose: ${prev} -> ${now}`);
        prev = now;
        if (a.currentState() === "retired")
            break;
    }
    assert.ok(a.currentState() === "retired", "agent with no succession retires at the floor");
    assert.ok(r0 > 0);
});
//# sourceMappingURL=agent.test.js.map