import { test } from "node:test";
import assert from "node:assert/strict";
import { agentId, type AgentId } from "../src/agent.js";
import {
  yieldOf,
  separationCost,
  clearMarket,
  forcedUtilisation,
  type Slot,
  type Assignment,
} from "../src/market.js";

const TICK = 1e-3;

function unitSlots(n: number): Slot[] {
  return Array.from({ length: n }, (_, i) => ({ id: `e${i}`, capacity: 1, maxRate: 1 }));
}

function agents(n: number): AgentId[] {
  return Array.from({ length: n }, (_, i) => agentId(`x${i}`));
}

// Deterministic payoff matrix payoff(x_i, e_j) from a fixed pseudo-random seed.
function payoffMatrix(nA: number, nS: number, seed = 12345) {
  const M: number[][] = [];
  let s = seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return 1 + (s % 9000) / 1000; // in [1, 10)
  };
  for (let i = 0; i < nA; i++) {
    const row: number[] = [];
    for (let j = 0; j < nS; j++) row.push(Math.round(rnd() * 1000) / 1000);
    M.push(row);
  }
  const ags = agents(nA);
  const idx = new Map(ags.map((a, i) => [a, i]));
  const payoff = (a: AgentId, slot: string) => M[idx.get(a)!]![Number(slot.slice(1))]!;
  return { payoff, ags, M };
}

// enumerate all permutations of [0..n)
function permutations(n: number): number[][] {
  if (n === 0) return [[]];
  const out: number[][] = [];
  const rec = (cur: number[], rest: number[]) => {
    if (rest.length === 0) { out.push(cur); return; }
    for (let i = 0; i < rest.length; i++) {
      rec([...cur, rest[i]!], [...rest.slice(0, i), ...rest.slice(i + 1)]);
    }
  };
  rec([], Array.from({ length: n }, (_, i) => i));
  return out;
}

test("separation cost is non-negative", () => {
  const slots = unitSlots(4);
  const { payoff, ags } = payoffMatrix(4, 4);
  const { assignment } = clearMarket(ags, slots, payoff, TICK);
  for (const s of slots) {
    assert.ok(separationCost(s.id, assignment, ags, slots, payoff, TICK) >= 0);
  }
});

test("Three-way Equivalence: clearMarket optimum is yield-optimal, closed, and clearing (Thm 7.1)", () => {
  const n = 4;
  const slots = unitSlots(n);
  const { payoff, ags } = payoffMatrix(n, n);
  const { assignment, prices } = clearMarket(ags, slots, payoff, TICK);

  // brute-force the global yield optimum over all permutations
  const perms = permutations(n);
  const ty = (p: number[]) => {
    const a: Assignment = new Map(ags.map((ag, i) => [ag, `e${p[i]}`]));
    return yieldOf(a, ags, slots, payoff, TICK);
  };
  let yMax = -Infinity;
  for (const p of perms) yMax = Math.max(yMax, ty(p));

  const yCleared = yieldOf(assignment, ags, slots, payoff, TICK);

  // (i) yield-optimality up to tau_0
  assert.ok(yCleared >= yMax - TICK, `cleared yield ${yCleared} within tau of max ${yMax}`);

  // (ii) deterministic closure: no single swap improves yield by > tau_0
  const base = yCleared;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const alt = new Map(assignment);
      const si = assignment.get(ags[i]!)!;
      const sj = assignment.get(ags[j]!)!;
      alt.set(ags[i]!, sj);
      alt.set(ags[j]!, si);
      assert.ok(
        yieldOf(alt, ags, slots, payoff, TICK) <= base + TICK,
        "closure violated: a swap improved yield by > tau_0",
      );
    }
  }

  // (iii) market clearing (network-yield Def. clearing): prices ARE the
  // separation costs, prices are non-negative, and no agent can be moved to a
  // slot in a way that raises total yield by more than tau_0 (the clearing/
  // closure content of the theorem — this is the swap-stability the fixed point
  // guarantees, not per-agent Walrasian IR, which sep-prices do not promise).
  for (const s of slots) {
    assert.equal(
      prices.get(s.id),
      separationCost(s.id, assignment, ags, slots, payoff, TICK),
      "price must equal separation cost",
    );
    assert.ok(prices.get(s.id)! >= 0, "prices are non-negative");
  }
  // no single move (to any slot, swapping the occupant) beats closure by > tau_0
  for (const a of ags) {
    const cur = assignment.get(a)!;
    for (const s of slots) {
      if (s.id === cur) continue;
      const alt = new Map(assignment);
      // move a to s; whoever holds s takes a's old slot (swap, keeps it valid)
      let occ: AgentId | undefined;
      for (const b of ags) if (b !== a && assignment.get(b) === s.id) { occ = b; break; }
      alt.set(a, s.id);
      if (occ !== undefined) alt.set(occ, cur);
      assert.ok(
        yieldOf(alt, ags, slots, payoff, TICK) <= base + TICK,
        "clearing: no move raises yield by > tau_0",
      );
    }
  }
});

test("forced optimal utilisation: unique interior v* (corrected net-yield form, Thm 7.x)", () => {
  const vstar = forcedUtilisation({ pressure: 3, capacity: 1, tick: 1, maxRate: 1 });
  // interior: strictly inside (0, vbar)
  assert.ok(vstar > 0.01 && vstar < 0.99, `v* should be interior, got ${vstar}`);
  // for P=3, c=1, tick=1, b=log(1+v), g_u=v^2/(1-v): v* ~ 0.43 (matches Python check)
  assert.ok(Math.abs(vstar - 0.43) < 0.05, `v* ~ 0.43 expected, got ${vstar}`);
});

test("comparative-advantage: at closure each slot holds the agent that values it most", () => {
  // With a diagonal-dominant payoff, the clearing assignment is the diagonal.
  const n = 3;
  const slots = unitSlots(n);
  const ags = agents(n);
  const payoff = (a: AgentId, slot: string) => {
    const i = Number(a.slice(1));
    const j = Number(slot.slice(1));
    return i === j ? 10 : 1; // each agent strongly prefers its own-index slot
  };
  const { assignment } = clearMarket(ags, slots, payoff, TICK);
  for (let i = 0; i < n; i++) {
    assert.equal(assignment.get(ags[i]!), `e${i}`, "diagonal comparative-advantage assignment");
  }
});
