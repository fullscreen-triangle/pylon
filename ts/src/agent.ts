/**
 * Process agents (network-yield §9) — the load-bearing unification.
 *
 * An allocated compute packet is not passive data the scheduler moves; it is a
 * standing, goal-directed agent that:
 *   - owns its completion target tau(x) as a standing goal;
 *   - descends its own residual r(x) by at least tau_0 each live tick (occupancy,
 *     Thm 9.1 "occupancy");
 *   - carries a strictly-monotone committed-step counter M (incorruptibility);
 *   - persists past r <= beta by GOAL SUCCESSION (Def 9.x): draws a fresh target
 *     from its occupation gamma instead of terminating;
 *   - returns a distinct response each interaction because its monitoring cell
 *     has advanced (ever-fresh response, Prop 9.x);
 *   - retires only when gamma returns null.
 *
 * The four musande invariants map directly:
 *   I1 conserved identity chi   -> AgentId + fixed PartitionCoords self-frame
 *   I2 monotone committed count -> M (never decremented)
 *   I3 search-not-fetch         -> each step reads the current advancing state
 *   I4 exclusive phases         -> observe (read residual/cell) then commit (descend)
 *
 * Residual/monotone-M semantics ported from `crates/srn-node/src/scheduler.rs`.
 */

import type { PartitionCoords } from "./coords.js";

/** The compute-tick tau_0 — the single quantum (physical/algorithmic/market). */
export const TICK = 1e-3;

/** Resolution floor beta: a goal is "attained" once residual <= FLOOR. beta >= tau_0. */
export const FLOOR = TICK * 0.5;

/** Opaque, unique, conserved agent identity (invariant I1, chi). */
export type AgentId = string & { readonly __brand: "AgentId" };

/** Build an AgentId from a string. */
export function agentId(s: string): AgentId {
  return s as AgentId;
}

/** A completion target tau(x) in the agent's disposition space (R^d). */
export type Target = ReadonlyArray<number>;

/**
 * Occupation gamma (Def 9.x): given the just-attained target and the finite
 * interaction history, produce the next target — or null to retire.
 */
export type Occupation = (attained: Target, history: ReadonlyArray<Target>) => Target | null;

/** Which phase the agent is in (invariant I4: the two are exclusive). */
export type Phase = "observe" | "commit";

/** The agent's lifecycle state. */
export type AgentState = "pursuing" | "attained" | "stalled" | "retired";

/** Euclidean distance — the residual metric r(x) = ||loc - goal||. */
function dist(a: Target, b: Target): number {
  const n = Math.max(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0);
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Quantise a state vector into a monitoring cell of the given width. */
function cellOf(loc: Target, width: number): string {
  return loc.map((x) => Math.floor(x / width)).join(":");
}

/**
 * A persistent, goal-directed process agent.
 *
 * Construct with a self-frame (I1), an initial internal state `loc`, an initial
 * standing goal, and an occupation gamma for succession. Then drive it with
 * step(); it descends its residual, attains, and succeeds to fresh goals until
 * gamma retires it.
 */
export class ProcessAgent {
  readonly id: AgentId;
  /** Conserved self-frame (invariant I1, chi). */
  readonly frame: PartitionCoords;

  private loc: number[];
  private goal: Target;
  private readonly occupation: Occupation;
  private readonly cellWidth: number;

  /** Monotone committed-step counter M (invariant I2); never decremented. */
  private M = 0;
  private state: AgentState = "pursuing";
  private phase: Phase = "observe";
  private readonly history: Target[] = [];
  private cell: string | null = null;
  private cellCrossings = 0;
  private successions = 0;
  /** Smallest live-step residual decrement observed (for occupancy checks). */
  private minLiveStep = Infinity;
  private flatTicks = 0;

  constructor(args: {
    id: AgentId;
    frame: PartitionCoords;
    loc: Target;
    goal: Target;
    occupation?: Occupation;
    cellWidth?: number;
  }) {
    this.id = args.id;
    this.frame = args.frame;
    this.loc = [...args.loc];
    this.goal = args.goal;
    this.occupation = args.occupation ?? (() => null); // no succession -> retire at first completion
    this.cellWidth = args.cellWidth ?? 0.1;
  }

  /** Current residual r(x) = distance from internal state to standing goal. */
  residual(): number {
    return dist(this.loc, this.goal);
  }

  /** Monotone committed-step count M (invariant I2). */
  committedStep(): number {
    return this.M;
  }

  currentState(): AgentState {
    return this.state;
  }

  currentPhase(): Phase {
    return this.phase;
  }

  goalsAttained(): number {
    return this.successions;
  }

  cellCrossingCount(): number {
    return this.cellCrossings;
  }

  /** The agent's felt drive is its residual; standing goal is its target. */
  standingGoal(): Target {
    return this.goal;
  }

  /** A read-only snapshot of internal state (for monitoring — cell index only). */
  monitoringCell(): string {
    return cellOf(this.loc, this.cellWidth);
  }

  /**
   * One tick of autonomous goal pursuit (I4: observe then commit).
   *
   * If the agent is already inside the target cell (r <= FLOOR), it SUCCEEDS to a
   * fresh goal (persistence). Otherwise it descends its residual toward the goal
   * by exactly one compute-tick — unless a full tau step would reach the target
   * cell, in which case it lands (the attainment step, not counted as a >=tau live
   * step). Every action (descent or succession) advances M.
   *
   * Returns the new lifecycle state.
   */
  step(): AgentState {
    if (this.state === "retired") return this.state;

    // -- observe phase (I4): read residual and cell without acting --
    this.phase = "observe";
    const r0 = this.residual();

    if (r0 <= FLOOR) {
      // already attained -> succeed to a fresh goal (persistence, Def 9.x)
      return this.succeed();
    }

    // -- commit phase (I4): descend the residual by one compute-tick --
    this.phase = "commit";
    const dir = this.unit(this.goal, this.loc, r0);
    const attaining = r0 - TICK <= FLOOR; // a full tau step reaches the target cell
    if (attaining) {
      this.loc = [...this.goal]; // land inside the target cell (attainment step)
    } else {
      for (let i = 0; i < this.loc.length; i++) {
        this.loc[i] = (this.loc[i] ?? 0) + (dir[i] ?? 0) * TICK;
      }
      const dec = r0 - this.residual();
      if (dec < this.minLiveStep) this.minLiveStep = dec; // occupancy: only live steps
      // stall detection: no measurable descent
      if (dec < 1e-12) this.flatTicks++;
      else this.flatTicks = 0;
    }

    // monitoring-cell crossing (ever-fresh response, Prop 9.x)
    const newCell = this.monitoringCell();
    if (this.cell !== null && newCell !== this.cell) this.cellCrossings++;
    this.cell = newCell;

    // committed action -> advance M (invariant I2)
    this.M++;

    this.state = this.flatTicks >= 5 ? "stalled" : "pursuing";
    if (this.residual() <= FLOOR) return this.succeed();
    return this.state;
  }

  /** Run up to `maxTicks` ticks (or until retired). Returns ticks actually run. */
  run(maxTicks: number): number {
    let t = 0;
    while (t < maxTicks && this.state !== "retired") {
      this.step();
      t++;
    }
    return t;
  }

  /** The smallest per-tick residual decrement over all live (non-attaining) steps. */
  minLiveStepSeen(): number {
    return this.minLiveStep;
  }

  /** A cell-indexed response — differs across interactions once the cell advances. */
  respond<T>(render: (cell: string, M: number) => T): T {
    return render(this.monitoringCell(), this.M);
  }

  // -- goal succession (Def 9.x): record the attained goal, draw the next --
  private succeed(): AgentState {
    this.history.push(this.goal);
    this.M++; // succession is a committed action (I2)
    this.successions++;
    this.flatTicks = 0;

    const next = this.occupation(this.goal, this.history);
    if (next === null) {
      this.state = "retired";
      return this.state;
    }
    // ensure the new goal is not already attained; if it is, retire rather than spin
    let g = next;
    if (dist(g, this.loc) <= FLOOR) {
      const g2 = this.occupation(g, this.history);
      if (g2 === null || dist(g2, this.loc) <= FLOOR) {
        this.state = "retired";
        return this.state;
      }
      g = g2;
    }
    this.goal = g;
    this.state = "pursuing";
    return this.state;
  }

  private unit(to: Target, from: number[], r: number): number[] {
    const out: number[] = [];
    const denom = r + 1e-12;
    const n = Math.max(to.length, from.length);
    for (let i = 0; i < n; i++) out.push(((to[i] ?? 0) - (from[i] ?? 0)) / denom);
    return out;
  }
}

/**
 * The musande-compatible Agent interface. A ProcessAgent satisfies it. Kept
 * minimal and structural (musande is not vendored in long-grass), so long-grass
 * can treat pylon compute-agents and musande vocational agents uniformly.
 */
export interface Agent {
  readonly id: AgentId;
  readonly frame: PartitionCoords;
  residual(): number;
  committedStep(): number;
  currentState(): AgentState;
  standingGoal(): Target;
  monitoringCell(): string;
}
