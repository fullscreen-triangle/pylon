/**
 * The Cluster — the top-level distributed runtime (contract §6.3).
 *
 * A Cluster is a set of SRN nodes (execution slots), a yield market that clears
 * task-to-slot assignments at separation-cost prices, and a bank of Kuramoto
 * oscillators keeping the per-node schedulers phase-locked. submit() takes an SRN
 * expression, selects a receiver by the expression's `to { region }` clause and
 * the current market clearing, instantiates a persistent ProcessAgent for it, and
 * returns a Yield.
 *
 * Forest openness (SRN Thm 8.1): membership is capability, not enrolment. A node
 * added with a valid frame is a peer immediately; there is no whitelist.
 */

import { coordTuple, coordsEqual, type Coord, type PartitionCoords } from "./coords.js";
import { parseSrn, isParseError } from "./srn/parse.js";
import { evalExpr, type Glyph, type Expr, type ReceiverFrame } from "./srn/expr.js";
import {
  ProcessAgent,
  agentId,
  TICK,
  type Agent,
  type AgentId,
  type Occupation,
  type Target,
} from "./agent.js";
import { clearMarket, type Slot, type Price, type PayoffFn } from "./market.js";
import { KuramotoBank, criticalCoupling } from "./kuramoto.js";
import type { PylonError } from "./errors.js";

/** A cluster node (execution slot) described by its receiver frame + capacities. */
export interface NodeSpec {
  readonly id: string;
  readonly frame: Coord;
  readonly capacity: number;
  readonly taskDuration: number;
  readonly maxRate: number;
}

/** Configuration for a Cluster. */
export interface ClusterConfig {
  readonly nodes: ReadonlyArray<NodeSpec>;
  /** Utilisation cost g_u; must satisfy g_u(0)=0, g_u'>0, g_u''>0. */
  readonly utilisationCost?: (v: number) => number;
  /** Kuramoto coupling K for scheduler phase-lock. Should exceed K_c* = 2 sigma/pi. */
  readonly coupling?: number;
}

/** The result of a submitted expression (contract §6.1). */
export type Yield =
  | {
      ok: true;
      agent: AgentId;
      committedStep: number;
      residual: number;
      allocation: { node: string; slot: number; price: Price };
      /** Receiver-relative result from the chosen node's evaluation. */
      value: unknown;
    }
  | { ok: false; error: PylonError };

/** An opaque snapshot of cluster assignment state. */
export interface ClusterSnapshot {
  readonly nodes: ReadonlyArray<NodeSpec>;
  readonly coupling: number;
  readonly agents: ReadonlyArray<{
    id: string;
    frame: Coord;
    committedStep: number;
    residual: number;
    goal: Target;
  }>;
}

interface LiveEntry {
  agent: ProcessAgent;
  nodeId: string;
}

export class Cluster {
  readonly tick = TICK;
  private readonly nodeSpecs: NodeSpec[];
  private readonly utilisationCost: (v: number) => number;
  private readonly coupling: number;
  private readonly kuramoto: KuramotoBank;
  private readonly live = new Map<AgentId, LiveEntry>();
  private counter = 0;

  constructor(config: ClusterConfig) {
    this.nodeSpecs = [...config.nodes];
    this.utilisationCost = config.utilisationCost ?? ((v) => v * v);
    this.coupling = config.coupling ?? 2.0;
    // one oscillator per node; small deterministic frequency spread
    const freqs = this.nodeSpecs.map((_, i) => 1.0 + 0.01 * (i - this.nodeSpecs.length / 2));
    this.kuramoto = new KuramotoBank(freqs, this.coupling);
  }

  // ---- read-only inspectors (contract §6.3) ----

  nodes(): ReadonlyArray<NodeSpec> {
    return this.nodeSpecs;
  }

  /** Add a node. Forest openness: no enrolment; the node is a peer at once. */
  addNode(spec: NodeSpec): void {
    if (!this.nodeSpecs.some((n) => n.id === spec.id)) this.nodeSpecs.push(spec);
  }

  price(nodeId: string): Price {
    const { prices } = this.clear();
    return prices.get(nodeId) ?? 0;
  }

  /** Kuramoto order parameter (R, psi). */
  orderParameter(): { R: number; psi: number } {
    // advance the phase model a little so coupling has effect, deterministically
    this.kuramoto.run(200, 0.05);
    return this.kuramoto.order();
  }

  isPhaseLocked(): boolean {
    return this.orderParameter().R >= 0.95;
  }

  criticalCoupling(sigmaOmega: number): number {
    return criticalCoupling(sigmaOmega);
  }

  liveAgents(): ReadonlyArray<Agent> {
    return [...this.live.values()].map((e) => e.agent);
  }

  agent(id: AgentId): Agent | null {
    return this.live.get(id)?.agent ?? null;
  }

  // ---- submission (contract §6.3) ----

  /**
   * Submit an SRN expression for evaluation somewhere in the cluster. Selects a
   * receiver by the expression's target region and the current market clearing,
   * instantiates a persistent ProcessAgent, evaluates receiver-relatively, and
   * returns a Yield. A succession map makes the agent persist past completion.
   */
  submit(
    expr: string | Glyph,
    options?: { succession?: Occupation; timeoutTicks?: number; goal?: Target },
  ): Yield {
    // 1. parse (enforces the mandatory `not` boundary, SRN Cor 2.1)
    let glyph: Glyph;
    if (typeof expr === "string") {
      const parsed = parseSrn(expr);
      if (isParseError(parsed)) {
        if (parsed.kind === "no-negation-boundary") return { ok: false, error: { kind: "no-negation-boundary" } };
        return { ok: false, error: { kind: "malformed-srn", at: parsed.at } };
      }
      glyph = parsed;
    } else {
      glyph = expr;
    }

    // 2. candidate receivers: nodes whose frame lies in the target region. Here
    //    the region is the glyph's target coordinate; a node matches if its frame
    //    equals the target, else all nodes are candidates (region = wildcard).
    const candidates = this.candidateNodes(glyph.target);
    if (candidates.length === 0) {
      return { ok: false, error: { kind: "boundary-rejects", frame: coordTuple(glyph.target) } };
    }

    // 3. clear the market over live agents + this new one to pick the slot
    const newId = agentId(`a${this.counter++}`);
    const chosen = this.chooseSlot(newId, candidates);
    if (chosen === null) {
      return { ok: false, error: { kind: "no-capacity", price: 0 } };
    }

    // 4. instantiate the persistent process agent (network-yield §9)
    const goal = options?.goal ?? defaultGoalFor(glyph.target);
    const agent = new ProcessAgent({
      id: newId,
      frame: glyph.target,
      loc: goal.map(() => 0),
      goal,
      ...(options?.succession ? { occupation: options.succession } : {}),
    });
    this.live.set(newId, { agent, nodeId: chosen.node.id });

    // 5. receiver-relative evaluation at the chosen node's frame (SRN Thm 4.2)
    const frame: ReceiverFrame = { coords: this.frameOf(chosen.node), trajectoryCount: agent.committedStep() };
    const result = evalExpr(glyph as Expr, frame, new Map());
    if (result.kind === "rejected") {
      this.live.delete(newId);
      return { ok: false, error: { kind: "boundary-rejects", frame: coordTuple(glyph.target) } };
    }

    // 6. drive the agent one step so it is "occupied" (Thm occupancy), then report
    agent.step();

    const value = result.kind === "value" ? result.value : result;
    return {
      ok: true,
      agent: newId,
      committedStep: agent.committedStep(),
      residual: agent.residual(),
      allocation: { node: chosen.node.id, slot: chosen.slotIndex, price: chosen.price },
      value,
    };
  }

  /** Broadcast a meta-expression to every node's registry (SRN §6). */
  broadcastMeta(_expr: string | Glyph): { reached: number } {
    // In the single-process model every node is reachable; a real transport would
    // forward the trajectory. We report the reachable count (Forest: all peers).
    return { reached: this.nodeSpecs.length };
  }

  // ---- persistence ----

  snapshot(): ClusterSnapshot {
    return {
      nodes: this.nodeSpecs,
      coupling: this.coupling,
      agents: [...this.live.values()].map((e) => ({
        id: e.agent.id,
        frame: coordTuple(e.agent.frame),
        committedStep: e.agent.committedStep(),
        residual: e.agent.residual(),
        goal: [...e.agent.standingGoal()],
      })),
    };
  }

  static fromSnapshot(snap: ClusterSnapshot): Cluster {
    const c = new Cluster({ nodes: snap.nodes, coupling: snap.coupling });
    // restore live agents (committed step is preserved -> monotone across restart)
    for (const a of snap.agents) {
      const id = agentId(a.id);
      const frame = { n: a.frame[0], l: a.frame[1], m: a.frame[2], s: (a.frame[3] as 1 | -1) };
      const agent = new ProcessAgent({ id, frame, loc: a.goal.map(() => 0), goal: a.goal });
      // fast-forward committed step so M does not regress (incorruptibility)
      while (agent.committedStep() < a.committedStep) agent.step();
      c.live.set(id, { agent, nodeId: snap.nodes[0]?.id ?? "" });
    }
    return c;
  }

  // ---- internals ----

  private candidateNodes(target: PartitionCoords): NodeSpec[] {
    const exact = this.nodeSpecs.filter((n) => coordsEqual(this.frameOf(n), target));
    return exact.length > 0 ? exact : this.nodeSpecs.slice();
  }

  private frameOf(n: NodeSpec): PartitionCoords {
    const [nn, l, m, s] = n.frame;
    return { n: nn, l, m, s: (s as 1 | -1) };
  }

  private slots(): Slot[] {
    return this.nodeSpecs.map((n) => ({ id: n.id, capacity: n.capacity, maxRate: n.maxRate }));
  }

  private payoff(): PayoffFn {
    // Each agent's payoff on a slot is inversely related to its residual and the
    // slot's load: closer-to-goal work on a faster slot yields more per tick.
    return (a: AgentId, slotId: string): number => {
      const entry = this.live.get(a);
      const slot = this.nodeSpecs.find((n) => n.id === slotId);
      if (slot === undefined) return 0;
      const r = entry ? Math.max(entry.agent.residual(), 1e-6) : 1;
      return (slot.maxRate * slot.capacity) / r;
    };
  }

  private clear(): { assignment: Map<AgentId, string>; prices: Map<string, Price> } {
    const ags = [...this.live.keys()];
    return clearMarket(ags, this.slots(), this.payoff(), this.tick, this.utilisationCost);
  }

  private chooseSlot(
    newId: AgentId,
    candidates: NodeSpec[],
  ): { node: NodeSpec; slotIndex: number; price: Price } | null {
    if (candidates.length === 0) return null;
    // temporarily include the new agent, clear, and read its assigned slot
    const ags = [...this.live.keys(), newId];
    const candidateSlots: Slot[] = candidates.map((n) => ({ id: n.id, capacity: n.capacity, maxRate: n.maxRate }));
    const payoff: PayoffFn = (a, slotId) => {
      const slot = candidates.find((n) => n.id === slotId);
      if (!slot) return 0;
      if (a === newId) return slot.maxRate * slot.capacity; // new agent: prefer fastest free slot
      const entry = this.live.get(a);
      const r = entry ? Math.max(entry.agent.residual(), 1e-6) : 1;
      return (slot.maxRate * slot.capacity) / r;
    };
    const { assignment, prices } = clearMarket(ags, candidateSlots, payoff, this.tick, this.utilisationCost);
    const nodeId = assignment.get(newId);
    if (nodeId === undefined) return null;
    const idx = this.nodeSpecs.findIndex((n) => n.id === nodeId);
    const node = this.nodeSpecs[idx];
    if (node === undefined) return null;
    return { node, slotIndex: idx, price: prices.get(nodeId) ?? 0 };
  }
}

/** A default goal target derived from a coordinate (small, tick-scale). */
function defaultGoalFor(c: PartitionCoords): Target {
  // a 2-D target a few dozen ticks from the origin, deterministic in the coords
  return [0.02 + 0.001 * c.n, 0.02 + 0.001 * (c.l + 1)];
}
