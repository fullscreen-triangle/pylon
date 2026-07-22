/**
 * SRN node — the top-level single-node runtime combining all subsystems.
 *
 * A node N = (Gamma, M, F) where:
 *   Gamma — live cell registry (content-addressed by partition coords)
 *   M     — trajectory count (strictly monotone; the committed-step counter)
 *   F     — receiver frame (this node's identity)
 *
 * Forest Theorem: membership in the SRN network = capability to evaluate SRN
 * expressions. No registration, no administrator (SRN Thm 8.1).
 *
 * Ported from `crates/srn-node/src/node.rs`.
 */

import type { PartitionCoords } from "./coords.js";
import { evalExpr, type Expr, type EvalResult, type ReceiverFrame, type Env, serializeExpr } from "./srn/expr.js";
import { Registry } from "./registry.js";
import { Scheduler, Task, type TickResult } from "./scheduler.js";
import { TrajectoryAddress, digestOf, type EvalRecord } from "./label.js";

export interface NodeConfig {
  readonly coords: PartitionCoords;
  /** Branching factor for the trajectory-address tree (default 3). */
  readonly addressBranching: number;
  /** Scheduler dispatch budget per schedule call. */
  readonly tickBudget: number;
}

/** A single SRN node. */
export class SrnNode {
  readonly config: NodeConfig;
  private frame: ReceiverFrame;
  private readonly registry = new Registry();
  private readonly scheduler = new Scheduler();
  private readonly address: TrajectoryAddress;
  /** Append-only evaluation log. */
  readonly evalLog: EvalRecord[] = [];
  private readonly env = new Map<string, unknown>();
  private clock = 0; // deterministic monotonic timestamp source

  constructor(config: NodeConfig) {
    this.config = config;
    this.frame = { coords: config.coords, trajectoryCount: 0 };
    this.address = TrajectoryAddress.root(config.addressBranching);
  }

  /** A default reference node at (1,0,0,+1). */
  static reference(coords: PartitionCoords): SrnNode {
    return new SrnNode({ coords, addressBranching: 3, tickBudget: 8 });
  }

  /** The node's current receiver frame (identity + committed count). */
  receiverFrame(): ReceiverFrame {
    return this.frame;
  }

  /** Monotone committed-step count M. */
  trajectoryCount(): number {
    return this.frame.trajectoryCount;
  }

  addressKey(): string {
    return this.address.key();
  }

  /**
   * Evaluate an SRN expression in this node's receiver frame (receiver-relative).
   * Each call advances M, appends to Gamma, and logs the record.
   */
  evaluate(expr: Expr): EvalResult {
    const result = evalExpr(expr, this.frame, this.env as Env);

    // advance the trajectory address (one digit per committed unit) -> M grows
    const digit = this.frame.trajectoryCount % this.config.addressBranching;
    this.address.advance(digit);
    this.frame = { coords: this.frame.coords, trajectoryCount: this.address.count };

    const record: EvalRecord = {
      exprDigest: digestOf(serializeExpr(expr)).hex,
      address: this.address.key(),
      frame: this.frame,
      result,
      timestampNs: this.clock++,
    };
    this.registry.append(this.config.coords, record);
    this.evalLog.push(record);
    return result;
  }

  /** Submit a task and run up to tickBudget dispatches. */
  scheduleAndRun(task: Task, workFn: (t: Task) => number): TickResult[] {
    this.scheduler.addTask(task);
    return this.scheduler.tick(this.config.tickBudget, workFn);
  }

  /** Fetch the latest successful evaluation for a coord (Fetch / Fetch-Miss). */
  fetch(coords: PartitionCoords): EvalRecord | undefined {
    return this.registry.fetchLatest(coords)?.record;
  }

  /** Install a binding into the evaluation environment. */
  install(key: string, value: unknown): void {
    this.env.set(key, value);
  }

  registrySize(): number {
    return this.registry.cellCount();
  }

  totalCommitted(): number {
    return this.registry.totalCount;
  }
}
