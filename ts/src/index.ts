/**
 * @buhera/pylon — distributed resource-allocation runtime for Buhera OS.
 *
 * Speaks Sango Rine Shumba (SRN) as its transmission unit, clears a
 * compute-tick-bounded yield market, and instantiates each allocated task as a
 * persistent process agent with goal succession (network-yield §9).
 *
 * Public surface (contract §6.5). TS-native reimplementation; the Rust
 * `crates/srn-node` is the reference the semantics are ported from.
 */

// ---- constants ----
export { TICK } from "./agent.js";

// ---- coordinates ----
export {
  makeCoords,
  coordsFromTuple,
  coordTuple,
  referenceCoords,
  shell,
  shellCapacity,
  coordKey,
  coordsEqual,
  isCoordError,
  type PartitionCoords,
  type Coord,
  type CoordError,
} from "./coords.js";

// ---- SRN language ----
export {
  glyph,
  composed,
  literal,
  evalExpr,
  serializeExpr,
  type Expr,
  type Glyph,
  type ComposedExpr,
  type LiteralExpr,
  type Operator,
  type EvalResult,
  type ReceiverFrame,
  type Env,
} from "./srn/expr.js";
export { parseSrn, isParseError, type ParseError } from "./srn/parse.js";
export {
  encodeTrajectory,
  decodeTrajectory,
  encodeCoords,
  decodeCoords,
  type Trajectory,
} from "./srn/trajectory.js";

// ---- agents (contract re-exports Agent/AgentId; pylon defines them, since
//      @buhera/musande is not vendored — the interface is musande-compatible) ----
export {
  ProcessAgent,
  agentId,
  FLOOR as AGENT_FLOOR,
  type Agent,
  type AgentId,
  type Target,
  type Occupation,
  type Phase,
  type AgentState,
} from "./agent.js";

// ---- yield market (pure operators, contract §6.4) ----
export {
  yieldOf,
  separationCost,
  clearMarket,
  forcedUtilisation,
  defaultUtilisationCost,
  type Slot,
  type Price,
  type Assignment,
  type PayoffFn,
  type UtilisationCost,
} from "./market.js";

// ---- single-node runtime ----
export { SrnNode, type NodeConfig } from "./node.js";
export { Scheduler, Task, type TaskState, type TickResult } from "./scheduler.js";
export { Registry, type RegistryEntry } from "./registry.js";
export {
  TrajectoryAddress,
  digestOf,
  type ContentDigest,
  type ProcessLabel,
  type EvalRecord,
} from "./label.js";

// ---- cluster ----
export {
  Cluster,
  type ClusterConfig,
  type ClusterSnapshot,
  type NodeSpec,
  type Yield,
} from "./cluster.js";

// ---- Kuramoto phase-lock ----
export {
  orderParameter,
  criticalCoupling,
  KuramotoBank,
  LOCK_THRESHOLD,
} from "./kuramoto.js";

// ---- typed errors ----
export type { PylonError } from "./errors.js";
