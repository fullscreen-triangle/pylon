/**
 * Content-addressing labels and committed evaluation records.
 *
 * Every committed unit is annotated with a paired (digest, address) label:
 *   - digest: a content hash of the expression (content addressing);
 *   - address: the node's trajectory-address path at commit time.
 *
 * Ported from `crates/srn-node/src/label.rs`. Uses a small pure-JS FNV-1a hash
 * (no crypto dependency; content addressing here needs stability, not security —
 * SRN's structural incorruptibility comes from the absence of a parser, not from
 * a cryptographic primitive).
 */

import type { ReceiverFrame, EvalResult } from "./srn/expr.js";

/** A stable content digest of some bytes. */
export interface ContentDigest {
  readonly hex: string;
}

/** FNV-1a 64-bit over a UTF-8 string, returned as hex. Stable, non-cryptographic. */
export function digestOf(data: string): ContentDigest {
  // 64-bit FNV-1a using BigInt for stability across platforms.
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < data.length; i++) {
    hash ^= BigInt(data.charCodeAt(i) & 0xff);
    hash = (hash * prime) & mask;
  }
  return { hex: hash.toString(16).padStart(16, "0") };
}

/**
 * A trajectory address — a path in the node's committed-unit tree. Advancing it
 * once per committed unit yields a strictly growing, replay-resistant address.
 */
export class TrajectoryAddress {
  private readonly branching: number;
  private readonly path: number[] = [];
  /** Monotone count of advances (== committed units addressed). */
  count = 0;

  constructor(branching = 3) {
    this.branching = branching;
  }

  static root(branching = 3): TrajectoryAddress {
    return new TrajectoryAddress(branching);
  }

  /** Advance by one digit (0..branching-1). Monotone; never rewinds. */
  advance(digit: number): void {
    this.path.push(((digit % this.branching) + this.branching) % this.branching);
    this.count++;
  }

  key(): string {
    return this.path.length === 0 ? "root" : this.path.join(".");
  }
}

/** A paired process label for one committed unit. */
export interface ProcessLabel {
  readonly digest: ContentDigest;
  readonly address: string;
}

/** A committed evaluation record — one entry in the append-only eval log. */
export interface EvalRecord {
  readonly exprDigest: string;
  readonly address: string;
  readonly frame: ReceiverFrame;
  readonly result: EvalResult;
  readonly timestampNs: number;
}
