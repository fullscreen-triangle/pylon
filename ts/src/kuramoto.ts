/**
 * Kuramoto phase-lock for scheduler synchronisation (network-yield §swarm).
 *
 * K scheduler instances each carry a phase phi_i in [0, 2pi). The synchronisation
 * order parameter is
 *   R e^{i psi} = (1/K) sum_i e^{i phi_i},
 * with R = 1 perfect phase-lock and R = 0 maximal desynchronisation. Under
 * Kuramoto coupling K_c the system locks (R -> R_inf > 0) once K_c exceeds the
 * critical coupling K_c* = 2 sigma_omega / pi. A drop of R below 0.95 despite
 * adequate coupling is a first-class fault signal (partition or scheduler death).
 */

/** The phase-lock threshold: R >= 0.95 counts as locked. */
export const LOCK_THRESHOLD = 0.95;

/** Order parameter of a set of phases: magnitude R and mean phase psi. */
export function orderParameter(phases: ReadonlyArray<number>): { R: number; psi: number } {
  const k = phases.length;
  if (k === 0) return { R: 0, psi: 0 };
  let sx = 0;
  let sy = 0;
  for (const p of phases) {
    sx += Math.cos(p);
    sy += Math.sin(p);
  }
  sx /= k;
  sy /= k;
  return { R: Math.hypot(sx, sy), psi: Math.atan2(sy, sx) };
}

/** Critical coupling K_c* = 2 sigma_omega / pi for a frequency spread sigma_omega. */
export function criticalCoupling(sigmaOmega: number): number {
  return (2 * sigmaOmega) / Math.PI;
}

/**
 * A bank of Kuramoto oscillators (one per scheduler instance). Deterministic
 * integrator (explicit Euler); natural frequencies are supplied, not randomised,
 * so runs are reproducible (the contract leaves the integrator open, §13).
 */
export class KuramotoBank {
  private readonly omega: number[];
  private phases: number[];
  private readonly coupling: number;

  constructor(naturalFreqs: ReadonlyArray<number>, coupling: number, initialPhases?: ReadonlyArray<number>) {
    this.omega = [...naturalFreqs];
    this.coupling = coupling;
    this.phases = initialPhases
      ? [...initialPhases]
      : naturalFreqs.map((_, i) => (i * 2 * Math.PI) / naturalFreqs.length);
  }

  /** Advance all phases by dt using the mean-field Kuramoto update. */
  step(dt: number): void {
    const k = this.phases.length;
    if (k === 0) return;
    const { R, psi } = orderParameter(this.phases);
    const next = this.phases.map((phi, i) => {
      const dphi = this.omega[i]! + this.coupling * R * Math.sin(psi - phi);
      return wrap(phi + dphi * dt);
    });
    this.phases = next;
  }

  /** Run for `steps` steps of size `dt`. */
  run(steps: number, dt: number): void {
    for (let i = 0; i < steps; i++) this.step(dt);
  }

  order(): { R: number; psi: number } {
    return orderParameter(this.phases);
  }

  isLocked(): boolean {
    return this.order().R >= LOCK_THRESHOLD;
  }

  currentPhases(): ReadonlyArray<number> {
    return this.phases;
  }
}

function wrap(x: number): number {
  const twoPi = 2 * Math.PI;
  let v = x % twoPi;
  if (v < 0) v += twoPi;
  return v;
}
