//! Partition coordinates (n, l, m, s) — node identity in the SRN address space.
//!
//! Derived from SO(3) representation theory:
//!   n ≥ 1  — partition depth (shell)
//!   0 ≤ l < n  — structural complexity (angular momentum index)
//!   -l ≤ m ≤ l  — orientation index
//!   s ∈ {-1, +1}  — residue parity (spin)
//!
//! Shell capacity: C(n) = 2n²  (each shell n holds exactly 2n² valid addresses)

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionCoords {
    /// Depth / shell index (≥ 1)
    pub n: u32,
    /// Structural complexity  (0 ≤ l < n)
    pub l: u32,
    /// Orientation index  (-l ≤ m ≤ l, stored as i32)
    pub m: i32,
    /// Residue parity  (+1 or -1)
    pub s: i32,
}

impl PartitionCoords {
    /// Construct, enforcing the angular-momentum constraints.
    pub fn new(n: u32, l: u32, m: i32, s: i32) -> anyhow::Result<Self> {
        anyhow::ensure!(n >= 1, "n must be ≥ 1, got {n}");
        anyhow::ensure!(l < n, "l must be < n, got l={l} n={n}");
        anyhow::ensure!(m.unsigned_abs() <= l, "|m| must be ≤ l, got m={m} l={l}");
        anyhow::ensure!(s == 1 || s == -1, "s must be ±1, got {s}");
        Ok(Self { n, l, m, s })
    }

    /// Chromebook reference node: the minimal-depth address (1,0,0,+1)
    pub fn chromebook_reference() -> Self {
        Self { n: 1, l: 0, m: 0, s: 1 }
    }

    /// Shell capacity at this depth: C(n) = 2n²
    pub fn shell_capacity(n: u32) -> u64 {
        2 * (n as u64) * (n as u64)
    }

    /// All valid coordinates at depth n (enumerates the shell).
    pub fn shell(n: u32) -> Vec<Self> {
        let mut out = Vec::with_capacity(Self::shell_capacity(n) as usize);
        for l in 0..n {
            for m in -(l as i32)..=(l as i32) {
                for s in [-1i32, 1] {
                    out.push(Self { n, l, m, s });
                }
            }
        }
        out
    }

    /// Compact key string used for content-addressing in the registry.
    pub fn key(&self) -> String {
        format!("({},{},{},{})", self.n, self.l, self.m, if self.s > 0 { "+" } else { "-" })
    }
}

impl fmt::Display for PartitionCoords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(n={}, l={}, m={}, s={})", self.n, self.l, self.m, if self.s > 0 { "+" } else { "-" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_capacity_formula() {
        for n in 1u32..=10 {
            let enumerated = PartitionCoords::shell(n).len() as u64;
            assert_eq!(enumerated, PartitionCoords::shell_capacity(n),
                "C({n}) mismatch");
        }
    }

    #[test]
    fn reference_node_valid() {
        PartitionCoords::new(1, 0, 0, 1).unwrap();
    }

    #[test]
    fn invalid_coords_rejected() {
        assert!(PartitionCoords::new(1, 1, 0, 1).is_err()); // l ≥ n
        assert!(PartitionCoords::new(2, 1, 2, 1).is_err()); // |m| > l
        assert!(PartitionCoords::new(2, 1, 0, 0).is_err()); // s ∉ {±1}
    }
}
