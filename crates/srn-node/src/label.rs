//! Paired content–time process label from the scheduling-mechanism paper.
//!
//! Label Orthogonality Theorem: no single label serves both identity and
//! relatedness — they demand opposite locality.
//!
//! Construction (§8 of scheduling-mechanism):
//!   label(p) = (digest(content(p)), address(M(p)))
//!
//! • digest  — SHA-1 of content bytes; avalanche → identity-faithful,
//!             time-blind (Thm: Content hashes are time-blind)
//! • address — base-b digit string of trajectory count; prefix = proximity
//!             (Thm: Prefix is proximity)
//!
//! The two components are kept separate; mixing them would destroy both axes.

use sha1::{Digest, Sha1};
use std::fmt;

/// Cryptographic content digest (20-byte SHA-1, hex-encoded).
/// Identity-faithful by construction (strict avalanche); time-blind.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentDigest(pub [u8; 20]);

impl ContentDigest {
    pub fn of(content: &[u8]) -> Self {
        let mut h = Sha1::new();
        h.update(content);
        let result = h.finalize();
        let mut bytes = [0u8; 20];
        bytes.copy_from_slice(&result);
        Self(bytes)
    }

    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Display for ContentDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.hex()[..8]) // abbreviated for display
    }
}

/// Trajectory address — base-3 digit string of the monotone count.
///
/// Each committed work unit appends one digit.  Two processes that share
/// a structural subtree to depth j share a prefix of length ≥ j.
/// lcp(addr1, addr2) is the depth of their lowest common ancestor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrajectoryAddress {
    /// Branching factor (fixed at 3 for a ternary work-tree)
    pub branching: u32,
    /// Digits from root to current node (monotone, append-only)
    pub digits: Vec<u32>,
    /// The raw trajectory count this address encodes
    pub count: u64,
}

impl TrajectoryAddress {
    pub fn root(branching: u32) -> Self {
        Self { branching, digits: vec![], count: 0 }
    }

    /// Extend by one digit — called on every committed work unit.
    pub fn advance(&mut self, choice: u32) {
        debug_assert!(choice < self.branching, "digit out of range");
        self.digits.push(choice % self.branching);
        self.count += 1;
    }

    /// Longest common prefix length with another address.
    pub fn lcp(&self, other: &TrajectoryAddress) -> usize {
        self.digits
            .iter()
            .zip(other.digits.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Relatedness queries (O(1) in label width):
    /// recurrence:    same digest, distinct address  → same work, later occurrence
    /// co-temporality: distinct digest, large lcp   → different work, same burst
    /// refinement:    small content change, large lcp → successive structural step
    pub fn co_temporal_with(&self, other: &TrajectoryAddress, min_lcp: usize) -> bool {
        self.lcp(other) >= min_lcp
    }

    pub fn key(&self) -> String {
        let s: String = self.digits.iter().map(|d| d.to_string()).collect();
        format!("{}/{}", self.branching, s)
    }
}

impl fmt::Display for TrajectoryAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:[", self.branching)?;
        for (i, d) in self.digits.iter().enumerate() {
            if i > 0 { write!(f, ".")?; }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// The paired label — the full annotation for one committed work unit.
#[derive(Debug, Clone)]
pub struct ProcessLabel {
    pub digest: ContentDigest,
    pub address: TrajectoryAddress,
}

impl ProcessLabel {
    pub fn new(content: &[u8], address: TrajectoryAddress) -> Self {
        Self {
            digest: ContentDigest::of(content),
            address,
        }
    }

    /// Recurrence: same computation, occurring again at a later count.
    pub fn is_recurrence_of(&self, other: &ProcessLabel) -> bool {
        self.digest == other.digest && self.address != other.address
    }

    /// Co-temporality: different content, same structural neighbourhood.
    pub fn is_cotemporal_with(&self, other: &ProcessLabel, min_lcp: usize) -> bool {
        self.digest != other.digest && self.address.lcp(&other.address) >= min_lcp
    }
}

impl fmt::Display for ProcessLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.digest, self.address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_avalanche_minimal_check() {
        let a = ContentDigest::of(b"hello");
        let b = ContentDigest::of(b"iello"); // one byte difference
        // avalanche: digests must differ
        assert_ne!(a, b);
    }

    #[test]
    fn address_monotone() {
        let mut addr = TrajectoryAddress::root(3);
        for i in 0..10u64 {
            addr.advance(0);
            assert_eq!(addr.count, i + 1);
        }
    }

    #[test]
    fn lcp_equals_shared_depth() {
        let mut a = TrajectoryAddress::root(3);
        let mut b = TrajectoryAddress::root(3);
        // share first 3 digits
        for _ in 0..3 { a.advance(0); b.advance(0); }
        let shared = a.lcp(&b);
        a.advance(1);
        b.advance(2); // diverge here
        assert_eq!(a.lcp(&b), shared); // lcp didn't change after divergence
        assert_eq!(shared, 3);
    }

    #[test]
    fn time_blindness() {
        // Same content, different counts → same digest
        let d1 = ContentDigest::of(b"expression-text");
        let d2 = ContentDigest::of(b"expression-text");
        assert_eq!(d1, d2); // time-blind: count is not an input
    }
}
