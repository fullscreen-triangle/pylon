//! Sango Rine Shumba (SRN) node runtime.
//!
//! A single self-contained node that:
//!   1. Holds a partition-coordinate identity (n, l, m, s)
//!   2. Evaluates SRN expressions in its own receiver frame
//!   3. Runs a residue-driven scheduler (scheduling-mechanism paper)
//!   4. Maintains a live cell registry Γ, content-addressed by coords
//!   5. Annotates every committed unit with a paired (digest, address) label

pub mod coords;
pub mod expression;
pub mod label;
pub mod peers;
pub mod propagation;
pub mod registry;
pub mod scheduler;
pub mod node;
pub mod api;
