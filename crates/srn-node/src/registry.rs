//! Live cell registry Γ — content-addressed by partition coordinates.
//!
//! Γ is the local, append-biased store of evaluated glyphs.  It maps
//! PartitionCoords → list of EvalRecord (append-only; newest first for
//! lookup, but the log never shrinks).
//!
//! "Live" means the registry reflects the current evaluation state of this
//! node — it is not a global shared store.  Every node has its own Γ.

use std::collections::HashMap;
use crate::coords::PartitionCoords;
use crate::expression::{EvalRecord, EvalResult};

/// A single entry in the registry.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub record: EvalRecord,
    /// Sequential index within this coord's history (0-based, monotone).
    pub seq: u64,
}

/// The live cell registry.
#[derive(Debug, Default)]
pub struct Registry {
    /// coord key → append-only list of evaluation records
    cells: HashMap<String, Vec<RegistryEntry>>,
    /// total committed record count (monotone — never decremented)
    pub total_count: u64,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an evaluation record for the given coordinates.
    /// This is the only mutation — the log is append-only.
    pub fn append(&mut self, coords: &PartitionCoords, record: EvalRecord) {
        let key = coords.key();
        let entries = self.cells.entry(key).or_default();
        let seq = entries.len() as u64;
        entries.push(RegistryEntry { record, seq });
        self.total_count += 1;
    }

    /// Fetch the most recent successful evaluation for these coordinates.
    /// Returns None if no successful evaluation exists (Fetch-Miss).
    pub fn fetch_latest(&self, coords: &PartitionCoords) -> Option<&RegistryEntry> {
        let entries = self.cells.get(&coords.key())?;
        entries.iter().rev().find(|e| {
            matches!(e.record.result, EvalResult::Value(_))
        })
    }

    /// Fetch all records for these coordinates (full history).
    pub fn fetch_all(&self, coords: &PartitionCoords) -> Vec<&RegistryEntry> {
        self.cells
            .get(&coords.key())
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// True if any successful evaluation exists for these coordinates.
    pub fn has_cell(&self, coords: &PartitionCoords) -> bool {
        self.fetch_latest(coords).is_some()
    }

    /// Number of live cells (coordinates with at least one record).
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Summary for display / API.
    pub fn summary(&self) -> serde_json::Value {
        let cells: Vec<serde_json::Value> = self.cells.iter().map(|(k, v)| {
            serde_json::json!({
                "coords": k,
                "record_count": v.len(),
                "latest_seq": v.last().map(|e| e.seq),
            })
        }).collect();
        serde_json::json!({
            "total_committed": self.total_count,
            "cell_count": self.cell_count(),
            "cells": cells,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{ReceiverFrame, EvalResult};

    fn dummy_record(n: u32) -> EvalRecord {
        use crate::label::TrajectoryAddress;
        EvalRecord {
            expr_digest: format!("digest-{n}"),
            address: TrajectoryAddress::root(3).key(),
            frame: ReceiverFrame {
                coords: PartitionCoords::new(n, 0, 0, 1).unwrap(),
                trajectory_count: 0,
            },
            result: EvalResult::Value(serde_json::json!({"ok": true})),
            timestamp_ns: 0,
        }
    }

    #[test]
    fn append_and_fetch() {
        let mut reg = Registry::new();
        let coords = PartitionCoords::chromebook_reference();
        reg.append(&coords, dummy_record(1));
        reg.append(&coords, dummy_record(1));
        assert_eq!(reg.fetch_all(&coords).len(), 2);
        assert!(reg.fetch_latest(&coords).is_some());
        assert_eq!(reg.total_count, 2);
    }

    #[test]
    fn fetch_miss_for_unknown_coords() {
        let reg = Registry::new();
        let coords = PartitionCoords::new(3, 1, 0, 1).unwrap();
        assert!(reg.fetch_latest(&coords).is_none());
    }

    #[test]
    fn total_count_monotone() {
        let mut reg = Registry::new();
        let c1 = PartitionCoords::chromebook_reference();
        let c2 = PartitionCoords::new(2, 1, 0, 1).unwrap();
        for i in 1u32..=5 {
            reg.append(&c1, dummy_record(i.min(1)));
            reg.append(&c2, dummy_record(2));
            assert_eq!(reg.total_count, (2 * i) as u64);
        }
    }
}
