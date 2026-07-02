//! SRN node — the top-level runtime combining all subsystems.
//!
//! A node N = (Γ, M, F) where:
//!   Γ — live cell registry (content-addressed by partition coords)
//!   M — trajectory count (strictly monotone; owned by the scheduler)
//!   F — receiver frame (this node's identity)
//!
//! Forest Theorem: membership in the SRN network = capability to evaluate
//! SRN expressions.  No registration, no administrator.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use crate::coords::PartitionCoords;
use crate::expression::{Env, Expr, EvalRecord, EvalResult, ReceiverFrame};
use crate::label::{ProcessLabel, TrajectoryAddress};
use crate::registry::Registry;
use crate::scheduler::{Scheduler, Task};

/// Node configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// This node's partition identity.
    pub coords: PartitionCoords,
    /// Branching factor for the trajectory address tree (default 3).
    pub address_branching: u32,
    /// How many scheduler ticks to run per evaluation request.
    pub tick_budget: usize,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            coords: PartitionCoords::chromebook_reference(),
            address_branching: 3,
            tick_budget: 8,
        }
    }
}

/// A single SRN node.
pub struct SrnNode {
    pub config: NodeConfig,
    /// Receiver frame — bound to this node's coordinates.
    frame: ReceiverFrame,
    /// Live cell registry Γ.
    registry: Registry,
    /// Residue-driven scheduler.
    scheduler: Scheduler,
    /// Trajectory address for annotation (paired-label scheme).
    address: TrajectoryAddress,
    /// Append-only evaluation log (process labels + results).
    pub eval_log: Vec<EvalRecord>,
    /// Environment bindings visible during evaluation.
    pub env: Env,
}

impl SrnNode {
    pub fn new(config: NodeConfig) -> Self {
        let frame = ReceiverFrame {
            coords: config.coords,
            trajectory_count: 0,
        };
        let address = TrajectoryAddress::root(config.address_branching);
        Self {
            config,
            frame,
            registry: Registry::new(),
            scheduler: Scheduler::new(),
            address,
            eval_log: Vec::new(),
            env: HashMap::new(),
        }
    }

    /// Create a default Chromebook reference node at (1,0,0,+1).
    pub fn chromebook() -> Self {
        Self::new(NodeConfig::default())
    }

    /// Evaluate an SRN expression in this node's receiver frame.
    ///
    /// This is the primary entry point.  Every call:
    ///   1. Evaluates the expression receiver-relatively.
    ///   2. Commits one work unit (increments M via the scheduler).
    ///   3. Appends the result to the live cell registry Γ.
    ///   4. Annotates the record with a paired (digest, address) label.
    ///   5. Appends the record to the append-only eval log.
    pub fn eval(&mut self, expr: &Expr) -> EvalResult {
        // Receiver-relative evaluation
        let result = expr.eval(&self.frame, &self.env);

        // Advance the trajectory address (one digit per committed unit)
        let digit = (self.frame.trajectory_count % self.config.address_branching as u64) as u32;
        self.address.advance(digit);
        self.frame.trajectory_count = self.address.count;

        // Build the paired label
        let label = ProcessLabel::new(&serde_json::to_vec(expr).unwrap_or_default(), self.address.clone());

        // Build the eval record
        let record = EvalRecord {
            expr_digest: label.digest.hex(),
            address: label.address.key(),
            frame: self.frame.clone(),
            result: result.clone(),
            timestamp_ns: current_ns(),
        };

        // Append to registry (keyed by this node's own coords)
        self.registry.append(&self.config.coords, record.clone());

        // Append to eval log
        self.eval_log.push(record);

        result
    }

    /// Submit a task to the scheduler and run up to tick_budget ticks.
    ///
    /// The `work_fn` closure is the actual computation — it receives the
    /// current task and returns a new residue reading.  This decouples the
    /// scheduler (which only cares about residue) from the work itself.
    pub fn schedule_and_run(
        &mut self,
        task: Task,
        mut work_fn: impl FnMut(&Task) -> f64,
    ) -> Vec<crate::scheduler::TickResult> {
        self.scheduler.add_task(task);
        self.scheduler.tick(self.config.tick_budget, &mut work_fn)
    }

    /// Fetch the latest successful evaluation for a given coord (Fetch / Fetch-Miss).
    pub fn fetch(&self, coords: &PartitionCoords) -> Option<&EvalRecord> {
        self.registry.fetch_latest(coords).map(|e| &e.record)
    }

    /// Install a binding into the evaluation environment.
    pub fn install(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.env.insert(key.into(), value);
    }

    /// Record that this node emitted a glyph to a peer (append-only log entry).
    pub fn record_emit(&mut self, peer_name: &str, glyph_name: &str) {
        use crate::expression::EvalResult;
        let digit = (self.frame.trajectory_count % self.config.address_branching as u64) as u32;
        self.address.advance(digit);
        self.frame.trajectory_count = self.address.count;
        let record = crate::expression::EvalRecord {
            expr_digest: format!("emit:{}:{}", peer_name, glyph_name),
            address: self.address.key(),
            frame: self.frame.clone(),
            result: EvalResult::Forward {
                target: self.config.coords, // placeholder — actual target is the peer
                expr: Box::new(crate::expression::Expr::Literal(
                    serde_json::json!({ "emitted_to": peer_name, "glyph": glyph_name })
                )),
            },
            timestamp_ns: current_ns(),
        };
        self.registry.append(&self.config.coords, record.clone());
        self.eval_log.push(record);
    }

    pub fn trajectory_count(&self) -> u64 {
        self.frame.trajectory_count
    }

    pub fn address_key(&self) -> String {
        self.address.key()
    }

    /// Status snapshot.
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "coords": self.config.coords.key(),
            "trajectory_count": self.frame.trajectory_count,
            "address": self.address.key(),
            "registry": self.registry.summary(),
            "eval_log_length": self.eval_log.len(),
            "scheduler": {
                "running_tasks": self.scheduler.running_count(),
                "stalled_tasks": self.scheduler.stalled_tasks().len(),
            }
        })
    }
}

fn current_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Thread-safe shared node handle for use in the API layer.
pub type SharedNode = Arc<Mutex<SrnNode>>;

pub fn shared(node: SrnNode) -> SharedNode {
    Arc::new(Mutex::new(node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{Glyph, Expr};

    #[test]
    fn eval_increments_trajectory_count() {
        let mut node = SrnNode::chromebook();
        assert_eq!(node.frame.trajectory_count, 0);

        let glyph = Glyph::new(
            "test",
            PartitionCoords::chromebook_reference(),
            "never-set-guard",
            "body",
            "self",
        );
        node.eval(&Expr::Glyph(glyph.clone()));
        assert_eq!(node.frame.trajectory_count, 1);
        node.eval(&Expr::Glyph(glyph));
        assert_eq!(node.frame.trajectory_count, 2);
    }

    #[test]
    fn eval_log_append_only() {
        let mut node = SrnNode::chromebook();
        let glyph = Glyph::new("g", PartitionCoords::chromebook_reference(), "x", "y", "z");
        let expr = Expr::Glyph(glyph);
        for i in 1..=5 {
            node.eval(&expr);
            assert_eq!(node.eval_log.len(), i);
        }
    }

    #[test]
    fn fetch_miss_before_eval() {
        let node = SrnNode::chromebook();
        let other = PartitionCoords::new(2, 1, 0, 1).unwrap();
        assert!(node.fetch(&other).is_none());
    }

    #[test]
    fn install_and_guard_fires() {
        let mut node = SrnNode::chromebook();
        node.install("forbidden", serde_json::json!(true));
        let glyph = Glyph::new(
            "g",
            PartitionCoords::chromebook_reference(),
            "forbidden",
            "body",
            "self",
        );
        let result = node.eval(&Expr::Glyph(glyph));
        assert!(matches!(result, EvalResult::Rejected { .. }));
    }
}
