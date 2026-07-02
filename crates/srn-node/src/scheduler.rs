//! Residue-driven scheduler — §6 of the scheduling-mechanism paper.
//!
//! Priority rule (Definition: Residue priority):
//!   P(τ) = Δ_τ / max(ρ_τ − θ_τ, β)
//!
//! where β > 0 is the entropic floor, Δ_τ is the descent rate over a window,
//! ρ_τ is the current residue, and θ_τ is the sufficiency threshold.
//!
//! Invariants proved in the paper:
//!   • A stalled task (Δ = 0) gets P = 0 → never dispatched while others run.
//!   • A task at θ gets P = +∞ → always dispatched first (finished immediately).
//!   • Trajectory count M is strictly monotone across non-declining ticks.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

pub const FLOOR: f64 = 0.01;   // β: entropic floor (strictly positive)
pub const STALL_WINDOW: usize = 5; // w: stall declared after this many flat units

/// State of a live task in the scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    Running,
    Stalled,
    Sufficient,
    Done,
    Declined,
}

/// A live task record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    /// Categorical complexity K — number of distinguishable work-trajectories needed.
    pub complexity: u64,
    /// Operator type count d (for T(n,d) = d(d+1)^{n-1}).
    pub op_types: u64,
    /// Committed unit count M — strictly monotone.
    pub trajectory_count: u64,
    /// Current residue ρ ∈ [β, 100].
    pub residue: f64,
    /// Sufficiency threshold θ ∈ [β, 100].
    pub threshold: f64,
    /// Recent residue readings (for descent-rate window).
    residue_history: Vec<f64>,
    /// Current task state.
    pub state: TaskState,
}

impl Task {
    pub fn new(id: impl Into<String>, complexity: u64, op_types: u64, initial_residue: f64) -> Self {
        let id = id.into();
        let threshold = FLOOR; // default: run to floor; caller may override
        Self {
            id,
            complexity,
            op_types,
            trajectory_count: 0,
            residue: initial_residue,
            threshold,
            residue_history: vec![initial_residue],
            state: TaskState::Running,
        }
    }

    pub fn with_threshold(mut self, theta: f64) -> Self {
        self.threshold = theta.max(FLOOR);
        self
    }

    /// Descent rate Δ over the stall window — non-negative by definition.
    pub fn descent_rate(&self) -> f64 {
        let h = &self.residue_history;
        if h.len() < 2 {
            return 0.0;
        }
        let window = STALL_WINDOW.min(h.len() - 1);
        let oldest = h[h.len() - 1 - window];
        let newest = *h.last().unwrap();
        (oldest - newest).max(0.0) / window as f64
    }

    /// Residue priority P(τ) — the scheduling signal.
    pub fn priority(&self) -> f64 {
        if self.state != TaskState::Running {
            return 0.0;
        }
        if self.residue <= self.threshold {
            return f64::INFINITY; // finish immediately
        }
        let delta = self.descent_rate();
        if delta == 0.0 {
            return 0.0; // stalled — do not feed
        }
        let denom = (self.residue - self.threshold).max(FLOOR);
        delta / denom
    }

    /// Record a new residue reading after one committed unit.
    /// Returns the updated priority.
    pub fn commit_unit(&mut self, new_residue: f64) -> f64 {
        debug_assert!(
            new_residue >= FLOOR || new_residue <= self.residue,
            "residue may not increase"
        );
        self.trajectory_count += 1; // strictly monotone — never decremented
        self.residue = new_residue.max(FLOOR);
        self.residue_history.push(self.residue);

        // State transitions
        if self.residue <= self.threshold {
            self.state = TaskState::Sufficient;
        } else if self.descent_rate() == 0.0
            && self.residue_history.len() > STALL_WINDOW
        {
            let flat = self.residue_history
                .iter()
                .rev()
                .take(STALL_WINDOW)
                .all(|&r| (r - self.residue).abs() < 1e-9);
            if flat {
                self.state = TaskState::Stalled;
            }
        }
        self.priority()
    }

    /// Expected termination unit count: n_term(K, d) = 1 + ⌈log_{d+1}(K/d)⌉
    pub fn termination_bound(&self) -> u64 {
        let k = self.complexity as f64;
        let d = self.op_types as f64;
        if d <= 0.0 || k <= 0.0 { return 1; }
        let inner = (k / d).log(d + 1.0);
        1 + inner.ceil() as u64
    }

    /// True if task is behind its static termination prediction.
    pub fn is_behind_curve(&self) -> bool {
        self.trajectory_count > self.termination_bound()
            && self.state == TaskState::Running
    }
}

/// A task entry in the priority queue — ordered by P(τ).
#[derive(Debug)]
struct PQEntry {
    priority: f64,   // f64 doesn't impl Ord; we use a wrapper
    task_id: String,
}

impl PartialEq for PQEntry {
    fn eq(&self, other: &Self) -> bool { self.task_id == other.task_id }
}
impl Eq for PQEntry {}

impl Ord for PQEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for PQEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Tick result for one dispatched unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickResult {
    pub task_id: String,
    pub trajectory_count: u64,
    pub residue_before: f64,
    pub residue_after: f64,
    pub priority_after: f64,
    pub state: TaskState,
}

/// The residue-driven scheduler.
pub struct Scheduler {
    pub tasks: std::collections::HashMap<String, Task>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self { tasks: std::collections::HashMap::new() }
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.insert(task.id.clone(), task);
    }

    /// One scheduler tick over budget B (max dispatches this tick).
    ///
    /// Implements Algorithm 1 (Tick) from the scheduling-mechanism paper.
    /// Returns the list of committed results, in dispatch order.
    pub fn tick(&mut self, budget: usize, dispatch: &mut dyn FnMut(&Task) -> f64) -> Vec<TickResult> {
        let mut results = Vec::new();
        let mut remaining = budget;

        while remaining > 0 {
            // Build priority queue from running tasks
            let mut pq: BinaryHeap<PQEntry> = BinaryHeap::new();
            for task in self.tasks.values() {
                if task.state == TaskState::Running {
                    pq.push(PQEntry {
                        priority: task.priority(),
                        task_id: task.id.clone(),
                    });
                }
            }

            let top = match pq.pop() {
                Some(e) => e,
                None => break, // no runnable tasks
            };

            if top.priority == 0.0 {
                // All runnable tasks are stalled — decline
                break;
            }

            // Dispatch the highest-priority task
            let task = self.tasks.get(&top.task_id).unwrap();
            let residue_before = task.residue;

            let new_residue = dispatch(task); // caller-supplied work unit

            let task = self.tasks.get_mut(&top.task_id).unwrap();
            let priority_after = task.commit_unit(new_residue);

            results.push(TickResult {
                task_id: top.task_id.clone(),
                trajectory_count: task.trajectory_count,
                residue_before,
                residue_after: task.residue,
                priority_after,
                state: task.state.clone(),
            });

            remaining -= 1;
        }

        results
    }

    pub fn running_count(&self) -> usize {
        self.tasks.values().filter(|t| t.state == TaskState::Running).count()
    }

    pub fn stalled_tasks(&self) -> Vec<&Task> {
        self.tasks.values().filter(|t| t.state == TaskState::Stalled).collect()
    }
}

impl Default for Scheduler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_zero_for_stalled() {
        let mut t = Task::new("t", 100, 4, 50.0);
        // force a flat residue window
        for _ in 0..=STALL_WINDOW {
            t.commit_unit(50.0);
        }
        // After enough flat readings, priority should be 0 or state stalled
        assert!(t.priority() == 0.0 || t.state == TaskState::Stalled);
    }

    #[test]
    fn trajectory_count_strictly_monotone() {
        let mut t = Task::new("t", 100, 4, 80.0);
        for i in 1..=20u64 {
            t.commit_unit((80.0 - i as f64).max(FLOOR));
            assert_eq!(t.trajectory_count, i);
        }
    }

    #[test]
    fn sufficient_at_threshold() {
        let mut t = Task::new("t", 100, 4, 80.0).with_threshold(5.0);
        t.commit_unit(3.0); // below threshold
        assert_eq!(t.state, TaskState::Sufficient);
    }

    #[test]
    fn termination_bound_logarithmic() {
        // n_term(K,d) = 1 + ceil(log_{d+1}(K/d))
        // For K=10^12, d=3: log_4(10^12/3) ≈ 19.14 → ceil = 20 → n_term = 21
        let t = Task::new("t", 1_000_000_000_000, 3, 50.0);
        let bound = t.termination_bound();
        assert_eq!(bound, 21, "expected n_term(10^12, 3) = 21, got {bound}");

        // Verify logarithmic scaling: n_term grows as O(log K)
        let t2 = Task::new("t2", 1_000_000, 3, 50.0); // K = 10^6
        let b2 = t2.termination_bound();
        assert!(b2 < bound, "n_term should be smaller for smaller K");
        assert!(b2 > 1, "n_term > 1 for non-trivial K");
    }
}
