//! SRN expression model — glyph syntax, receiver-relative evaluation.
//!
//! Grammar (simplified; full grammar in the SRN paper §3):
//!
//!   expr   ::= glyph | let-in | if-then-else | compose | catalyst | par | seq
//!   glyph  ::= |name : (n,l,m,s)| not { guard } do { body } to { target } [as { alias }]
//!
//! Receiver-relative evaluation (Theorem: Receiver-Relativity):
//!   The same glyph expression produces different output at every node
//!   because 'self' binds to the evaluating node's own coordinates.
//!
//! Every expression carries a mandatory 'not' boundary — individuation by
//! negation, a structural requirement of the SRN language.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::coords::PartitionCoords;
use crate::label::ContentDigest;

/// A receiver frame — the evaluating node's identity at evaluation time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiverFrame {
    pub coords: PartitionCoords,
    pub trajectory_count: u64,
}

/// The four primitive SRN operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    Compose,   // <>
    Catalyst,  // >>
    Parallel,  // ||
    Sequential, // ;
}

/// A glyph — the transmission unit of SRN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Glyph {
    pub name: String,
    /// Target partition address (which node this is addressed to).
    pub target: PartitionCoords,
    /// Guard expression text (the 'not' boundary — what this is NOT).
    pub not_guard: String,
    /// Body expression text (the 'do' clause — what to evaluate).
    pub body: String,
    /// Destination expression text (the 'to' clause — where result goes).
    pub to_target: String,
    /// Optional alias (the 'as' clause).
    pub alias: Option<String>,
}

impl Glyph {
    pub fn new(
        name: impl Into<String>,
        target: PartitionCoords,
        not_guard: impl Into<String>,
        body: impl Into<String>,
        to_target: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            target,
            not_guard: not_guard.into(),
            body: body.into(),
            to_target: to_target.into(),
            alias: None,
        }
    }

    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Serialise to bytes for content-addressing.
    pub fn as_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn digest(&self) -> ContentDigest {
        ContentDigest::of(&self.as_bytes())
    }
}

/// A composed expression — two glyphs or sub-expressions joined by an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedExpr {
    pub left: Box<Expr>,
    pub op: Operator,
    pub right: Box<Expr>,
}

/// Top-level SRN expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Glyph(Glyph),
    Composed(ComposedExpr),
    Literal(serde_json::Value),
}

impl Expr {
    /// Compute the SHA-1 content digest of this expression.
    pub fn digest(&self) -> ContentDigest {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        ContentDigest::of(&bytes)
    }

    /// Receiver-relative evaluation.
    ///
    /// The result depends on the evaluating node's frame — same expression,
    /// different nodes, different (all simultaneously valid) outputs.
    pub fn eval(&self, frame: &ReceiverFrame, env: &Env) -> EvalResult {
        match self {
            Expr::Literal(v) => EvalResult::Value(v.clone()),
            Expr::Glyph(g) => eval_glyph(g, frame, env),
            Expr::Composed(c) => eval_composed(c, frame, env),
        }
    }
}

/// Evaluation environment — bindings visible in this receiver frame.
pub type Env = HashMap<String, serde_json::Value>;

/// Result of evaluating an SRN expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalResult {
    /// A concrete value produced.
    Value(serde_json::Value),
    /// Guard rejected — the 'not' boundary fired.
    Rejected { reason: String },
    /// The expression emits to another node.
    Emit { target: PartitionCoords, payload: serde_json::Value },
    /// A forward (relay) to another node.
    Forward { target: PartitionCoords, expr: Box<Expr> },
    /// Error during evaluation.
    Error(String),
}

fn eval_glyph(g: &Glyph, frame: &ReceiverFrame, env: &Env) -> EvalResult {
    // Receiver-relative guard check — the 'not' boundary.
    // In this reference implementation the guard is a simple env-key lookup.
    // A full implementation would parse and evaluate an SRN guard expression.
    if env.get(&g.not_guard).is_some() {
        return EvalResult::Rejected {
            reason: format!("guard '{}' fired at {}", g.not_guard, frame.coords),
        };
    }

    // Receiver-relative body: the value of 'self' is this node's coords.
    // We encode the receiver frame into the output so the caller can observe
    // the Receiver-Relativity theorem: same glyph, different coords → different result.
    let output = serde_json::json!({
        "glyph": g.name,
        "self_n": frame.coords.n,
        "self_l": frame.coords.l,
        "self_m": frame.coords.m,
        "self_s": frame.coords.s,
        "trajectory_count": frame.trajectory_count,
        "body": g.body,
    });

    EvalResult::Value(output)
}

fn eval_composed(c: &ComposedExpr, frame: &ReceiverFrame, env: &Env) -> EvalResult {
    let left_result = c.left.eval(frame, env);
    let right_result = c.right.eval(frame, env);

    match c.op {
        Operator::Sequential => {
            // Left runs, then right; result is right's result
            match left_result {
                EvalResult::Rejected { reason } => EvalResult::Rejected { reason },
                _ => right_result,
            }
        }
        Operator::Compose => {
            // Pipe left result into right context
            match (left_result, right_result) {
                (EvalResult::Value(lv), EvalResult::Value(rv)) => {
                    EvalResult::Value(serde_json::json!({
                        "composed": [lv, rv]
                    }))
                }
                (EvalResult::Rejected { reason }, _) => EvalResult::Rejected { reason },
                (_, EvalResult::Rejected { reason }) => EvalResult::Rejected { reason },
                (_, r) => r,
            }
        }
        Operator::Catalyst => {
            // Right acts on left's residual; catalytic power κ_comb = 1-(1-κ1)(1-κ2)
            match (left_result, right_result) {
                (EvalResult::Value(lv), EvalResult::Value(rv)) => {
                    EvalResult::Value(serde_json::json!({
                        "catalysed": { "base": lv, "catalyst": rv }
                    }))
                }
                (EvalResult::Rejected { .. }, rv) => rv,
                (lv, EvalResult::Rejected { .. }) => lv,
                (lv, _) => lv,
            }
        }
        Operator::Parallel => {
            // Both sides run independently; both results are valid simultaneously
            match (left_result, right_result) {
                (EvalResult::Value(lv), EvalResult::Value(rv)) => {
                    EvalResult::Value(serde_json::json!({
                        "parallel": [lv, rv]
                    }))
                }
                (EvalResult::Rejected { .. }, rv) => rv,
                (lv, EvalResult::Rejected { .. }) => lv,
                (lv, _) => lv,
            }
        }
    }
}

/// A committed evaluation record — one entry in the append-only evaluation log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRecord {
    pub expr_digest: String,
    pub address: String,
    pub frame: ReceiverFrame,
    pub result: EvalResult,
    pub timestamp_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_frame() -> ReceiverFrame {
        ReceiverFrame {
            coords: PartitionCoords::chromebook_reference(),
            trajectory_count: 0,
        }
    }

    #[test]
    fn receiver_relativity() {
        let glyph = Glyph::new(
            "ping",
            PartitionCoords::new(2, 1, 0, 1).unwrap(),
            "blocked",
            "echo self",
            "return",
        );
        let expr = Expr::Glyph(glyph);
        let env = Env::new();

        let frame1 = ReceiverFrame {
            coords: PartitionCoords::new(1, 0, 0, 1).unwrap(),
            trajectory_count: 0,
        };
        let frame2 = ReceiverFrame {
            coords: PartitionCoords::new(2, 1, 0, 1).unwrap(),
            trajectory_count: 0,
        };

        let r1 = expr.eval(&frame1, &env);
        let r2 = expr.eval(&frame2, &env);

        // Same expression, different frames → different results (Receiver-Relativity)
        match (r1, r2) {
            (EvalResult::Value(v1), EvalResult::Value(v2)) => {
                assert_ne!(v1["self_n"], v2["self_n"]);
            }
            _ => panic!("expected Value results"),
        }
    }

    #[test]
    fn not_guard_fires() {
        let glyph = Glyph::new(
            "blocked-glyph",
            PartitionCoords::chromebook_reference(),
            "forbidden-key",
            "body",
            "nowhere",
        );
        let mut env = Env::new();
        env.insert("forbidden-key".to_string(), serde_json::json!(true));
        let result = Expr::Glyph(glyph).eval(&ref_frame(), &env);
        assert!(matches!(result, EvalResult::Rejected { .. }));
    }
}
