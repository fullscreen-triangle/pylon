//! HTTP API for the local SRN node — axum-based, minimal surface.
//!
//! Local routes:
//!   GET  /status                    — node identity + registry summary
//!   POST /eval                      — evaluate an SRN expression locally
//!   GET  /fetch/:n/:l/:m/:s         — fetch latest Γ record for coords (local)
//!   POST /install                   — install env binding
//!   POST /schedule                  — residue-driven scheduler
//!
//! Peer/network routes:
//!   GET  /peers                     — list known peers
//!   POST /peers                     — add a peer
//!   DELETE /peers/:name             — remove a peer by name
//!   GET  /network/probe             — ping all peers; proves private network
//!   POST /network/emit              — delegate a glyph to a specific peer
//!   POST /network/gossip            — install a binding on ALL peers
//!   GET  /network/fetch-remote/:peer_name/:n/:l/:m/:s
//!                                   — fetch a Γ record from a remote peer

use std::sync::Arc;
use tokio::sync::RwLock;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};

use crate::coords::PartitionCoords;
use crate::expression::{Expr, Glyph};
use crate::node::SharedNode;
use crate::peers::{Peer, PeerTable, probe_all};
use crate::propagation::{emit_to_peer, fetch_remote, gossip};
use crate::scheduler::Task;

/// Shared peer table — separate from the node so the API can read/write it
/// without locking the evaluator.
pub type SharedPeers = Arc<RwLock<PeerTable>>;

pub fn shared_peers(table: PeerTable) -> SharedPeers {
    Arc::new(RwLock::new(table))
}

/// Combined app state.
#[derive(Clone)]
pub struct AppState {
    pub node: SharedNode,
    pub peers: SharedPeers,
}

/// Build the axum router.
pub fn router(node: SharedNode, peers: SharedPeers) -> Router {
    let state = AppState { node, peers };
    Router::new()
        // ── local ──────────────────────────────────────────────────────────
        .route("/status",          get(status_handler))
        .route("/eval",            post(eval_handler))
        .route("/fetch/:n/:l/:m/:s", get(fetch_handler))
        .route("/install",         post(install_handler))
        .route("/schedule",        post(schedule_handler))
        // ── peers ──────────────────────────────────────────────────────────
        .route("/peers",           get(list_peers_handler).post(add_peer_handler))
        .route("/peers/:name",     delete(remove_peer_handler))
        // ── network ────────────────────────────────────────────────────────
        .route("/network/probe",   get(probe_handler))
        .route("/network/emit",    post(emit_handler))
        .route("/network/gossip",  post(gossip_handler))
        .route("/network/fetch-remote/:peer_name/:n/:l/:m/:s",
                                   get(fetch_remote_handler))
        .with_state(state)
}

// ── /status ──────────────────────────────────────────────────────────────────

async fn status_handler(State(s): State<AppState>) -> Json<serde_json::Value> {
    let node = s.node.lock().await;
    let peers = s.peers.read().await;
    let mut status = node.status();
    status["peer_count"] = serde_json::json!(peers.peers.len());
    Json(status)
}

// ── /eval ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalRequest {
    pub name: String,
    pub target_n: u32,
    pub target_l: u32,
    pub target_m: i32,
    pub target_s: i32,
    pub not_guard: String,
    pub body: String,
    pub to_target: String,
    pub alias: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EvalResponse {
    pub result: serde_json::Value,
    pub trajectory_count: u64,
    pub address: String,
}

async fn eval_handler(
    State(s): State<AppState>,
    Json(req): Json<EvalRequest>,
) -> Result<Json<EvalResponse>, (StatusCode, String)> {
    let target = PartitionCoords::new(req.target_n, req.target_l, req.target_m, req.target_s)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let mut glyph = Glyph::new(req.name, target, req.not_guard, req.body, req.to_target);
    if let Some(alias) = req.alias { glyph = glyph.with_alias(alias); }

    let expr = Expr::Glyph(glyph);
    let mut node = s.node.lock().await;
    let result = node.eval(&expr);
    let count = node.trajectory_count();
    let address = node.address_key();

    Ok(Json(EvalResponse {
        result: serde_json::to_value(result)
            .unwrap_or(serde_json::json!({"error": "serialisation failed"})),
        trajectory_count: count,
        address,
    }))
}

// ── /fetch/:n/:l/:m/:s ───────────────────────────────────────────────────────

async fn fetch_handler(
    State(s): State<AppState>,
    Path((n, l, m, sv)): Path<(u32, u32, i32, i32)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let coords = PartitionCoords::new(n, l, m, sv)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let node = s.node.lock().await;
    match node.fetch(&coords) {
        Some(record) => Ok(Json(serde_json::to_value(record).unwrap_or_default())),
        None => Err((StatusCode::NOT_FOUND, format!("no record for {}", coords.key()))),
    }
}

// ── /install ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct InstallRequest {
    pub key: String,
    pub value: serde_json::Value,
}

async fn install_handler(
    State(s): State<AppState>,
    Json(req): Json<InstallRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut node = s.node.lock().await;
    node.install(req.key.clone(), req.value);
    (StatusCode::OK, Json(serde_json::json!({ "installed": req.key })))
}

// ── /schedule ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ScheduleRequest {
    pub id: String,
    pub complexity: u64,
    pub op_types: u64,
    pub initial_residue: f64,
    pub threshold: Option<f64>,
    pub descent_per_tick: f64,
}

async fn schedule_handler(
    State(s): State<AppState>,
    Json(req): Json<ScheduleRequest>,
) -> Json<serde_json::Value> {
    let mut task = Task::new(req.id, req.complexity, req.op_types, req.initial_residue);
    if let Some(theta) = req.threshold { task = task.with_threshold(theta); }
    let descent = req.descent_per_tick;
    let mut node = s.node.lock().await;
    let results = node.schedule_and_run(task, |t| {
        (t.residue - descent).max(crate::scheduler::FLOOR)
    });
    Json(serde_json::json!({ "ticks": results.len(), "results": results }))
}

// ── /peers ───────────────────────────────────────────────────────────────────

async fn list_peers_handler(State(s): State<AppState>) -> Json<serde_json::Value> {
    let peers = s.peers.read().await;
    Json(serde_json::json!({ "peers": peers.peers }))
}

async fn add_peer_handler(
    State(s): State<AppState>,
    Json(peer): Json<Peer>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Validate coords before accepting
    if let Err(e) = peer.coords() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })));
    }
    let name = peer.name.clone();
    let mut peers = s.peers.write().await;
    peers.add(peer);
    (StatusCode::OK, Json(serde_json::json!({ "added": name, "total": peers.peers.len() })))
}

async fn remove_peer_handler(
    State(s): State<AppState>,
    Path(name): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut peers = s.peers.write().await;
    peers.remove_by_name(&name);
    (StatusCode::OK, Json(serde_json::json!({ "removed": name })))
}

// ── /network/probe ───────────────────────────────────────────────────────────
//
// Pings every known peer and returns a reachability report.
// A successful run proves the private network: every reachable peer responds
// with its status, including its partition coordinates and trajectory count.

async fn probe_handler(State(s): State<AppState>) -> Json<serde_json::Value> {
    let peers = s.peers.read().await;
    let node = s.node.lock().await;
    let own_coords = node.config.coords;
    let others: Vec<&Peer> = peers.others(&own_coords);

    let results = probe_all(&others).await;

    let reachable = results.iter().filter(|r| r.reachable).count();
    let total = results.len();

    Json(serde_json::json!({
        "own_coords": own_coords.key(),
        "peers_probed": total,
        "peers_reachable": reachable,
        "network_healthy": reachable == total && total > 0,
        "results": results,
    }))
}

// ── /network/emit ─────────────────────────────────────────────────────────────
//
// Delegate a glyph expression to a named peer.
// The peer evaluates it in its own receiver frame (Receiver-Relativity).
// This node records the emit in its own log; the peer's Γ is sovereign.

#[derive(Debug, Deserialize)]
pub struct EmitRequest {
    /// Name of the target peer (must be in the peer table).
    pub peer_name: String,
    /// The glyph to emit.
    #[serde(flatten)]
    pub glyph: EvalRequest,
}

async fn emit_handler(
    State(s): State<AppState>,
    Json(req): Json<EmitRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let peers = s.peers.read().await;
    let peer = peers.peers.iter()
        .find(|p| p.name == req.peer_name)
        .cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND,
            format!("no peer named '{}'", req.peer_name)))?;
    drop(peers);

    let result = emit_to_peer(&peer, &req.glyph).await;

    // Record the emit in the local eval log as a forward event.
    {
        let mut node = s.node.lock().await;
        node.record_emit(&peer.name, &req.glyph.name);
    }

    Ok(Json(serde_json::to_value(result).unwrap_or_default()))
}

// ── /network/gossip ───────────────────────────────────────────────────────────
//
// Push a key-value binding to ALL peers simultaneously.
// Proves information propagation: after this call, every reachable peer's
// environment contains the binding and will apply it in future evaluations.

#[derive(Debug, Deserialize)]
pub struct GossipRequest {
    pub key: String,
    pub value: serde_json::Value,
}

async fn gossip_handler(
    State(s): State<AppState>,
    Json(req): Json<GossipRequest>,
) -> Json<serde_json::Value> {
    let peers = s.peers.read().await;
    let node = s.node.lock().await;
    let own_coords = node.config.coords;
    let others: Vec<&Peer> = peers.others(&own_coords);
    drop(node);

    let results = gossip(&others, req.key.clone(), req.value).await;

    let delivered = results.iter().filter(|r| r.delivered).count();
    Json(serde_json::json!({
        "key": req.key,
        "peers_attempted": results.len(),
        "peers_delivered": delivered,
        "results": results,
    }))
}

// ── /network/fetch-remote/:peer_name/:n/:l/:m/:s ─────────────────────────────
//
// Pull the latest Γ record for the given coords from a named remote peer.
// This is how one node reads information that only another node has acquired
// (e.g. laptop reads phone-only contacts from the phone's registry).

async fn fetch_remote_handler(
    State(s): State<AppState>,
    Path((peer_name, n, l, m, sv)): Path<(String, u32, u32, i32, i32)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let coords = PartitionCoords::new(n, l, m, sv)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let peers = s.peers.read().await;
    let peer = peers.peers.iter()
        .find(|p| p.name == peer_name)
        .cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND,
            format!("no peer named '{}'", peer_name)))?;
    drop(peers);

    fetch_remote(&peer, &coords)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::BAD_GATEWAY, e))
}
