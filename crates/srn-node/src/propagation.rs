//! Information propagation — emit and gossip.
//!
//! Two primitives:
//!
//! 1. emit(glyph, target_peer)
//!    Posts the glyph to the target peer's /eval endpoint.
//!    The peer evaluates it in its own receiver frame and records the result
//!    in its own Γ.  The originating node records that it emitted (in its
//!    own log) but does not store the peer's result — the peer's Γ is
//!    sovereign.
//!
//! 2. gossip(key, value, peers)
//!    Installs a key-value binding on every reachable peer via /install.
//!    Used to propagate information acquired by one node to all others.
//!    Example: phone acquires a contact; gossip spreads it to all nodes.
//!
//! 3. fetch_remote(coords, peer)
//!    Fetches the latest Γ record for `coords` from a specific peer.
//!    Used to pull information that only a remote node has.

use serde::{Deserialize, Serialize};
use crate::api::EvalRequest;
use crate::peers::Peer;

/// Result of emitting a glyph to a remote peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitResult {
    pub peer_name: String,
    pub peer_addr: String,
    pub accepted: bool,
    pub response: Option<serde_json::Value>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Emit a glyph expression to a single remote peer.
pub async fn emit_to_peer(peer: &Peer, req: &EvalRequest) -> EmitResult {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();

    let start = std::time::Instant::now();
    let url = peer.eval_url();

    match client.post(&url).json(req).send().await {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            let accepted = resp.status().is_success();
            let response = resp.json::<serde_json::Value>().await.ok();
            EmitResult {
                peer_name: peer.name.clone(),
                peer_addr: peer.addr.clone(),
                accepted,
                response,
                latency_ms,
                error: None,
            }
        }
        Err(e) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            EmitResult {
                peer_name: peer.name.clone(),
                peer_addr: peer.addr.clone(),
                accepted: false,
                response: None,
                latency_ms,
                error: Some(e.to_string()),
            }
        }
    }
}

/// Gossip result for one peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipResult {
    pub peer_name: String,
    pub peer_addr: String,
    pub delivered: bool,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Install a key-value binding on every listed peer concurrently.
pub async fn gossip(
    peers: &[&Peer],
    key: String,
    value: serde_json::Value,
) -> Vec<GossipResult> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let payload = serde_json::json!({ "key": key, "value": value });

    let mut handles = Vec::new();
    for peer in peers {
        let peer = (*peer).clone();
        let client = client.clone();
        let payload = payload.clone();
        handles.push(tokio::spawn(async move {
            let start = std::time::Instant::now();
            let url = format!("{}/install", peer.base_url());
            match client.post(&url).json(&payload).send().await {
                Ok(resp) => GossipResult {
                    peer_name: peer.name.clone(),
                    peer_addr: peer.addr.clone(),
                    delivered: resp.status().is_success(),
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: None,
                },
                Err(e) => GossipResult {
                    peer_name: peer.name.clone(),
                    peer_addr: peer.addr.clone(),
                    delivered: false,
                    latency_ms: start.elapsed().as_millis() as u64,
                    error: Some(e.to_string()),
                },
            }
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        if let Ok(r) = h.await { results.push(r); }
    }
    results
}

/// Fetch the latest Γ record for `coords` from a single remote peer.
pub async fn fetch_remote(
    peer: &Peer,
    coords: &crate::coords::PartitionCoords,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let url = peer.fetch_url(coords);
    client
        .get(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}
