//! Peer table — maps partition coords to reachable host:port addresses.
//!
//! This is the only piece of configuration that must be shared across devices:
//! each node needs to know where its peers are.  Everything else is local.
//!
//! The table is loaded from a TOML file (peers.toml) at startup and can be
//! updated at runtime via the /peers API without restarting the node.
//!
//! No peer registration protocol: you add a peer by writing its IP:port.
//! The Forest Theorem holds — membership is capability, not registration.

use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::coords::PartitionCoords;

/// One peer entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    /// Human label for this device (e.g. "chromebook", "iphone").
    pub name: String,
    /// HTTP address: "192.168.1.42:7700"
    pub addr: String,
    /// Partition coordinates.
    pub n: u32,
    pub l: u32,
    pub m: i32,
    pub s: i32,
}

impl Peer {
    pub fn coords(&self) -> anyhow::Result<PartitionCoords> {
        PartitionCoords::new(self.n, self.l, self.m, self.s)
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn status_url(&self) -> String {
        format!("{}/status", self.base_url())
    }

    pub fn eval_url(&self) -> String {
        format!("{}/eval", self.base_url())
    }

    pub fn fetch_url(&self, coords: &PartitionCoords) -> String {
        format!("{}/fetch/{}/{}/{}/{}", self.base_url(),
            coords.n, coords.l, coords.m, coords.s)
    }
}

/// The full peer table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerTable {
    pub peers: Vec<Peer>,
}

impl PeerTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from a TOML file.  Returns an empty table if the file doesn't exist yet.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let text = std::fs::read_to_string(path)?;
        let table: Self = toml::from_str(&text)?;
        Ok(table)
    }

    /// Save to a TOML file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn add(&mut self, peer: Peer) {
        // Replace if same coord key already exists.
        let key = format!("({},{},{},{})", peer.n, peer.l, peer.m, peer.s);
        self.peers.retain(|p| {
            format!("({},{},{},{})", p.n, p.l, p.m, p.s) != key
        });
        self.peers.push(peer);
    }

    pub fn remove_by_name(&mut self, name: &str) {
        self.peers.retain(|p| p.name != name);
    }

    /// Find a peer by partition coords.
    pub fn find(&self, coords: &PartitionCoords) -> Option<&Peer> {
        self.peers.iter().find(|p| {
            p.n == coords.n && p.l == coords.l && p.m == coords.m && p.s == coords.s
        })
    }

    /// All peers except self (identified by own_coords).
    pub fn others(&self, own_coords: &PartitionCoords) -> Vec<&Peer> {
        self.peers.iter().filter(|p| {
            !(p.n == own_coords.n && p.l == own_coords.l
              && p.m == own_coords.m && p.s == own_coords.s)
        }).collect()
    }
}

/// Result of probing one peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub peer_name: String,
    pub peer_addr: String,
    pub coords: String,
    pub reachable: bool,
    pub status: Option<serde_json::Value>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Probe all peers concurrently; return one result per peer.
pub async fn probe_all(peers: &[&Peer]) -> Vec<ProbeResult> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let mut handles = Vec::new();
    for peer in peers {
        let peer = (*peer).clone();
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            probe_one(&client, &peer).await
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        match h.await {
            Ok(r) => results.push(r),
            Err(e) => results.push(ProbeResult {
                peer_name: "?".into(),
                peer_addr: "?".into(),
                coords: "?".into(),
                reachable: false,
                status: None,
                latency_ms: 0,
                error: Some(format!("task error: {e}")),
            }),
        }
    }
    results
}

async fn probe_one(client: &reqwest::Client, peer: &Peer) -> ProbeResult {
    let url = peer.status_url();
    let coords = format!("({},{},{},{})", peer.n, peer.l, peer.m, peer.s);
    let start = std::time::Instant::now();

    match client.get(&url).send().await {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            let status = resp.json::<serde_json::Value>().await.ok();
            ProbeResult {
                peer_name: peer.name.clone(),
                peer_addr: peer.addr.clone(),
                coords,
                reachable: true,
                status,
                latency_ms,
                error: None,
            }
        }
        Err(e) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            ProbeResult {
                peer_name: peer.name.clone(),
                peer_addr: peer.addr.clone(),
                coords,
                reachable: false,
                status: None,
                latency_ms,
                error: Some(e.to_string()),
            }
        }
    }
}
