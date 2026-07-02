//! srn-node — local SRN node binary.
//!
//! Usage:
//!   srn-node [OPTIONS]
//!
//! On first run, copy peers.example.toml → peers.toml and edit the IP
//! addresses to match your local network.  Then start the node on each device.

use clap::Parser;
use std::path::PathBuf;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use srn_node::{
    api::{router, shared_peers},
    coords::PartitionCoords,
    node::{NodeConfig, SrnNode, shared},
    peers::PeerTable,
};

#[derive(Parser, Debug)]
#[command(name = "srn-node", about = "Sango Rine Shumba — personal mesh node")]
struct Args {
    /// HTTP bind address
    #[arg(long, default_value = "0.0.0.0:7700", env = "SRN_BIND")]
    bind: String,

    /// Partition depth n (≥ 1)
    #[arg(long, default_value_t = 1, env = "SRN_N")]
    n: u32,

    /// Structural complexity l (0 ≤ l < n)
    #[arg(long, default_value_t = 0, env = "SRN_L")]
    l: u32,

    /// Orientation index m (−l ≤ m ≤ l)
    #[arg(long, default_value_t = 0, env = "SRN_M")]
    m: i32,

    /// Residue parity s (+1 or −1)
    #[arg(long, default_value_t = 1, env = "SRN_S")]
    s: i32,

    /// Trajectory address branching factor
    #[arg(long, default_value_t = 3, env = "SRN_BRANCH")]
    branching: u32,

    /// Scheduler tick budget per request
    #[arg(long, default_value_t = 8, env = "SRN_TICK_BUDGET")]
    tick_budget: usize,

    /// Path to peers TOML file
    #[arg(long, default_value = "peers.toml", env = "SRN_PEERS")]
    peers: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    let coords = PartitionCoords::new(args.n, args.l, args.m, args.s)
        .map_err(|e| anyhow::anyhow!("invalid partition coordinates: {e}"))?;

    let config = NodeConfig {
        coords,
        address_branching: args.branching,
        tick_budget: args.tick_budget,
    };

    // Load peer table (empty if file doesn't exist yet).
    let peer_table = match PeerTable::load(&args.peers) {
        Ok(t) => {
            info!("loaded {} peers from {}", t.peers.len(), args.peers.display());
            t
        }
        Err(e) => {
            warn!("could not load {}: {e} — starting with empty peer table", args.peers.display());
            PeerTable::new()
        }
    };

    let node = SrnNode::new(config);
    let node_handle = shared(node);
    let peers_handle = shared_peers(peer_table);

    info!("SRN node {} — listening on {}", coords, args.bind);

    let app = router(node_handle, peers_handle);
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;

    info!("ready — POST /peers to register devices, GET /network/probe to test connectivity");
    axum::serve(listener, app).await?;

    Ok(())
}
