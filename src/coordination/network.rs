//! Network layer for Pylon coordination

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

use async_trait::async_trait;
use crate::types::{
    PylonId, NetworkNode, NodeStatus, NodeMetrics,
    CoordinationRequest, CoordinationResponse, TemporalCoordinate,
};
use crate::errors::{PylonError, NetworkError};
use crate::config::NetworkConfig;

/// Network layer for Pylon coordination
pub struct NetworkLayer {
    /// Network configuration
    config: NetworkConfig,
    /// Local node information
    local_node: NetworkNode,
    /// Connected nodes registry
    connected_nodes: Arc<RwLock<HashMap<PylonId, ConnectedNode>>>,
    /// Network message handlers
    message_handlers: Arc<RwLock<HashMap<MessageType, Arc<dyn MessageHandler>>>>,
    /// Network event sender
    event_sender: mpsc::UnboundedSender<NetworkEvent>,
    /// Network metrics
    metrics: Arc<RwLock<NetworkMetrics>>,
}

/// Connected node information
#[derive(Debug, Clone)]
pub struct ConnectedNode {
    /// Node information
    pub node: NetworkNode,
    /// Connection handle
    pub connection: Arc<NetworkConnection>,
    /// Connection timestamp
    pub connected_at: TemporalCoordinate,
    /// Last activity timestamp
    pub last_activity: TemporalCoordinate,
    /// Connection metrics
    pub metrics: ConnectionMetrics,
}

/// Network connection handle
#[derive(Debug)]
pub struct NetworkConnection {
    /// Connection identifier
    pub connection_id: PylonId,
    /// Remote address
    pub remote_addr: SocketAddr,
    /// Message sender
    pub message_sender: mpsc::UnboundedSender<NetworkMessage>,
    /// Connection status
    pub status: Arc<RwLock<ConnectionStatus>>,
}

/// Connection status
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    /// Connection is active
    Active,
    /// Connection is being established
    Connecting,
    /// Connection is being closed
    Closing,
    /// Connection is closed
    Closed,
    /// Connection has error
    Error(String),
}

/// Connection performance metrics
#[derive(Debug, Clone)]
pub struct ConnectionMetrics {
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Average latency
    pub average_latency: Duration,
    /// Error count
    pub error_count: u64,
}

/// Network message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    /// Coordination request message
    CoordinationRequest,
    /// Coordination response message
    CoordinationResponse,
    /// Node discovery message
    NodeDiscovery,
    /// Node announcement message
    NodeAnnouncement,
    /// Heartbeat message
    Heartbeat,
    /// Fragment distribution message
    FragmentDistribution,
    /// Fragment reconstruction message
    FragmentReconstruction,
    /// System status message
    SystemStatus,
}

/// Network message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    /// Message identifier
    pub message_id: PylonId,
    /// Message type
    pub message_type: MessageType,
    /// Source node identifier
    pub source_node: PylonId,
    /// Target node identifier (None for broadcast)
    pub target_node: Option<PylonId>,
    /// Message payload
    pub payload: MessagePayload,
    /// Message timestamp
    pub timestamp: TemporalCoordinate,
    /// Message priority
    pub priority: MessagePriority,
}

/// Message payload data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    /// Coordination request payload
    CoordinationRequest(CoordinationRequest),
    /// Coordination response payload
    CoordinationResponse(CoordinationResponse),
    /// Node discovery payload
    NodeDiscovery(NodeDiscoveryPayload),
    /// Node announcement payload
    NodeAnnouncement(NodeAnnouncementPayload),
    /// Heartbeat payload
    Heartbeat(HeartbeatPayload),
    /// Fragment payload
    Fragment(FragmentPayload),
    /// System status payload
    SystemStatus(SystemStatusPayload),
    /// Raw data payload
    Raw(Vec<u8>),
}

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    /// Low priority (background operations)
    Low = 1,
    /// Normal priority (standard operations)
    Normal = 2,
    /// High priority (time-sensitive operations)
    High = 3,
    /// Critical priority (system-critical operations)
    Critical = 4,
}

/// Node discovery payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDiscoveryPayload {
    /// Requesting node information
    pub requesting_node: NetworkNode,
    /// Requested capabilities
    pub requested_capabilities: Vec<String>,
    /// Discovery radius (hop count)
    pub discovery_radius: u32,
}

/// Node announcement payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnnouncementPayload {
    /// Announcing node information
    pub node: NetworkNode,
    /// Available services
    pub available_services: Vec<String>,
    /// Network topology information
    pub topology_info: NetworkTopologyInfo,
}

/// Heartbeat payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    /// Node status
    pub status: NodeStatus,
    /// Current metrics
    pub metrics: NodeMetrics,
    /// System load information
    pub system_load: SystemLoadInfo,
}

/// Fragment payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentPayload {
    /// Fragment data
    pub fragment_data: Vec<u8>,
    /// Fragment metadata
    pub metadata: FragmentMetadata,
    /// Reconstruction information
    pub reconstruction_info: ReconstructionInfo,
}

/// System status payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatusPayload {
    /// System status information
    pub status: String,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Alert information
    pub alerts: Vec<String>,
}

/// Network topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopologyInfo {
    /// Connected neighbors
    pub neighbors: Vec<PylonId>,
    /// Network distance to other nodes
    pub distances: HashMap<PylonId, u32>,
    /// Topology update timestamp
    pub last_update: TemporalCoordinate,
}

/// System load information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemLoadInfo {
    /// CPU utilization (0.0 - 1.0)
    pub cpu_usage: f64,
    /// Memory utilization (0.0 - 1.0)
    pub memory_usage: f64,
    /// Network utilization (0.0 - 1.0)
    pub network_usage: f64,
    /// Active coordination sessions
    pub active_sessions: u32,
}

/// Fragment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentMetadata {
    /// Fragment identifier
    pub fragment_id: PylonId,
    /// Fragment sequence number
    pub sequence_number: u32,
    /// Total fragments in group
    pub total_fragments: u32,
    /// Fragment checksum
    pub checksum: u64,
}

/// Reconstruction information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionInfo {
    /// Required fragments for reconstruction
    pub required_fragments: Vec<PylonId>,
    /// Reconstruction algorithm
    pub algorithm: String,
    /// Reconstruction parameters
    pub parameters: HashMap<String, String>,
}

/// Network event types
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Node connected
    NodeConnected {
        node_id: PylonId,
        node_info: NetworkNode,
    },
    /// Node disconnected
    NodeDisconnected {
        node_id: PylonId,
        reason: String,
    },
    /// Message received
    MessageReceived {
        message: NetworkMessage,
        connection_id: PylonId,
    },
    /// Message sent
    MessageSent {
        message_id: PylonId,
        target_node: PylonId,
    },
    /// Connection error
    ConnectionError {
        connection_id: PylonId,
        error: String,
    },
    /// Network topology changed
    TopologyChanged {
        added_nodes: Vec<PylonId>,
        removed_nodes: Vec<PylonId>,
    },
}

/// Message handler trait
#[async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle incoming network message
    async fn handle_message(
        &self,
        message: NetworkMessage,
        connection: &NetworkConnection,
    ) -> Result<Option<NetworkMessage>, PylonError>;
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    /// Total connections established
    pub total_connections: u64,
    /// Active connections
    pub active_connections: usize,
    /// Total messages sent
    pub total_messages_sent: u64,
    /// Total messages received
    pub total_messages_received: u64,
    /// Total bytes transferred
    pub total_bytes_transferred: u64,
    /// Average message latency
    pub average_message_latency: Duration,
    /// Network errors
    pub network_errors: u64,
}

use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// ---------------------------------------------------------------------------
// Wire protocol helpers — 4-byte big-endian length prefix + JSON body
// ---------------------------------------------------------------------------

async fn write_message(stream: &mut tokio::io::WriteHalf<TcpStream>, msg: &NetworkMessage) -> std::io::Result<()> {
    let json = serde_json::to_vec(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let len = json.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(&json).await?;
    stream.flush().await
}

async fn read_message(stream: &mut tokio::io::ReadHalf<TcpStream>) -> std::io::Result<NetworkMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "message too large"));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    serde_json::from_slice(&buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------

impl NetworkLayer {
    /// Create new network layer
    pub fn new(config: NetworkConfig, local_node: NetworkNode) -> Result<(Self, mpsc::UnboundedReceiver<NetworkEvent>), PylonError> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        let network_layer = Self {
            config,
            local_node,
            connected_nodes: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            metrics: Arc::new(RwLock::new(NetworkMetrics::new())),
        };

        Ok((network_layer, event_receiver))
    }

    /// Start network layer
    pub async fn start(&self) -> Result<(), PylonError> {
        info!("Starting network layer");

        // Start TCP listener for incoming connections
        self.start_listener().await?;

        // Start network discovery
        self.start_discovery().await?;

        // Start heartbeat mechanism
        self.start_heartbeat().await?;

        info!("Network layer started successfully");
        Ok(())
    }

    /// Start TCP listener for incoming connections
    async fn start_listener(&self) -> Result<(), PylonError> {
        let bind_addr = format!("0.0.0.0:{}", self.config.listen_port);
        let listener = TcpListener::bind(&bind_addr).await
            .map_err(|e| NetworkError::ConnectionFailure {
                address: bind_addr.clone(),
                error: e.to_string(),
            })?;

        info!("Network listener started on {}", bind_addr);

        let connected_nodes = Arc::clone(&self.connected_nodes);
        let event_sender = self.event_sender.clone();
        let message_handlers = Arc::clone(&self.message_handlers);
        let local_node = self.local_node.clone();
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        debug!("Incoming connection from {}", addr);
                        let cn = Arc::clone(&connected_nodes);
                        let es = event_sender.clone();
                        let mh = Arc::clone(&message_handlers);
                        let ln = local_node.clone();
                        let mx = Arc::clone(&metrics);
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_incoming_connection(
                                stream, addr, cn, es, mh, ln, mx,
                            ).await {
                                error!("Connection from {} failed: {}", addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle incoming TCP connection: handshake → register → message loop.
    async fn handle_incoming_connection(
        stream: TcpStream,
        addr: SocketAddr,
        connected_nodes: Arc<RwLock<HashMap<PylonId, ConnectedNode>>>,
        event_sender: mpsc::UnboundedSender<NetworkEvent>,
        message_handlers: Arc<RwLock<HashMap<MessageType, Arc<dyn MessageHandler>>>>,
        local_node: NetworkNode,
        metrics: Arc<RwLock<NetworkMetrics>>,
    ) -> Result<(), PylonError> {
        debug!("Handling connection from {}", addr);

        let (mut reader, mut writer) = tokio::io::split(stream);

        // ── Handshake: send NodeAnnouncement, receive theirs ─────────────────
        let announce = NetworkMessage {
            message_id: PylonId::new_v4(),
            message_type: MessageType::NodeAnnouncement,
            source_node: local_node.node_id,
            target_node: None,
            payload: MessagePayload::NodeAnnouncement(NodeAnnouncementPayload {
                node: local_node.clone(),
                available_services: vec!["coordination".into(), "fragment".into()],
                topology_info: NetworkTopologyInfo {
                    neighbors: {
                        connected_nodes.read().await.keys().cloned().collect()
                    },
                    distances: HashMap::new(),
                    last_update: TemporalCoordinate::now(),
                },
            }),
            timestamp: TemporalCoordinate::now(),
            priority: MessagePriority::High,
        };

        write_message(&mut writer, &announce).await
            .map_err(|e| NetworkError::TransmissionFailure {
                message_id: announce.message_id,
                error: e.to_string(),
            })?;

        // Read remote's announcement
        let remote_msg = read_message(&mut reader).await
            .map_err(|e| NetworkError::ProtocolError {
                protocol: "pylon-handshake".into(),
                error: e.to_string(),
            })?;

        let remote_node = match &remote_msg.payload {
            MessagePayload::NodeAnnouncement(a) => a.node.clone(),
            _ => {
                return Err(NetworkError::ProtocolError {
                    protocol: "pylon-handshake".into(),
                    error: "expected NodeAnnouncement".into(),
                }.into());
            }
        };

        let remote_id = remote_node.node_id;
        info!("Handshake complete with node {} ({})", remote_id, addr);

        // ── Register connection ───────────────────────────────────────────────
        let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<NetworkMessage>();
        let connection = Arc::new(NetworkConnection {
            connection_id: PylonId::new_v4(),
            remote_addr: addr,
            message_sender: msg_tx,
            status: Arc::new(RwLock::new(ConnectionStatus::Active)),
        });

        {
            let mut nodes = connected_nodes.write().await;
            nodes.insert(remote_id, ConnectedNode {
                node: remote_node.clone(),
                connection: Arc::clone(&connection),
                connected_at: TemporalCoordinate::now(),
                last_activity: TemporalCoordinate::now(),
                metrics: ConnectionMetrics::new(),
            });
            let mut m = metrics.write().await;
            m.total_connections += 1;
            m.active_connections = nodes.len();
        }

        let _ = event_sender.send(NetworkEvent::NodeConnected {
            node_id: remote_id,
            node_info: remote_node,
        });

        // ── Outbound writer task ──────────────────────────────────────────────
        tokio::spawn(async move {
            while let Some(msg) = msg_rx.recv().await {
                if let Err(e) = write_message(&mut writer, &msg).await {
                    warn!("Write to {} failed: {}", addr, e);
                    break;
                }
            }
        });

        // ── Inbound message loop ──────────────────────────────────────────────
        loop {
            match read_message(&mut reader).await {
                Ok(msg) => {
                    debug!("Received {:?} from {}", msg.message_type, remote_id);

                    // Update last_activity
                    {
                        let mut nodes = connected_nodes.write().await;
                        if let Some(cn) = nodes.get_mut(&remote_id) {
                            cn.last_activity = TemporalCoordinate::now();
                            cn.metrics.messages_received += 1;
                        }
                    }

                    let _ = event_sender.send(NetworkEvent::MessageReceived {
                        message: msg.clone(),
                        connection_id: connection.connection_id,
                    });

                    // Dispatch to registered handler
                    let msg_type = msg.message_type;
                    let handlers = message_handlers.read().await;
                    if let Some(handler) = handlers.get(&msg_type) {
                        if let Err(e) = handler.handle_message(msg, &connection).await {
                            warn!("Handler error for {:?}: {}", msg_type, e);
                        }
                    }
                }
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        warn!("Read from {} failed: {}", remote_id, e);
                    }
                    break;
                }
            }
        }

        // ── Cleanup ───────────────────────────────────────────────────────────
        {
            let mut status = connection.status.write().await;
            *status = ConnectionStatus::Closed;
        }
        {
            let mut nodes = connected_nodes.write().await;
            nodes.remove(&remote_id);
            let mut m = metrics.write().await;
            m.active_connections = nodes.len();
        }
        let _ = event_sender.send(NetworkEvent::NodeDisconnected {
            node_id: remote_id,
            reason: "connection closed".into(),
        });

        Ok(())
    }

    /// Start network discovery: connect to each seed node and maintain connections.
    async fn start_discovery(&self) -> Result<(), PylonError> {
        let seeds = self.config.seed_nodes.clone();
        if seeds.is_empty() {
            debug!("No seed nodes configured — running as bootstrap node");
            return Ok(());
        }

        let connected_nodes = Arc::clone(&self.connected_nodes);
        let event_sender = self.event_sender.clone();
        let message_handlers = Arc::clone(&self.message_handlers);
        let local_node = self.local_node.clone();
        let metrics = Arc::clone(&self.metrics);
        let timeout = self.config.connection_timeout;

        tokio::spawn(async move {
            // Stagger initial connections to avoid thundering-herd on a shared bootstrap
            for (i, addr) in seeds.iter().enumerate() {
                let addr = addr.clone();
                let cn = Arc::clone(&connected_nodes);
                let es = event_sender.clone();
                let mh = Arc::clone(&message_handlers);
                let ln = local_node.clone();
                let mx = Arc::clone(&metrics);
                tokio::time::sleep(Duration::from_millis(100 * i as u64)).await;
                tokio::spawn(async move {
                    Self::connect_to_peer(addr, cn, es, mh, ln, mx, timeout).await;
                });
            }
        });

        Ok(())
    }

    /// Outbound connection to a peer: connect, handshake, register, run message loop.
    async fn connect_to_peer(
        addr: String,
        connected_nodes: Arc<RwLock<HashMap<PylonId, ConnectedNode>>>,
        event_sender: mpsc::UnboundedSender<NetworkEvent>,
        message_handlers: Arc<RwLock<HashMap<MessageType, Arc<dyn MessageHandler>>>>,
        local_node: NetworkNode,
        metrics: Arc<RwLock<NetworkMetrics>>,
        timeout: Duration,
    ) {
        use tokio::time::timeout as with_timeout;

        let socket_addr: SocketAddr = match addr.parse() {
            Ok(a) => a,
            Err(e) => {
                warn!("Invalid peer address {}: {}", addr, e);
                return;
            }
        };

        let stream = match with_timeout(timeout, TcpStream::connect(socket_addr)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                warn!("Failed to connect to peer {}: {}", addr, e);
                return;
            }
            Err(_) => {
                warn!("Connection to peer {} timed out", addr);
                return;
            }
        };

        info!("Outbound connection to {} established", addr);

        if let Err(e) = Self::handle_incoming_connection(
            stream,
            socket_addr,
            connected_nodes,
            event_sender,
            message_handlers,
            local_node,
            metrics,
        ).await {
            warn!("Peer session with {} ended: {}", addr, e);
        }
    }

    /// Start heartbeat mechanism
    async fn start_heartbeat(&self) -> Result<(), PylonError> {
        debug!("Starting heartbeat mechanism");
        
        let connected_nodes = Arc::clone(&self.connected_nodes);
        let local_node = self.local_node.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Send heartbeat to all connected nodes
                let nodes = connected_nodes.read().await;
                for (node_id, connected_node) in nodes.iter() {
                    let heartbeat_message = NetworkMessage {
                        message_id: PylonId::new_v4(),
                        message_type: MessageType::Heartbeat,
                        source_node: local_node.node_id,
                        target_node: Some(*node_id),
                        payload: MessagePayload::Heartbeat(HeartbeatPayload {
                            status: local_node.status,
                            metrics: local_node.metrics.clone(),
                            system_load: SystemLoadInfo {
                                cpu_usage: 0.1, // TODO: Get actual system metrics
                                memory_usage: 0.2,
                                network_usage: 0.05,
                                active_sessions: 0,
                            },
                        }),
                        timestamp: TemporalCoordinate::now(),
                        priority: MessagePriority::Low,
                    };
                    
                    if let Err(e) = connected_node.connection.message_sender.send(heartbeat_message) {
                        warn!("Failed to send heartbeat to node {}: {}", node_id, e);
                    }
                }
            }
        });
        
        Ok(())
    }

    /// Register message handler
    pub async fn register_message_handler(
        &self,
        message_type: MessageType,
        handler: Arc<dyn MessageHandler>,
    ) {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(message_type, handler);
    }

    /// Send message to specific node
    pub async fn send_message(
        &self,
        target_node: PylonId,
        message: NetworkMessage,
    ) -> Result<(), PylonError> {
        let nodes = self.connected_nodes.read().await;
        
        if let Some(connected_node) = nodes.get(&target_node) {
            connected_node.connection.message_sender.send(message)
                .map_err(|e| NetworkError::TransmissionFailure {
                    message_id: PylonId::new_v4(),
                    error: e.to_string(),
                })?;
        } else {
            return Err(NetworkError::ConnectionFailure {
                address: target_node.to_string(),
                error: "Node not connected".to_string(),
            }.into());
        }

        Ok(())
    }

    /// Broadcast message to all connected nodes
    pub async fn broadcast_message(&self, message: NetworkMessage) -> Result<(), PylonError> {
        let nodes = self.connected_nodes.read().await;
        
        for connected_node in nodes.values() {
            if let Err(e) = connected_node.connection.message_sender.send(message.clone()) {
                warn!("Failed to broadcast message to node {}: {}", 
                      connected_node.node.node_id, e);
            }
        }

        Ok(())
    }

    /// Get connected nodes
    pub async fn get_connected_nodes(&self) -> Vec<NetworkNode> {
        self.connected_nodes.read().await
            .values()
            .map(|cn| cn.node.clone())
            .collect()
    }

    /// Get network metrics
    pub async fn get_metrics(&self) -> NetworkMetrics {
        self.metrics.read().await.clone()
    }

    /// Disconnect from node
    pub async fn disconnect_node(&self, node_id: PylonId) -> Result<(), PylonError> {
        let mut nodes = self.connected_nodes.write().await;
        
        if let Some(connected_node) = nodes.remove(&node_id) {
            // Update connection status
            {
                let mut status = connected_node.connection.status.write().await;
                *status = ConnectionStatus::Closing;
            }

            // Send disconnect event
            let _ = self.event_sender.send(NetworkEvent::NodeDisconnected {
                node_id,
                reason: "Manual disconnect".to_string(),
            });

            info!("Disconnected from node {}", node_id);
        }

        Ok(())
    }
}

impl NetworkMetrics {
    /// Create new network metrics
    pub fn new() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            total_bytes_transferred: 0,
            average_message_latency: Duration::from_millis(0),
            network_errors: 0,
        }
    }
}

impl ConnectionMetrics {
    /// Create new connection metrics
    pub fn new() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            average_latency: Duration::from_millis(0),
            error_count: 0,
        }
    }
}
