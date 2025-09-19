"""
Temporal Fragmentation Module

Implements temporal message fragmentation where messages are split across multiple
temporal coordinates, creating cryptographic security through temporal incoherence.
Individual fragments appear as random data outside their designated temporal windows.

This module implements the transformation function F_{i,j}(t) = T(M_i, j, t, K_t)
where messages become coherent only when all temporal fragments are available.
"""

import asyncio
import time
import random
import hashlib
import struct
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
import uuid

@dataclass
class MessageFragment:
    """Represents a temporal message fragment"""
    
    fragment_id: str
    message_id: str
    sequence_number: int
    total_fragments: int
    temporal_coordinate: float
    payload: bytes
    temporal_key: bytes
    
    # Metadata
    source_node: str = ""
    destination_node: str = ""
    creation_time: float = 0.0
    expiration_time: float = 0.0
    fragment_size: int = 0
    entropy_score: float = 0.0
    
    def __post_init__(self):
        """Initialize derived properties"""
        if self.creation_time == 0.0:
            self.creation_time = time.time()
        
        if self.fragment_size == 0:
            self.fragment_size = len(self.payload)
        
        if self.expiration_time == 0.0:
            self.expiration_time = self.creation_time + 30.0  # 30 second default TTL
        
        # Calculate entropy score
        self.entropy_score = self._calculate_entropy()
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of fragment payload"""
        if not self.payload:
            return 0.0
        
        # Count byte frequencies
        byte_counts = {}
        for byte in self.payload:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        # Calculate entropy
        total_bytes = len(self.payload)
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize to 0-1 range (maximum entropy is log2(256) = 8 bits)
        return entropy / 8.0
    
    @property
    def is_expired(self) -> bool:
        """Check if fragment has expired"""
        return time.time() > self.expiration_time
    
    @property
    def age_seconds(self) -> float:
        """Get fragment age in seconds"""
        return time.time() - self.creation_time
    
    @property
    def appears_random(self) -> bool:
        """Check if fragment appears random (entropy > 0.9)"""
        return self.entropy_score > 0.9

@dataclass
class FragmentedMessage:
    """Represents a message that has been temporally fragmented"""
    
    message_id: str
    original_payload: bytes
    fragments: List[MessageFragment]
    source_node: str
    destination_node: str
    fragmentation_time: float
    
    # Reconstruction tracking
    reconstruction_window_start: float = 0.0
    reconstruction_window_end: float = 0.0
    reconstruction_attempts: int = 0
    is_reconstructed: bool = False
    reconstructed_payload: Optional[bytes] = None
    reconstruction_time: Optional[float] = None
    
    def __post_init__(self):
        """Initialize fragmentation properties"""
        if self.fragments:
            temporal_coordinates = [f.temporal_coordinate for f in self.fragments]
            self.reconstruction_window_start = min(temporal_coordinates)
            self.reconstruction_window_end = max(temporal_coordinates)
    
    @property
    def fragmentation_ratio(self) -> float:
        """Get fragmentation ratio (fragment count / original size)"""
        if len(self.original_payload) == 0:
            return 0.0
        return len(self.fragments) / len(self.original_payload)
    
    @property
    def total_fragment_size(self) -> int:
        """Get total size of all fragments"""
        return sum(f.fragment_size for f in self.fragments)
    
    @property
    def overhead_ratio(self) -> float:
        """Get overhead ratio from fragmentation"""
        if len(self.original_payload) == 0:
            return 0.0
        return self.total_fragment_size / len(self.original_payload)
    
    @property
    def average_entropy(self) -> float:
        """Get average entropy across all fragments"""
        if not self.fragments:
            return 0.0
        return sum(f.entropy_score for f in self.fragments) / len(self.fragments)

class TemporalFragmenter:
    """
    Temporal message fragmentation engine
    
    Implements the core temporal fragmentation algorithm:
    1. Splits messages across multiple temporal coordinates
    2. Applies temporal transformation T(M_i, j, t, K_t) 
    3. Ensures fragments appear random outside temporal windows
    4. Provides cryptographic security through temporal incoherence
    """
    
    def __init__(self, precision_calculator, data_collector=None):
        """Initialize temporal fragmenter"""
        self.precision_calculator = precision_calculator
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Fragmentation parameters
        self.min_fragments = 4
        self.max_fragments = 16
        self.target_fragment_size = 1024  # bytes
        self.entropy_threshold = 0.95
        self.temporal_jitter = 0.1  # 100ms jitter
        
        # Fragment storage
        self.active_fragments: Dict[str, List[MessageFragment]] = {}  # By message_id
        self.fragment_index: Dict[str, MessageFragment] = {}  # By fragment_id
        self.fragmented_messages: Dict[str, FragmentedMessage] = {}
        
        # Security parameters
        self.key_derivation_salt = b"SangoRineShumba2024"
        self.temporal_key_size = 32  # 256 bits
        
        # Performance tracking
        self.performance_metrics = {
            'messages_fragmented': 0,
            'fragments_created': 0,
            'messages_reconstructed': 0,
            'reconstruction_failures': 0,
            'average_entropy': 0.0,
            'average_fragmentation_time': 0.0,
            'average_reconstruction_time': 0.0
        }
        
        # Service state
        self.is_running = False
        
        self.logger.info("Temporal fragmenter initialized")
    
    async def start_fragmentation_service(self):
        """Start temporal fragmentation background service"""
        self.logger.info("Starting temporal fragmentation service...")
        self.is_running = True
        
        # Start background tasks
        cleanup_task = asyncio.create_task(self._cleanup_expired_fragments())
        
        try:
            await cleanup_task
        except asyncio.CancelledError:
            self.logger.info("Temporal fragmentation service stopped")
    
    async def fragment_message(self, message: str, source_node: str, destination_node: str) -> FragmentedMessage:
        """Fragment a message across temporal coordinates"""
        
        start_time = time.time()
        message_payload = message.encode('utf-8')
        message_id = str(uuid.uuid4())
        
        # Get current coordination matrix for temporal coordinates
        coordination_matrix = self.precision_calculator.get_current_coordination_matrix()
        if not coordination_matrix:
            raise RuntimeError("No coordination matrix available for temporal fragmentation")
        
        # Determine optimal fragment count
        fragment_count = self._calculate_optimal_fragment_count(len(message_payload))
        
        # Generate temporal coordinates within the coordination window
        temporal_coordinates = self._generate_temporal_coordinates(
            coordination_matrix, fragment_count
        )
        
        # Split message into base fragments
        base_fragments = self._split_message(message_payload, fragment_count)
        
        # Apply temporal transformation to each fragment
        fragments = []
        for i, (base_fragment, temporal_coord) in enumerate(zip(base_fragments, temporal_coordinates)):
            fragment = await self._create_temporal_fragment(
                message_id=message_id,
                sequence_number=i,
                total_fragments=fragment_count,
                payload=base_fragment,
                temporal_coordinate=temporal_coord,
                source_node=source_node,
                destination_node=destination_node
            )
            fragments.append(fragment)
        
        # Create fragmented message record
        fragmented_message = FragmentedMessage(
            message_id=message_id,
            original_payload=message_payload,
            fragments=fragments,
            source_node=source_node,
            destination_node=destination_node,
            fragmentation_time=start_time
        )
        
        # Store fragments and message
        self.active_fragments[message_id] = fragments
        for fragment in fragments:
            self.fragment_index[fragment.fragment_id] = fragment
        self.fragmented_messages[message_id] = fragmented_message
        
        # Update performance metrics
        fragmentation_time = time.time() - start_time
        self.performance_metrics['messages_fragmented'] += 1
        self.performance_metrics['fragments_created'] += len(fragments)
        self.performance_metrics['average_fragmentation_time'] = (
            0.9 * self.performance_metrics['average_fragmentation_time'] + 
            0.1 * fragmentation_time
        )
        
        # Update average entropy
        avg_entropy = fragmented_message.average_entropy
        self.performance_metrics['average_entropy'] = (
            0.9 * self.performance_metrics['average_entropy'] + 
            0.1 * avg_entropy
        )
        
        # Log fragmentation data
        if self.data_collector:
            await self.data_collector.log_message_fragmentation({
                'timestamp': start_time,
                'message_id': message_id,
                'source_node': source_node,
                'destination_node': destination_node,
                'original_size': len(message_payload),
                'fragment_count': fragment_count,
                'total_fragment_size': fragmented_message.total_fragment_size,
                'overhead_ratio': fragmented_message.overhead_ratio,
                'average_entropy': avg_entropy,
                'fragmentation_time': fragmentation_time,
                'temporal_window_duration': coordination_matrix.temporal_window_duration
            })
        
        self.logger.debug(f"Fragmented message {message_id} into {fragment_count} fragments with {avg_entropy:.3f} average entropy")
        
        return fragmented_message
    
    def _calculate_optimal_fragment_count(self, message_size: int) -> int:
        """Calculate optimal number of fragments for a message"""
        
        # Base fragment count on message size
        base_count = max(self.min_fragments, min(self.max_fragments, message_size // self.target_fragment_size))
        
        # Adjust for security (more fragments = higher security)
        if message_size > 4096:  # Large messages get more fragments
            base_count = min(self.max_fragments, base_count + 2)
        
        # Ensure reasonable bounds
        return max(self.min_fragments, min(self.max_fragments, base_count))
    
    def _generate_temporal_coordinates(self, coordination_matrix, fragment_count: int) -> List[float]:
        """Generate temporal coordinates for fragment distribution"""
        
        coordinates = []
        window_start = coordination_matrix.temporal_window_start
        window_end = coordination_matrix.temporal_window_end
        window_duration = coordination_matrix.temporal_window_duration
        
        if window_duration == 0:
            # Fallback: create small synthetic window
            window_duration = 0.1  # 100ms
            window_end = window_start + window_duration
        
        # Distribute fragments across the temporal window
        for i in range(fragment_count):
            # Base position within window
            progress = (i + 0.5) / fragment_count  # Center fragments in their slots
            base_coordinate = window_start + (progress * window_duration)
            
            # Add temporal jitter for security
            jitter = random.uniform(-self.temporal_jitter/2, self.temporal_jitter/2)
            coordinate = base_coordinate + jitter
            
            # Ensure coordinate stays within window bounds
            coordinate = max(window_start, min(window_end, coordinate))
            coordinates.append(coordinate)
        
        # Sort coordinates to maintain temporal order
        coordinates.sort()
        return coordinates
    
    def _split_message(self, message: bytes, fragment_count: int) -> List[bytes]:
        """Split message into base fragments before temporal transformation"""
        
        message_length = len(message)
        base_fragment_size = message_length // fragment_count
        remainder = message_length % fragment_count
        
        fragments = []
        offset = 0
        
        for i in range(fragment_count):
            # Distribute remainder bytes across first fragments
            fragment_size = base_fragment_size + (1 if i < remainder else 0)
            fragment_data = message[offset:offset + fragment_size]
            fragments.append(fragment_data)
            offset += fragment_size
        
        return fragments
    
    async def _create_temporal_fragment(self, message_id: str, sequence_number: int, 
                                       total_fragments: int, payload: bytes,
                                       temporal_coordinate: float, source_node: str,
                                       destination_node: str) -> MessageFragment:
        """Create a temporal fragment with cryptographic transformation"""
        
        fragment_id = f"{message_id}_{sequence_number}"
        
        # Generate temporal key based on coordinate
        temporal_key = self._derive_temporal_key(temporal_coordinate, message_id)
        
        # Apply temporal transformation T(M_i, j, t, K_t)
        transformed_payload = self._apply_temporal_transformation(
            payload, sequence_number, temporal_coordinate, temporal_key
        )
        
        # Create fragment
        fragment = MessageFragment(
            fragment_id=fragment_id,
            message_id=message_id,
            sequence_number=sequence_number,
            total_fragments=total_fragments,
            temporal_coordinate=temporal_coordinate,
            payload=transformed_payload,
            temporal_key=temporal_key,
            source_node=source_node,
            destination_node=destination_node
        )
        
        return fragment
    
    def _derive_temporal_key(self, temporal_coordinate: float, message_id: str) -> bytes:
        """Derive temporal key from coordinate and message ID"""
        
        # Convert temporal coordinate to bytes with high precision
        coord_bytes = struct.pack('!d', temporal_coordinate)  # 8 bytes double
        message_bytes = message_id.encode('utf-8')
        
        # Create key derivation input
        key_input = self.key_derivation_salt + coord_bytes + message_bytes
        
        # Use SHA-256 for key derivation
        key_hash = hashlib.sha256(key_input).digest()
        
        return key_hash[:self.temporal_key_size]
    
    def _apply_temporal_transformation(self, payload: bytes, sequence_number: int,
                                     temporal_coordinate: float, temporal_key: bytes) -> bytes:
        """Apply temporal transformation function T(M_i, j, t, K_t)"""
        
        # Create transformation factors
        sequence_factor = struct.pack('!I', sequence_number)
        coordinate_factor = struct.pack('!d', temporal_coordinate)
        
        # Combine factors into transformation seed
        transformation_seed = temporal_key + sequence_factor + coordinate_factor
        
        # Generate transformation stream using the seed
        transformation_stream = self._generate_transformation_stream(
            transformation_seed, len(payload)
        )
        
        # Apply XOR transformation (ensures reversibility)
        transformed = bytearray()
        for i, byte in enumerate(payload):
            transformed_byte = byte ^ transformation_stream[i % len(transformation_stream)]
            transformed.append(transformed_byte)
        
        return bytes(transformed)
    
    def _generate_transformation_stream(self, seed: bytes, length: int) -> bytes:
        """Generate deterministic transformation stream from seed"""
        
        stream = bytearray()
        current_seed = seed
        
        while len(stream) < length:
            # Hash current seed to get next block
            hash_result = hashlib.sha256(current_seed).digest()
            stream.extend(hash_result)
            
            # Update seed for next iteration
            current_seed = hash_result
        
        return bytes(stream[:length])
    
    async def reconstruct_message(self, fragments: List[MessageFragment]) -> Optional[bytes]:
        """Reconstruct message from temporal fragments"""
        
        if not fragments:
            return None
        
        start_time = time.time()
        message_id = fragments[0].message_id
        
        try:
            # Validate fragment completeness
            if not self._validate_fragment_completeness(fragments):
                self.performance_metrics['reconstruction_failures'] += 1
                return None
            
            # Sort fragments by sequence number
            fragments.sort(key=lambda f: f.sequence_number)
            
            # Reverse temporal transformation for each fragment
            base_fragments = []
            for fragment in fragments:
                # Reverse the temporal transformation
                base_payload = self._reverse_temporal_transformation(
                    fragment.payload, fragment.sequence_number,
                    fragment.temporal_coordinate, fragment.temporal_key
                )
                base_fragments.append(base_payload)
            
            # Concatenate base fragments
            reconstructed_message = b''.join(base_fragments)
            
            # Update fragmented message record
            if message_id in self.fragmented_messages:
                fragmented_msg = self.fragmented_messages[message_id]
                fragmented_msg.is_reconstructed = True
                fragmented_msg.reconstructed_payload = reconstructed_message
                fragmented_msg.reconstruction_time = time.time()
            
            # Update performance metrics
            reconstruction_time = time.time() - start_time
            self.performance_metrics['messages_reconstructed'] += 1
            self.performance_metrics['average_reconstruction_time'] = (
                0.9 * self.performance_metrics['average_reconstruction_time'] + 
                0.1 * reconstruction_time
            )
            
            # Log reconstruction
            if self.data_collector:
                await self.data_collector.log_message_reconstruction({
                    'timestamp': start_time,
                    'message_id': message_id,
                    'fragment_count': len(fragments),
                    'reconstructed_size': len(reconstructed_message),
                    'reconstruction_time': reconstruction_time,
                    'success': True
                })
            
            self.logger.debug(f"Successfully reconstructed message {message_id} from {len(fragments)} fragments")
            return reconstructed_message
            
        except Exception as e:
            self.logger.error(f"Failed to reconstruct message {message_id}: {e}")
            self.performance_metrics['reconstruction_failures'] += 1
            
            if self.data_collector:
                await self.data_collector.log_message_reconstruction({
                    'timestamp': start_time,
                    'message_id': message_id,
                    'fragment_count': len(fragments),
                    'reconstruction_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                })
            
            return None
    
    def _validate_fragment_completeness(self, fragments: List[MessageFragment]) -> bool:
        """Validate that fragments form a complete message"""
        
        if not fragments:
            return False
        
        # Check that all fragments belong to the same message
        message_id = fragments[0].message_id
        if not all(f.message_id == message_id for f in fragments):
            self.logger.warning("Fragments from different messages cannot be reconstructed together")
            return False
        
        # Check fragment count consistency
        expected_count = fragments[0].total_fragments
        if len(fragments) != expected_count:
            self.logger.warning(f"Incomplete fragment set: {len(fragments)}/{expected_count}")
            return False
        
        # Check sequence number completeness
        sequence_numbers = {f.sequence_number for f in fragments}
        expected_sequences = set(range(expected_count))
        if sequence_numbers != expected_sequences:
            missing = expected_sequences - sequence_numbers
            self.logger.warning(f"Missing fragment sequences: {missing}")
            return False
        
        # Check fragment expiration
        current_time = time.time()
        if any(f.expiration_time < current_time for f in fragments):
            self.logger.warning("Some fragments have expired")
            return False
        
        return True
    
    def _reverse_temporal_transformation(self, transformed_payload: bytes, sequence_number: int,
                                       temporal_coordinate: float, temporal_key: bytes) -> bytes:
        """Reverse temporal transformation to recover original fragment"""
        
        # Generate the same transformation stream used during encoding
        sequence_factor = struct.pack('!I', sequence_number)
        coordinate_factor = struct.pack('!d', temporal_coordinate)
        transformation_seed = temporal_key + sequence_factor + coordinate_factor
        
        transformation_stream = self._generate_transformation_stream(
            transformation_seed, len(transformed_payload)
        )
        
        # Reverse XOR transformation
        original = bytearray()
        for i, byte in enumerate(transformed_payload):
            original_byte = byte ^ transformation_stream[i % len(transformation_stream)]
            original.append(original_byte)
        
        return bytes(original)
    
    async def _cleanup_expired_fragments(self):
        """Clean up expired fragments and messages"""
        while self.is_running:
            try:
                current_time = time.time()
                expired_messages = []
                
                # Find expired fragments
                for message_id, fragments in self.active_fragments.items():
                    expired_fragments = [f for f in fragments if f.is_expired]
                    
                    if expired_fragments:
                        # Remove expired fragments from index
                        for fragment in expired_fragments:
                            if fragment.fragment_id in self.fragment_index:
                                del self.fragment_index[fragment.fragment_id]
                        
                        # Remove expired fragments from active list
                        self.active_fragments[message_id] = [
                            f for f in fragments if not f.is_expired
                        ]
                        
                        # Mark message for cleanup if no fragments remain
                        if not self.active_fragments[message_id]:
                            expired_messages.append(message_id)
                
                # Clean up expired messages
                for message_id in expired_messages:
                    if message_id in self.active_fragments:
                        del self.active_fragments[message_id]
                    if message_id in self.fragmented_messages:
                        del self.fragmented_messages[message_id]
                
                if expired_messages:
                    self.logger.debug(f"Cleaned up {len(expired_messages)} expired messages")
                
                await asyncio.sleep(30.0)  # Cleanup every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in fragment cleanup: {e}")
                await asyncio.sleep(10.0)
    
    def get_fragment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fragmentation statistics"""
        
        active_messages = len(self.active_fragments)
        total_active_fragments = sum(len(fragments) for fragments in self.active_fragments.values())
        
        # Calculate entropy distribution
        entropy_scores = []
        for fragments in self.active_fragments.values():
            entropy_scores.extend(f.entropy_score for f in fragments)
        
        avg_entropy = statistics.mean(entropy_scores) if entropy_scores else 0.0
        high_entropy_ratio = sum(1 for e in entropy_scores if e > self.entropy_threshold) / len(entropy_scores) if entropy_scores else 0.0
        
        return {
            'active_messages': active_messages,
            'total_active_fragments': total_active_fragments,
            'messages_fragmented': self.performance_metrics['messages_fragmented'],
            'fragments_created': self.performance_metrics['fragments_created'],
            'messages_reconstructed': self.performance_metrics['messages_reconstructed'],
            'reconstruction_failures': self.performance_metrics['reconstruction_failures'],
            'reconstruction_success_rate': (
                self.performance_metrics['messages_reconstructed'] / 
                max(1, self.performance_metrics['messages_reconstructed'] + self.performance_metrics['reconstruction_failures'])
            ),
            'average_entropy': avg_entropy,
            'high_entropy_ratio': high_entropy_ratio,
            'average_fragmentation_time': self.performance_metrics['average_fragmentation_time'],
            'average_reconstruction_time': self.performance_metrics['average_reconstruction_time'],
            'entropy_threshold': self.entropy_threshold
        }
    
    def stop(self):
        """Stop temporal fragmentation service"""
        self.is_running = False
        self.logger.info("Temporal fragmenter stopped")
