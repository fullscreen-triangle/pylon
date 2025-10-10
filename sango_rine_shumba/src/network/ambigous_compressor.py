import time
import json
import logging
import zlib
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class AmbiguousCompressor:
    """Extracts gear ratios from compression-resistant patterns"""
    
    def __init__(self, compression_threshold: float = 0.7):
        self.logger = logging.getLogger(__name__)
        self.compression_threshold = compression_threshold
        self.window_size = 64  # bytes
        self.overlap_size = 32  # bytes
        
        # Statistics tracking
        self.total_data_processed = 0
        self.segments_analyzed = 0
        self.compression_resistant_segments = 0
        self.gear_ratios_extracted = 0
        self.start_time = time.time()
        
        self.logger.info(f"Ambiguous compressor initialized with threshold {compression_threshold}")
    
    def extract_gear_ratios_from_ambiguous_bits(self, data: bytes) -> List[float]:
        """Find compression-resistant patterns that become gear ratios"""
        start_time = time.time()
        self.total_data_processed += len(data)
        
        # Find high entropy segments
        compression_resistant_segments = self.find_high_entropy_segments(data)
        self.compression_resistant_segments += len(compression_resistant_segments)
        
        gear_ratios = []
        
        for segment in compression_resistant_segments:
            self.segments_analyzed += 1
            
            if self.has_multiple_meanings(segment):
                ratio = self.extract_gear_ratio_from_ambiguity(segment)
                gear_ratios.append(ratio)
                self.gear_ratios_extracted += 1
        
        processing_time = time.time() - start_time
        self.logger.debug(f"Extracted {len(gear_ratios)} gear ratios from {len(data)} bytes in {processing_time*1000:.2f}ms")
        
        return gear_ratios
    
    def find_high_entropy_segments(self, data: bytes) -> List[bytes]:
        """Find compression-resistant segments using sliding window analysis"""
        if len(data) < self.window_size:
            # Test entire data if smaller than window
            if self._is_compression_resistant(data):
                return [data]
            else:
                return []
        
        resistant_segments = []
        
        # Sliding window with overlap
        for i in range(0, len(data) - self.window_size + 1, self.window_size - self.overlap_size):
            window = data[i:i + self.window_size]
            
            if self._is_compression_resistant(window):
                resistant_segments.append(window)
        
        return resistant_segments
    
    def _is_compression_resistant(self, segment: bytes) -> bool:
        """Check if segment resists compression"""
        if len(segment) == 0:
            return False
        
        try:
            # Use zlib compression
            compressed = zlib.compress(segment, level=9)  # Maximum compression
            compression_ratio = len(compressed) / len(segment)
            
            # Segment is compression-resistant if ratio > threshold
            return compression_ratio > self.compression_threshold
            
        except Exception as e:
            self.logger.warning(f"Compression test failed: {e}")
            return False
    
    def has_multiple_meanings(self, segment: bytes) -> bool:
        """Check if segment has multiple possible interpretations"""
        if len(segment) < 4:
            return False
        
        # Test for ambiguity through multiple interpretation approaches
        ambiguity_tests = []
        
        # Test 1: Byte pattern distribution
        byte_counts = {}
        for byte in segment:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        # High entropy in byte distribution suggests multiple meanings
        byte_entropy = self._calculate_byte_entropy(byte_counts, len(segment))
        ambiguity_tests.append(byte_entropy > 7.0)  # Near maximum entropy
        
        # Test 2: Multiple valid interpretations as different data types
        interpretations = []
        
        # Interpret as integers
        try:
            if len(segment) >= 4:
                int_val = int.from_bytes(segment[:4], byteorder='big')
                interpretations.append(f"int_{int_val}")
        except:
            pass
        
        # Interpret as floats (if possible)
        try:
            if len(segment) >= 4:
                import struct
                float_val = struct.unpack('>f', segment[:4])[0]
                if not (np.isnan(float_val) or np.isinf(float_val)):
                    interpretations.append(f"float_{float_val:.6f}")
        except:
            pass
        
        # Interpret as ASCII (if printable)
        try:
            ascii_val = segment.decode('ascii')
            if ascii_val.isprintable():
                interpretations.append(f"ascii_{ascii_val}")
        except:
            pass
        
        # Interpret as hex string
        hex_val = segment.hex()
        interpretations.append(f"hex_{hex_val}")
        
        ambiguity_tests.append(len(interpretations) >= 2)
        
        # Test 3: Hash-based ambiguity (multiple hash meanings)
        hash_meanings = []
        hash_meanings.append(hashlib.md5(segment).hexdigest()[:8])
        hash_meanings.append(hashlib.sha1(segment).hexdigest()[:8])
        hash_meanings.append(hashlib.sha256(segment).hexdigest()[:8])
        
        # Check if hashes produce different patterns
        unique_patterns = len(set(hash_meanings))
        ambiguity_tests.append(unique_patterns >= 2)
        
        # Segment has multiple meanings if it passes at least 2 ambiguity tests
        return sum(ambiguity_tests) >= 2
    
    def _calculate_byte_entropy(self, byte_counts: Dict[int, int], total_bytes: int) -> float:
        """Calculate Shannon entropy of byte distribution"""
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            if probability > 0:
                entropy -= probability * np.log2(probability)
        return entropy
    
    def extract_gear_ratio_from_ambiguity(self, segment: bytes) -> float:
        """Extract gear ratio from ambiguous segment"""
        try:
            # Method 1: Hash-based ratio extraction
            segment_hash = hashlib.sha256(segment).hexdigest()
            hash_int = int(segment_hash[:16], 16)
            
            # Normalize to meaningful gear ratio range [0.001, 1000.0]
            # Use logarithmic scaling for gear ratios
            normalized_hash = (hash_int % 10000000) / 10000000.0  # [0, 1)
            
            # Convert to gear ratio with logarithmic distribution
            min_log = np.log10(0.001)  # Minimum gear ratio
            max_log = np.log10(1000.0)  # Maximum gear ratio
            log_ratio = min_log + normalized_hash * (max_log - min_log)
            gear_ratio = 10 ** log_ratio
            
            # Method 2: Entropy-based modulation
            byte_counts = {}
            for byte in segment:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1
            
            entropy = self._calculate_byte_entropy(byte_counts, len(segment))
            entropy_factor = entropy / 8.0  # Normalize to [0, 1]
            
            # Modulate gear ratio based on entropy
            modulated_ratio = gear_ratio * (0.5 + 0.5 * entropy_factor)
            
            # Method 3: Segment length influence
            length_factor = min(2.0, len(segment) / 32.0)  # Longer segments get higher ratios
            final_ratio = modulated_ratio * length_factor
            
            # Ensure ratio stays in valid range
            final_ratio = max(0.001, min(1000.0, final_ratio))
            
            self.logger.debug(f"Extracted gear ratio {final_ratio:.6f} from {len(segment)}-byte ambiguous segment")
            return final_ratio
            
        except Exception as e:
            self.logger.error(f"Error extracting gear ratio: {e}")
            return 1.0  # Default neutral ratio
    
    def analyze_compression_resistance_batch(self, data_batch: List[bytes]) -> Dict[str, Any]:
        """Analyze compression resistance across multiple data samples"""
        batch_start_time = time.time()
        
        batch_results = {
            'total_samples': len(data_batch),
            'compression_analysis': [],
            'aggregate_statistics': {},
            'extracted_gear_ratios': []
        }
        
        compression_ratios = []
        segment_counts = []
        gear_ratio_counts = []
        
        for i, data in enumerate(data_batch):
            # Analyze individual sample
            sample_segments = self.find_high_entropy_segments(data)
            sample_gear_ratios = self.extract_gear_ratios_from_ambiguous_bits(data)
            
            # Calculate overall compression ratio for sample
            try:
                compressed_data = zlib.compress(data, level=9)
                overall_compression_ratio = len(compressed_data) / max(1, len(data))
            except:
                overall_compression_ratio = 1.0
            
            sample_analysis = {
                'sample_index': i,
                'data_size': len(data),
                'overall_compression_ratio': overall_compression_ratio,
                'resistant_segments_found': len(sample_segments),
                'gear_ratios_extracted': len(sample_gear_ratios),
                'compression_resistant': overall_compression_ratio > self.compression_threshold
            }
            
            batch_results['compression_analysis'].append(sample_analysis)
            batch_results['extracted_gear_ratios'].extend(sample_gear_ratios)
            
            compression_ratios.append(overall_compression_ratio)
            segment_counts.append(len(sample_segments))
            gear_ratio_counts.append(len(sample_gear_ratios))
        
        # Calculate aggregate statistics
        if compression_ratios:
            batch_results['aggregate_statistics'] = {
                'mean_compression_ratio': np.mean(compression_ratios),
                'std_compression_ratio': np.std(compression_ratios),
                'compression_resistant_samples': sum(1 for ratio in compression_ratios if ratio > self.compression_threshold),
                'compression_resistance_rate': sum(1 for ratio in compression_ratios if ratio > self.compression_threshold) / len(compression_ratios),
                'total_resistant_segments': sum(segment_counts),
                'total_gear_ratios_extracted': sum(gear_ratio_counts),
                'average_segments_per_sample': np.mean(segment_counts),
                'average_gear_ratios_per_sample': np.mean(gear_ratio_counts),
                'batch_processing_time': time.time() - batch_start_time
            }
        
        return batch_results
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression analysis statistics"""
        current_time = time.time()
        runtime = current_time - self.start_time
        
        # Calculate processing rates
        data_rate = self.total_data_processed / max(1, runtime)  # bytes per second
        segment_rate = self.segments_analyzed / max(1, runtime)  # segments per second
        gear_ratio_rate = self.gear_ratios_extracted / max(1, runtime)  # ratios per second
        
        # Calculate efficiency metrics
        resistance_detection_rate = self.compression_resistant_segments / max(1, self.segments_analyzed)
        gear_ratio_extraction_rate = self.gear_ratios_extracted / max(1, self.compression_resistant_segments)
        
        return {
            'processing_statistics': {
                'runtime_seconds': runtime,
                'total_data_processed_bytes': self.total_data_processed,
                'total_segments_analyzed': self.segments_analyzed,
                'compression_resistant_segments_found': self.compression_resistant_segments,
                'gear_ratios_extracted': self.gear_ratios_extracted
            },
            'processing_rates': {
                'data_processing_rate_bytes_per_second': data_rate,
                'segment_analysis_rate_per_second': segment_rate,
                'gear_ratio_extraction_rate_per_second': gear_ratio_rate
            },
            'efficiency_metrics': {
                'resistance_detection_rate': resistance_detection_rate,
                'gear_ratio_extraction_rate': gear_ratio_extraction_rate,
                'overall_extraction_efficiency': resistance_detection_rate * gear_ratio_extraction_rate
            },
            'configuration': {
                'compression_threshold': self.compression_threshold,
                'window_size': self.window_size,
                'overlap_size': self.overlap_size
            }
        }
    
    def export_compression_data(self, filepath: str):
        """Export compression analysis data to JSON"""
        export_data = {
            'export_metadata': {
                'export_timestamp': time.time(),
                'compressor_version': '1.0',
                'analysis_type': 'ambiguous_compression'
            },
            'compression_statistics': self.get_compression_statistics(),
            'theoretical_foundation': {
                'compression_principle': 'Compression-resistant patterns contain maximum semantic density',
                'ambiguity_principle': 'Multiple meanings indicate higher information content',
                'gear_ratio_principle': 'Ambiguous segments encode hierarchical frequency relationships'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Compression analysis data exported to {filepath}")
    
    def create_compression_visualization(self, data_samples: List[bytes], output_dir: str):
        """Create visualizations of compression analysis"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Analyze samples for visualization data
            batch_results = self.analyze_compression_resistance_batch(data_samples)
            
            import matplotlib.pyplot as plt
            
            # Plot 1: Compression ratio distribution
            compression_ratios = [analysis['overall_compression_ratio'] for analysis in batch_results['compression_analysis']]
            
            plt.figure(figsize=(10, 6))
            plt.hist(compression_ratios, bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=self.compression_threshold, color='red', linestyle='--', 
                       label=f'Compression Threshold ({self.compression_threshold})')
            plt.xlabel('Compression Ratio')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Compression Ratios')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / "compression_ratio_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Gear ratios extracted
            if batch_results['extracted_gear_ratios']:
                gear_ratios = batch_results['extracted_gear_ratios']
                
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(gear_ratios)), gear_ratios, alpha=0.6, color='green')
                plt.xlabel('Extraction Order')
                plt.ylabel('Gear Ratio Value')
                plt.title('Extracted Gear Ratios from Ambiguous Segments')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path / "extracted_gear_ratios.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Compression visualizations saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
    
    def __repr__(self):
        return f"AmbiguousCompressor(threshold={self.compression_threshold}, ratios_extracted={self.gear_ratios_extracted})"