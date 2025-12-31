"""
GPU Monitoring Utility for tracking GPU utilization and memory usage.

Uses pynvml (NVIDIA Management Library) to monitor GPU metrics.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Try importing pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. Install with: pip install nvidia-ml-py3")


@dataclass
class GPUSnapshot:
    """Single GPU measurement snapshot."""
    timestamp: float
    utilization_gpu: int  # GPU utilization percentage (0-100)
    utilization_memory: int  # Memory utilization percentage (0-100)
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[int] = None
    power_watts: Optional[float] = None


class GPUMonitor:
    """Monitor GPU utilization and memory usage over time.
    
    Usage:
        monitor = GPUMonitor(device_id=0)
        monitor.start()
        # ... run GPU workload ...
        monitor.stop()
        stats = monitor.get_statistics()
        print(f"Average GPU utilization: {stats['avg_gpu_util']:.1f}%")
    """
    
    def __init__(self, device_id: int = 0):
        """Initialize GPU monitor.
        
        Args:
            device_id: CUDA device ID to monitor (default: 0)
        """
        self.device_id = device_id
        self.handle = None
        self.snapshots: List[GPUSnapshot] = []
        self.monitoring = False
        self._monitor_thread = None
        
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml not available - GPU monitoring disabled")
            return
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            logger.info(f"GPU Monitor initialized for device {device_id}: {device_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GPU monitor: {e}")
            self.handle = None
    
    def start(self, interval_seconds: float = 0.5):
        """Start monitoring GPU in background thread.
        
        Args:
            interval_seconds: Sampling interval (default: 0.5s)
        """
        if not PYNVML_AVAILABLE or self.handle is None:
            logger.warning("GPU monitoring not available")
            return
        
        if self.monitoring:
            logger.warning("GPU monitoring already started")
            return
        
        self.monitoring = True
        self.snapshots = []
        
        import threading
        
        def monitor_loop():
            while self.monitoring:
                try:
                    snapshot = self._capture_snapshot()
                    if snapshot:
                        self.snapshots.append(snapshot)
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
                
                time.sleep(interval_seconds)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop(self):
        """Stop monitoring GPU."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        logger.info(f"GPU monitoring stopped ({len(self.snapshots)} samples)")
    
    def _capture_snapshot(self) -> Optional[GPUSnapshot]:
        """Capture a single GPU snapshot."""
        if not PYNVML_AVAILABLE or self.handle is None:
            return None
        
        try:
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used_mb = mem_info.used / (1024 ** 2)
            memory_total_mb = mem_info.total / (1024 ** 2)
            memory_util = int((mem_info.used / mem_info.total) * 100)
            
            # Optional metrics
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            except:
                power = None
            
            return GPUSnapshot(
                timestamp=time.time(),
                utilization_gpu=util.gpu,
                utilization_memory=memory_util,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                temperature_c=temp,
                power_watts=power,
            )
        except Exception as e:
            logger.debug(f"Failed to capture GPU snapshot: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics from monitoring session.
        
        Returns:
            Dictionary with keys:
                - avg_gpu_util: Average GPU utilization (%)
                - peak_gpu_util: Peak GPU utilization (%)
                - avg_memory_util: Average memory utilization (%)
                - peak_memory_mb: Peak memory usage (MB)
                - avg_power_watts: Average power consumption (W)
                - duration_seconds: Monitoring duration
        """
        if not self.snapshots:
            return {
                'avg_gpu_util': 0.0,
                'peak_gpu_util': 0.0,
                'avg_memory_util': 0.0,
                'peak_memory_mb': 0.0,
                'avg_power_watts': 0.0,
                'duration_seconds': 0.0,
            }
        
        gpu_utils = [s.utilization_gpu for s in self.snapshots]
        mem_utils = [s.utilization_memory for s in self.snapshots]
        mem_used = [s.memory_used_mb for s in self.snapshots]
        powers = [s.power_watts for s in self.snapshots if s.power_watts is not None]
        
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        
        return {
            'avg_gpu_util': sum(gpu_utils) / len(gpu_utils),
            'peak_gpu_util': max(gpu_utils),
            'avg_memory_util': sum(mem_utils) / len(mem_utils),
            'peak_memory_mb': max(mem_used),
            'avg_power_watts': sum(powers) / len(powers) if powers else 0.0,
            'duration_seconds': duration,
        }
    
    def get_avg_utilization(self) -> float:
        """Get average GPU utilization percentage."""
        stats = self.get_statistics()
        return stats['avg_gpu_util']
    
    def get_peak_memory(self) -> float:
        """Get peak GPU memory usage in MB."""
        stats = self.get_statistics()
        return stats['peak_memory_mb']
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Cleanup on deletion."""
        if PYNVML_AVAILABLE and self.handle is not None:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
