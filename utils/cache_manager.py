"""
Intelligent Caching System for CAN-Graph Training

Implements automatic caching and loading of preprocessing and model predictions
to dramatically reduce training startup time from 50+ minutes to under 1 minute.
"""

import os
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import hashlib
import json

class CacheManager:
    """Manages caching of preprocessing results and model predictions."""
    
    def __init__(self, cache_dir: str = "cache", dataset_name: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_name = dataset_name
        
        # Cache configuration
        self.cache_config = {
            'id_mapping': {'enabled': True, 'max_age_hours': 24*7},  # 1 week
            'raw_dataset': {'enabled': True, 'max_age_hours': 24*7},
            'fusion_predictions': {'enabled': True, 'max_age_hours': 24*3},  # 3 days
            'gpu_states': {'enabled': True, 'max_age_hours': 24*1}  # 1 day
        }
    
    def _get_cache_path(self, cache_type: str, dataset_name: str = None) -> Path:
        """Generate cache file path."""
        name = dataset_name or self.dataset_name or "default"
        return self.cache_dir / f"{name}_{cache_type}.pkl"
    
    def _get_metadata_path(self, cache_type: str, dataset_name: str = None) -> Path:
        """Generate metadata file path."""
        name = dataset_name or self.dataset_name or "default"
        return self.cache_dir / f"{name}_{cache_type}.meta.json"
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for integrity checking."""
        if isinstance(data, torch.Tensor):
            data_bytes = data.cpu().numpy().tobytes()
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, (list, tuple)):
            data_bytes = str(data).encode()
        elif isinstance(data, dict):
            data_bytes = str(sorted(data.items())).encode()
        else:
            data_bytes = str(data).encode()
        
        return hashlib.md5(data_bytes).hexdigest()
    
    def _is_cache_valid(self, cache_type: str, dataset_name: str = None) -> bool:
        """Check if cache is valid and not expired."""
        cache_path = self._get_cache_path(cache_type, dataset_name)
        meta_path = self._get_metadata_path(cache_type, dataset_name)
        
        if not cache_path.exists() or not meta_path.exists():
            return False
        
        # Check age
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            cache_time = metadata.get('timestamp', 0)
            max_age_sec = self.cache_config[cache_type]['max_age_hours'] * 3600
            current_time = time.time()
            
            if current_time - cache_time > max_age_sec:
                print(f"🗑️  Cache expired for {cache_type} (age: {(current_time - cache_time)/3600:.1f}h)")
                return False
                
            return True
            
        except Exception as e:
            print(f"⚠️  Cache metadata error for {cache_type}: {e}")
            return False
    
    def save_cache(self, data: Any, cache_type: str, dataset_name: str = None,
                   metadata: Dict = None) -> bool:
        """Save data to cache with metadata."""
        if not self.cache_config[cache_type]['enabled']:
            return False
        
        cache_path = self._get_cache_path(cache_type, dataset_name)
        meta_path = self._get_metadata_path(cache_type, dataset_name)
        
        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            meta_data = {
                'timestamp': time.time(),
                'cache_type': cache_type,
                'dataset_name': dataset_name or self.dataset_name,
                'data_hash': self._compute_data_hash(data),
                'size_mb': cache_path.stat().st_size / (1024**2),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                'torch_version': torch.__version__
            }
            
            if metadata:
                meta_data.update(metadata)
            
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            print(f"💾 Cached {cache_type}: {meta_data['size_mb']:.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Failed to cache {cache_type}: {e}")
            return False
    
    def load_cache(self, cache_type: str, dataset_name: str = None) -> Optional[Any]:
        """Load data from cache if valid."""
        if not self.cache_config[cache_type]['enabled']:
            return None
            
        if not self._is_cache_valid(cache_type, dataset_name):
            return None
        
        cache_path = self._get_cache_path(cache_type, dataset_name)
        
        try:
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            load_time = time.time() - start_time
            size_mb = cache_path.stat().st_size / (1024**2)
            
            print(f"📥 Loaded {cache_type}: {size_mb:.1f} MB in {load_time:.2f}s")
            return data
            
        except Exception as e:
            print(f"❌ Failed to load {cache_type}: {e}")
            return None
    
    def clear_cache(self, cache_type: str = None, dataset_name: str = None):
        """Clear specific cache or all caches."""
        if cache_type:
            # Clear specific cache
            cache_types = [cache_type]
        else:
            # Clear all caches
            cache_types = list(self.cache_config.keys())
        
        cleared_count = 0
        for ct in cache_types:
            cache_path = self._get_cache_path(ct, dataset_name)
            meta_path = self._get_metadata_path(ct, dataset_name)
            
            for path in [cache_path, meta_path]:
                if path.exists():
                    try:
                        path.unlink()
                        cleared_count += 1
                    except Exception as e:
                        print(f"⚠️  Failed to remove {path}: {e}")
        
        print(f"🗑️  Cleared {cleared_count} cache files")
    
    def get_cache_info(self) -> Dict:
        """Get information about all caches."""
        info = {'total_size_mb': 0, 'caches': {}}
        
        for cache_type in self.cache_config.keys():
            cache_path = self._get_cache_path(cache_type)
            meta_path = self._get_metadata_path(cache_type)
            
            cache_info = {
                'exists': cache_path.exists(),
                'valid': False,
                'size_mb': 0,
                'age_hours': 0
            }
            
            if cache_path.exists():
                cache_info['size_mb'] = cache_path.stat().st_size / (1024**2)
                info['total_size_mb'] += cache_info['size_mb']
                
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                        cache_info['age_hours'] = (time.time() - metadata['timestamp']) / 3600
                        cache_info['valid'] = self._is_cache_valid(cache_type)
                    except Exception:
                        pass
            
            info['caches'][cache_type] = cache_info
        
        return info
    
    def print_cache_status(self):
        """Print detailed cache status."""
        info = self.get_cache_info()
        
        print(f"\n📊 CACHE STATUS ({self.dataset_name or 'default'})")
        print("-" * 50)
        print(f"Total Cache Size: {info['total_size_mb']:.1f} MB")
        print()
        
        for cache_type, cache_info in info['caches'].items():
            status = "✅ Valid" if cache_info['valid'] else ("❌ Invalid" if cache_info['exists'] else "⭕ Missing")
            size = f"{cache_info['size_mb']:.1f} MB" if cache_info['exists'] else "N/A"
            age = f"{cache_info['age_hours']:.1f}h" if cache_info['exists'] else "N/A"
            
            print(f"{cache_type:20} | {status:10} | {size:10} | Age: {age}")


def estimate_time_savings() -> Dict[str, float]:
    """Estimate time savings from caching."""
    return {
        'id_mapping': 0.25,      # 15 seconds
        'raw_dataset': 23.0,     # Graph creation
        'fusion_predictions': 25.0,  # Model inference  
        'gpu_states': 2.0,       # State computation
        'total_without_cache': 50.25,
        'total_with_cache': 1.0   # Just loading time
    }


if __name__ == "__main__":
    # Demo the cache manager
    cache_mgr = CacheManager(dataset_name="set_04")
    
    print("🔍 CAN-Graph Cache Manager Demo")
    print("=" * 50)
    
    # Show current cache status
    cache_mgr.print_cache_status()
    
    # Show time savings estimate
    savings = estimate_time_savings()
    print(f"\n⏱️  TIME SAVINGS ESTIMATE:")
    print(f"Without cache: {savings['total_without_cache']:.1f} minutes")
    print(f"With cache: {savings['total_with_cache']:.1f} minutes")
    print(f"Time saved: {savings['total_without_cache'] - savings['total_with_cache']:.1f} minutes")
    print(f"Speedup: {savings['total_without_cache'] / savings['total_with_cache']:.1f}x faster")