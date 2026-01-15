"""
Simple cache manager for preprocessing data.
"""

import os
import pickle
from typing import Any, Optional, Dict


class CacheManager:
    """Simple cache manager for preprocessing data storage."""
    
    def __init__(self, dataset_name: str, cache_dir: str = "cache"):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{self.dataset_name}_{cache_key}.pkl")
    
    def load_cache(self, cache_key: str) -> Optional[Any]:
        """Load cached data."""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_key}: {e}")
                return None
        return None
    
    def save_cache(self, data: Any, cache_key: str, metadata: Optional[Dict] = None) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            if metadata:
                print(f"💾 Cached {cache_key} with metadata: {metadata}")
            else:
                print(f"💾 Cached {cache_key}")
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")