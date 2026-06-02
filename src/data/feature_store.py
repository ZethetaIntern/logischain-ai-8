"""Feature store: versioned feature caching and retrieval for LogisChain AI.

Provides a file-backed registry with:
- Parquet-based storage (columnar, ~10× smaller than CSV)
- Semantic versioning and rich metadata tracking
- In-process LRU cache to skip repeated disk reads
- Cache invalidation based on file modification time
- Dependency graph for tracking which raw files a feature set was built from
"""
import functools
import hashlib
import json
import logging
import os
import pickle
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class _LRUCache:
    """Simple in-process LRU cache for DataFrames (by feature key)."""

    def __init__(self, max_size: int = 8):
        self._cache: OrderedDict[str, Tuple[pd.DataFrame, float]] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str, mtime: float) -> Optional[pd.DataFrame]:
        if key in self._cache:
            df, cached_mtime = self._cache[key]
            if mtime <= cached_mtime:          # file unchanged → cache valid
                self._cache.move_to_end(key)
                self.hits += 1
                return df
            else:
                del self._cache[key]           # file updated → invalidate
        self.misses += 1
        return None

    def put(self, key: str, df: pd.DataFrame, mtime: float):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (df, mtime)
        if len(self._cache) > self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"LRU evicted: {evicted_key}")

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self):
        self._cache.clear()

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": self.size}


class FeatureStore:
    """File-based versioned feature store with LRU caching and dependency tracking.

    Usage
    ─────
    store = FeatureStore()

    # Save
    store.save(my_df, name="supplier_features", version="v2",
               metadata={"source": "suppliers_500.csv"},
               dependencies=["data/raw/suppliers_500.csv"])

    # Load (uses in-memory LRU cache on repeated calls)
    df = store.load("supplier_features", version="v2")

    # Check freshness
    if not store.is_fresh("supplier_features", deps=["data/raw/suppliers_500.csv"]):
        df = recompute_features()
        store.save(df, "supplier_features")

    # Browse registry
    print(store.list_features())
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        cache_size: int = 8,
    ):
        self.store_path = Path(
            store_path or os.getenv("FEATURES_DATA_PATH", "./data/features")
        )
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.store_path / "_registry.json"
        self._registry: Dict[str, dict] = self._load_registry()
        self._cache = _LRUCache(max_size=cache_size)

    # ── Registry I/O ───────────────────────────────────────────────────────

    def _load_registry(self) -> Dict[str, dict]:
        if self._registry_path.exists():
            try:
                with open(self._registry_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Registry corrupted ({e}), starting fresh.")
        return {}

    def _save_registry(self):
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    # ── Key & path utilities ───────────────────────────────────────────────

    @staticmethod
    def _key(name: str, version: str) -> str:
        return f"{name}:{version}"

    @staticmethod
    def _fid(name: str, version: str) -> str:
        return hashlib.md5(f"{name}:{version}".encode()).hexdigest()[:8]

    def _parquet_path(self, name: str, version: str) -> Path:
        fid = self._fid(name, version)
        return self.store_path / f"{name}_{version}_{fid}.parquet"

    # ── Core API ──────────────────────────────────────────────────────────

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        version: str = "latest",
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        overwrite: bool = True,
    ) -> str:
        """Persist a DataFrame to Parquet and register it.

        Parameters
        ──────────
        name          : feature set name (e.g. "supplier_risk_features")
        version       : semantic version string (e.g. "v1", "v2", "latest")
        metadata      : arbitrary JSON-serializable dict stored with the feature set
        dependencies  : list of raw file paths this feature set was built from
        overwrite     : if False and the key already exists, raise ValueError

        Returns the path to the saved Parquet file.
        """
        key = self._key(name, version)
        if not overwrite and key in self._registry:
            raise ValueError(
                f"Feature set '{key}' already exists. Pass overwrite=True to replace."
            )

        filepath = self._parquet_path(name, version)
        df.to_parquet(filepath, index=False, compression="snappy")
        mtime = filepath.stat().st_mtime

        dep_mtimes = {}
        for dep in (dependencies or []):
            p = Path(dep)
            dep_mtimes[dep] = p.stat().st_mtime if p.exists() else 0.0

        entry = {
            "name":          name,
            "version":       version,
            "fid":           self._fid(name, version),
            "filepath":      str(filepath),
            "shape":         list(df.shape),
            "columns":       df.columns.tolist(),
            "dtypes":        {c: str(df[c].dtype) for c in df.columns},
            "created_at":    datetime.utcnow().isoformat(),
            "size_bytes":    filepath.stat().st_size,
            "metadata":      metadata or {},
            "dependencies":  dependencies or [],
            "dep_mtimes":    dep_mtimes,
        }
        self._registry[key] = entry
        self._save_registry()

        # Populate LRU cache
        self._cache.put(key, df, mtime)

        size_mb = entry["size_bytes"] / 1_048_576
        logger.info(
            f"Saved '{key}' → {filepath.name}  "
            f"shape={df.shape}  size={size_mb:.2f}MB"
        )
        return str(filepath)

    def load(
        self, name: str, version: str = "latest", use_cache: bool = True
    ) -> pd.DataFrame:
        """Load a feature set from Parquet (or LRU cache if unchanged).

        Parameters
        ──────────
        use_cache : if False, bypass LRU and always read from disk
        """
        key = self._key(name, version)
        if key not in self._registry:
            raise KeyError(
                f"Feature set '{key}' not in registry. "
                f"Available: {list(self._registry.keys())}"
            )
        entry = self._registry[key]
        filepath = Path(entry["filepath"])
        if not filepath.exists():
            raise FileNotFoundError(
                f"Parquet file missing for '{key}': {filepath}\n"
                "Re-run feature engineering to regenerate."
            )

        mtime = filepath.stat().st_mtime

        if use_cache:
            cached = self._cache.get(key, mtime)
            if cached is not None:
                logger.debug(f"Cache hit for '{key}'.")
                return cached

        df = pd.read_parquet(filepath)
        if use_cache:
            self._cache.put(key, df, mtime)
        logger.info(f"Loaded '{key}' — shape={df.shape}")
        return df

    def is_fresh(
        self, name: str, version: str = "latest", deps: Optional[List[str]] = None
    ) -> bool:
        """Check whether the stored feature set is up-to-date relative to its source files.

        Returns True if the feature set exists and all dependency files are
        unchanged since the feature set was saved.
        """
        key = self._key(name, version)
        if key not in self._registry:
            return False
        entry = self._registry[key]
        filepath = Path(entry["filepath"])
        if not filepath.exists():
            return False

        feature_mtime = filepath.stat().st_mtime
        check_deps = deps or entry.get("dependencies", [])
        for dep in check_deps:
            dep_path = Path(dep)
            if dep_path.exists() and dep_path.stat().st_mtime > feature_mtime:
                logger.info(f"Dependency '{dep}' is newer than '{key}' — stale.")
                return False
        return True

    # ── Registry management ────────────────────────────────────────────────

    def exists(self, name: str, version: str = "latest") -> bool:
        """Return True if the feature set key is registered."""
        return self._key(name, version) in self._registry

    def delete(self, name: str, version: str = "latest", delete_file: bool = True):
        """Remove a feature set from the registry (and optionally delete its Parquet file)."""
        key = self._key(name, version)
        if key not in self._registry:
            logger.warning(f"'{key}' not in registry — nothing to delete.")
            return
        if delete_file:
            path = Path(self._registry[key]["filepath"])
            if path.exists():
                path.unlink()
                logger.info(f"Deleted Parquet file: {path.name}")
        del self._registry[key]
        self._save_registry()
        self._cache._cache.pop(key, None)
        logger.info(f"Removed '{key}' from registry.")

    def list_features(self) -> pd.DataFrame:
        """Return a summary DataFrame of all registered feature sets."""
        if not self._registry:
            return pd.DataFrame()
        rows = []
        for key, e in self._registry.items():
            fp = Path(e["filepath"])
            rows.append(
                {
                    "key":         key,
                    "name":        e["name"],
                    "version":     e["version"],
                    "rows":        e["shape"][0],
                    "cols":        e["shape"][1],
                    "size_mb":     round(e.get("size_bytes", 0) / 1_048_576, 3),
                    "created_at":  e["created_at"],
                    "file_exists": fp.exists(),
                    "n_deps":      len(e.get("dependencies", [])),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("created_at", ascending=False)
            .reset_index(drop=True)
        )

    def get_metadata(self, name: str, version: str = "latest") -> dict:
        """Return the full metadata entry for a registered feature set."""
        key = self._key(name, version)
        if key not in self._registry:
            raise KeyError(f"'{key}' not found.")
        return self._registry[key].copy()

    def latest_version(self, name: str) -> Optional[str]:
        """Return the most recently created version of a named feature set."""
        candidates = [
            (k, e["created_at"])
            for k, e in self._registry.items()
            if e["name"] == name
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0].split(":")[1]

    def rename(self, name: str, version: str, new_version: str):
        """Rename a version in the registry without moving the file."""
        old_key = self._key(name, version)
        new_key = self._key(name, new_version)
        if old_key not in self._registry:
            raise KeyError(f"'{old_key}' not found.")
        if new_key in self._registry:
            raise ValueError(f"'{new_key}' already exists.")
        entry = self._registry.pop(old_key)
        entry["version"] = new_version
        self._registry[new_key] = entry
        self._save_registry()
        logger.info(f"Renamed '{old_key}' → '{new_key}'")

    def promote_to_latest(self, name: str, version: str):
        """Promote a specific version to 'latest' (copies registry entry)."""
        src_key = self._key(name, version)
        if src_key not in self._registry:
            raise KeyError(f"'{src_key}' not found.")
        latest_key = self._key(name, "latest")
        self._registry[latest_key] = dict(self._registry[src_key])
        self._registry[latest_key]["version"] = "latest"
        self._save_registry()
        logger.info(f"Promoted '{version}' to latest for '{name}'.")

    # ── Artifact store (non-DataFrame objects) ─────────────────────────────

    def save_artifact(self, obj: Any, name: str) -> str:
        """Pickle any Python object (e.g., fitted encoders, scalers) to the store."""
        filepath = self.store_path / f"{name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved artifact '{name}' ({filepath.stat().st_size} bytes).")
        return str(filepath)

    def load_artifact(self, name: str) -> Any:
        """Load a pickled artifact by name."""
        filepath = self.store_path / f"{name}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact '{name}' not found at {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # ── Cache management ──────────────────────────────────────────────────

    def clear_cache(self):
        """Evict all entries from the in-process LRU cache."""
        self._cache.clear()
        logger.info("LRU cache cleared.")

    def cache_stats(self) -> Dict[str, int]:
        """Return LRU cache statistics (hits, misses, size)."""
        return self._cache.stats()

    def warm_cache(self, names: Optional[List[str]] = None):
        """Pre-load registered feature sets into the LRU cache.

        Parameters
        ──────────
        names : list of feature names to warm. If None, warms all registered sets
                up to the cache capacity.
        """
        keys = (
            [self._key(n, self.latest_version(n) or "latest") for n in names]
            if names
            else list(self._registry.keys())
        )
        loaded = 0
        for key in keys[: self._cache.max_size]:
            name, version = key.split(":", 1)
            try:
                self.load(name, version, use_cache=True)
                loaded += 1
            except Exception as e:
                logger.warning(f"Could not warm '{key}': {e}")
        logger.info(f"Cache warmed: {loaded} feature sets loaded.")

    def purge_orphaned_files(self) -> List[str]:
        """Delete Parquet files in the store directory that have no registry entry.

        Returns list of deleted file paths.
        """
        registered_files = {Path(e["filepath"]) for e in self._registry.values()}
        deleted = []
        for f in self.store_path.glob("*.parquet"):
            if f not in registered_files:
                f.unlink()
                deleted.append(str(f))
                logger.info(f"Purged orphaned file: {f.name}")
        return deleted

    # ── Storage summary ───────────────────────────────────────────────────

    def storage_summary(self) -> Dict[str, Any]:
        """Return a summary of total disk usage for all registered feature sets."""
        total_bytes = 0
        for e in self._registry.values():
            p = Path(e["filepath"])
            if p.exists():
                total_bytes += p.stat().st_size
        return {
            "n_feature_sets":    len(self._registry),
            "total_size_mb":     round(total_bytes / 1_048_576, 3),
            "store_path":        str(self.store_path.resolve()),
            "cache_stats":       self.cache_stats(),
        }
