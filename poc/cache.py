"""Artifact caching for tokens, oracle damage, and importance scores."""
import os
import hashlib
import numpy as np


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def _cache_key(*parts) -> str:
    """Generate MD5-based cache key from string parts."""
    raw = ":".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def cache_tokens(codec_name: str, audio_path: str, tokens: np.ndarray):
    """Save encoded tokens to cache."""
    key = _cache_key(codec_name, os.path.basename(audio_path), "tokens")
    path = os.path.join(CACHE_DIR, "tokens", f"{key}.npy")
    _ensure_dir(path)
    np.save(path, tokens)


def load_cached_tokens(codec_name: str, audio_path: str):
    """Load cached tokens, or None if not cached."""
    key = _cache_key(codec_name, os.path.basename(audio_path), "tokens")
    path = os.path.join(CACHE_DIR, "tokens", f"{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def cache_oracle(codec_name: str, audio_path: str, oracle_damage: np.ndarray):
    """Save oracle damage scores to cache."""
    key = _cache_key(codec_name, os.path.basename(audio_path), "oracle")
    path = os.path.join(CACHE_DIR, "oracle", f"{key}.npy")
    _ensure_dir(path)
    np.save(path, oracle_damage)


def load_cached_oracle(codec_name: str, audio_path: str):
    """Load cached oracle damage, or None if not cached."""
    key = _cache_key(codec_name, os.path.basename(audio_path), "oracle")
    path = os.path.join(CACHE_DIR, "oracle", f"{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def cache_importance(codec_name: str, audio_path: str, scores: dict):
    """Save individual importance scores dict to cache."""
    key = _cache_key(codec_name, os.path.basename(audio_path), "importance")
    path = os.path.join(CACHE_DIR, "importance", f"{key}.npz")
    _ensure_dir(path)
    np.savez(path, **scores)


def load_cached_importance(codec_name: str, audio_path: str):
    """Load cached importance scores dict, or None if not cached."""
    key = _cache_key(codec_name, os.path.basename(audio_path), "importance")
    path = os.path.join(CACHE_DIR, "importance", f"{key}.npz")
    if os.path.exists(path):
        data = np.load(path)
        return {k: data[k] for k in data.files}
    return None
