"""Abstract base class for codec wrappers."""
from abc import ABC, abstractmethod
import numpy as np


class CodecInterface(ABC):
    """Minimal interface for encode/decode with frame-level token access."""

    @abstractmethod
    def encode(self, pcm: np.ndarray) -> np.ndarray:
        """Encode PCM to token indices. Returns shape (n_frames, n_codebooks)."""

    @abstractmethod
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode token indices back to PCM."""

    @abstractmethod
    def decode_with_mask(
        self, tokens: np.ndarray, mask: np.ndarray, concealment: str = "zero_fill"
    ) -> np.ndarray:
        """Decode tokens with a boolean mask; lost frames are concealed.
        mask: shape (n_frames,), True = frame available, False = lost.
        concealment: 'zero_fill', 'neighbor_copy', or 'linear_interp'."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native sample rate of the codec."""

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Samples per frame."""

    @property
    @abstractmethod
    def n_codebooks(self) -> int:
        """Number of RVQ codebooks."""
