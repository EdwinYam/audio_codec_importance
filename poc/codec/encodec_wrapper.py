"""EnCodec 24kHz causal wrapper implementing CodecInterface."""
import numpy as np
import torch
from transformers import EncodecModel, AutoProcessor

from .base import CodecInterface


class EnCodecWrapper(CodecInterface):
    """Wraps facebook/encodec_24khz for frame-level encode/decode."""

    def __init__(self, bandwidth: float = 3.0, device: str = "cpu"):
        self.device = device
        self.bandwidth = bandwidth
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.model.eval()
        # EnCodec 24k: 320 samples per frame at 24kHz = 13.3ms
        self._frame_size = 320
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def n_codebooks(self) -> int:
        # Depends on bandwidth: 3kbps -> 4 codebooks, 6kbps -> 8, etc.
        bw_to_cb = {1.5: 2, 3.0: 4, 6.0: 8, 12.0: 16, 24.0: 32}
        return bw_to_cb.get(self.bandwidth, 4)

    def encode(self, pcm: np.ndarray) -> np.ndarray:
        """Encode PCM float32 [-1,1] to tokens (n_frames, n_codebooks)."""
        pcm = pcm.astype(np.float32)
        if pcm.ndim == 1:
            pcm = pcm[np.newaxis, :]  # (1, T)

        inputs = self.processor(
            raw_audio=pcm[0],
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.encode(
                input_values, padding_mask, bandwidth=self.bandwidth
            )
        # audio_codes: (batch=1, n_chunks=1, n_codebooks, n_frames)
        codes = encoder_outputs.audio_codes[0]  # (n_chunks, n_codebooks, n_frames)
        # Flatten across chunks: (n_codebooks, total_frames)
        n_codebooks = codes.shape[1]
        codes = codes.reshape(n_codebooks, -1)
        codes_np = codes.cpu().numpy()
        # Store scales for decoding
        self._last_scales = encoder_outputs.audio_scales
        self._last_padding_mask = padding_mask
        return codes_np.T  # (n_frames, n_codebooks)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode tokens (n_frames, n_codebooks) to PCM."""
        # tokens: (n_frames, n_codebooks) -> need (1, 1, n_codebooks, n_frames)
        codes = torch.from_numpy(tokens.T.astype(np.int64)).unsqueeze(0).unsqueeze(0)
        codes = codes.to(self.device)
        with torch.no_grad():
            audio = self.model.decode(codes, self._last_scales, self._last_padding_mask)
        return audio.audio_values[0, 0].cpu().numpy()

    def decode_with_mask(
        self, tokens: np.ndarray, mask: np.ndarray, concealment: str = "zero_fill"
    ) -> np.ndarray:
        """Decode with frame-level mask. Lost frames are concealed."""
        from .concealment import apply_concealment
        tokens_masked = apply_concealment(tokens, mask, method=concealment)
        return self.decode(tokens_masked)

    def encode_decode_with_loss(
        self, pcm: np.ndarray, frame_mask: np.ndarray
    ) -> np.ndarray:
        """Convenience: encode, apply frame mask, decode."""
        tokens = self.encode(pcm)
        n_frames = min(len(frame_mask), tokens.shape[0])
        mask = frame_mask[:n_frames]
        tokens = tokens[:n_frames]
        return self.decode_with_mask(tokens, mask)
