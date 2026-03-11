"""HILCodec ONNX wrapper implementing CodecInterface."""
import os
import numpy as np
import onnxruntime

from .base import CodecInterface


class HILCodecWrapper(CodecInterface):
    """Wraps HILCodec speech model (ONNX) for frame-level encode/decode.

    Uses the hil_speech ONNX models: encoder, per-codebook VQ/deQ, decoder.
    At 3 kbps we use 4 codebooks (each ~750 bps).
    """

    def __init__(self, n_quantizers: int = 4, onnx_dir: str = None):
        if onnx_dir is None:
            onnx_dir = os.path.join(os.path.dirname(__file__), "onnx")
        self._onnx_dir = onnx_dir
        self._n_quantizers = n_quantizers
        self._hop_size = 320
        self._sr = 24000
        self._name = "hil_speech"

        so = onnxruntime.SessionOptions()
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3  # suppress warnings

        # Load encoder
        self._enc = onnxruntime.InferenceSession(
            os.path.join(onnx_dir, f"{self._name}_enc.onnx"), sess_options=so
        )
        # Load VQ models
        self._vq = {}
        for i in range(n_quantizers):
            self._vq[i] = onnxruntime.InferenceSession(
                os.path.join(onnx_dir, f"{self._name}_vq{i}.onnx"), sess_options=so
            )
        # Load dequantizer models
        self._deq = {}
        for i in range(n_quantizers):
            self._deq[i] = onnxruntime.InferenceSession(
                os.path.join(onnx_dir, f"{self._name}_deq{i}.onnx"), sess_options=so
            )
        # Load decoder
        self._dec = onnxruntime.InferenceSession(
            os.path.join(onnx_dir, f"{self._name}_dec.onnx"), sess_options=so
        )
        # Load cache templates
        self._enc_cache_template = dict(
            np.load(os.path.join(onnx_dir, f"{self._name}_cache_enc.npz"))
        )
        self._dec_cache_template = dict(
            np.load(os.path.join(onnx_dir, f"{self._name}_cache_dec.npz"))
        )

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def frame_size(self) -> int:
        return self._hop_size

    @property
    def n_codebooks(self) -> int:
        return self._n_quantizers

    def encode(self, pcm: np.ndarray) -> np.ndarray:
        """Encode PCM float32 [-1,1] to tokens (n_frames, n_codebooks)."""
        pcm = pcm.astype(np.float32)
        length = len(pcm) // self._hop_size * self._hop_size
        pcm = pcm[:length]
        wav = pcm[np.newaxis, np.newaxis, :]  # (1, 1, T)

        enc_input = {k: v.copy() for k, v in self._enc_cache_template.items()}
        indices_list = [[] for _ in range(self._n_quantizers)]

        for i in range(0, length, self._hop_size):
            enc_input["wav_in"] = wav[:, :, i:i + self._hop_size]
            out = self._enc.run(None, enc_input)
            x = out[0]
            for j in range(len(out[1:])):
                enc_input[f"e_in{j}"] = out[j + 1]

            # RVQ
            residual = x
            for j in range(self._n_quantizers):
                quantized, index = self._vq[j].run(None, {"x": residual})
                indices_list[j].append(index)
                residual = residual - quantized

        for j in range(self._n_quantizers):
            indices_list[j] = np.concatenate(indices_list[j], axis=1)

        indices = np.stack(indices_list)  # [n_q, 1, T]
        return indices[:, 0, :].T.astype(np.int64)  # (n_frames, n_codebooks)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode tokens (n_frames, n_codebooks) to PCM.

        Uses batch dequantization + single-pass decoder for speed.
        """
        indices = tokens.T[:, np.newaxis, :].astype(np.int64)  # [n_q, 1, T]

        # Batch dequantize all frames at once
        quantized = self._deq[0].run(None, {"idx": indices[0]})[0]
        for j in range(1, self._n_quantizers):
            quantized += self._deq[j].run(None, {"idx": indices[j]})[0]

        # Decode all frames in one pass
        dec_input = {k: v.copy() for k, v in self._dec_cache_template.items()}
        dec_input["q"] = quantized
        out = self._dec.run(None, dec_input)
        return np.squeeze(out[0])

    def decode_with_mask(
        self, tokens: np.ndarray, mask: np.ndarray, concealment: str = "zero_fill"
    ) -> np.ndarray:
        """Decode with frame-level mask. Lost frames are concealed."""
        from .concealment import apply_concealment
        tokens_masked = apply_concealment(tokens, mask, method=concealment)
        return self.decode(tokens_masked)
