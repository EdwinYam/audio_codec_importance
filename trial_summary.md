# Trial Summary — Importance-Aware Frame Protection PoC

> This file consolidates all experiment context, results, findings, and next-step recommendations
> so a new Claude agent/session can pick up where we left off.

---

## 1. Project Goal

**Research question:** In a fixed redundancy budget, does importance-aware frame protection
improve IMS/VoIP audio quality under packet loss compared to random or heuristic protection?

**Codec scope:** Neural audio codecs (EnCodec, HILCodec) at 3 kbps, 24 kHz, 320 samples/frame (13.3 ms).

---

## 2. Architecture & Key Files

```
poc/
  codec/
    base.py                  # CodecInterface ABC
    encodec_wrapper.py       # EnCodec 24kHz (HuggingFace transformers)
    hilcodec_wrapper.py      # HILCodec speech (ONNX, batch decode)
    onnx/                    # HILCodec ONNX models (gitignored, must be re-downloaded)
  importance/
    a1_vad_onset.py          # VAD + talkspurt onset boost
    a2_spectral_flux.py      # Spectral flux / transient detection
    a4_evs_criticality.py    # EVS-style frame criticality
    b1_token_novelty.py      # Token change-rate / novelty
    composite.py             # Equal-weighted composite (A1+A2+B1+A4) / 4
  network/
    random_loss.py           # Bernoulli packet loss
    burst_loss.py            # Gilbert-Elliott burst loss
    jitter_discard.py        # Jitter-based late-arrival discard
  protection/
    no_protection.py         # Baseline: no protection
    random_protection.py     # Baseline: random frame selection
    heuristic_protection.py  # Baseline: top-K by A1 score
    importance_aware.py      # Proposed: top-K by composite importance
  eval/
    audio_quality.py         # PESQ, STOI, ESTOI, SI-SDR
    oracle.py                # Leave-one-out oracle importance (ground truth)
    diagnostics.py           # Spearman correlation, precision@K, concealment stats
  pipeline.py               # Single experiment: encode → impair → protect → decode → eval
  run_experiment.py          # Full matrix runner (multi-codec)
  report.py                  # Report + plot generator (multi-codec)
results/
  results.csv               # 1920 experiment rows
  oracle_diagnostics.csv     # Per-file per-codec method diagnostics
  report.md                  # Generated report
  plots/                     # PESQ, STOI, SI-SDR, post-repair loss, oracle diagnostics
```

### Pipeline flow (per experiment)
```
clean PCM → codec.encode() → tokens (n_frames, n_codebooks)
  → compute importance scores (composite of A1+A2+B1+A4)
  → select protected frames (by method)
  → apply network loss (Bernoulli)
  → apply protection (repair lost protected frames)
  → codec.decode_with_mask() → degraded PCM
  → compute metrics (PESQ, STOI, ESTOI, SI-SDR)
```

### Key optimization
- `pipeline.py` accepts pre-encoded `tokens` and `importance_scores` to avoid redundant
  encoding/scoring across experiments sharing the same file+codec.
- HILCodec uses batch dequantization + single-pass decoder (~1s for 300 frames)
  instead of frame-by-frame ONNX (~2.5s).

---

## 3. Experiment Configuration (Latest Run)

| Parameter | Value |
|-----------|-------|
| Codecs | EnCodec 3kbps, HILCodec 3kbps |
| Audio | 12 LibriSpeech files, 4s each, 24kHz mono |
| PLRs | 0%, 1%, 3%, 5%, 10%, 20%, 30%, 40% |
| Network | random_loss (Bernoulli) |
| Protection methods | none, random, heuristic, importance_aware, importance_selective |
| Protection budget | 10% of frames |
| Seeds | 42, 123 (averaged) |
| Total experiments | 1920 |

### Protection Methods Explained

| Method | Description |
|--------|-------------|
| `none` | No frames protected |
| `random` | Random K frames protected (K = 10% budget) |
| `heuristic` | Top-K by A1 (VAD onset) score |
| `importance_aware` | Top-K by composite importance; loss hits all frames, then protected frames repaired |
| `importance_selective` | Loss only hits non-important frames; important frames (top-K) are immune; PLR adjusted so expected total lost frames matches uniform PLR |

---

## 4. Results Summary

### Codec Intrinsic Quality (PLR=0, no loss)

| Codec | PESQ | STOI | SI-SDR |
|-------|------|------|--------|
| EnCodec 3kbps | **2.05** | 0.901 | 3.25 dB |
| HILCodec 3kbps | **2.75** | 0.928 | 3.24 dB |

**Key insight:** PESQ was "low" (user expected >3) because EnCodec at 3kbps intrinsically
degrades to PESQ=2.05 even without loss. HILCodec at same bitrate achieves 2.75 — significantly better.

### EnCodec 3kbps — PESQ under loss

| PLR | none | random | heuristic | importance_aware | importance_selective |
|-----|------|--------|-----------|-----------------|---------------------|
| 0%  | 2.051 | 2.051 | 2.051 | 2.051 | 2.051 |
| 1%  | 1.702 | 1.679 | 1.706 | 1.706 | **1.773** |
| 5%  | 1.242 | 1.290 | 1.255 | 1.261 | 1.281 |
| 10% | 1.141 | 1.167 | 1.148 | 1.153 | 1.158 |
| 20% | 1.067 | 1.077 | 1.072 | 1.073 | 1.070 |
| 40% | 1.037 | 1.041 | 1.041 | 1.042 | 1.037 |

### HILCodec 3kbps — PESQ under loss

| PLR | none | random | heuristic | importance_aware | importance_selective |
|-----|------|--------|-----------|-----------------|---------------------|
| 0%  | 2.751 | 2.751 | 2.751 | 2.751 | 2.751 |
| 1%  | 2.390 | 2.425 | 2.391 | 2.391 | **2.542** |
| 5%  | 1.690 | 1.857 | 1.715 | 1.738 | 1.821 |
| 10% | 1.437 | 1.511 | 1.465 | 1.485 | 1.484 |
| 20% | 1.198 | 1.232 | 1.216 | 1.219 | 1.233 |
| 40% | 1.099 | 1.115 | 1.113 | 1.112 | 1.106 |

### STOI Best-Method Winners (highest STOI per condition)

| Codec | Random wins | Importance Aware wins | Importance Selective wins | Heuristic wins |
|-------|------------|----------------------|--------------------------|---------------|
| EnCodec | 3/8 | 3/8 | 1/8 | 1/8 |
| HILCodec | 4/8 | 2/8 | 1/8 | 1/8 |

### Oracle Diagnostics (importance method vs. ground truth)

| Method | Spearman (EnCodec) | Spearman (HILCodec) | P@20% (EnCodec) | P@20% (HILCodec) |
|--------|-------------------|--------------------|-----------------|--------------------|
| A1 (VAD onset) | 0.309 | 0.314 | 0.251 | 0.264 |
| A2 (spectral flux) | 0.151 | 0.171 | **0.389** | **0.374** |
| A4 (EVS criticality) | **-0.245** | **-0.255** | 0.118 | 0.111 |
| B1 (token novelty) | 0.119 | 0.016 | 0.200 | 0.199 |
| composite | 0.145 | 0.118 | 0.249 | 0.247 |

---

## 5. Key Findings & Diagnosis

### What works
1. **HILCodec >> EnCodec** at same 3kbps — PESQ 2.75 vs 2.05 baseline, and maintains
   advantage under all loss conditions.
2. **importance_selective shines at low PLR (1%)** — best PESQ for both codecs
   (EnCodec: 1.773 vs 1.706; HILCodec: 2.542 vs 2.391).
3. **importance_aware consistently beats "none"** across almost all conditions.

### What doesn't work well
1. **A4 (EVS criticality) has negative oracle correlation** (-0.25) — it's actively
   anti-correlated with actual frame damage. This drags down the composite score.
2. **B1 (token novelty) is near-zero for HILCodec** (Spearman=0.016) — HILCodec tokens
   may not change the same way as EnCodec tokens.
3. **Random protection often wins at medium-high PLR (3-40%)** — because:
   - The importance scoring is inaccurate (composite Spearman ~0.12-0.15)
   - With inaccurate importance, protecting "wrong" important frames wastes budget
   - Random protection has no bias, so it doesn't systematically miss
4. **PLC is naive (zero-fill)** — real PLC would mask differences between methods more.

### Root cause of importance method weakness
The **composite equal weighting (A1+A2+B1+A4)/4** is suboptimal:
- A4 should be **removed or weight reduced** (negative correlation hurts)
- A2 has the best precision@20% (0.37-0.39) but only moderate Spearman
- A1 has the best Spearman (0.31) but moderate precision
- The optimal weighting is likely **heavy A1 + A2, minimal B1, zero A4**

---

## 6. What Was NOT Done (from eval_method.md recommendations)

These were planned but not yet implemented:

- [ ] **Burst loss** (Gilbert-Elliott) — code exists but not in experiment matrix
- [ ] **Jitter discard** — code exists but not in experiment matrix
- [ ] **Mixed MTSI profiles** — not implemented
- [ ] **Damage-by-region analysis** (onset vs. stable vs. silence)
- [ ] **Better PLC** (frame repeat, interpolation instead of zero-fill)
- [ ] **Higher bitrate codecs** (6kbps EnCodec, 6kbps HILCodec)
- [ ] **Importance weight tuning** (A4 removal, optimized A1/A2 weights)
- [ ] **Partial protection** (protect only critical codebook layers, not full frame)
- [ ] **Subjective listening tests**
- [ ] **A3 (phonetic boundary) and B3 (cross-codebook disagreement) methods**

---

## 7. Recommended Next Steps (priority order)

### High priority — likely to produce significant improvement
1. **Fix importance scoring**: Remove A4 or set its weight to 0. Re-weight to favor A1+A2.
   Re-run experiments and check if importance_aware starts beating random.
2. **Add burst loss and jitter discard** to experiment matrix (code already exists in
   `poc/network/`). This is where importance-aware should show larger advantage.
3. **Better PLC**: Replace zero-fill with frame-repeat or simple interpolation.
   Zero-fill is unrealistically harsh.

### Medium priority
4. **Higher bitrate**: Test 6kbps (8 codebooks) for both codecs to get PESQ > 3.
5. **Partial protection**: Protect only first 2 codebooks (low-frequency content)
   instead of entire frame — more efficient use of budget.
6. **Increase protection budget**: Test 15%, 20% to see if importance methods
   benefit more from larger budgets.

### Lower priority
7. Add A3, B3 importance methods.
8. Damage-by-region analysis.
9. Subjective evaluation.

---

## 8. Technical Notes for Resuming

### HILCodec ONNX models
The ONNX files in `poc/codec/onnx/` are gitignored (too large for git).
To restore them:
```bash
git clone https://github.com/aask1357/hilcodec.git /tmp/hilcodec
cp /tmp/hilcodec/onnx/hil_speech_*.onnx /tmp/hilcodec/onnx/hil_speech_cache_*.npz poc/codec/onnx/
pip install onnxruntime
```

### Running experiments
```bash
python3 -m poc.run_experiment --data-dir poc/data/audio --results-dir results
```
Runtime: ~60 min (dominated by HILCodec ONNX inference; EnCodec is fast).

### Generating report
```bash
python3 -m poc.report
```

### LyraV2 status
LyraV2 was investigated but **not feasible** for this project:
- C++ only (Bazel build), no Python bindings, no pip package
- Public API returns opaque bytes, not RVQ token indices
- Project dormant since Dec 2022
- Replaced by HILCodec which has similar profile but Python/ONNX support

---

## 9. Repository
- GitHub: `EdwinYam/audio_codec_importance`
- Branch: `master`
- Latest commit: `3f18e70` — "Add HILCodec codec, importance_selective method, extend PLR range"
