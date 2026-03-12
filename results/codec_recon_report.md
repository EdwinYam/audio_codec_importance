# Codec Reconstruction Quality Report

**Test corpus**: 12 LibriSpeech utterances (16 kHz, 4 s each)  
**Metrics**: PESQ-WB (ITU-T P.862.2, 16 kHz) and PESQ-NB (ITU-T P.862, 8 kHz)  
**Reference**: original 16 kHz PCM

## Summary (averaged over 12 files)

| Codec | Type | Bitrate | PESQ-WB | PESQ-NB |
|:------|:-----|--------:|--------:|--------:|
| EnCodec 3kbps | Neural | 3.00 kbps | 2.050 ± 0.215 | 2.660 ± 0.276 |
| HILCodec 3kbps | Neural | 3.00 kbps | 2.752 ± 0.363 | 3.405 ± 0.176 |
| Lyra v2 3.2kbps | Neural | 3.20 kbps | 2.292 ± 0.194 | 2.942 ± 0.192 |
| AMR-NB 4.75kbps | Traditional | 4.75 kbps | 1.938 ± 0.339 | 3.208 ± 0.134 |
| EVS 5.9kbps VBR | Traditional | 5.90 kbps | 2.802 ± 0.325 | 3.540 ± 0.210 |
| Opus 6kbps | Traditional | 6.00 kbps | 2.034 ± 0.364 | 3.054 ± 0.171 |
| AMR-WB 6.6kbps | Traditional | 6.60 kbps | 2.589 ± 0.263 | 3.346 ± 0.176 |
| EnCodec 6kbps | Neural | 6.00 kbps | 2.692 ± 0.349 | 3.281 ± 0.321 |
| EVS 7.2kbps | Traditional | 7.20 kbps | 3.049 ± 0.318 | 3.668 ± 0.179 |

## Ranking by PESQ-WB

| Rank | Codec | Bitrate | PESQ-WB |
|-----:|:------|--------:|--------:|
| 1 | EVS 7.2kbps | 7.20 kbps | 3.049 |
| 2 | EVS 5.9kbps VBR | 5.90 kbps | 2.802 |
| 3 | HILCodec 3kbps | 3.00 kbps | 2.752 |
| 4 | EnCodec 6kbps | 6.00 kbps | 2.692 |
| 5 | AMR-WB 6.6kbps | 6.60 kbps | 2.589 |
| 6 | Lyra v2 3.2kbps | 3.20 kbps | 2.292 |
| 7 | EnCodec 3kbps | 3.00 kbps | 2.050 |
| 8 | Opus 6kbps | 6.00 kbps | 2.034 |
| 9 | AMR-NB 4.75kbps | 4.75 kbps | 1.938 |

## Ranking by PESQ-NB

| Rank | Codec | Bitrate | PESQ-NB |
|-----:|:------|--------:|--------:|
| 1 | EVS 7.2kbps | 7.20 kbps | 3.668 |
| 2 | EVS 5.9kbps VBR | 5.90 kbps | 3.540 |
| 3 | HILCodec 3kbps | 3.00 kbps | 3.405 |
| 4 | AMR-WB 6.6kbps | 6.60 kbps | 3.346 |
| 5 | EnCodec 6kbps | 6.00 kbps | 3.281 |
| 6 | AMR-NB 4.75kbps | 4.75 kbps | 3.208 |
| 7 | Opus 6kbps | 6.00 kbps | 3.054 |
| 8 | Lyra v2 3.2kbps | 3.20 kbps | 2.942 |
| 9 | EnCodec 3kbps | 3.00 kbps | 2.660 |

## Per-file PESQ-WB

| File | EnCodec 3kbps | HILCodec 3kbps | Lyra v2 3.2kbps | AMR-NB 4.75kbps | EVS 5.9kbps VBR | Opus 6kbps | AMR-WB 6.6kbps | EnCodec 6kbps | EVS 7.2kbps |
|:-----|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1089-134686-0000 | 2.359 | 3.242 | 2.488 | 2.434 | 3.306 | 2.614 | 3.022 | 3.258 | 3.507 |
| 1188-133604-0000 | 2.214 | 2.532 | 2.235 | 2.068 | 2.308 | 2.228 | 2.696 | 3.037 | 2.920 |
| 121-121726-0000 | 1.985 | 3.142 | 2.232 | 2.128 | 3.301 | 2.058 | 2.833 | 2.737 | 3.437 |
| 1221-135766-0000 | 1.879 | 2.563 | 2.182 | 1.266 | 2.430 | 1.277 | 2.525 | 2.495 | 2.463 |
| 1284-1180-0000 | 2.343 | 2.860 | 2.655 | 2.376 | 3.125 | 2.296 | 2.959 | 2.984 | 3.460 |
| 1320-122612-0000 | 2.341 | 2.958 | 2.563 | 1.827 | 3.014 | 1.890 | 2.633 | 3.030 | 3.145 |
| 1580-141083-0000 | 1.895 | 2.770 | 2.144 | 1.716 | 2.739 | 1.775 | 2.351 | 2.405 | 2.983 |
| 1995-1826-0000 | 2.155 | 3.016 | 2.261 | 1.621 | 2.544 | 1.774 | 2.087 | 2.814 | 3.212 |
| 2094-142345-0000 | 1.913 | 2.504 | 2.119 | 1.589 | 2.475 | 1.701 | 2.266 | 2.487 | 2.606 |
| 2300-131720-0000 | 1.930 | 2.812 | 2.292 | 2.101 | 2.750 | 2.453 | 2.497 | 2.623 | 2.945 |
| 237-126133-0000 | 1.923 | 2.830 | 1.940 | 2.284 | 2.976 | 2.390 | 2.537 | 2.526 | 3.125 |
| 260-123286-0000 | 1.670 | 1.799 | 2.397 | 1.849 | 2.656 | 1.956 | 2.662 | 1.908 | 2.787 |
| **Mean** | **2.050** | **2.752** | **2.292** | **1.938** | **2.802** | **2.034** | **2.589** | **2.692** | **3.049** |

## Per-file PESQ-NB

| File | EnCodec 3kbps | HILCodec 3kbps | Lyra v2 3.2kbps | AMR-NB 4.75kbps | EVS 5.9kbps VBR | Opus 6kbps | AMR-WB 6.6kbps | EnCodec 6kbps | EVS 7.2kbps |
|:-----|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1089-134686-0000 | 2.956 | 3.666 | 3.034 | 3.368 | 3.870 | 3.482 | 3.543 | 3.853 | 3.833 |
| 1188-133604-0000 | 2.961 | 3.349 | 2.974 | 3.294 | 3.361 | 3.051 | 3.428 | 3.568 | 3.758 |
| 121-121726-0000 | 2.513 | 3.616 | 3.119 | 3.457 | 3.854 | 3.048 | 3.511 | 3.200 | 3.895 |
| 1221-135766-0000 | 2.409 | 3.160 | 2.731 | 3.176 | 3.204 | 2.755 | 3.342 | 3.010 | 3.382 |
| 1284-1180-0000 | 2.915 | 3.338 | 3.068 | 3.294 | 3.669 | 3.244 | 3.667 | 3.543 | 3.877 |
| 1320-122612-0000 | 2.948 | 3.613 | 3.028 | 3.315 | 3.530 | 3.027 | 3.410 | 3.465 | 3.754 |
| 1580-141083-0000 | 2.252 | 3.139 | 2.576 | 3.222 | 3.374 | 2.917 | 3.325 | 2.768 | 3.600 |
| 1995-1826-0000 | 2.608 | 3.534 | 2.948 | 3.009 | 3.670 | 3.037 | 3.219 | 3.300 | 3.816 |
| 2094-142345-0000 | 2.468 | 3.286 | 3.096 | 3.119 | 3.356 | 2.907 | 2.993 | 2.969 | 3.580 |
| 2300-131720-0000 | 2.466 | 3.453 | 2.952 | 3.045 | 3.325 | 3.056 | 3.149 | 3.154 | 3.419 |
| 237-126133-0000 | 2.355 | 3.213 | 2.594 | 3.052 | 3.726 | 3.055 | 3.234 | 2.904 | 3.697 |
| 260-123286-0000 | 3.066 | 3.498 | 3.180 | 3.150 | 3.537 | 3.072 | 3.330 | 3.634 | 3.410 |
| **Mean** | **2.660** | **3.405** | **2.942** | **3.208** | **3.540** | **3.054** | **3.346** | **3.281** | **3.668** |

## Key Observations

1. **HILCodec 3 kbps** is the best neural codec at ≤3.2 kbps on both metrics (WB=2.752, NB=3.405)
2. **HILCodec 3 kbps vs EVS 5.9 kbps**: HILCodec matches or approaches EVS at roughly half the bitrate (WB: 2.752 vs 2.802, NB: 3.405 vs 3.540)
3. **Lyra v2 3.2 kbps** (WB=2.292, NB=2.942) sits between EnCodec and HILCodec at similar bitrate
4. **EnCodec 3 kbps** (WB=2.050, NB=2.660) lags behind all other codecs except AMR-NB
5. **EVS 7.2 kbps** remains the overall leader (WB=3.049, NB=3.668) but at >2× the bitrate of HILCodec
6. **AMR-NB 4.75 kbps** scores relatively better on NB (3.208) than WB, as expected for a narrowband codec

## Methodology

- 12 LibriSpeech utterances, each truncated to 4 s, evaluated at 16 kHz
- **PESQ-WB** (P.862.2): wideband mode, computed at 16 kHz; range [−0.5, 4.5]
- **PESQ-NB** (P.862): narrowband mode, computed at 8 kHz; range [−0.5, 4.5]
- Neural codecs operating at 24 kHz (EnCodec, HILCodec): resampled 16→24→16 kHz
- Lyra v2 operates natively at 16 kHz (SoundStream encoder + RVQ + LyraGAN decoder)
- EVS 5.9 kbps requires DTX; all other EVS modes use default settings
- AMR-NB internally operates at 8 kHz; output upsampled to 16 kHz for WB scoring

## Codec Details

| Codec | Native SR | Framework | Notes |
|:------|----------:|:----------|:------|
| AMR-NB | 8 kHz | OpenCORE (ffmpeg) | 3GPP TS26.071 narrowband |
| AMR-WB | 16 kHz | VisualOn (ffmpeg) | 3GPP TS26.171 wideband |
| Opus | 48 kHz | libopus (ffmpeg) | IETF RFC 6716 |
| EVS | 8–48 kHz | 3GPP TS26.443 C ref | Enhanced Voice Services |
| Lyra v2 | 16 kHz | TFLite (Google) | SoundStream-based neural codec |
| EnCodec | 24 kHz | HuggingFace (Meta) | RVQ neural codec |
| HILCodec | 24 kHz | ONNX | RVQ neural codec |