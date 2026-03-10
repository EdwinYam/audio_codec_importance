# Importance-Aware Frame Protection: Experiment Report

**Generated from**: `/home/smartedwin/Projects/importance_eval_for_codec/results/results.csv`

**Total experiments**: 288

**Audio files**: 12

**Seeds**: [np.int64(42), np.int64(123)]


## Experiment Configuration

- **Codec**: EnCodec 24kHz causal, 3 kbps
- **Protection budget**: 10% extra frames
- **Network types**: random_loss
- **PLRs**: 1%, 3%, 5%
- **Protection methods**: heuristic, importance_aware, none, random

## Summary Results (averaged over files and seeds)

### Random Loss

**STOI:**

|   target_plr |   heuristic |   importance_aware |   none |   random |
|-------------:|------------:|-------------------:|-------:|---------:|
|         0.01 |      0.8922 |             0.8924 | 0.892  |   0.8914 |
|         0.03 |      0.8662 |             0.8671 | 0.8638 |   0.8744 |
|         0.05 |      0.8504 |             0.8519 | 0.8467 |   0.863  |

**SI-SDR (dB):**

|   target_plr |   heuristic |   importance_aware |   none |   random |
|-------------:|------------:|-------------------:|-------:|---------:|
|         0.01 |        2.87 |               2.9  |   2.86 |     2.88 |
|         0.03 |        2.19 |               2.31 |   2.17 |     2.35 |
|         0.05 |        1.44 |               1.59 |   1.39 |     1.78 |

**Post-Repair Loss Rate:**

|   target_plr |   heuristic |   importance_aware |   none |   random |
|-------------:|------------:|-------------------:|-------:|---------:|
|         0.01 |      0.0081 |             0.0081 | 0.0083 |   0.01   |
|         0.03 |      0.0285 |             0.0278 | 0.03   |   0.0267 |
|         0.05 |      0.0468 |             0.0457 | 0.05   |   0.0417 |

## Importance Method Diagnostics (vs Oracle)

| method    |   spearman_corr |   precision_at_20pct |
|:----------|----------------:|---------------------:|
| A1        |          0.3092 |               0.2514 |
| A2        |          0.1508 |               0.3889 |
| A4        |         -0.2446 |               0.1181 |
| B1        |          0.1191 |               0.2    |
| composite |          0.1454 |               0.2486 |

## Interpretation

- **Importance-aware wins 1/3 conditions** (highest STOI among all methods).
- **Random Loss**: importance-aware STOI advantage over random = -0.0058

## Plots

- [STOI vs PLR](plots/stoi_vs_plr.png)
- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)
- [Post-Repair Loss](plots/post_repair_loss.png)
- [Oracle Diagnostics](plots/oracle_diagnostics.png)