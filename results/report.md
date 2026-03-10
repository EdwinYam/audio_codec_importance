# Importance-Aware Frame Protection: Experiment Report

**Generated from**: `/home/smartedwin/Projects/importance_eval_for_codec/results/results.csv`

**Total experiments**: 480

**Audio files**: 12

**Seeds**: [np.int64(42), np.int64(123)]


## Experiment Configuration

- **Codec**: EnCodec 24kHz causal, 3 kbps
- **Protection budget**: 10% extra frames
- **Network types**: random_loss
- **PLRs**: 0%, 1%, 3%, 5%
- **Protection methods**: heuristic, importance_aware, importance_selective, none, random

## Summary Results (averaged over files and seeds)

### Random Loss

**PESQ (MOS-LQO):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |       2.051 |              2.051 |                  2.051 |  2.051 |    2.051 |
|         0.01 |       1.706 |              1.706 |                  1.773 |  1.702 |    1.679 |
|         0.03 |       1.383 |              1.387 |                  1.364 |  1.371 |    1.392 |
|         0.05 |       1.255 |              1.261 |                  1.281 |  1.242 |    1.29  |

**STOI:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0.9013 |             0.9013 |                 0.9013 | 0.9013 |   0.9013 |
|         0.01 |      0.8922 |             0.8924 |                 0.8943 | 0.892  |   0.8914 |
|         0.03 |      0.8662 |             0.8671 |                 0.8634 | 0.8638 |   0.8744 |
|         0.05 |      0.8504 |             0.8519 |                 0.8533 | 0.8467 |   0.863  |

**SI-SDR (dB):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |        3.25 |               3.25 |                   3.25 |   3.25 |     3.25 |
|         0.01 |        2.87 |               2.9  |                   3.05 |   2.86 |     2.88 |
|         0.03 |        2.19 |               2.31 |                   2.01 |   2.17 |     2.35 |
|         0.05 |        1.44 |               1.59 |                   1.65 |   1.39 |     1.78 |

**Post-Repair Loss Rate:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0      |             0      |                 0      | 0      |   0      |
|         0.01 |      0.0081 |             0.0081 |                 0.0067 | 0.0083 |   0.01   |
|         0.03 |      0.0285 |             0.0278 |                 0.0333 | 0.03   |   0.0267 |
|         0.05 |      0.0468 |             0.0457 |                 0.0467 | 0.05   |   0.0417 |

## Importance Method Diagnostics (vs Oracle)

| method    |   spearman_corr |   precision_at_20pct |
|:----------|----------------:|---------------------:|
| A1        |          0.3092 |               0.2514 |
| A2        |          0.1508 |               0.3889 |
| A4        |         -0.2446 |               0.1181 |
| B1        |          0.1191 |               0.2    |
| composite |          0.1454 |               0.2486 |

## Interpretation

- **Importance-aware wins 0/4 conditions** (highest STOI among all methods).
- **Random Loss**: importance-aware STOI advantage over random = -0.0044

## Plots

- [PESQ vs PLR](plots/pesq_vs_plr.png)
- [STOI vs PLR](plots/stoi_vs_plr.png)
- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)
- [Post-Repair Loss](plots/post_repair_loss.png)
- [Oracle Diagnostics](plots/oracle_diagnostics.png)