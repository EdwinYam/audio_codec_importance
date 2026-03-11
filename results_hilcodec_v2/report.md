# Importance-Aware Frame Protection: Experiment Report

**Generated from**: `results_hilcodec_v2/results.csv`

**Total experiments**: 1920

**Audio files**: 12

**Codecs**: HILCodec_3kbps

**Seeds**: [np.int64(42), np.int64(123)]

**Concealments**: neighbor_copy, zero_fill


## Experiment Configuration

- **Codecs**: HILCodec_3kbps
- **Protection budget**: 10% extra frames
- **Network types**: random_loss
- **PLRs**: 0%, 1%, 3%, 5%, 10%, 20%, 30%, 40%
- **Protection methods**: heuristic, importance_aware, importance_selective, none, random
- **Concealment methods**: neighbor_copy, zero_fill

## HILCodec_3kbps Results (averaged over files, seeds, concealments)

### Random Loss

**PESQ WB (MOS-LQO):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |       2.751 |              2.751 |                  2.751 |  2.751 |    2.751 |
|         0.01 |       2.473 |              2.474 |                  2.576 |  2.472 |    2.478 |
|         0.03 |       2.115 |              2.14  |                  2.096 |  2.084 |    2.135 |
|         0.05 |       1.889 |              1.927 |                  1.973 |  1.852 |    1.981 |
|         0.1  |       1.604 |              1.642 |                  1.633 |  1.571 |    1.63  |
|         0.2  |       1.296 |              1.31  |                  1.308 |  1.266 |    1.312 |
|         0.3  |       1.199 |              1.205 |                  1.19  |  1.171 |    1.21  |
|         0.4  |       1.155 |              1.16  |                  1.145 |  1.133 |    1.159 |

**PESQ NB (MOS-LQO):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |       3.406 |              3.406 |                  3.406 |  3.406 |    3.406 |
|         0.01 |       3.178 |              3.181 |                  3.259 |  3.176 |    3.198 |
|         0.03 |       2.819 |              2.864 |                  2.782 |  2.792 |    2.876 |
|         0.05 |       2.577 |              2.638 |                  2.666 |  2.539 |    2.721 |
|         0.1  |       2.237 |              2.295 |                  2.291 |  2.198 |    2.286 |
|         0.2  |       1.777 |              1.811 |                  1.824 |  1.718 |    1.825 |
|         0.3  |       1.588 |              1.621 |                  1.599 |  1.52  |    1.606 |
|         0.4  |       1.489 |              1.518 |                  1.483 |  1.426 |    1.496 |

**STOI:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0.9275 |             0.9275 |                 0.9275 | 0.9275 |   0.9275 |
|         0.01 |      0.9213 |             0.9215 |                 0.9233 | 0.9213 |   0.9234 |
|         0.03 |      0.9067 |             0.9091 |                 0.9046 | 0.9055 |   0.9123 |
|         0.05 |      0.8926 |             0.8963 |                 0.8998 | 0.8895 |   0.904  |
|         0.1  |      0.8713 |             0.8753 |                 0.872  | 0.8659 |   0.87   |
|         0.2  |      0.8006 |             0.8048 |                 0.8146 | 0.7855 |   0.8043 |
|         0.3  |      0.7454 |             0.7524 |                 0.7464 | 0.7206 |   0.7426 |
|         0.4  |      0.7046 |             0.7144 |                 0.6968 | 0.6746 |   0.6933 |

**SI-SDR (dB):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |        3.24 |               3.24 |                   3.24 |   3.24 |     3.24 |
|         0.01 |        2.9  |               2.92 |                   2.98 |   2.89 |     2.89 |
|         0.03 |        2.31 |               2.47 |                   2.11 |   2.29 |     2.31 |
|         0.05 |        1.7  |               1.92 |                   1.87 |   1.66 |     1.88 |
|         0.1  |        0.66 |               0.89 |                   0.7  |   0.55 |     0.48 |
|         0.2  |       -2.35 |              -2.12 |                  -2.22 |  -2.6  |    -1.92 |
|         0.3  |       -4.78 |              -4.5  |                  -5.03 |  -5.39 |    -4.31 |
|         0.4  |       -6.77 |              -6.38 |                  -7.73 |  -7.58 |    -6.55 |

**Post-Repair Loss Rate:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0      |             0      |                 0      | 0      |   0      |
|         0.01 |      0.0081 |             0.0081 |                 0.0067 | 0.0083 |   0.01   |
|         0.03 |      0.0285 |             0.0278 |                 0.0333 | 0.03   |   0.0267 |
|         0.05 |      0.0468 |             0.0457 |                 0.0467 | 0.05   |   0.0417 |
|         0.1  |      0.0907 |             0.09   |                 0.0917 | 0.0983 |   0.0867 |
|         0.2  |      0.2042 |             0.2042 |                 0.2117 | 0.2233 |   0.195  |
|         0.3  |      0.3015 |             0.3019 |                 0.33   | 0.3333 |   0.2967 |
|         0.4  |      0.3779 |             0.3775 |                 0.4183 | 0.4167 |   0.385  |

## Concealment Method Comparison

### HILCodec_3kbps

| concealment   |   PESQ |   PESQ_NB |   STOI |   SI-SDR |
|:--------------|-------:|----------:|-------:|---------:|
| neighbor_copy | 1.9033 |    2.538  | 0.8705 |  -0.1181 |
| zero_fill     | 1.7324 |    2.2571 | 0.8207 |  -0.6662 |

## Importance Method Diagnostics (vs Oracle)

### HILCodec_3kbps

| method    |   spearman_corr |   precision_at_20pct |
|:----------|----------------:|---------------------:|
| A1        |          0.3143 |               0.2639 |
| A2        |          0.1707 |               0.3736 |
| A4        |         -0.2545 |               0.1111 |
| B1        |          0.0161 |               0.1986 |
| composite |          0.1425 |               0.2514 |

## Interpretation

- **HILCodec_3kbps**: Heuristic wins 1/8 conditions (highest STOI).
- **HILCodec_3kbps**: Importance Aware wins 3/8 conditions (highest STOI).
- **HILCodec_3kbps**: Importance Selective wins 1/8 conditions (highest STOI).
- **HILCodec_3kbps**: Random wins 3/8 conditions (highest STOI).

## Plots

- [PESQ vs PLR](plots/pesq_vs_plr.png)
- [STOI vs PLR](plots/stoi_vs_plr.png)
- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)
- [Post-Repair Loss](plots/post_repair_loss.png)
- [PESQ-NB vs PLR](plots/pesq_nb_vs_plr.png)
- [PESQ by Concealment](plots/pesq_by_concealment.png)
- [STOI by Concealment](plots/stoi_by_concealment.png)
- [SI-SDR by Concealment](plots/si_sdr_by_concealment.png)
- [Oracle Diagnostics](plots/oracle_diagnostics.png)