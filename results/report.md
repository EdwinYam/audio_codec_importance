# Importance-Aware Frame Protection: Experiment Report

**Generated from**: `/home/smartedwin/Projects/importance_eval_for_codec/results/results.csv`

**Total experiments**: 1920

**Audio files**: 12

**Codecs**: EnCodec_3kbps, HILCodec_3kbps

**Seeds**: [np.int64(42), np.int64(123)]


## Experiment Configuration

- **Codecs**: EnCodec_3kbps, HILCodec_3kbps
- **Protection budget**: 10% extra frames
- **Network types**: random_loss
- **PLRs**: 0%, 1%, 3%, 5%, 10%, 20%, 30%, 40%
- **Protection methods**: heuristic, importance_aware, importance_selective, none, random

## EnCodec_3kbps Results (averaged over files and seeds)

### Random Loss

**PESQ (MOS-LQO):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |       2.051 |              2.051 |                  2.051 |  2.051 |    2.051 |
|         0.01 |       1.706 |              1.706 |                  1.773 |  1.702 |    1.679 |
|         0.03 |       1.383 |              1.387 |                  1.364 |  1.371 |    1.392 |
|         0.05 |       1.255 |              1.261 |                  1.281 |  1.242 |    1.29  |
|         0.1  |       1.148 |              1.153 |                  1.158 |  1.141 |    1.167 |
|         0.2  |       1.072 |              1.073 |                  1.07  |  1.067 |    1.077 |
|         0.3  |       1.05  |              1.051 |                  1.046 |  1.046 |    1.051 |
|         0.4  |       1.041 |              1.042 |                  1.037 |  1.037 |    1.041 |

**STOI:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0.9013 |             0.9013 |                 0.9013 | 0.9013 |   0.9013 |
|         0.01 |      0.8922 |             0.8924 |                 0.8943 | 0.892  |   0.8914 |
|         0.03 |      0.8662 |             0.8671 |                 0.8634 | 0.8638 |   0.8744 |
|         0.05 |      0.8504 |             0.8519 |                 0.8533 | 0.8467 |   0.863  |
|         0.1  |      0.8268 |             0.8279 |                 0.8231 | 0.8207 |   0.8274 |
|         0.2  |      0.7604 |             0.7625 |                 0.7654 | 0.7504 |   0.766  |
|         0.3  |      0.7167 |             0.7207 |                 0.712  | 0.7031 |   0.7193 |
|         0.4  |      0.6845 |             0.69   |                 0.6726 | 0.6699 |   0.6816 |

**SI-SDR (dB):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |        3.25 |               3.25 |                   3.25 |   3.25 |     3.25 |
|         0.01 |        2.87 |               2.9  |                   3.05 |   2.86 |     2.88 |
|         0.03 |        2.19 |               2.31 |                   2.01 |   2.17 |     2.35 |
|         0.05 |        1.44 |               1.59 |                   1.65 |   1.39 |     1.78 |
|         0.1  |        0.13 |               0.26 |                   0.24 |  -0.05 |     0.14 |
|         0.2  |       -3.47 |              -3.35 |                  -3.49 |  -3.96 |    -3.08 |
|         0.3  |       -6.2  |              -6.02 |                  -6.84 |  -6.95 |    -5.82 |
|         0.4  |       -8.3  |              -8.01 |                  -9.46 |  -9.2  |    -8.19 |

**Post-Repair Loss Rate:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0      |             0      |                 0      | 0      |   0      |
|         0.01 |      0.0081 |             0.0081 |                 0.0067 | 0.0083 |   0.01   |
|         0.03 |      0.0285 |             0.0278 |                 0.0333 | 0.03   |   0.0267 |
|         0.05 |      0.0468 |             0.0457 |                 0.0467 | 0.05   |   0.0417 |
|         0.1  |      0.0907 |             0.0899 |                 0.0917 | 0.0983 |   0.0867 |
|         0.2  |      0.2042 |             0.2037 |                 0.2117 | 0.2233 |   0.195  |
|         0.3  |      0.3015 |             0.3015 |                 0.33   | 0.3333 |   0.2967 |
|         0.4  |      0.3779 |             0.3772 |                 0.4183 | 0.4167 |   0.385  |

## HILCodec_3kbps Results (averaged over files and seeds)

### Random Loss

**PESQ (MOS-LQO):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |       2.751 |              2.751 |                  2.751 |  2.751 |    2.751 |
|         0.01 |       2.391 |              2.391 |                  2.542 |  2.39  |    2.425 |
|         0.03 |       1.961 |              1.979 |                  1.95  |  1.937 |    2.026 |
|         0.05 |       1.715 |              1.738 |                  1.821 |  1.69  |    1.857 |
|         0.1  |       1.465 |              1.485 |                  1.484 |  1.437 |    1.511 |
|         0.2  |       1.216 |              1.219 |                  1.233 |  1.198 |    1.232 |
|         0.3  |       1.149 |              1.151 |                  1.137 |  1.129 |    1.163 |
|         0.4  |       1.113 |              1.112 |                  1.106 |  1.099 |    1.115 |

**STOI:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0.9275 |             0.9275 |                 0.9275 | 0.9275 |   0.9275 |
|         0.01 |      0.9183 |             0.9184 |                 0.921  | 0.9183 |   0.9217 |
|         0.03 |      0.8972 |             0.8998 |                 0.8933 | 0.8956 |   0.9053 |
|         0.05 |      0.876  |             0.8805 |                 0.8868 | 0.8718 |   0.8942 |
|         0.1  |      0.8471 |             0.8518 |                 0.8497 | 0.8398 |   0.8474 |
|         0.2  |      0.7574 |             0.7602 |                 0.7751 | 0.7429 |   0.7656 |
|         0.3  |      0.693  |             0.6972 |                 0.6888 | 0.6727 |   0.6993 |
|         0.4  |      0.6488 |             0.6533 |                 0.636  | 0.6249 |   0.6443 |

**SI-SDR (dB):**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |        3.24 |               3.24 |                   3.24 |   3.24 |     3.24 |
|         0.01 |        2.9  |               2.91 |                   2.98 |   2.9  |     2.87 |
|         0.03 |        2.32 |               2.47 |                   2.07 |   2.31 |     2.33 |
|         0.05 |        1.64 |               1.85 |                   1.82 |   1.6  |     1.79 |
|         0.1  |        0.51 |               0.74 |                   0.54 |   0.36 |     0.3  |
|         0.2  |       -2.79 |              -2.59 |                  -2.62 |  -3.15 |    -2.35 |
|         0.3  |       -5.37 |              -5.17 |                  -5.67 |  -6.15 |    -4.9  |
|         0.4  |       -7.53 |              -7.23 |                  -8.55 |  -8.54 |    -7.4  |

**Post-Repair Loss Rate:**

|   target_plr |   heuristic |   importance_aware |   importance_selective |   none |   random |
|-------------:|------------:|-------------------:|-----------------------:|-------:|---------:|
|         0    |      0      |             0      |                 0      | 0      |   0      |
|         0.01 |      0.0081 |             0.0081 |                 0.0067 | 0.0083 |   0.01   |
|         0.03 |      0.0285 |             0.0278 |                 0.0333 | 0.03   |   0.0267 |
|         0.05 |      0.0468 |             0.0457 |                 0.0467 | 0.05   |   0.0417 |
|         0.1  |      0.0907 |             0.0899 |                 0.0917 | 0.0983 |   0.0867 |
|         0.2  |      0.2042 |             0.2037 |                 0.2117 | 0.2233 |   0.195  |
|         0.3  |      0.3015 |             0.3015 |                 0.33   | 0.3333 |   0.2967 |
|         0.4  |      0.3779 |             0.3772 |                 0.4183 | 0.4167 |   0.385  |

## Importance Method Diagnostics (vs Oracle)

### EnCodec_3kbps

| method    |   spearman_corr |   precision_at_20pct |
|:----------|----------------:|---------------------:|
| A1        |          0.3092 |               0.2514 |
| A2        |          0.1508 |               0.3889 |
| A4        |         -0.2446 |               0.1181 |
| B1        |          0.1191 |               0.2    |
| composite |          0.1454 |               0.2486 |

### HILCodec_3kbps

| method    |   spearman_corr |   precision_at_20pct |
|:----------|----------------:|---------------------:|
| A1        |          0.3143 |               0.2639 |
| A2        |          0.1707 |               0.3736 |
| A4        |         -0.2545 |               0.1111 |
| B1        |          0.0161 |               0.1986 |
| composite |          0.1184 |               0.2472 |

## Interpretation

- **EnCodec_3kbps**: Random wins 3/8 conditions (highest STOI).
- **EnCodec_3kbps**: Heuristic wins 1/8 conditions (highest STOI).
- **EnCodec_3kbps**: Importance Aware wins 3/8 conditions (highest STOI).
- **EnCodec_3kbps**: Importance Selective wins 1/8 conditions (highest STOI).
- **HILCodec_3kbps**: Random wins 4/8 conditions (highest STOI).
- **HILCodec_3kbps**: Heuristic wins 1/8 conditions (highest STOI).
- **HILCodec_3kbps**: Importance Aware wins 2/8 conditions (highest STOI).
- **HILCodec_3kbps**: Importance Selective wins 1/8 conditions (highest STOI).

## Plots

- [PESQ vs PLR](plots/pesq_vs_plr.png)
- [STOI vs PLR](plots/stoi_vs_plr.png)
- [SI-SDR vs PLR](plots/si_sdr_vs_plr.png)
- [Post-Repair Loss](plots/post_repair_loss.png)
- [Oracle Diagnostics](plots/oracle_diagnostics.png)