# Unified transfer benchmark (v0.6.3a)

This report re-evaluates the saved v0.6.1 / v0.6.2 checkpoints on a single common benchmark:

- LAB2 all 13 active
- Top 5 head-only subsets for k=5
- Top 5 head-only subsets for k=3
- Top 5 head-only subsets for k=2

## Mean R² by regime

### Temperature

| Mode | All13 | Top5 k=5 mean | Top5 k=3 mean | Top5 k=2 mean |
|---|---:|---:|---:|---:|
| lab1_reference | 0.966 | 0.945 | 0.924 | 0.913 |
| zero_shot | 0.903 | 0.895 | 0.893 | 0.888 |
| head_only | 0.943 | 0.938 | 0.938 | 0.934 |
| full_finetune | 0.985 | 0.981 | 0.975 | 0.976 |

### Humidity

| Mode | All13 | Top5 k=5 mean | Top5 k=3 mean | Top5 k=2 mean |
|---|---:|---:|---:|---:|
| lab1_reference | 0.986 | 0.977 | 0.970 | 0.962 |
| zero_shot | 0.871 | 0.860 | 0.845 | 0.855 |
| head_only | 0.899 | 0.889 | 0.895 | 0.872 |
| full_finetune | 0.972 | 0.968 | 0.963 | 0.960 |

## Selected subsets

| k | Rank | Protocol | Sensors | Head-only source score |
|---:|---:|---|---|---:|
| 5 | 1.0 | lab2_active_2_4_5_6_9 | {2, 4, 5, 6, 9} | 0.921 |
| 5 | 2.0 | lab2_active_3_4_5_6_7 | {3, 4, 5, 6, 7} | 0.914 |
| 5 | 3.0 | lab2_active_6_7_8_10_12 | {6, 7, 8, 10, 12} | 0.911 |
| 5 | 4.0 | lab2_active_4_6_8_10_12 | {4, 6, 8, 10, 12} | 0.911 |
| 5 | 5.0 | lab2_active_1_6_7_8_9 | {1, 6, 7, 8, 9} | 0.911 |
| 3 | 1.0 | lab2_active_4_6_7 | {4, 6, 7} | 0.919 |
| 3 | 2.0 | lab2_active_4_6_9 | {4, 6, 9} | 0.916 |
| 3 | 3.0 | lab2_active_2_4_7 | {2, 4, 7} | 0.916 |
| 3 | 4.0 | lab2_active_5_6_9 | {5, 6, 9} | 0.916 |
| 3 | 5.0 | lab2_active_6_9_10 | {6, 9, 10} | 0.915 |
| 2 | 1.0 | lab2_active_5_9 | {5, 9} | 0.905 |
| 2 | 2.0 | lab2_active_2_7 | {2, 7} | 0.905 |
| 2 | 3.0 | lab2_active_8_10 | {8, 10} | 0.903 |
| 2 | 4.0 | lab2_active_5_7 | {5, 7} | 0.901 |
| 2 | 5.0 | lab2_active_10_12 | {10, 12} | 0.900 |
