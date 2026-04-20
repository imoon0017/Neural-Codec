# Split-Test Evaluation Summary

**Date:** 2026-04-20
**Split:** test (50 samples each, 18 configurations)
**Targets:** CR ≥ 10×, ADR ≤ 2%, encode ≤ 50 ms, decode ≤ 100 ms

---

## Parameter Count Formula

All configs use `encoder_channels C = 64`. Parameters depend only on `D` and `c` (not `B`):

```
Params = 1,345 + 184,576 × N + 130 × D    where N = log2(c)
```

---

## Results by Compaction Ratio Group

### c = 8 — N = 3 down/up-blocks (~556K–559K params)

| Run | Params | CR (mean) | ADR mean | ADR p95 | Encode ms | Decode ms |
|---|---|---|---|---|---|---|
| D32_c8_B8  | 559,233 | 1.14× | **3.00%** | 3.56% | 5.7 | 1.5 |
| D32_c8_B16 | 559,233 | 1.01× | 3.15% | 3.79% | 52.3 ⚠ | 3.5 |
| D8_c8_B16  | 556,113 | 3.13× | 3.24% | 3.74% | 6.6 | 1.5 |
| D16_c8_B16 | 557,153 | 1.75× | 3.37% | 4.15% | 5.7 | 1.5 |
| D8_c8_B8   | 556,113 | 3.48× | 3.44% | 4.16% | 5.8 | 1.5 |
| D16_c8_B8  | 557,153 | 2.14× | 3.46% | 4.29% | 5.7 | 1.5 |

⚠ D32_c8_B16 encode mean is inflated by a one-time GPU warm-up spike; p95 = 1.26 ms.

### c = 16 — N = 4 down/up-blocks (~741K–744K params)

| Run | Params | CR (mean) | ADR mean | ADR p95 | Encode ms | Decode ms |
|---|---|---|---|---|---|---|
| D32_c16_B16 | 743,809 | 2.94× | 4.59% | 5.42% | 5.7 | 1.8 |
| D32_c16_B8  | 743,809 | 3.40× | 4.86% | 5.78% | 5.7 | 1.8 |
| D8_c16_B16  | 740,689 | 9.38× | 5.19% | 5.86% | 5.7 | 1.7 |
| D8_c16_B8   | 740,689 | 10.86× | 5.24% | 5.99% | 5.9 | 1.8 |
| D16_c16_B16 | 741,729 | 5.38× | 5.33% | 6.13% | 5.9 | 1.8 |
| D16_c16_B8  | 741,729 | 6.14× | 5.35% | 6.35% | 5.9 | 1.8 |

### c = 32 — N = 5 down/up-blocks (~925K–928K params)

| Run | Params | CR (mean) | ADR mean | ADR p95 | Encode ms | Decode ms |
|---|---|---|---|---|---|---|
| D32_c32_B8  | 928,385 | 12.17× | 13.81% | 14.90% | 5.8 | 1.9 |
| D32_c32_B16 | 928,385 | 10.56× | 14.14% | 15.33% | 5.9 | 1.9 |
| D16_c32_B16 | 926,305 | 17.96× | 14.07% | 15.36% | 8.0 | 1.9 |
| D16_c32_B8  | 926,305 | 21.12× | 14.73% | 16.14% | 5.9 | 1.9 |
| D8_c32_B16  | 925,265 | 29.66× | 25.50% | 27.73% | 6.3 | 3.1 |
| D8_c32_B8   | 925,265 | 36.04× | 25.31% | 27.38% | 6.0 | 2.0 |

*Baseline (validation split):* CR 0.98×, ADR 3.27%

---

## Target Pass/Fail Summary

| Target | Passes |
|---|---|
| CR ≥ 10× | D8_c16_B8 (10.86×), D8_c32_B16 (29.66×), D8_c32_B8 (36.04×), D16_c32_B16 (17.96×), D16_c32_B8 (21.12×), D32_c32_B8 (12.17×), D32_c32_B16 (10.56×) |
| ADR ≤ 2% | **None** |
| CR ≥ 10× **AND** ADR ≤ 2% | **None** |
| Encode ≤ 50 ms (p95) | All (max p95 = 1.56 ms, excl. warm-up outlier) |
| Decode ≤ 100 ms (p95) | All |

---

## Key Findings

### 1. No run meets both primary targets simultaneously
CR ≥ 10× requires `c ≥ 32`, but every c=32 run produces ADR ≥ 13.8% — nearly 7× the 2% target. The tradeoff is a hard architectural constraint, not a training artifact.

### 2. `c` is the dominant factor
ADR scales cleanly with compaction ratio regardless of D or B:

| c | Median ADR |
|---|---|
| 8  | ~3.2% |
| 16 | ~5.1% |
| 32 | ~17%  |

### 3. B=8 gives free compression with negligible quality cost
Within any (D, c) group, B=8 boosts CR by ~10–20% while adding only ~0.1–0.2 pp ADR. However, the expected 2× file-size reduction from halving the dtype does **not** materialise because `ZIP_DEFLATED` already exploits the high-byte redundancy of small int16 values, compressing both dtypes to nearly the same size. The raw 2× benefit would require storing `dna.bin` uncompressed or using a domain-specific entropy coder.

### 4. D has negligible impact
Within c=8 and B=8, D=32 (ADR 3.00%) vs D=8 (ADR 3.44%) is only 0.44 pp across a 3× parameter difference at the bottleneck. Capacity is not the binding constraint.

### 5. Training converged — more epochs will not close gaps
All runs ended at `lr ≈ lr_min = 1e-6` (cosine schedule exhausted) with train ≈ val loss (no underfitting gap). The ADR differences between D values are capacity-driven, not convergence-driven.

---

## Recommended Configs

| Goal | Config | Params | CR | ADR |
|---|---|---|---|---|
| Best reconstruction quality | D32_c8_B8 | 559,233 | 1.14× | 3.00% |
| Best CR that clears 10× | D8_c16_B8 | 740,689 | 10.86× | 5.24% |

---

## Path Forward

The results expose a gap between the current architecture and project targets:

- **CR gap at c=8:** The codec compresses spatial extent by 8× but `D` is small, so net bit-rate is near 1:1 versus OASIS (which is already an efficient format). Entropy coding of the quantized values is needed to close this gap.
- **ADR gap at all c:** Best ADR is 3.0% vs the 2.0% target. Adding **bottleneck self-attention** (single MHSA layer between the encoder projection and quantizer at the small `S/c × S/c` resolution) is the most promising next step, allowing global shape context without crossing the encoder–decoder boundary.
