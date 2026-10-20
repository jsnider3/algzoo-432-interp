# Prespecified anchor-gradient experiment

## Design and mechanism

I fixed 34 position arrangements before fitting: the union of max-at-last-step and max-immediately-after-second-max, each with R at the first or last remaining position. I used MF4 rows only, at S slices 0 and K−1 (one slice for K=1), giving 68 rows at K=2/4 and 34 rows at K=1. There is no target sampling or target-dependent row selection.

The selected output rival is the largest non-target logit, with NumPy's first-index tie rule. Backpropagation through the actual anchor ReLU gates gives all ten input derivatives of the signed target-minus-rival margin. I retain the six lower-input derivatives. The lower conditional standard deviation at Q=0.3715875745 is 0.6724137472, from `Var(X|X<Q)=1+Q*mu(Q)-mu(Q)**2`.

Let `z_j = sigma_lower * d(margin)/dx_j / s` for the six lower positions and zero for the other four, where s is G's existing RMS anchor margin. Eight summaries are averaged separately in each selected S slice:

1. L2 norm of z: local spread under independent lower-input variation.
2. L1 norm divided by sqrt(6): sensitivity to coherent worst-direction lower-input changes.
3. Absolute sum divided by sqrt(6): sensitivity to a common lower-input shift.
4. L2 norm restricted to time positions 8 and 9: late-input sensitivity.
5. RMS adjacent-pair sum, over pairs whose two entries are lower inputs: sensitivity to adjacent same-sign disturbances.
6. RMS adjacent-pair difference over those pairs: sensitivity to adjacent alternating disturbances.
7. Fraction with a strictly positive anchor margin no larger than its local L2 spread.
8. Mean bounded ratio `L2/(abs(margin)/s + L2 + 1e-12)`.

These are local sensitivity descriptors. They are not certified error bounds or rigorous probabilities: the ReLU pattern and output rival may change away from the anchor. The adjacent features measure designed directions, not an assumption that conditionally independent lower inputs actually have a nonzero covariance.

The block has 16 dimensions for K=2/4 and 8 for K=1. All dimensions were retained. The readout gamma/alpha/family are nested exactly as in the audited G baseline, with outer permutation seed 0 and stable inner denominator 1024. External models are predicted after eight-fold selection on the 1,024 training networks and refitting on all of them.

## Cost

For each selected row, reverse propagation costs 9×256 recurrent MACs, 10×16 gate multiplications, 10×16 input-gradient MACs, and 16 output-weight differences: 2,640 primitive operations, or 0.91667 FE. There is a conservative 160-operation feature/aggregation allowance per row, plus final feature means. Additional standardized RBF dimensions cost `(2*1024+2)*d_extra/2880` FE.

If G's forward pass retains the selected paths' gate masks and output rivals, the additional costs are:

| K | Selected MF4 rows already in G | Backward FE | Feature FE | Additional readout FE | Total added FE |
|---|---:|---:|---:|---:|---:|
| 1 | 34 | 31.1667 | 1.8917 | 5.6944 | 38.7528 |
| 2 or 4 | 68 | 62.3333 | 3.7833 | 11.3889 | 77.5056 |

The supplied standalone NumPy extractor re-runs those rows to obtain gates. Its conservative charge is 3,108 operations per extra forward row, including explicit gate comparisons (the first state's known-zero recurrent product is omitted). This adds another **36.6917 FE at K1** and **73.3833 FE at K2/4**. The authoritative JSON files include these standalone costs. Initial printed logs used 2,948 operations before I added the explicit mask-comparison charge; the fitted features and predictions are unchanged.

## Results

| Recipe | K | Baseline G RMS | G + gradients RMS | Mean | Total FE with gate reuse | Ratio with reuse | Standalone total FE | Standalone ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unif | 1 | 0.96355 | 0.95791 | +0.01321 | 657.2983 | 0.71665 | 693.9899 | 0.75666 |
| Unif | 2 | 0.67802 | 0.66750 | +0.01199 | 1,207.1691 | 0.63909 | 1,280.5524 | 0.67794 |
| Unif | 4 | 0.58306 | **0.51626** | +0.00200 | 2,229.4052 | **0.70602** | 2,302.7885 | **0.72926** |
| Zoo | 1 | 0.59673 | 0.63405 | +0.01162 | 657.2983 | 0.38238 | 693.9899 | 0.40373 |
| Zoo | 2 | 0.51503 | 0.51153 | +0.00798 | 1,207.1691 | 0.45710 | 1,280.5524 | 0.48488 |
| Zoo | 4 | 0.43451 | **0.40564** | +0.00340 | 2,229.4052 | **0.53083** | 2,302.7885 | **0.54831** |

The gradient block improves cost-normalized performance at K4 on both populations, including the standalone cost. It does not pay at K1/K2. This is consistent with local fragility being more informative near the tie boundaries reached by the extreme K4 slices; the result does not identify causality or establish that this is the unique reason.

External mystery-recipe sets, with RMS and mean in points (ratios use gate reuse):

| K | b128 RMS / mean / ratio | b64 RMS / mean / ratio | Mystery error |
|---|---|---|---:|
| 1 | 0.86131 / +0.11753 / 0.57658 | 1.21025 / +0.74766 / 1.15639 | -0.62630 |
| 2 | 0.57598 / +0.05618 / 0.47355 | 0.94658 / +0.53468 / 1.29920 | -0.65171 |
| 4 | **0.50260 / +0.06945 / 0.66591** | **0.58807 / +0.18313 / 0.92606** | -0.15062 |

For K4 standalone cost, the ratios are 0.68783 (b128) and 0.95654 (b64). The mystery's individual error worsens relative to G's -0.06583; no setting was chosen from that model's result.

Zoo seeds 0–4 errors:

- K1: +0.55038, -0.03011, -0.20781, -0.22493, +0.34196.
- K2: -0.75933, +0.32444, -0.00931, -0.37331, +0.08737.
- K4: -0.66805, +0.77503, -0.20628, -0.54450, -0.43892.

## Verification and reproducibility

`gradient_features.py` is the compact extractor and evaluation driver. It imports the existing audited readout via `audit_check.py`. Run from the project directory:

`python /path/to/gradient_features.py 4`

The all-input Jacobian satisfies the homogeneous-network Euler identity `sum_j x_j*dmargin/dx_j = margin` to a maximum absolute residual below 4e-13. For 192 deterministic central finite-difference checks on eight networks, including the mystery and zoo seed4, maximum derivative error is below 1.5e-7. No selected cached margin is exactly zero in any of the six recipe/K combinations. Validation output is in `gradients_validation.log`.

All six population/CV runs took about 164 seconds with one BLAS thread. `gradients_{unif,zoo}_k{1,2,4}.npz` contain the fixed extra feature block for every network, outer predictions, and external predictions. Matching JSON files have exact costs and metrics. This was one prespecified feature block examined at the three existing budgets; no further gradient variants were selected or tuned on outer labels.
