# Result

**I did not find sampling parity.** I did find a deterministic two-level estimator that improves the matched `z38` result substantially:

- **Development-selected estimator:** RMS error **0.2254 percentage points on development** and **0.2493 on held out**.
- **Direct comparison with `z38`:** the preplanned interpolation-based version has approximately **6.28× lower MSE while using 8.92% less counted work**.
- **Remaining gap:** the selected estimator’s held-out MSE is about **222× sampling’s** under the stated arithmetic convention. Using only its matrix-operation cost—a lower bound for this implementation—still gives **219.9×**.
- **Theory:** there is an explicit, fixed-dimension lower bound for deterministic **Boolean cone-membership queries**. It does **not** establish a lower bound for algorithms inspecting these RNNs’ weights, or for average error over the trained population.

The complete deliverables are:

- [algzoo_r6_report.txt](sandbox:/workspace/scratch/5c1a427569dc/deliverables/algzoo_r6_report.txt) — detailed findings, every tested setting, full proofs, reproduction commands, and source code.
- [algzoo_r6_results.json](sandbox:/workspace/scratch/5c1a427569dc/deliverables/algzoo_r6_results.json) — all numerical tables, individual-network errors, diagnostics, and cost metadata.
- [algzoo_r6_reproduction.zip](sandbox:/workspace/scratch/5c1a427569dc/deliverables/algzoo_r6_reproduction.zip) — the original repository plus the new code, results, and reproduction guide.

All new numerical code is also included inline below.

## 1. Evaluation and short audit

I used the requested **64 development networks** `0,16,…,1008`, **64 held-out networks** `8,24,…,1016`, and all six zoo models. The correction shortlist and selected high-compute method were frozen before held-out scoring. The held-out curve evaluates the methods selected on development; it does not select a separate held-out optimum.

Errors below are **estimate minus reference**, in percentage points. Zoo order is always **mystery, seed0, seed1, seed2, seed3, seed4**. The numerical source for the tables is the [results file](sandbox:/workspace/scratch/5c1a427569dc/deliverables/algzoo_r6_results.json).

### Audit findings

| Script or issue | Finding | Consequence |
|---|---|---|
| `z21`, `z37` | They evaluate dense S grids but attach the charge of an optimized exact prefix integrator absent from the archive. At `K=256`, actual row counts exceed the quoted per-node charges by about **138.7×** and **120.6×**, respectively. | The numerical anchor integrals remain informative; the speed claims are conditional on implementing the missing optimization. |
| MF2 cost extrapolation | Its smaller inputs move as \(\mu(S)\), so the input path is curved. | An affine MF3 line cost cannot simply be scaled by the number of arrangements. |
| `z18` pair projection | The symmetric pair tensor and factor \(1/2\) are consistent. At `q=8`, adjacent pairs average **241 rows**; separation at most two averages **411⅔**, with range **241–625**; all pairs cost **1,009**. The script evaluates all pairs before masking. | Reduced subset costs require actually omitting those evaluations. There is no factor-of-two projection bug. |
| `z18` convolution | The threshold searches and related target-dependent work are additional to its forward rows. | A forward-row-only count is incomplete. |
| `z38` | Its forward count and sampling comparator check out. | Its failure survives a direct matched rerun. |
| Reference noise | RMS reference SE is **0.02643 points on development**, **0.02580 on held out**. | A close claim near \(10^7\) forwards needs a larger reference or explicit noise accounting. The present \(0.25\)-point errors are much too large to be explained by this noise. |

At \(C=10^7\), reference variance alone contributes approximately \(C/2^{20}=9.54\) to a raw MSE/sampling comparison. Also, shared reference inputs correlate network errors; treating the 64 reference errors as independent would overstate precision.

### Cost convention

For constant cost \(C\), I use

\[
\operatorname{SamplingRMS}(C)
=
100\sqrt{\frac{\operatorname{mean}_i p_i(1-p_i)}{C}}.
\]

For exact line tracing, costs vary by network, so I use the matched comparison

\[
100\sqrt{\operatorname{mean}_i\frac{p_i(1-p_i)}{C_i}},
\]

while reporting mean \(C_i\) on the horizontal axis.

There are two qualifications:

1. The supplied line tracer counts matrix/readout MACs but omits scalar roots, sorting, and CDF work. Its count is a **lower bound for the implemented computation**.
2. Other routines use declared scalar allowances: **200 scalar equivalents per root/CDF interval**, and **1% additional work for correction-grid processing**. These are accounting conventions, not measured hardware upper bounds.

Consequently, an exact-anchor combination **including** the 1% correction allowance is an optimistic count, not itself a proved total-cost lower bound. The result files separately provide the pure-MAC lower bound.

## 2. The successful improvement: two levels

Let \(I_3\) and \(I_4\) denote correctness at the MF3 and MF4 anchors. For an MF4 configuration, define the clamped singleton model

\[
F_4
=
\operatorname{clip}\!\left(
I_4+\sum_{j=1}^{6}(E_j-I_4),\,0,\,1
\right),
\]

where \(E_j\) integrates only lower input \(j\), using eight Gauss–Legendre nodes in its truncated-normal CDF coordinate.

The new estimator is

\[
\boxed{
\widehat p
=
Q_{\mathrm{fine}}^{\mathrm{MF3}}(I_3)
+
Q_{\mathrm{coarse}}^{\mathrm{MF4}}(F_4-I_3).
}
\]

The fine anchor uses all 720 MF3 arrangements and outer order six. The selected version integrates S-lines exactly with the supplied tracer. The preplanned companion uses 32 S intervals, evaluates logits with actual shared prefixes, and integrates correctness exactly for the resulting **linear-logit interpolant**.

The coarse correction uses:

- Order two in each of the three MF4 outer coordinates: eight outer nodes.
- Eight conditional-CDF S midpoints.
- Eight lower-input Gauss–Legendre nodes.
- Eight balanced \((R,Q)\) placements for each of the 90 \((M,S)\) classes.

Each configuration evaluates 49 singleton-model rows and one separate MF3 anchor row. Thus the correction evaluates exactly

\[
8\times8\times720\times50
=
2{,}304{,}000
\]

forward rows. Its declared charge is \(2{,}327{,}040\) after the 1% allowance.

### End-to-end comparison

| Method | Split | Counted forwards | RMS error | Mean error | Sampling RMS | MSE ratio |
|---|---|---:|---:|---:|---:|---:|
| Two levels, exact-S MF3 anchor | Development | 2,491,426* | **0.225373** | +0.055505 | 0.017183 | 172.0 |
| Two levels, exact-S MF3 anchor | Held out | 2,497,761* | **0.249308** | +0.044037 | 0.016735 | 221.9 |
| Two levels, interpolated MF3 anchor | Development | 2,718,305 | 0.247840 | −0.019012 | 0.016418 | 227.9 |
| Two levels, interpolated MF3 anchor | Held out | 2,718,305 | 0.247615 | −0.022789 | 0.016024 | 238.8 |
| Matched `z38`, `n3/K8/q8` | Development | 2,984,429 | 0.621258 | +0.512445 | 0.015668 | 1,572.1 |
| Matched `z38`, `n3/K8/q8` | Held out | 2,984,429 | 0.620461 | +0.522291 | 0.015293 | 1,646.0 |

\* These counts omit exact-anchor scalar work and include the correction allowance. Removing that allowance gives pure-MAC lower bounds of **2,468,386 development / 2,474,721 held**, with MSE-ratio lower bounds **170.4 / 219.9** for this implementation.

### Zoo errors

| Method | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---|---:|---:|---:|---:|---:|---:|
| Two levels, exact-S anchor | +0.02332 | +0.35282 | +0.06921 | +0.36046 | −0.14904 | +0.19130 |
| Two levels, interpolated anchor | −0.04131 | +0.35534 | +0.01853 | +0.03659 | −0.19393 | +0.18649 |
| Matched `z38` | +0.52172 | +0.55571 | +0.20352 | +0.16719 | +0.22872 | +0.56521 |

The interpolation-based comparison gives **6.28× lower MSE at 8.92% less declared work**. Its error relative to sampling improves by about **6.9×**. That is a useful improvement, but it leaves a large gap.

## 3. What the correction experiments showed

### (a) Amortizing across arrangements

I measured the actual true-minus-anchor correction with **256 independent draws in every MF4 arrangement**, on the 64 development networks and the zoo. Each network therefore received 1,290,240 diagnostic inputs, evaluated at truth, MF3, and MF4: **3,870,720 diagnostic forward rows**. These samples are absent from every estimator.

After subtracting measurement variance:

| Correction | Median arrangement SD | Network 10th–90th percentile | Variance explained by M,S positions | Including relative rank order |
|---|---:|---:|---:|---:|
| True minus MF3 | **3.939** | 2.508–5.414 | 21.9% | 45.3% |
| True minus MF4 | **2.048** | 1.147–3.079 | 16.9% | 34.9% |

The “1% of integrand variance” clue does not mean arrangement corrections agree to \(0.05\)–\(0.1\) points.

For designed subsets, the following are **measurements granting exact integration over values**, rather than complete estimators:

| Correction and subset | Arrangements | Noise-corrected RMS loss | Measurement SE per network |
|---|---:|---:|---:|
| MF3: one R placement per M,S class | 90 | 0.582 | 0.052 |
| MF3: two symmetric R placements | 180 | 0.281 | 0.035 |
| MF3: four symmetric R placements | 360 | 0.228 | 0.020 |
| MF4: four R,Q placements | 360 | 0.266 | 0.058 |
| MF4: eight balanced R,Q placements | 720 | **0.110** | 0.040 |
| MF4 coordinates, true minus MF3, eight placements | 720 | **0.186** | 0.049 |

The balanced pairs, in ranks among the eight remaining positions, are

```text
(0,7), (7,0), (1,3), (3,1), (2,5), (5,2), (4,6), (6,4)
```

Every free position appears once as R and once as Q.

For the eight-placement MF4 correction measurement, zoo deviations are

```text
−0.038, −0.018, +0.128, −0.206, −0.054, +0.023
```

with measurement SEs about \(0.028\)–\(0.049\) points. The \(0.110\)-point RMS result therefore does not resolve a sharp \(0.100\)-point threshold.

This supports moderate amortization, but not the claim that the correction can be copied almost unchanged across arrangements.

### (b) How coarse can the correction integral be?

The next table grants a **free reference anchor**, isolating correction error. These are favorable diagnostics, not end-to-end scores. Entries separated by `/` are development/held out.

| Correction rule | Counted forwards | RMS D/H | Mean D/H | Sampling RMS D/H | MSE ratio D/H |
|---|---:|---:|---:|---:|---:|
| MF3 `n1/K8/q4`, all 720 | 168,710 | 0.455 / 0.468 | −0.160 / −0.149 | 0.06590 / 0.06432 | 47.7 / 52.9 |
| MF3 `n2/K16/q8`, all 720 | 2,652,826 | 0.360 / 0.341 | −0.267 / −0.224 | 0.01662 / 0.01622 | 468.8 / 441.8 |
| MF4 minus MF4, `n1/K8/q8`, all 5,040 | 2,036,160 | **0.162 / 0.137** | +0.005 / +0.015 | 0.01897 / 0.01852 | 73.3 / 54.6 |
| MF4 minus MF3, `n2/K8/q8`, balanced 720 | 2,327,040 | 0.234 / 0.251 | +0.099 / +0.088 | 0.01774 / 0.01732 | 174.5 / 210.3 |

Their zoo errors are:

| Rule, in the same order | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---|---:|---:|---:|---:|---:|---:|
| MF3 coarse | +0.0828 | −0.3527 | −1.4573 | −0.0632 | −0.6336 | −0.5254 |
| MF3 finer | −0.1708 | −0.3902 | −1.0463 | −0.1237 | −0.3192 | −0.2765 |
| MF4 minus MF4 | +0.0345 | −0.0088 | −0.2345 | +0.0653 | −0.1370 | −0.0001 |
| MF4 minus MF3 | +0.0891 | +0.3466 | +0.1604 | +0.4371 | −0.0772 | +0.2233 |

Increasing the MF3 correction cost **15.7×** makes only a small difference to error. The pure MF4 correction is easier, but it also misses the requested \(0.1\)-point partial result.

I stopped this family after the frozen comparisons. A later development-only combination using the already computed `z38` correction did not improve the selected result; it was not used to retune held-out estimates.

## 4. (c) State-distribution propagation

I implemented a deterministic component recursion carrying the full hidden vector, \(M,S\), and **both** top positions. The max position is needed for a future handoff.

At each represented state:

- The three event branches receive their exact Gaussian probabilities.
- A branch uses its exact truncated mean, or two Gauss nodes in conditional-CDF coordinates.
- Optional merging groups by positions and CDF bins of \(M,S\), retaining the weighted mean of the **joint hidden vector**.
- The final input is integrated using a gate sweep, including changes in the true target position.

This is a **point/mean closure**. I did not complete the proposed exact affine-component law preserving every past-input dependency.

| Recursion | Counted forwards | RMS D/H | Mean D/H | Sampling RMS D/H | MSE ratio D/H |
|---|---:|---:|---:|---:|---:|
| One node, one value bin | 200.6 | 4.983 / 5.090 | +4.275 / +4.605 | 1.911 / 1.865 | 6.80 / 7.45 |
| Two nodes, eight value bins | 3,199.8 | 5.579 / 5.260 | +5.167 / +4.923 | 0.479 / 0.467 | 135.9 / 126.8 |
| One node, no merging | 18,394.9 | 5.256 / 5.057 | +4.968 / +4.857 | 0.200 / 0.195 | 693.6 / 674.0 |

| Recursion | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---|---:|---:|---:|---:|---:|---:|
| One node, one bin | +2.046 | +6.286 | +7.410 | +3.624 | +3.649 | +6.160 |
| Two nodes, eight bins | +2.777 | +6.380 | +7.713 | +3.422 | +4.120 | +7.618 |
| One node, no merging | +3.345 | +6.153 | +4.866 | +4.003 | +4.481 | +6.952 |

Keeping more components does not repair this closure. These results reject the implemented approximation, **not correlated affine mixtures**.

## 5. (d) Error geometry and component proliferation

I used 65,536 independent diagnostic inputs per network, split equally between discovery and coverage measurement, on eight development networks and all six zoo models.

| Time | Median observed gate histories | Test mass in unseen histories | Test mass covered by top 1,024 discovery histories |
|---|---:|---:|---:|
| 3 | 474 | 0.18% | 99.82% |
| 5 | 7,774 | 8.90% | 77.73% |
| 7 | 20,980 | 27.76% | 46.22% |
| 9 | 38,936 | 55.74% | 19.65% |
| 10 | 46,867 | 68.84% | 12.76% |

At time ten, retaining sixteen histories per age pair covers only **6.77%** of test mass; retaining sixteen per event history covers **8.83%**.

Fixing a full activation pattern, the two true top positions, and the predicted position yields a particular convex polyhedral error-cone partition. I observed a median **5,846 error fragments**, yet **97.29% of independent test errors** fell in fragments absent from discovery.

The corresponding zoo unseen-error fractions are:

```text
97.52%, 88.77%, 86.83%, 98.37%, 98.38%, 76.40%
```

This is evidence against retaining a short list of complete histories or discovered local fragments. It does not bound a method that merges histories with equivalent future behavior.

There is also a definitional issue with “count maximal convex pieces.” Even

\[
E=\{x\le0\}\cup\{y\le0\}
\]

has continuously many maximal convex subcones

\[
H_{a,b}=\{ax+by\le0\},\qquad a,b>0.
\]

Adding any point outside \(H_{a,b}\) lets a convex combination enter the excluded positive quadrant. A finite chosen partition, minimum convex cover, or explicit merging procedure is a well-defined alternative.

**I did not construct and integrate a merged Gaussian error-cone cover.** The diagnostic costs 65,536 forward evaluations per network, plus analysis; it is not an accuracy estimator, so assigning it an estimator RMS/sampling comparison would be misleading.

## 6. (e) Low budgets and the full curve

I evaluated six fixed MF2 rules, six MF3 interpolation rules, and the event rules above. Representative points are:

| Rule | Counted forwards | RMS D/H | Mean D/H | Sampling RMS D/H | MSE ratio D/H |
|---|---:|---:|---:|---:|---:|
| MF2 `n1/q2` | 95.8 | 7.770 / 7.413 | +7.211 / +6.978 | 2.765 / 2.699 | 7.89 / 7.54 |
| MF3 interpolant `n1/K2` | 683.1 | 2.484 / 2.251 | +0.424 / +0.596 | 1.036 / 1.011 | 5.75 / 4.96 |
| MF3 interpolant `n2/K4` | 5,448.7 | 1.174 / 1.232 | +0.205 / +0.279 | 0.367 / 0.358 | 10.25 / 11.84 |
| MF3 interpolant `n4/K16` | 86,980.8 | 0.588 / 0.610 | +0.152 / +0.191 | 0.0918 / 0.0896 | 41.10 / 46.40 |

| Rule | Mystery | Seed0 | Seed1 | Seed2 | Seed3 | Seed4 |
|---|---:|---:|---:|---:|---:|---:|
| MF2 `n1/q2` | +4.695 | +9.019 | +10.193 | +5.158 | +6.798 | +11.023 |
| MF3 `n1/K2` | +0.242 | +2.387 | +0.294 | −5.847 | +0.863 | +2.053 |
| MF3 `n2/K4` | +0.114 | +1.145 | +0.977 | −2.798 | +0.110 | +0.812 |
| MF3 `n4/K16` | −0.053 | +0.176 | +1.062 | −0.704 | −0.230 | −0.155 |

None reaches parity. Fine anchor integration eventually leaves the anchor’s mean-field bias.

The measured frontier extends from approximately **48 to 3 million forwards**. The sampling line extends to \(10^7\); no new estimator performance is extrapolated there. All 72 preplanned anchor/correction combinations per split are retained in the report and results.

![Development and held-out accuracy-error curves compared with sampling](sandbox:/workspace/scratch/5c1a427569dc/deliverables/algzoo_r6_curve.png)

[algzoo_r6_curve.png](sandbox:/workspace/scratch/5c1a427569dc/deliverables/algzoo_r6_curve.png)

The low-cost side remains several times worse than sampling in MSE. The two-level correction improves the high-cost side, but its cost rises much faster than the precision required to stay near sampling.

## 7. A concrete restricted lower bound

Classical volume-estimation lower bounds distinguish access through membership or separation queries from access to a geometric description. That distinction is essential here. Bárány–Füredi’s result uses an oracle model and growing dimension; it should not be imported directly as a fixed-size RNN lower bound. genui{"citation":{"ref":"turn2view0"}}

Here is a self-contained fixed-dimension statement.

### Theorem

Let the unknown **error set** be one closed polyhedral cone \(E\subset\mathbb R^{10}\), with unrestricted number of facets and Gaussian mass at most

\[
q_*=\frac{1-\sqrt{1-4(0.069)}}2
=0.074559052276\ldots.
\]

The algorithm receives only Boolean answers to adaptive membership queries. Intermediate computation is free.

Define

\[
G(r)=\Pr\{X_{10}\ge \|X_{1:9}\|/r\}.
\]

For \(\alpha=\arctan r\),

\[
G(r)=
\frac{
35\alpha-28\sin(2\alpha)+7\sin(4\alpha)
-\frac43\sin(6\alpha)+\frac18\sin(8\alpha)
}{35\pi}.
\]

Let \(R=1.901649869866\ldots\) satisfy \(G(R)=q_*\), and put

\[
\theta_N=\left(\frac{256}{175N}\right)^{1/8}.
\]

Any deterministic algorithm making at most \(N\) membership queries has worst-case absolute error at least

\[
\boxed{
L_N=\frac25\left[G(R)-G(R\cos\theta_N)\right].
}
\]

“Worst case” is a supremum: for every \(\eta>0\), an admissible polyhedral cone gives error at least \(L_N-\eta\).

| Queries \(N\) | Absolute-error lower bound, points | Sampling RMS upper bound, points | Squared lower bound / sampling MSE |
|---:|---:|---:|---:|
| 100 | 1.179407 | 2.626785 | 0.202 |
| 1,000 | 0.685273 | 0.830662 | 0.681 |
| 10,000 | 0.391097 | 0.262679 | 2.217 |
| 100,000 | 0.221534 | 0.083066 | 7.113 |
| 1,000,000 | 0.125048 | 0.026268 | 22.662 |
| 10,000,000 | 0.070462 | 0.008307 | 71.955 |

Asymptotically,

\[
L_N=(0.0397228583\ldots+o(1))N^{-1/4}.
\]

Thus worst-case MSE is \(\Omega(N^{-1/2})\), whereas sampling MSE on this class is at most \(0.069/N\). This is a lower bound, not a matching upper-bound theorem.

### Proof

For a compact convex \(B\subset\mathbb R^9\), take its projective cone

\[
C(B)=\{t(z,1):t>0,\ z\in B\}\cup\{0\}.
\]

Changing variables in the Gaussian integral gives

\[
\gamma_{10}(C(B))
=
\frac{12}{\pi^5}
\int_B (1+\|z\|^2)^{-5}\,dz.
\]

For the radius-\(r\) ball \(B_r\), this equals \(G(r)\).

Run the algorithm against membership in the circular cone \(C(B_R)\). This reference cone is not polyhedral, but its finite transcript will be realized by two polyhedral cones.

Project the positive queries to points \(z_i\in B_R\), and define

\[
P_-=\operatorname{conv}\bigl(\{0\}\cup\{z_i\}\bigr).
\]

It agrees with every answer: it contains all positive queried points and, being inside the ball, contains no negative queried point.

A cap of angular radius \(\theta\) on \(S^8\) has normalized area

\[
\frac{35}{32}\int_0^\theta \sin^7 t\,dt
\le \frac{35}{256}\theta^8.
\]

The \(N\) caps centered on the nonzero \(z_i\) directions, with radius \(\theta_N\), therefore cover at most \(1/5\) of the sphere.

For the remaining \(4/5\) of directions \(u\),

\[
\sup_{z\in P_-}u\cdot z\le R\cos\theta_N.
\]

Hence the radial shell from \(R\cos\theta_N\) to \(R\) is absent from \(P_-\) on at least \(4/5\) of directions. Since the projective density is radial,

\[
\gamma_{10}(C(B_R))-\gamma_{10}(C(P_-))
\ge
\frac45\left[G(R)-G(R\cos\theta_N)\right].
\]

Now form a sufficiently fine inscribed polytope \(P_+\) containing \(P_-\). Its measure can approach that of \(B_R\), while preserving every query answer. Both \(C(P_-)\) and \(C(P_+)\) are legal polyhedral error cones with mass at most \(q_*\).

The algorithm returns the same estimate on both. At least one error is half their mass difference. Taking the approximation error to zero gives \(L_N\).

The proof also covers **whole label-only affine-segment traces**: the positive part of a projected segment has at most two endpoints, so replace \(N\) by \(2L\), where \(L\) is the number of queried segments. It does not cover traces revealing logits, affine coefficients, or gate inequalities.

### Why this does not settle the requested problem

A minimax lower bound for a large class containing the RNN integrands does not lower-bound the smaller family:

\[
\inf_A\sup_{f\in\mathcal F_{\rm RNN}}|A(f)-I(f)|
\le
\inf_A\sup_{f\in\mathcal F_{\rm large}}|A(f)-I(f)|.
\]

The inequality goes in the unhelpful direction. A transfer needs hard instances **inside the architecture**, under the actual weight-access cost model. A statement about the average over trained networks needs another argument concerning that population or training distribution.

### Three narrower corrections to “floor” arguments

**ANOVA projection error is not a universal additive accuracy floor.** For independent standard Gaussians \(X,Y,Z\), let

\[
m=|X+Y|-|X-Y|+2|Z|.
\]

Its first-order additive \(L^2\) projection is \(m_A=2|Z|\), so \(\Pr(m_A>0)=1\). But \(m<0\) exactly when \(XY<0\) and \(|Z|\) is the smallest of the three absolute values, giving

\[
\Pr(m>0)=1-\frac12\frac13=\frac56.
\]

The different additive margin

\[
g=X+\sqrt3|Z|
\]

has \(\Pr(g>0)=5/6\) exactly. Thus an additive model can have the correct sign probability while the optimal additive margin projection has a \(1/6\) bias.

This does not furnish an allowed calibrated estimator for these networks. It refutes the inference from `z34` to a floor for every additive method. The narrower statement is valid: if \(g_n\to m_A\) in \(L^2\) and \(\Pr(m_A=0)=0\), then their sign probabilities converge, because

\[
|\Pr(g_n>0)-\Pr(m_A>0)|
\le
\Pr(|m_A|\le\delta)
+\frac{\mathbb E(g_n-m_A)^2}{\delta^2}.
\]

**Uncovered mass is a certificate radius, not necessarily actual error.** If known correct mass is \(a\) and completely unresolved mass is \(u\), the admissible interval is \([a,a+u]\), with optimal worst-case radius \(u/2\). A fallback can nevertheless be accurate on the actual network. Further, `z26`’s tails are measured estimates, not certified masses.

**Nonlocality of a proxy is not a universal accuracy lower bound.** Omitting a known correction contribution gives its signed integral as truncation error. But `z22`’s reported contribution concerns its margin proxy and finite S-grid crossing test. It does not lower-bound every estimator of true RNN accuracy.

## 8. Verdict and the next experiment

The new two-level estimator generalizes to the held-out cohort and materially improves `z38`. It still fails both requested targets: **sampling parity** and **a correction accurate to 0.1 points at no more than ten times the anchor cost**.

Exact correlated state propagation and integration of a compact merged error-cone cover remain open possibilities. The diagnostic measurements do not rule them out.

**The single next experiment I would prioritize is exact S-integration of the two-level correction.**

At each MF4 outer configuration, trace the correctness intervals of all 49 singleton-model streams and the MF3 baseline. Merge every interval endpoint. Between adjacent endpoints, the entire clamped correction is constant, so its S-integral is an exact sum of Gaussian CDF differences.

Clipping must occur **on each common interval**. Integrating the streams separately and then clipping their averages computes a different model.

This experiment includes off-boundary effects and directly isolates the unresolved eight-point S-grid error. I would freeze an eight-development-network pilot and continue only if exact S either:

- Reduces correction RMS by at least **0.10 points**, or
- Reduces charged correction cost by at least **4×**.

The subsequent partial-success gate is held-out correction RMS at most \(0.10\) points with fully counted correction cost at most \(10C_{\rm anchor}\). Full sampling parity remains a separate end-to-end gate. I am not claiming this experiment will bridge the remaining \(220\times\) gap.

## 9. Reproduction and complete inline code

Save the following blocks under their stated filenames in the extracted repository, beside the supplied `lines.py`. The new code uses CPU NumPy/SciPy; Numba is needed for the supplied exact tracer.

The recorded environment was Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Numba 0.68.0, llvmlite 0.50.0, threadpoolctl 3.6.0, and matplotlib 3.10.8.

A compact full-run sequence follows. The manifest bootstrap is sufficient when reconstructing solely from the inline code; the bundle also contains the original timestamped freeze record.

```bash
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMBA_NUM_THREADS=2

python - <<'PY'
import json
from pathlib import Path
p = Path("r6_results")
p.mkdir(exist_ok=True)
settings = []
for n, K, q in [(1, 8, 4), (2, 16, 8)]:
    for keep in [1, 2, 4, 8]:
        settings.append(dict(k=3, n=n, K=K, q=q, keep=keep))
for keep in [4, 8, 56]:
    settings.append(dict(k=4, n=1, K=8, q=8, keep=keep))
settings.append(dict(k=4, n=2, K=8, q=8, keep=8))
f = p / "correction_frozen.json"
if not f.exists():
    f.write_text(json.dumps({
        "settings": settings,
        "note": "Replays the fixed round-6 settings; no new tuning."
    }, indent=2))
PY

python r6_verify.py
python r6_low.py

for split in dev held; do
  python r6_core.py --split "$split" \
    --out "r6_results/anchors_${split}.json"
  python r6_event.py --split "$split" \
    --out "r6_results/event_${split}.json"
  python r6_exact_anchor.py --split "$split" --orders 3,4,6 \
    --out "r6_results/exact_anchor_${split}.json"

  python r6_correction.py --split "$split" --k 3 --n 1 --K 8 --q 4 --keep 8
  python r6_correction.py --split "$split" --k 3 --n 2 --K 16 --q 8 --keep 8
  python r6_correction.py --split "$split" --k 4 --n 1 --K 8 --q 8 --keep 56
  python r6_correction.py --split "$split" --k 4 --n 2 --K 8 --q 8 --keep 8

  # Fixed matched z38 comparator:
  python r6_correction.py --split "$split" --k 3 --n 3 --K 8 --q 8 --keep 8
  python r6_correction.py --split "$split" --report
done

python r6_summary.py
python r6_bounds_minimal.py
python r6_bounds_check.py

# Measurement only; these outputs never enter an estimator:
python r6_correction.py --measure 256
python r6_geometry.py --dev-count 8 --log2-inputs 16

# Optional dense verification of an exact S node:
python r6_exact_anchor.py --split dev --orders 3,4,6 \
  --resume --verify --out r6_results/exact_anchor_dev.json
```

`--report` expects all four correction source runs and both anchor families. The included reproduction guide checks those prerequisites and verifies that it produces 72 combinations per split.

The implementation checks passed: shared-prefix logits matched dense logits exactly; final-input probabilities matched the supplied tracer within \(2.45\times10^{-14}\); the all-zero tie was correct; and forced overflow retries reproduced the unrestricted result while charging all discarded attempts.

### `r6_core.py`

```python
"""CPU deterministic MF3 quadrature with shared prefixes and linear-logit roots.
Run from this repository. All rules are fixed; reference values only score them.
"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, roots_jacobi
from threadpoolctl import threadpool_limits

ARR = np.array(list(itertools.permutations(range(10), 3)))

def load(ids):
    z = np.load('pop1k/seed0.npz')
    keys = ['Wih', 'Whh', 'Wo']
    pop = [z['s1048576_' + k] for k in keys]
    zoo = []
    for f in ['weights.npz'] + [f'weights_seed{i}.npz' for i in range(5)]:
        w = np.load(f)
        zoo.append([w['rnn.weight_ih_l0'][:, 0],
                    w['rnn.weight_hh_l0'], w['linear.weight']])
    return [np.ascontiguousarray(np.stack([
        pop[j][i] if i < 1024 else zoo[i-1024][j] for i in ids
    ]), dtype=np.float64) for j in range(3)]

def rule(n, a=1, b=1):
    x, w = roots_jacobi(n, b-1, a-1)
    return (x+1)/2, w/w.sum()

def forward(W, X):
    wi, wh, wo = W
    h = np.zeros((len(wi), len(X), 16))
    for t in range(10):
        h = np.maximum(h @ wh.transpose(0, 2, 1)
                       + wi[:, None, :] * X[None, :, t, None], 0)
    return h @ wo.transpose(0, 2, 1)

def prefix_forward(W, tokens, values):
    """Evaluate exactly the same rows, deduplicating identical input prefixes.
    Returns logits and actual dense multiply/add work, in 2880-MAC units.
    Also charges scalar adds and ReLU comparisons, unlike the base MAC count.
    """
    wi, wh, wo = W
    drive = wi[:, None, :] * values[None, :, None]
    h = np.zeros((len(wi), 1, 16))
    inv = np.zeros(len(tokens), dtype=int)
    work = 16*len(values)
    for t in range(10):
        edges, inv = np.unique(np.column_stack([inv, tokens[:, t]]),
                               axis=0, return_inverse=True)
        if t == 0:
            h = np.maximum(drive[:, edges[:, 1]], 0)
            work += len(edges)*16
        else:
            h = np.maximum(h[:, edges[:, 0]] @ wh.transpose(0, 2, 1)
                           + drive[:, edges[:, 1]], 0)
            work += len(edges)*(256+32)
    L = h @ wo.transpose(0, 2, 1)
    work += len(h[0])*160
    return L[:, inv], work/2880

def line_probability(L, s, y):
    """Gaussian measure of correctness for logits linearly interpolated in x_S.
    The integral of this interpolant is exact; the network need not be affine.
    """
    target = np.take_along_axis(L, y[None, :, None, None], axis=-1)
    d = target-L
    d0, dd = d[:, :, :-1], np.diff(d, axis=2)
    nz = dd != 0
    r = np.divide(-d0, dd, out=np.zeros_like(dd), where=nz)
    low = np.maximum(0, np.max(np.where(dd > 0, r, -np.inf), axis=-1))
    high = np.minimum(1, np.min(np.where(dd < 0, r, np.inf), axis=-1))
    bad = np.any((~nz) & ((d0 < 0) | ((d0 == 0) &
        (np.arange(10)[None, None, None, :] < y[None, :, None, None]))), axis=-1)
    low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
    a = s[:-1]+np.diff(s)*low
    b = s[:-1]+np.diff(s)*high
    mass = np.where((high > low) & ~bad, ndtr(b)-ndtr(a), 0)
    return mass.sum(axis=-1)/(ndtr(s[-1])-ndtr(s[0]))

def mf3(W, n, K):
    """n x n Beta outer rule; K affine interpolation intervals along S."""
    est = np.zeros(len(W[0])); cost = 0.0
    ur, wr = rule(n, 8, 3); ug, wg = rule(n, 2, 1)
    for a, wa in zip(ur, wr):
        for g, w in zip(ug, wg):
            b = a+(1-a)*g
            R, M = ndtri(a), ndtri(b)
            s = ndtri(a+(b-a)*np.arange(K+1)/K)
            mu = -np.exp(-R*R/2)/np.sqrt(2*np.pi)/a
            # symbols 0=M, 1=R, 2=mu; interior S nodes have distinct symbols.
            vals = np.r_[M, R, mu, s[1:-1]]
            st = np.r_[1, np.arange(3, K+2), 0]
            tok = np.full((720, K+1, 10), 2, dtype=int)
            for j in range(720):
                tok[j, :, ARR[j, 0]] = 0
                tok[j, :, ARR[j, 1]] = st
                tok[j, :, ARR[j, 2]] = 1
            L, c = prefix_forward(W, tok.reshape(-1, 10), vals)
            p = line_probability(L.reshape(len(W[0]), 720, K+1, 10),
                                 s, ARR[:, 1])
            est += wa*w*p.mean(axis=1)
            # 200 scalar-equivalent operations/interval: margins, roots,
            # reductions, masks and both normal CDFs. Report separately.
            cost += c+720*K*200/2880
    return est, cost

def score(est, ids, cost):
    ref = np.load('pop1k/reference.npz')
    p = ref['acc'][ids]; err = (est-p)*100
    ans = {'cost': float(cost), 'ids': list(map(int, ids)),
           'estimates': est.tolist(), 'errors_pp': err.tolist()}
    for name, mask in [('dev', ids < 1024),
                       ('held', (ids < 1024) & (ids % 16 == 8))]:
        if name == 'dev': mask &= ids % 16 == 0
        if mask.any():
            e = err[mask]; samp = 100*np.sqrt(np.mean(p[mask]*(1-p[mask]))/cost)
            ans[name] = dict(rms=float(np.sqrt(np.mean(e*e))),
                             mean=float(e.mean()), sampling=float(samp),
                             ratio=float(np.mean(e*e)/samp**2))
    ans['zoo_errors'] = err[ids >= 1024].tolist()
    return ans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['dev', 'held', 'all'], default='dev')
    ap.add_argument('--settings', default='1:1,1:2,2:4,3:8,4:16,6:32')
    ap.add_argument('--out', default='r6_results/anchors_dev.json')
    a = ap.parse_args()
    ids = np.r_[np.arange(0, 1024, 16) if a.split == 'dev' else
                np.arange(8, 1024, 16) if a.split == 'held' else
                np.sort(np.r_[np.arange(0, 1024, 16), np.arange(8, 1024, 16)]),
                np.arange(1024, 1030)]
    W = load(ids); results = []
    for n, K in [tuple(map(int, s.split(':'))) for s in a.settings.split(',')]:
        start = time.time(); est, cost = mf3(W, n, K)
        row = score(est, ids, cost); row.update(n=n, K=K)
        results.append(row)
        Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(n, K, 'seconds', round(time.time()-start, 1),
              {k: v for k, v in row.items() if k not in
               ('ids', 'estimates', 'errors_pp')}, flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
```

### `r6_low.py`

```python
"""Small MF2 product-Beta rules, evaluated with actual shared-prefix work."""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from threadpoolctl import threadpool_limits
from r6_core import load, rule, prefix_forward, score

def mf2(W, n, q):
    arr = np.array(list(itertools.permutations(range(10), 2)))
    um, wm = rule(n, 10, 1); us, ws = rule(q, 9, 1)
    est = np.zeros(len(W[0])); cost = 0.0
    for a, weight in zip(um, wm):
        M, S = ndtri(a), ndtri(a*us)
        mu = -np.exp(-S*S/2)/np.sqrt(2*np.pi)/(a*us)
        vals = np.r_[M, S, mu]
        tok = np.tile((1+q+np.arange(q))[None, :, None], (90, 1, 10))
        for j, (mp, sp) in enumerate(arr):
            tok[j, :, mp] = 0; tok[j, :, sp] = 1+np.arange(q)
        L, c = prefix_forward(W, tok.reshape(-1, 10), vals)
        I = L.reshape(len(W[0]), 90, q, 10).argmax(-1) == arr[None, :, None, 1]
        est += weight*(I*ws).sum(-1).mean(1)
        cost += c+12*90*q/2880
    return est, cost

if __name__ == '__main__':
    # The same six rules are fixed before evaluating either split.
    with threadpool_limits(limits=2):
        for split, offset in [('dev',0),('held',8)]:
            ids = np.r_[np.arange(offset,1024,16),np.arange(1024,1030)]
            W = load(ids); out = []
            for n,q in [(1,1),(1,2),(2,2),(2,4),(3,4),(4,8)]:
                est,cost = mf2(W,n,q); r = score(est,ids,cost)
                r.update(n=n,q=q); out.append(r)
                print(split,n,q,{k:v for k,v in r.items() if k not in ('ids','estimates','errors_pp')},flush=True)
            Path(f'r6_results/low_{split}.json').write_text(json.dumps(out,indent=2))
```

### `r6_event.py`

```python
"""Deterministic event-branch quadrature, optional age/value-bin mean merging.
The ninth-step mixture is integrated exactly over the final Gaussian input.
This is a point/mean closure, not an exact affine-component distribution law.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits
from r6_core import load, rule, score

def phi(x):
    return np.exp(-x*x/2)/np.sqrt(2*np.pi)

def last_input(W, h, M, S, ip, jp):
    """Exact gate sweep with lowest-index output ties. No random numbers."""
    probs = np.zeros(h.shape[:2]); B = len(M); rows = np.arange(B)
    for n, (wi, wh, wo) in enumerate(zip(*W)):
        a = h[n] @ wh.T
        gates = np.divide(-a, wi, out=np.full_like(a, np.inf), where=wi != 0)
        bounds = np.column_stack([gates, S, M])
        order = np.argsort(bounds, axis=1)
        bounds = np.take_along_axis(bounds, order, axis=1)
        on = (wi < 0) | ((wi == 0) & (a > 0))
        c = (a*on) @ wo.T
        d = (wi*on) @ wo.T
        low = np.full(B, -np.inf); total = np.zeros(B)
        for k in range(19):
            high = bounds[:, k] if k < 18 else np.full(B, np.inf)
            tgt = np.where(low >= M, ip, np.where(low >= S, 9, jp))
            dc = c[rows, tgt, None]-c
            dd = d[rows, tgt, None]-d
            roots = np.divide(-dc, dd, out=np.zeros_like(dd), where=dd != 0)
            l = np.maximum(low, np.max(np.where(dd > 0, roots, -np.inf), axis=1))
            u = np.minimum(high, np.min(np.where(dd < 0, roots, np.inf), axis=1))
            bad = np.any((dd == 0) & ((dc < 0) | ((dc == 0) &
                          (np.arange(10)[None, :] < tgt[:, None]))), axis=1)
            total += np.where((u > l) & ~bad, ndtr(u)-ndtr(l), 0)
            if k < 18:
                gate = order[:, k]; j = np.minimum(gate, 15)
                sign = np.where(gate < 16, np.sign(wi[j]), 0)
                c += (sign*a[rows, j])[:, None]*wo[:, j].T
                d += (sign*wi[j])[:, None]*wo[:, j].T
                flip = (gate < 16) & (wi[j] != 0)
                on[rows[flip], j[flip]] ^= True
                # Subtracting gate contributions can leave roundoff when the
                # exact hidden state is zero. Restore the exact all-zero tie.
                empty = ~on.any(axis=1)
                c[empty] = 0; d[empty] = 0
            low = high
        probs[n] = total
    # Includes final recurrent map, both initial readouts, gate sorting/updates,
    # and 200 scalar-equivalents per root/CDF interval; conservatively charged.
    cost = B*(256+320+16+128+18*(40+20)+19*200)/2880
    return probs, cost

def merge(h, M, S, ip, jp, weights, bins):
    if bins == 0:
        return h, M, S, ip, jp, weights, 0.0
    keys = np.column_stack([ip, jp, np.minimum(bins-1, (bins*ndtr(M)).astype(int)),
                             np.minimum(bins-1, (bins*ndtr(S)).astype(int))])
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    ix = np.argsort(inv, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(inv[ix]))+1]
    total = np.add.reduceat(weights[ix], starts)
    merged = np.add.reduceat(h[:, ix]*weights[None, ix, None], starts, axis=1)
    merged /= total[None, :, None]
    avg = lambda x: np.add.reduceat(x[ix]*weights[ix], starts)/total
    first = ix[starts]
    cost = (32*len(weights)+16*len(total))/2880
    return merged, avg(M), avg(S), ip[first], jp[first], total, cost

def event(W, q=1, bins=0):
    """q=1 uses exact truncated means; q>1 Gauss-Legendre in conditional CDF.
    bins=0 retains every component. Positive bins merge joint hidden vectors
    by the two positions and CDF bins of M,S, keeping the entire vector mean.
    """
    nodes, weights = rule(q)
    u = np.r_[nodes/2, .5+nodes/2]
    x = np.r_[-np.sqrt(2/np.pi), np.sqrt(2/np.pi)] if q == 1 else ndtri(u)
    mass = np.tile(weights/2, 2)
    wi, wh, wo = W
    h = np.maximum(wi[:, None, :]*x[None, :, None], 0)
    M = x.copy(); S = np.full(len(x), -np.inf)
    ip = np.zeros(len(x), dtype=int); jp = np.full(len(x), -1, dtype=int)
    cost = len(x)*32/2880; counts = [len(x)]
    for t in range(1, 9):
        bounds = np.column_stack([np.zeros(len(M)), ndtr(S), ndtr(M), np.ones(len(M))])
        widths = np.diff(bounds, axis=1)
        parent, branch = np.where(widths > 1e-15)
        left = bounds[parent, branch]; width = widths[parent, branch]
        if q == 1:
            z = (phi(ndtri(left))-phi(ndtri(left+width)))/width
            w = mass[parent]*width
        else:
            z = ndtri(left[:, None]+width[:, None]*nodes).ravel()
            w = (mass[parent, None]*width[:, None]*weights).ravel()
            parent = np.repeat(parent, q); branch = np.repeat(branch, q)
        h = np.maximum(h[:, parent] @ wh.transpose(0, 2, 1)
                       + wi[:, None, :]*z[None, :, None], 0)
        cost += len(z)*(272+32)/2880
        newmax = branch == 2; newsecond = branch == 1
        S = np.where(newmax, M[parent], np.where(newsecond, z, S[parent]))
        M = np.maximum(M[parent], z)
        jp = np.where(newmax, ip[parent], np.where(newsecond, t, jp[parent]))
        ip = np.where(newmax, t, ip[parent]); mass = w
        h, M, S, ip, jp, mass, c = merge(h, M, S, ip, jp, mass, bins)
        cost += c; counts.append(len(M))
    p, c = last_input(W, h, M, S, ip, jp)
    assert abs(mass.sum()-1) < 1e-10
    return p@mass, cost+c, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['dev', 'held'], default='dev')
    ap.add_argument('--settings', default='1:1,1:2,1:4,1:8,1:0,2:2,2:4,2:8')
    ap.add_argument('--out', default='r6_results/event_dev.json')
    a = ap.parse_args()
    ids = np.r_[np.arange(0 if a.split == 'dev' else 8, 1024, 16), np.arange(1024, 1030)]
    W = load(ids); out = []
    for q, bins in [tuple(map(int, s.split(':'))) for s in a.settings.split(',')]:
        start = time.time(); est, cost, counts = event(W, q, bins)
        r = score(est, ids, cost); r.update(q=q, bins=bins, counts=counts)
        out.append(r); Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(q, bins, 'seconds', round(time.time()-start, 1),
              {k:v for k,v in r.items() if k not in ('ids','estimates','errors_pp')}, flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
```

### `r6_exact_anchor.py`

```python
"""MF3 anchor: tensor Beta quadrature and exact Gaussian S-line integration.

Uses the supplied lines.trace_line; reports its measured matrix/readout MAC
count (2,880 MACs/forward), a lower bound excluding roots, sorting and CDF work.
No assumed prefix-tree acceleration. Failed overflow attempts are also charged.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import json
import time
from pathlib import Path
import numpy as np
import numba
from scipy.special import ndtri
from lines import trace_line
from r6_core import ARR, load, rule, forward, score


def nodes(n):
    ur, wr = rule(n,8,3)
    ug, wg = rule(n,2,1)
    out = []
    for a,wa in zip(ur,wr):
        for g,wb in zip(ug,wg):
            b = a+(1-a)*g
            R,M = ndtri(a),ndtri(b)
            mu = -np.exp(-R*R/2)/np.sqrt(2*np.pi)/a
            x = np.full((len(ARR),10),mu)
            x[np.arange(len(ARR)),ARR[:,0]] = M
            x[np.arange(len(ARR)),ARR[:,1]] = R
            x[np.arange(len(ARR)),ARR[:,2]] = R
            out.append((x,R,M,float(wa*wb),float(a),float(b)))
    return out


def exact_node(W,x,R,M,cap=512):
    remaining = np.arange(len(x))
    prob = np.empty(len(x))
    total_cost = 0.0
    discarded = 0
    p = np.ascontiguousarray(ARR[:,1],dtype=np.int64)
    while len(remaining):
        pp,cc,overflow = trace_line(*W,x[remaining],p[remaining],
            np.full(len(remaining),R),np.full(len(remaining),M),p[remaining],cap)
        total_cost += float(cc.sum())
        good = overflow == 0
        if np.any((pp[good] < -1e-10) | (pp[good] > 1+1e-10)):
            raise ArithmeticError("Invalid exact line probability")
        prob[remaining[good]] = pp[good]
        remaining = remaining[~good]
        discarded += len(remaining)
        if len(remaining):
            cap *= 2
    return prob,total_cost,discarded,cap


def mf3_exact(W,n,cap=512):
    est,cost,discarded,maxcap = 0.0,0.0,0,cap
    for x,R,M,w,a,b in nodes(n):
        p,c,d,used = exact_node(W,x,R,M,cap)
        est += w*p.mean()
        cost += c
        discarded += d
        maxcap = max(maxcap,used)
    return est,cost,discarded,maxcap


def verify(W,n=3,K=4096,cap=512):
    """One central outer node, all720 lines, checked against a dense CDF grid."""
    x,R,M,w,a,b = nodes(n)[n*n//2]
    exact,c,d,used = exact_node(W,x,R,M,cap)
    s = ndtri(a+(b-a)*(np.arange(K)+.5)/K)
    dense = np.empty(len(ARR))
    for start in range(0,len(ARR),8):
        part = np.arange(start,min(start+8,len(ARR)))
        xx = np.repeat(x[part,None,:],K,axis=1)
        xx[np.arange(len(part)),:,ARR[part,1]] = s
        logits = forward([v[None] for v in W],xx.reshape(-1,10))[0]
        dense[part] = (logits.argmax(1).reshape(len(part),K)==ARR[part,1,None]).mean(1)
    return dict(n=n,K=K,lines=len(ARR),mean_error_pp=float(100*(dense-exact).mean()),
                max_abs_error_pp=float(100*np.max(np.abs(dense-exact))),
                rms_error_pp=float(100*np.sqrt(np.mean((dense-exact)**2))),
                exact_node_mac_forwards=c,discarded_lines=d,maxcap=used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",choices=("dev","held"),default="dev")
    ap.add_argument("--orders",default="3,4")
    ap.add_argument("--dev-count",type=int,default=64)
    ap.add_argument("--maxp",type=int,default=512)
    ap.add_argument("--verify",action="store_true")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--out",default="r6_results/exact_anchor_dev.json")
    a = ap.parse_args()
    pop = np.arange(0 if a.split=="dev" else 8,1024,16)[:a.dev_count]
    ids = np.r_[pop,np.arange(1024,1030)]
    W = load(ids)
    result = []
    output = Path(a.out)
    output.parent.mkdir(exist_ok=True)
    if a.resume and output.exists():
        result = json.loads(output.read_text())
        if any(row["ids"] != ids.tolist() for row in result):
            raise ValueError("Cannot resume a different network cohort")
    if a.verify:
        v = verify([w[0] for w in W],cap=a.maxp)
        output.with_suffix(".verification.json").write_text(json.dumps(v,indent=2))
        print("verification",v,flush=True)
    for n in map(int,a.orders.split(",")):
        if any(row["n"]==n for row in result):
            continue
        start = time.time()
        est,cost,discarded,maxcaps = [],[],[],[]
        for i in range(len(ids)):
            e,c,d,m = mf3_exact([w[i] for w in W],n,a.maxp)
            est.append(e); cost.append(c); discarded.append(d); maxcaps.append(m)
            if i%8==0:
                print("order",n,"network",int(ids[i]),"MAC forwards",round(c,1),
                      "seconds",round(time.time()-start,1),flush=True)
        est,cost = np.array(est),np.array(cost)
        row = score(est,ids,float(cost[:len(pop)].mean()))
        row.update(n=n,cost_is_lower_bound=True,cost_definition=__doc__,
                   costs_per_network=cost.tolist(),discarded_lines=discarded,maxcaps=maxcaps,
                   seconds=time.time()-start)
        ref = np.load("pop1k/reference.npz")
        pq = ref["acc"][ids]*(1-ref["acc"][ids])
        sample = float(100*np.sqrt(np.mean(pq[:len(pop)]/cost[:len(pop)])))
        metric = row[a.split]
        metric.update(sampling_at_network_costs=sample,
                      ratio_at_network_costs=metric["rms"]**2/sample**2,
                      cost_min=float(cost[:len(pop)].min()),cost_max=float(cost[:len(pop)].max()),
                      anchor_rms_pp=float(100*np.sqrt(np.mean((est[:len(pop)]-ref["mf3"][pop])**2))))
        result.append(row)
        output.write_text(json.dumps(result,indent=2))
        print("completed",n,a.split,metric,flush=True)


if __name__ == "__main__":
    numba.set_num_threads(2)
    main()
```

### `r6_correction.py`

```python
"""Deterministic two-level MF3/MF4 correction; RNG is confined to measure().
Run from the repository, with OPENBLAS_NUM_THREADS=2. All forward rows charged.
Examples: python r6_correction.py --measure 256
 python r6_correction.py --split dev --k 3 --n 2 --K 16 --q 8 --keep 8
 python r6_correction.py --split held --k 4 --n 2 --K 8 --q 8 --keep 8
"""
import argparse, itertools, json, os, pathlib, time
import numpy as np
from scipy.special import ndtr, ndtri, roots_jacobi

OUT = pathlib.Path('r6_results'); OUT.mkdir(exist_ok=True)

def save_npz(path, **data):
    """Atomic replacement also keeps any open lazy NPZ reader valid."""
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**data)
    tmp.replace(path)

def load(split='dev'):
    ids = np.r_[np.arange(8 if split == 'held' else 0, 1024, 16), 1024:1030]
    p = np.load('pop1k/seed0.npz')
    z = [np.load(f) for f in ['weights.npz']+[f'weights_seed{i}.npz' for i in range(5)]]
    zz = [np.stack([v[key][:, 0] if j == 0 else v[key] for v in z]) for j, key in
          enumerate(['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'])]
    w = [np.concatenate([p['s1048576_'+key][ids[:-6]], b]).astype(np.float32) for key, b in zip(['Wih','Whh','Wo'],zz)]
    return ids, w

def predict(w, x):
    h = np.zeros((len(w[0]), len(x), 16), dtype=w[0].dtype)
    for t in range(10): h = np.maximum(h @ w[1].transpose(0,2,1) + x[None,:,t,None]*w[0][:,None], 0)
    return (h @ w[2].transpose(0,2,1)).argmax(-1)

def rule(n, a, b):
    x, w = roots_jacobi(n, b-1, a-1)
    return (x+1)/2, w/w.sum()

def mu(v): return -np.exp(-np.asarray(v)**2/2)/np.sqrt(2*np.pi)/ndtr(v)

def arrangements(k, keep):
    a = np.array(list(itertools.permutations(range(10), k)))
    if keep == 0 or keep == (8 if k == 3 else 56): return a
    chosen = []
    for row in a:
        f = [x for x in range(10) if x not in row[:2]]
        if k == 3: yes = f.index(row[2]) in {1:[3],2:[1,6],4:[0,2,5,7]}[keep]
        else:
            pairs = [(0,7),(7,0),(1,3),(3,1),(2,5),(5,2),(4,6),(6,4)]
            yes = (f.index(row[2]),f.index(row[3])) in pairs[:keep]
        if yes: chosen.append(row)
    return np.array(chosen)

def grid(w, k, n, K, q, keep):
    """Returns per-arrangement integrals: anchor3, anchor-k, corrected model.
    CDF Gauss-Jacobi outer rule, S midpoints, q Gauss-Legendre lower nodes.
    For k=4 the separately evaluated MF3 anchor allows MF4-model minus MF3.
    """
    arr = arrangements(k, keep); A = len(arr); N = len(w[0]); low = np.array([[i for i in range(10) if i not in a] for a in arr])
    rules = [rule(n,8,3),rule(n,2,1)] if k == 3 else [rule(n,7,4),rule(n,1,3),rule(n,2,1)]
    uq, wq = rule(q,1,1); out = np.zeros((N,A,3)); rows = 1+(10-k)*q+(k==4)
    block = max(1,4096//(K*rows)); count = 0
    for ij in itertools.product(range(n), repeat=len(rules)):
        u = [rules[t][0][j] for t,j in enumerate(ij)]; wt = np.prod([rules[t][1][j] for t,j in enumerate(ij)])
        if k == 3: fr = u[0]; fm = fr+(1-fr)*u[1]; vals = [ndtri(fm),None,ndtri(fr)]
        else: fq=u[0]; fr=fq+(1-fq)*u[1]; fm=fr+(1-fr)*u[2]; vals=[ndtri(fm),None,ndtri(fr),ndtri(fq)]
        s = ndtri(fr+(fm-fr)*(np.arange(K)+.5)/K); zn = ndtri(ndtr(vals[-1])*uq)
        for start in range(0,A,block):
            aa = arr[start:start+block]; ll = low[start:start+block]; b=len(aa)
            base = np.full((b,K,10),mu(vals[-1]),np.float32)
            for j,v in enumerate(vals): base[np.arange(b)[:,None],np.arange(K),aa[:,j,None]] = s if j==1 else v
            X = np.repeat(base[:,:,None,:],rows,axis=2)
            for j in range(10-k): X[np.arange(b)[:,None,None],np.arange(K)[None,:,None],1+j*q+np.arange(q),ll[:,j,None,None]] = zn
            if k == 4:
                X[:,:,-1] = mu(vals[2])
                for j in range(3): X[np.arange(b)[:,None],np.arange(K),-1,aa[:,j,None]] = s if j==1 else vals[j]
            I = (predict(w,X.reshape(-1,10)).reshape(N,b,K,rows)==aa[None,:,None,None,1]).astype(float)
            i0=I[...,0]; ej=(I[...,1:1+(10-k)*q].reshape(N,b,K,10-k,q)*wq).sum(-1)
            model=np.clip(i0+(ej-i0[...,None]).sum(-1),0,1)
            out[:,start:start+b,0]+=wt*(I[...,-1] if k==4 else i0).mean(-1)
            out[:,start:start+b,1]+=wt*i0.mean(-1); out[:,start:start+b,2]+=wt*model.mean(-1)
            count += b*K*rows
    return arr,out,count

def score(ids, est, rows, target='acc'):
    p=np.load('pop1k/reference.npz'); e=100*(est-p[target][ids]); pop=e[:-6]
    samp=100*np.sqrt(np.mean(p['acc'][ids[:-6]]*(1-p['acc'][ids[:-6]]))/(rows*1.01))
    return dict(forward_rows=int(rows),cost=rows*1.01,rms=float(np.sqrt(np.mean(pop**2))),mean=float(pop.mean()),
                sampling_rms=float(samp),mse_ratio=float(np.mean(pop**2)/samp**2),zoo=e[-6:].tolist())

def measure(reps):
    """Balanced independent measurement of true-minus-anchor arrangement means.
    This function never constructs an estimator; finite measurement noise is subtracted.
    """
    ids,w=load(); arr=arrangements(4,0); N=len(ids); A=len(arr); rng=np.random.default_rng(6007)
    sums=np.zeros((2,N,A)); squares=np.zeros_like(sums); parts=np.zeros((4,2,N,A))
    block=max(1,4096//reps); t0=time.time()
    for st in range(0,A,block):
        aa=arr[st:st+block]; b=len(aa); a=np.repeat(aa,reps,axis=0); B=len(a)
        v=np.sort(rng.standard_normal((B,10)),axis=1)[:,::-1][:,:4]
        lp=np.array([[p for p in range(10) if p not in row] for row in a]); x=np.zeros((B,10),np.float32)
        x[np.arange(B)[:,None],lp]=ndtri(ndtr(v[:,3,None])*rng.random((B,6)))
        x[np.arange(B)[:,None],a]=v; y=a[:,1]; truth=predict(w,x)==y
        for j,k in enumerate((3,4)):
            base=np.repeat(mu(v[:,k-1,None]),10,axis=1).astype(np.float32); base[np.arange(B)[:,None],a[:,:k]]=v[:,:k]
            d=(truth.astype(np.int8)-(predict(w,base)==y)).reshape(N,b,reps)
            sums[j,:,st:st+b]=d.sum(-1); squares[j,:,st:st+b]=(d*d).sum(-1)
            for h,pp in enumerate(np.array_split(np.arange(reps),4)): parts[h,j,:,st:st+b]=d[...,pp].sum(-1)
        if st%(block*80)==0: print('measurement',st,A,round(time.time()-t0,1),flush=True)
    result={}
    for j,k in enumerate((3,4)):
        groups=np.array([tuple(row[:k]) for row in arr]); _,inv=np.unique(groups,axis=0,return_inverse=True)
        G=inv.max()+1; s=np.zeros((N,G)); ss=np.zeros_like(s); nr=np.bincount(inv)*reps
        for i in range(N): s[i]=np.bincount(inv,weights=sums[j,i]); ss[i]=np.bincount(inv,weights=squares[j,i])
        means=s/nr; rawnoise=(squares[j]-sums[j]**2/reps)/(reps-1)/reps
        noise=np.stack([np.bincount(inv,weights=v) for v in rawnoise])/(nr/reps)**2
        var=means.var(1)-(1-1/G)*noise.mean(1)
        result[f'mf{k}']=dict(sd_points=(100*np.sqrt(np.maximum(var,0))).tolist(),raw_sd_points=(100*means.std(1)).tolist(),
            per_arrangement_se_points=(100*np.sqrt(noise.mean(1))).tolist(),correction_mean_points=(100*means.mean(1)).tolist(),arrangements=int(G),draws_per_arrangement=int(nr[0]))
    save_npz(OUT/'correction_measurement.npz',ids=ids,arr=arr,sums=sums,squares=squares,parts=parts,reps=reps)
    (OUT/'correction_measurement.json').write_text(json.dumps(result,indent=2)); arrangement_report()

def arrangement_report():
    """Describe the measurement only; noise-corrected subset errors are not estimates."""
    z=np.load(OUT/'correction_measurement.npz'); a=z['arr']; reps=int(z['reps']); result={}
    for j,k,label in [(0,3,'mf3'),(1,4,'mf4'),(0,4,'mf4_arr_minus_mf3')]:
        arr,inv=np.unique(a[:,:k],axis=0,return_inverse=True); G=len(arr); N=len(z['ids']); nr=np.bincount(inv)*reps
        s=np.stack([np.bincount(inv,weights=v,minlength=G) for v in z['sums'][j]])
        ss=np.stack([np.bincount(inv,weights=v,minlength=G) for v in z['squares'][j]])
        m=s/nr; rawnoise=(z['squares'][j]-z['sums'][j]**2/reps)/(reps-1)/reps
        noise=np.stack([np.bincount(inv,weights=v) for v in rawnoise])/(nr/reps)**2
        var=m.var(1)-(1-1/G)*noise.mean(1)
        r=dict(sd_points=(100*np.sqrt(np.maximum(var,0))).tolist(),classes={},subsets={})
        ms=10*arr[:,0]+arr[:,1]; rel=ms.copy()
        for col in range(2,k): rel=4*rel+2*(arr[:,col]>arr[:,0])+(arr[:,col]>arr[:,1])
        for class_label,gg in [('MS',ms),('MS_relative_order',rel)]:
            _,ii=np.unique(gg,return_inverse=True); cnt=np.bincount(ii); cm=np.stack([np.bincount(ii,weights=v)/cnt for v in m])
            residual=((m-cm[:,ii])**2).mean(1)-(noise*(1-1/cnt[ii])).mean(1)
            r['classes'][class_label]=dict(count=len(cnt),variance_share=(1-residual/var).tolist(),residual_sd_points=(100*np.sqrt(np.maximum(residual,0))).tolist())
        for keep in ([1,2,4] if k==3 else [4,8]):
            aa={tuple(row) for row in arrangements(k,keep)}; mask=np.array([tuple(row) in aa for row in arr]); w=mask/mask.sum()-1/G
            e=100*(m@w); nv=10000*(noise@(w*w)); rms=np.sqrt(max(np.mean((e*e-nv)[:-6]),0))
            r['subsets'][str(keep)]=dict(arrangements=int(mask.sum()),noise_corrected_rms=float(rms),raw_rms=float(np.sqrt(np.mean(e[:-6]**2))),
                mean=float(e[:-6].mean()),per_network_measurement_se=float(np.sqrt(nv[:-6].mean())),zoo=e[-6:].tolist(),zoo_se=np.sqrt(nv[-6:]).tolist())
        result[label]=r
    (OUT/'correction_arrangement_report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:dict(median_sd_points=float(np.median(v['sd_points'][:-6])),subsets=v['subsets']) for k,v in result.items()},indent=2))

def reports(split):
    """Read fixed subset rules from already evaluated arrangement integrals.
    Charge only rows required when that declared rule is deployed, plus actual anchor cost.
    """
    ref=np.load('pop1k/reference.npz'); corrections={}; output=[]
    for cfg in json.load(open(OUT/'correction_frozen.json'))['settings']:
        k,n,K,q,keep=[cfg[a] for a in ['k','n','K','q','keep']]
        stem=f'correction_{split}_k{k}_n{n}_K{K}_q{q}_keep'
        source=OUT/(stem+str(8 if k==3 or n==2 else 56)+'.npz')
        if not source.exists(): continue
        z=np.load(source); aa={tuple(x) for x in arrangements(k,keep)}; mask=np.array([tuple(x) in aa for x in z['arr']])
        ids=z['ids']; v=z['integrals'][:,mask]; rows=int(z['rows'])*mask.sum()//len(mask); es=v.mean(1); delta=es[:,2]-es[:,0]; delta4=es[:,2]-es[:,1]
        name=stem+str(keep); corrections[name]=(ids,delta,rows)
        save_npz(OUT/(name+'.npz'),ids=ids,arr=z['arr'][mask],integrals=v,rows=rows,delta=delta,delta4=delta4)
        rr=dict(settings=cfg,same_grid=score(ids,es[:,2],rows),correction_with_reference_anchor=score(ids,ref['mf3'][ids]+delta,rows))
        if k==4: rr['correction_with_reference_mf4']=score(ids,ref['mf4'][ids]+delta4,rows)
        (OUT/(name+'.json')).write_text(json.dumps(rr,indent=2))
    for af in ['anchors','exact_anchor']:
        f=OUT/(af+'_'+split+'.json')
        if not f.exists(): continue
        for a in json.load(open(f)):
            if a['n'] not in (3,4,6): continue
            for name,(ids,delta,rows) in corrections.items():
                assert np.array_equal(ids,a['ids'])
                costs=np.asarray(a.get('costs_per_network',np.full(len(ids),a['cost'])))+rows*1.01
                e=100*(np.array(a['estimates'])+delta-ref['acc'][ids]); pq=ref['acc'][ids]*(1-ref['acc'][ids])
                sm=100*np.sqrt(np.mean((pq/costs)[:-6])); rms=float(np.sqrt(np.mean(e[:-6]**2)))
                output.append(dict(anchor=af,n=a['n'],K=a.get('K'),correction=name,cost=float(costs[:-6].mean()),cost_is_lower_bound=False,
                    cost_min=float(costs[:-6].min()),cost_max=float(costs[:-6].max()),rms=rms,mean=float(e[:-6].mean()),
                    sampling_rms=float(sm),mse_ratio=float(rms*rms/sm**2),zoo=e[-6:].tolist(),ids=ids.tolist(),errors_pp=e.tolist()))
                output[-1]['exact_anchor_omits_scalar_work']=a.get('cost_is_lower_bound',False)
                if a.get('cost_is_lower_bound',False):
                    mac=np.asarray(a['costs_per_network'])+rows
                    ms=100*np.sqrt(np.mean((pq/mac)[:-6]))
                    output[-1].update(mac_only_cost_lower_bound=float(mac[:-6].mean()),
                        sampling_rms_at_mac_cost=float(ms),mse_ratio_at_mac_cost=float(rms*rms/ms**2))
    (OUT/('correction_combinations_'+split+'.json')).write_text(json.dumps(output,indent=2))
    print(split,'combinations saved:',len(output))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--measure',type=int,default=0); p.add_argument('--report',action='store_true'); p.add_argument('--split',default='dev',choices=['dev','held'])
    for a,d in [('k',3),('n',1),('K',8),('q',4),('keep',8)]: p.add_argument('--'+a,type=int,default=d)
    a=p.parse_args()
    if a.measure: measure(a.measure); return
    if a.report: reports(a.split); return
    ids,w=load(a.split); t=time.time(); arr,v,rows=grid(w,a.k,a.n,a.K,a.q,a.keep)
    est=v.mean(1); ref=np.load('pop1k/reference.npz'); delta=est[:,2]-est[:,0]
    report=dict(settings=vars(a),seconds=time.time()-t,arrangements=len(arr),same_grid=score(ids,est[:,2],rows),
                correction_with_reference_anchor=score(ids,ref['mf3'][ids]+delta,rows))
    delta4=est[:,2]-est[:,1]
    if a.k==4: report['correction_with_reference_mf4']=score(ids,ref['mf4'][ids]+delta4,rows)
    name=f'correction_{a.split}_k{a.k}_n{a.n}_K{a.K}_q{a.q}_keep{a.keep}'
    save_npz(OUT/(name+'.npz'),ids=ids,arr=arr,integrals=v,rows=rows,delta=delta,delta4=delta4)
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
```

### `r6_geometry.py`

```python
"""Measurement-only gate-history diagnostic; not an accuracy estimator.

The first half of independent iid Gaussian inputs discovers patterns; the second
half measures mass coverage. Error fragments also fix top-two and output indices,
so they form a particular convex-cone partition, not maximal convex error pieces.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import json
from pathlib import Path
import numpy as np


def keys(gates, group=None):
    """Opaque exact gate bitstrings, optionally prefixed by a grouping label."""
    raw = np.ascontiguousarray(gates).view(np.dtype((np.void, gates.shape[1]*2))).ravel()
    if group is None:
        return raw
    out = np.empty(len(raw), dtype=[("group", "u4"), ("pattern", raw.dtype)])
    out["group"], out["pattern"] = group, raw
    return out


def species(key, split, keep=None):
    """Discovery/test counts and held-out coverage of discovered species."""
    if keep is None:
        keep = np.ones(len(key), bool)
    which = np.arange(len(key))[keep] < split
    unique, inverse = np.unique(key[keep], return_inverse=True)
    a = np.bincount(inverse[which], minlength=len(unique))
    b = np.bincount(inverse[~which], minlength=len(unique))
    na, nb = int(a.sum()), int(b.sum())
    result = dict(observed_species=len(unique), discovery_species=int((a>0).sum()),
                  discovery_rows=na, test_rows=nb,
                  unseen_test_mass=float(b[a==0].sum()/max(1,nb)))
    order = np.argsort(-a, kind="stable")
    result["top_k_test_mass"] = {
        str(k): float(b[order[:min(k,int((a>0).sum()))]].sum()/max(1,nb))
        for k in (1,16,64,256,1024,4096)
    }
    if unique.dtype.names is not None:
        group = unique["group"]
        order = np.lexsort((-a, group))
        gs = group[order]
        starts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1])+1]
        rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts,len(order)]))
        result["observed_groups"] = len(starts)
        result["top_k_per_group_test_mass"] = {
            str(k): float(b[order[(rank<k)&(a[order]>0)]].sum()/max(1,nb))
            for k in (1,4,16)
        }
    return result


def input_groups(x, steps):
    M = np.full(len(x), -np.inf)
    S = M.copy()
    history = np.zeros(len(x), np.uint32)
    result = {}
    for t in range(10):
        nm, ns = x[:,t] > M, x[:,t] > S
        event = np.where(nm,2,np.where(ns,1,0))
        history = 3*history + event.astype(np.uint32)
        S, M = np.where(nm,M,np.where(ns,x[:,t],S)), np.maximum(M,x[:,t])
        if t+1 in steps:
            top = np.argsort(x[:,:t+1],axis=1)[:,-2:]
            result[t+1] = ((t-top[:,1])*(t+1)+(t-top[:,0]), history.copy())
    return result


def load_models(root, count):
    p = np.load(root/"pop1k/seed0.npz")
    w = [p["s1048576_"+k].astype(np.float64) for k in ("Wih","Whh","Wo")]
    indices = np.arange(0,64,max(1,64//count))[:count]*16
    for i in indices:
        yield "pop"+str(i), [v[i] for v in w]
    for name, file in [("mystery","weights.npz")]+[(f"seed{i}",f"weights_seed{i}.npz") for i in range(5)]:
        p = np.load(root/file)
        yield name, [p["rnn.weight_ih_l0"][:,0].astype(float),
                     p["rnn.weight_hh_l0"].astype(float), p["linear.weight"].astype(float)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--dev-count",type=int,default=8,choices=(8,16,32,64))
    ap.add_argument("--log2-inputs",type=int,default=16)
    ap.add_argument("--seed",type=int,default=2026101906)
    ap.add_argument("--out",type=Path,default=Path("r6_geometry.json"))
    args = ap.parse_args()
    n = 2**args.log2_inputs
    x = np.random.default_rng(args.seed).standard_normal((n,10))
    top = np.argsort(x,axis=1)[:,-2:]
    target, maxpos = top[:,0], top[:,1]
    steps = (3,5,7,9,10)
    groups = input_groups(x,steps)
    result = dict(description=__doc__, seed=args.seed, inputs=n, discovery_inputs=n//2,
                  dev_count=args.dev_count, measurement_forwards_per_network=n, networks=[])
    for name,(wi,wh,wo) in load_models(args.root,args.dev_count):
        h = np.zeros((n,16))
        gates = np.empty((n,10),np.uint16)
        bit = (1 << np.arange(16)).astype(np.uint16)
        for t in range(10):
            h = h@wh.T+x[:,t,None]*wi
            gates[:,t] = ((h>0)*bit).sum(1,dtype=np.uint16)
            np.maximum(h,0,out=h)
        pred = (h@wo.T).argmax(1)
        error = pred != target
        row = dict(name=name, accuracy=float(1-error.mean()), prefixes={})
        for t in steps:
            age, event = groups[t]
            row["prefixes"][str(t)] = dict(
                all=species(keys(gates[:,:t]),n//2),
                by_ages=species(keys(gates[:,:t],age),n//2),
                by_events=species(keys(gates[:,:t],event),n//2))
        row["error_patterns"] = species(keys(gates),n//2,error)
        fragment = ((maxpos*10+target)*10+pred).astype(np.uint32)
        row["error_cone_fragments"] = species(keys(gates,fragment),n//2,error)
        result["networks"].append(row)
        args.out.write_text(json.dumps(result,indent=2))
        end = row["prefixes"]["10"]["all"]
        print(name, "prefix_species", end["observed_species"], "unseen", round(end["unseen_test_mass"],4),
              "error_fragments", row["error_cone_fragments"]["observed_species"], flush=True)


if __name__ == "__main__":
    main()
```

### `r6_verify.py`

```python
"""Small regression checks only; randomized fixtures never enter an estimator."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "2"
import argparse
import json
from pathlib import Path
import numpy as np
import numba
from scipy.special import ndtr
from r6_core import load, forward, prefix_forward
from r6_event import last_input
from r6_exact_anchor import nodes, exact_node
from lines import trace_line


def verify():
    W = load(np.array([0,1024]))
    tokens = np.array([[0]*10,[0]*9+[1],[0]*8+[1,2],
        [0,1,2,0,1,2,0,1,2,0],[0,1,2,0,1,2,0,1,2,0],
        [1,0,2,2,0,1,1,2,0,0]])
    values = np.array([-1.25,.125,1.5])
    direct = forward(W,values[tokens])
    shared,cost = prefix_forward(W,tokens,values)
    assert np.allclose(direct,shared,rtol=1e-12,atol=1e-10)
    result = dict(prefix_max_logit_difference=float(np.max(np.abs(direct-shared))),
                  prefix_counted_cost=float(cost),direct_rows=len(tokens))

    # Independent Gaussian prefixes are a measurement fixture, not an estimator.
    x = np.zeros((6,10))
    x[:,:9] = np.random.default_rng(2026101907).standard_normal((6,9))
    top = np.argsort(x[:,:9],axis=1)[:,-2:]
    jp,ip = top[:,0],top[:,1]
    S,M = x[np.arange(6),jp],x[np.arange(6),ip]
    h = np.zeros((len(W[0]),6,16))
    for t in range(9):
        h = np.maximum(h@W[1].transpose(0,2,1)+W[0][:,None,:]*x[None,:,t,None],0)
    swept,_ = last_input(W,h,M,S,ip,jp)
    xx = np.repeat(x,3,axis=0)
    lo = np.column_stack([np.full(6,-12.),S,M]).ravel()
    hi = np.column_stack([S,M,np.full(6,12.)]).ravel()
    target = np.column_stack([jp,np.full(6,9),ip]).ravel().astype(np.int64)
    p = np.full(len(xx),9,np.int64)
    checked = []
    for i in range(len(W[0])):
        probability,_,overflow = trace_line(*[w[i] for w in W],xx,p,lo,hi,target)
        assert not overflow.any()
        checked.append((probability*(ndtr(hi)-ndtr(lo))).reshape(6,3).sum(1))
    checked = np.array(checked)
    # The omitted Gaussian tails beyond +/-12 have total mass below 4e-33.
    assert np.allclose(swept,checked,rtol=1e-10,atol=1e-10)
    result["last_input_max_probability_difference"] = float(np.max(np.abs(swept-checked)))

    wi = np.tile([1.,-1.],8)
    wh = np.zeros((16,16)); wh[:,0] = -1
    wo = np.random.default_rng(0).standard_normal((10,16)); wo[0] = 0
    hh = np.zeros((1,1,16)); hh[0,0,0] = 1
    probability,_ = last_input([wi[None],wh[None],wo[None]],hh,
        np.array([3.]),np.array([2.]),np.array([1]),np.array([0]))
    expected = float(ndtr(1)-ndtr(-1))
    assert abs(float(probability[0,0])-expected)<1e-14
    result["all_off_probability"] = float(probability[0,0])
    result["all_off_expected"] = expected

    x,R,M,_,_,_ = nodes(3)[4]
    ww = [w[0] for w in W]
    a,c,_,_ = exact_node(ww,x,R,M,512)
    b,retry_cost,discarded,capacity = exact_node(ww,x,R,M,4)
    assert np.array_equal(a,b) and discarded>0 and retry_cost>c
    result["overflow_retry"] = dict(max_probability_difference=float(np.max(np.abs(a-b))),
        baseline_mac_forwards=c,with_retries_mac_forwards=retry_cost,
        discarded_lines=discarded,final_capacity=capacity)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",default="r6_results/verify.json")
    args = parser.parse_args()
    numba.set_num_threads(2)
    result = verify()
    path = Path(args.out); path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
```

### `r6_bounds_minimal.py`

```python
"""Restricted cone-oracle bound and analytic ANOVA counterexample; no data."""
from math import atan, cos, pi, sin, sqrt


def G(r):
    a = atan(r)
    return (35*a - 28*sin(2*a) + 7*sin(4*a)
            - 4*sin(6*a)/3 + sin(8*a)/8)/(35*pi)


V = 0.069
Q = (1-sqrt(1-4*V))/2
lo, hi = 0., 10.
for _ in range(80):
    mid = (lo+hi)/2
    if G(mid) < Q:
        lo = mid
    else:
        hi = mid
R = (lo+hi)/2
print(f"Mass cap={Q:.12f}; R={R:.12f}")
print("Queries; worst-error lower(pp); sampling RMS(pp); squared ratio")
for N in (100, 1000, 10000, 100000, 1000000, 10000000):
    theta = (256/(175*N))**(1/8)
    lower = 2*(G(R)-G(R*cos(theta)))/5
    sampling = sqrt(V/N)
    print(N, 100*lower, 100*sampling, (lower/sampling)**2)

# iid Gaussian X,Y,Z: m=|X+Y|-|X-Y|+2|Z|, first ANOVA m_A=2|Z|.
# Failure of m means XY<0 and |Z| is smallest: probability (1/2)*(1/3).
# Additive g=X+sqrt(3)|Z| fails in a wedge of angle pi/3.
true_p = 1-1/6
anova_p = 1.
additive_p = 0.5+atan(sqrt(3))/pi
assert abs(true_p-additive_p) < 1e-15
print("ANOVA example: true, ANOVA, other additive:",
      true_p, anova_p, additive_p)
```

### `r6_bounds_check.py`

```python
"""Reproduce the original restricted-oracle bound in r6_bounds.md.

No data access, fitting, random numbers, or external packages are required.
These are minimax bounds for a larger geometric class, not measurements on
the 432-parameter RNN population.
"""
from math import atan, cos, pi, sin, sqrt


def cone_mass(radius):
    """P[X_10 >= ||X_1:9|| / radius], for X ~ N(0, I_10)."""
    angle = atan(radius)
    return (35 * angle - 28 * sin(2 * angle) + 7 * sin(4 * angle)
            - (4 / 3) * sin(6 * angle) + sin(8 * angle) / 8) / (35 * pi)


def bound_constants(variance=0.069):
    cap = (1 - sqrt(1 - 4 * variance)) / 2
    lower, upper = 0.0, 10.0
    for _ in range(80):
        middle = (lower + upper) / 2
        if cone_mass(middle) < cap:
            lower = middle
        else:
            upper = middle
    return cap, (lower + upper) / 2


def lower_bound(queries, radius):
    angle = (256 / (175 * queries)) ** (1 / 8)
    return (2 / 5) * (cone_mass(radius) - cone_mass(radius * cos(angle)))


def main():
    variance = 0.069
    cap, radius = bound_constants(variance)
    print(f"Error-mass cap: {cap:.12f}; cone radius: {radius:.12f}")
    print("queries  lower_pp  sampling_pp  squared_ratio")
    for queries in [100, 1000, 10000, 100000, 1000000, 10000000]:
        lower = lower_bound(queries, radius)
        sampling = sqrt(variance / queries)
        print(f"{queries:>8} {100*lower:>10.6f} {100*sampling:>12.6f} "
              f"{(lower/sampling)**2:>14.6f}")
    print("\nIllustrative certificate radii from z26's estimated median tails:")
    print("patterns  forwards  estimated_tail_pp  midpoint_radius_pp  sampling_pp")
    for patterns, tail in [(1, .80103), (3, .60671), (10, .33667),
                           (30, .15146), (100, .04989), (300, .01299),
                           (1000, .00291)]:
        cost = 5040 * patterns
        print(f"{patterns:>8} {cost:>9} {100*tail:>18.3f} "
              f"{50*tail:>19.4f} {100*sqrt(variance/cost):>12.6f}")
    print("\nTruncation crossover from 0.362 points:", variance / .00362**2)


if __name__ == "__main__":
    main()
```

### `r6_summary.py`

```python
"""Assemble tables and the development-selected frontier from saved results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path('r6_results')
def get(file): return json.loads((P/file).read_text())
records = {}
for split in ['dev', 'held']:
    rows = []
    for family in ['low', 'event', 'anchors', 'exact_anchor']:
        for r in get(f'{family}_{split}.json'):
            label = family+':'+':'.join(str(r.get(k, '')) for k in ['n','K','q','bins'])
            m = r[split]
            rows.append(dict(id=label, family=family, cost=r['cost'], rms=m['rms'], mean=m['mean'],
                sampling=m.get('sampling_at_network_costs',m['sampling']),
                ratio=m.get('ratio_at_network_costs',m['ratio']), zoo=r['zoo_errors'],
                lower_bound=r.get('cost_is_lower_bound',False)))
    for r in get(f'correction_combinations_{split}.json'):
        if 'k3_n3_K8_q8' in r['correction']: continue  # benchmark, outside frozen shortlist
        label = r['anchor']+str(r['n'])+':'+r['correction'].replace('_'+split+'_','_')
        rows.append(dict(id=label,family='two_level',cost=r['cost'],rms=r['rms'],mean=r['mean'],
                         sampling=r['sampling_rms'],ratio=r['mse_ratio'],zoo=r['zoo'],
                         lower_bound=r['cost_is_lower_bound'],
                         exact_anchor_omits_scalar_work=r.get('exact_anchor_omits_scalar_work',False),
                         mac_only_cost_lower_bound=r.get('mac_only_cost_lower_bound'),
                         sampling_rms_at_mac_cost=r.get('sampling_rms_at_mac_cost'),
                         mse_ratio_at_mac_cost=r.get('mse_ratio_at_mac_cost')))
    r=get(f'correction_{split}_k3_n3_K8_q8_keep8.json')['same_grid']
    rows.append(dict(id='z38',family='z38',cost=r['cost'],rms=r['rms'],mean=r['mean'],
                     sampling=r['sampling_rms'],ratio=r['mse_ratio'],zoo=r['zoo'],lower_bound=False))
    records[split] = rows

# Selection is a function of development results only; held values are just paired.
best = float('inf'); frontier=[]
held = {r['id']:r for r in records['held']}
for r in sorted(records['dev'],key=lambda x:x['cost']):
    if r['rms'] < best:
        frontier.append(dict(dev=r,held=held[r['id']]))
        best=r['rms']
summary=dict(units='percentage points',selection='development RMS Pareto frontier; no held retuning',
             arithmetic_caveat='Exact-line costs omit scalar work. Other costs use declared scalar allowances, not measured hardware upper bounds.',
             frontier=frontier,all_methods=records)
(P/'summary.json').write_text(json.dumps(summary,indent=2))

fig,ax=plt.subplots(figsize=(10.6,6.5))
c=np.geomspace(40,1e7,400)
ax.loglog(c,100*np.sqrt(.0698014658949404/c),'--',color='#222222',lw=1.8,label='Sampling (held-out variance)')
for split,color,marker in [('dev','#df8b23','o'),('held','#176d94','s')]:
    ax.loglog([r[split]['cost'] for r in frontier],[r[split]['rms'] for r in frontier],
              color=color,marker=marker,lw=1.7,ms=5,label=f'Frozen frontier: {split}')
for split,color in [('dev','#df8b23'),('held','#176d94')]:
    r=next(x for x in records[split] if x['id']=='z38')
    ax.scatter(r['cost'],r['rms'],color=color,marker='X',s=95,zorder=5)
ax.annotate('z38 reference rule',(2.984e6,.621),xytext=(7e5,1.25),
            arrowprops=dict(arrowstyle='-',color='#777777'),fontsize=10)
ax.axhline(.025800749,color='#999999',ls=':',lw=1.2,label='Reference RMS noise (not a hard error floor)')
ax.set(xlim=(40,1e7),ylim=(.006,12),xlabel='Counted forward equivalents per network (log scale)',
       ylabel='RMS accuracy error, percentage points (log scale)',
       title='Two-level quadrature improves the estimate, but remains above sampling')
ax.grid(which='major',alpha=.22);ax.legend(loc='lower left',fontsize=9)
fig.text(.12,.012,'64 development + 64 held-out networks. Exact-line x-coordinates are optimistic MAC counts; scalar work is additional.',fontsize=8,color='#444444')
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(P/'curve.png',dpi=180)
for item in frontier:
    print(item['dev']['id'], {s:{k:item[s][k] for k in ['cost','rms','mean','sampling','ratio']} for s in ['dev','held']})
```

---

Here's a quick summary of the conversation:

The conversation details the evaluation of deterministic multi-level estimators for predicting RNN network outputs, focusing on balancing computational cost and accuracy. A new two-level estimator using an integrated MF3 anchor and coarse MF4 correction improves accuracy significantly over previous methods like `z38`, achieving about 6.28× lower mean squared error with about 9% less counted work. Despite this improvement, a large error gap remains compared to sampling-based approaches, which still outperform the estimator by over 200× in mean squared error at similar computational costs.

Extensive audits reveal cost accounting caveats, variance issues, and the limits of amortizing corrections across arrangements. Numerical checks confirm the method's correctness, and a theoretical lower bound applies only to Boolean cone-membership queries, not to algorithms inspecting the RNN weights. The conversation concludes with full deliverables including reports, code, and data, emphasizing that while progress has been made, sampling parity has not been achieved.

