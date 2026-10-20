# Interpreting AlgZoo's 432-parameter 2nd-argmax RNN

Model: `alg_zoo.example_2nd_argmax()`. It is a one-layer ReLU RNN with 16 hidden neurons, no biases, and sequence length 10. Inputs are N(0,1), and it outputs the position of the second-largest value. Parameters: W_ih is 16, W_hh is 256 and W_out is 160, for 432 in total. Accuracy is 95.3%.

Prior work in ARC's blog post covered five neurons: n2 is the running max excluding the latest input, n4 is the increment of the running max, and n6, n7 and n1 are "leave-one-out max minus latest" features. This report covers the rest of the mechanism, its failure modes, and how it compares with the other zoo seeds and their training checkpoints.

All scripts are in the repository root and are run from there; `README.md` covers setup and maps scripts to sections. `common.py` is a NumPy copy of the RNN, exact to about 1e-12 against torch. Section 7.16 extends the estimation work to a population of 1,024 freshly trained networks. Section 10 lists corrections made after an audit of the report against its scripts.

---

## 1. The algorithm, in one paragraph

The network runs an **online, time-invariant 2nd-max tracker**. It never stores *where* the max and 2nd max are. It stores *how long ago* they appeared, as the **phases of two damped rotating clocks that turn at the same speed (about 43° per step)**. Each clock's amplitude scales with the value it tracks.

At each step, ReLU comparator neurons test the new input against thresholds that blend M (the max) and S (the 2nd max). Depending on the result:
- if the input is a new max, it restarts the max clock and hands the old max clock's phase to the 2nd-max clock;
- if it is a new 2nd max, it restarts the 2nd-max clock;
- otherwise both clocks just keep rotating.

Because both clocks turn at the same speed, the handoff from max to 2nd max is a fixed linear map. The readout decodes the 2nd-max clock's phase, along with harmonics at about 2×, 3× and 4× the base speed, into a position.

**A more fundamental view (section 7.9).** A sparse **write-pointer neuron (n5)** fires one step after each new max or new 2nd max. Forcing it moves the answer to the marked position in 88–96% of runs where no later input is a new max or 2nd max. The "clocks" are the eigenmode breakdown of the recurrent weights' impulse response to that pulse. From the weights alone, Wo·Whh^a·e₅ decodes every age correctly.

The other designs follow the same principle (pointer, impulse response, readout that decodes it), with a different shape of response: a non-rotating path in seed 2 (pointer n10) and time cells in the 32-neuron model. The readout is just the softmax-regression optimum for these features (7.12).

The comparators carry the max-to-2nd handoff. Whether training reaches this design at all, or a "remember every input" design, is an early, stochastic fork across a loss barrier, settled by about 2²⁴ sequences (7.14).

## 2. Evidence

### 2.1 The state stores ages, not positions (`a6.py`, `a7.py`)

Apply the final readout to the hidden state after step t and shift the answer by 9−t slots. This gives the 2nd argmax of the prefix with accuracy 0.66, 0.78, 0.86, 0.91, 0.94 and 0.95 for t = 3 to 8. Readout slot i therefore means "the 2nd max appeared 9−i steps ago". Linear probes on the final state recover the age of the max with 98% accuracy, the age of the 2nd max with 90%, and the age of the 3rd max with only 24%.

Fitting the final state from (age of max, age of 2nd max, M, S, x_9), with a separate linear fit for each pair of ages, and reading out from that fit gives 98.7% accuracy (`a9.py`). That number is circular: the age of the 2nd max is the answer, so the fit is handed it. The fitted state agrees with the model on only 95.5% of inputs (95.50% held out, `a16.py`), about the same as the true answer does (95.3%). So these five quantities describe the state well enough for the readout to decode the right age, but they do not explain which inputs the model gets wrong. Section 4 reaches the same conclusion.

### 2.2 Comparators with age-dependent thresholds (`a10.py`–`a12.py`)

Neurons n1, n4, n6, n7 and n8 have large input weights (about ±10–13). Each fires when x_t is on one side of a threshold. The threshold comes from the previous state. Within each group of inputs with the same max age, it is almost exactly a·M + b·S with a + b ≈ 1 (fit quality R² 0.94–0.999; fitted at one step, t = 8, `a12.py`):

| neuron | threshold ≈ S when … | otherwise ≈ M |
|---|---|---|
| n6 | the max arrived on the previous step | yes |
| n1 | the max is 1 step old (partly at 2 steps) | yes |
| n8 | the max is 0–3 steps old | ≈M from 5 steps old |
| n7, n4 | rarely: they test against ≈M (they detect a new max), drifting to about (0.6–0.75)·M + (0.2–0.3)·S when the max is 4–7 steps old | — |

Neurons n4 and n9 are sparse markers that the max is 0 or 1 steps old. n5 fires one step after a new max and also one step after a new 2nd max (84% of the time); it is the write pointer of section 7.9.

### 2.3 Two equal-speed clocks (`a19.py`–`a21.py`)

The usually-active neurons (13 of 16) have these eigenvalues:

| mode | size, angle per step | role |
|---|---|---|
| 0.70 ± 0.65i | 0.951, 42.7° | **max clock**: phase advances 41.5° ± 4.1° per step of max age, concentration 0.94 |
| 0.54 ± 0.50i | 0.735, 43.0° | **2nd-max clock**: phase advances 45.3° ± 6.0° per step of 2nd-max age, concentration 0.80 |
| 0.11 ± 0.93i | 83.3° | about the 2nd harmonic |
| −0.48 ± 0.69i | 124.8° | about the 3rd harmonic |
| −0.93 | 180° | about the 4th harmonic |
| 1.01, 0.97 | real | value registers (M and S scale) |

The neurons outside the comparator group (n0, n10, n11, n12, n14, n15) look like ReLU-clipped sinusoids of the max's age (`a19.py`).

**Phase handoff.** The 2nd-max clock's phase, for a given age, is the same whether the 2nd max was the displaced old max or was admitted directly. For ages 3–8 the differences are 9°, −3°, −9°, −10°, −5° and 10° (`a21.py`, `seeds_clocks.py`).

### 2.4 Causal tests (`e1.py`, `a21.py`, `y2.py`)

**Ageing the clock moves the answer.** In the final state, multiply the 2nd-max clock and its harmonics by λ^k, which is k steps of their own dynamics:

| intervention | answer moves by | fraction of inputs (answer in slots 2–7) |
|---|---|---|
| 2 steps older | −2 positions | 78% |
| 1 step older | −1 position | 65% |
| 1 step younger | +1 position | 48% |
| 2 steps younger | +2 positions | 38% |

Making the clock younger works less well. This is not because undoing the decay amplifies noise: rotating the phase alone, with amplitudes unchanged, shows the same asymmetry (one step older moves the answer by −1 for 64% of inputs, one step younger by +1 for 39%).

**Rotating the max clock alone changes only 5% of outputs.** So the max clock drives the updates, and the readout relies on the 2nd-max clock.

**Removing single modes from the readout.** Setting a mode to zero (`a21.py`) also removes its mean contribution, so the table also gives mean-ablation, where the mode is replaced by its average (`y2.py`):

| mode removed | accuracy, zero-ablated | accuracy, mean-ablated |
|---|---|---|
| 2nd-max clock (0.735, 43°) | 50% | 47% |
| 83° harmonic | 69% | 64% |
| 125° harmonic | 68% | 65% |
| 0.23 real mode | 76% | 73% |
| fastest-decaying mode (≈0) | 85% | 85% |
| −0.93 mode | 90% | 90% |
| max clock | 91% | 91% |
| 1.01 value mode | 86% | 90% |
| 0.97 value mode | 54% | **85%** |

The 2nd-max clock and its harmonics carry the answer. The 0.97 value mode's large zero-ablation effect is mostly its mean offset.

### 2.5 The readout is a Fourier decoder (`a23.py`)

Each readout slot i has a preferred phase on the 2nd-max clock. For slots 0–5 it matches the clock's phase at age 9−i to within about 10°: 6°/15°, 320°/316°, 273°/267°, 216°/227°, 175°/186° and 129°/139°. Readout weights grow for older ages, which makes up for the clock's decay (0.735 per step). Recent 2nd maxes (0–3 steps old) are decoded mostly from the sparse marker neurons and fast-decaying modes.

### 2.6 Length generalization (`e5.py`, `e6.py`)

On longer sequences, the unchanged readout still finds the 2nd max by its age (L is sequence length). The readout has 10 slots, so only sequences whose 2nd max lies in the last 10 positions are scored (`e5.py`); at L = 30 that is a third of them:

| L | accuracy |
|---|---|
| 12 | 92% |
| 17 | 85% |
| 30 | 89% |

Guessing would give 10%. This is consistent with the time-invariant account but does not single it out: memory-based seed 0 generalizes as well (0.90 at L=12, 0.92 at L=30), while seed 2, an online selector, falls to 0.73 at L=30. It is not perfect: when the max is also within the last 10 inputs, accuracy falls with L, reaching 81% at L=30. A likely cause is the mode with eigenvalue 1.01, which makes the max register drift.

## 3. Where and why it fails (`a13.py`, `a14.py`, `a17.py`, `e8.py`–`e10.py`)

The model is close to an exact algorithm, with comparisons whose **resolution scales with M** plus one blind spot:

| region | share of inputs | error rate there | share of all errors |
|---|---|---|---|
| S < 0 (2nd max is negative) | 1.1% | **77%** | 18% |
| M − S < 0.05·M (max and 2nd nearly tied) | — | — | 20% |
| S − R3 < 0.05·M (2nd and 3rd nearly tied) | — | — | 53% |
| everything else | — | — | 8% |
| "clean": S > 0 and both gaps > 0.1·M | 55% | **0.23%** | — |

With S/M in [0.2, 0.8) and (S − R3)/M ≥ 0.2, the error rate is 0.04%: 157 errors among the 377,560 such inputs in a 1.5M sample (`y3.py`).

The **S < 0 blind spot** is the "max with 0" artifact. The state starts at zero and the ReLUs clip negatives, so a negative value is never admitted as the 2nd max. In that case the model's picks are spread over all ranks: 21% the max, 22% the correct answer.

**Errors also depend on the lower-ranked inputs.** With the top-3 values held fixed, resampling the others flips 45% of the model's errors but only 1.8% of its correct answers (`b5.py`). The comparators respond linearly to every input below their threshold, so ordinary inputs leak slightly into the clocks.

## 4. Can a short program reproduce the model's outputs, mistakes included?

No, as far as I can tell (`b1.py`–`b7.py`).

**Rule-based programs don't capture the mistakes.** The true 2nd argmax agrees with the model 95.3% of the time. The best rule-based programs on top-k values, gaps and ages agree 95.85% of the time, even with decision trees of 4,700 leaves. The ceiling for programs that look only at the top 3 is 99.2%.

**The specific mistakes use nearly all the weights.** After rescaling each neuron so weights are comparable:

| simplification | agreement with model |
|---|---|
| prune 9 weights below 0.003 | 99.4% |
| prune 38 weights below 0.01 | 91.9% |
| prune 68 weights below 0.03 | 47% |
| round weights to 2 decimals (unrescaled weights, `b6.py`) | 90.5% |

Only 29 of the 256 recurrent weights can each be removed while keeping 99.5% agreement.

**Conclusion.** The mechanism explains the model's *correct* behaviour and the *rate and location* of its errors (section 3). Each individual mistake is the sum of many small leftover effects. Reproducing them input by input means re-describing the weights.

## 5. Other seeds use a different algorithm (`seeds_*.py`, `c1.py`–`c6.py`)

The zoo's 16-neuron, length-10 models (seeds 0–4) come from a separate training run, with accuracies 91.0%, 88.0%, 94.8%, 93.2% and 89.0%. I tested each for the same mechanisms:

**Shared by every seed:**
- the age code: the shifted readout's accuracy is within 1% of each model's own accuracy;
- a max clock: an age-indexed rotating plane explains 78–97% of the variance in the age averages.

**Clock speed differs:** about 43° per step in the mystery model, about 110° in seed 2, and about 150–170° in seeds 0, 1, 3 and 4. A fast clock of nearly a flip per step plus slow drift works because ten steps still land on distinct angles.

**A clean equal-speed 2nd-max clock with phase handoff appears only in the mystery model.**

**Two strategies, which differ most on clear cases:**

| model | accuracy | error at 2nd–3rd gap < 0.05 | error at gap > 0.5 | error on "clean" inputs | strategy |
|---|---|---|---|---|---|
| mystery | 95.3% | 22.6% | **1.0%** | 0.23% | online selection |
| seed 2 | 94.8% | 25.4% | **0.8%** | 0.50% | online selection |
| seed 3 | 93.2% | 29.0% | 2.1% | 0.65% | intermediate |
| seed 0 | 91.0% | 32.4% | 4.6% | 1.37% | memory |
| seed 4 | 89.0% | 35.5% | 5.7% | 2.72% | memory |
| seed 1 | 88.0% | 35.1% | 7.2% | 4.94% | memory |

- **Memory (seeds 0, 1, 4).** Always-on neurons keep a lossy linear memory of every input, and a nonlinear part marks where the max is. For seed 0, with the max at position p, logit i ≈ 23·x_i − 2.7·(sum of the other inputs) for i ≠ p, and the max's own logit is suppressed. Given the max's position, the logits are mostly linear in the raw inputs (R² 0.84–0.87). This is the same idea as ARC's handcrafted model, and it fails even on clear cases because the memory is lossy.
- **Online selection (mystery, seed 2).** Selection happens during the recurrence. Errors are confined to near-ties.

## 6. Training dynamics (`d1.py`, `d2.py`; 5 seeds × 12 checkpoints)

1. **Up to 2²² training sequences:** no stable clock (±30–58° jitter per step), and accuracy is at most 53%.
2. **2²³:** accuracy 58–70%, but the clock is still unstable (±28–49°). **2²⁴:** a max clock in every seed, with jitter ±12–22° and accuracy 73–77%; the clock speed settles.
3. **The memory score is not a clean phase marker.** It measures agreement of the "linear memory + max suppression" fit with the model on clear cases. It is already 0.37–0.80 at initialization, where the untrained network is close to linear in its inputs. It dips to 0.31–0.36 at 2²⁰ and reaches 0.73–0.83 by 2²³–2²⁴. Seeds 0 and 4 keep rising (0.92 and 0.89 at 2³⁰), while seed 2 falls (0.50).
4. **Seed 2 switches to online selection between 2²⁴ and 2²⁵.** The diagonal-to-off-diagonal ratio of a linear fit of its logits on the inputs (`d2.py`, a memory signature) falls from 1.02 to 0.63 and later to 0.53. After the switch, its error at large gaps falls from 15% to 0.8%. Seed 3 switches earlier (ratio 1.16 → 0.92 → 0.67). `d2.py` covers seeds 2, 3 and 0 only. Seeds 0 and 4 never switch, and their memory score keeps rising.

Interpretation: remembering everything is the easy solution that training finds first in most runs. Online selection is better but has to be discovered. The mystery model looks like a run that found it, with an especially tidy implementation using two equal-speed clocks.

## 7. Follow-up experiments

### 7.1 Patching modes confirms the mystery model's 2nd-max clock (`f1.py`, `f2.py`)

Take pairs of inputs that differ only in where the 2nd max sits (it is swapped with a lower value).

- **Copying single neurons** from one run into the other carries over at most 8% of answers, in each of the three models tested. The code is not organised neuron by neuron.
- **Copying eigenmodes** instead gives a clear answer for the mystery model. The most important single mode is the 2nd-max clock, which carries over 15% of answers. A greedy search then picks the 2nd-max clock, the 83° harmonic, the 125° harmonic and the fastest-decaying mode, in that order, with cumulative carry-over 15%, 27%, 47% and 56%. The first three are the predicted clock and harmonics. The fourth was not predicted, and the −0.93 "4th harmonic" is not picked.
- **Seed 2, seed 3 and seed 0 have no dominant mode.** The contrast is modest: seed 0's best single mode carries 13%, against 15% for the mystery model's 2nd-max clock.

### 7.2 How each model stores the 2nd max's age (`f4.py`)

Each rotating mode is fitted as z ≈ A[age of max]·M + B[age of 2nd max]·S, with a free complex coefficient for each age.

| model | storage |
|---|---|
| mystery | its own clock (0.73 mode): a 2nd-max hand turning 35–73° per step, gain +0.39. The 83° and 125° harmonics carry both hands. |
| seed 3 | **two hands on shared clocks.** On the 100°, 59° and 143° modes, the max and 2nd-max hands turn at the same speed. For example, on the 100° mode the max hand turns 97–109° per step over ages 2–7 and the 2nd-max hand 94–124°; both slow down at the oldest ages (to about 50° per step). Gain +0.27 to +0.29. |
| seed 2 (final) | a weak 2nd-max hand on the max clock (104°, gain +0.09). The rest lives in neurons that switch on and off. |
| seed 0 | no rotating 2nd-max hand. The information sits in non-rotating, memory-like modes. |

### 7.3 How the max gets suppressed (`f5.py`–`f7.py`)

In all four models tested (seeds 0, 2 and 4 and the mystery model), raising the max's value **lowers its own logit**. The gradient of the max's own logit with respect to its own value is −12 (seed 0), −9 (seed 4), −20 (mystery) and −96 (seed 2). For the 2nd and 3rd values it is +12 to +41.

**Mechanism:**
- Ordinary values are written through the comparators as ReLU(threshold − x), which carries −x.
- A new max switches those comparators off and is written instead through the new-max detector, as ReLU(x − old max), which has the opposite sign.
- So the max is written into the state with the opposite sign as it arrives, and suppression is built into the writing. The input-write norm in `f7.py` is unsigned, so it measures only strength, not sign. A new max is written more weakly than ordinary values in seed 0 (15.8 against 22.6) and the mystery model (13.5 against 22.7), about equally in seed 4 (15.8 against 16.4), and more strongly in seed 2 (24.9 against 11.4), which has a dedicated new-max neuron (input weight +26.9).

This explains why no single mode carries the suppression. Patching the max clock from a run where the max was moved barely changes it. Patching individual fast modes or sparse neurons undoes it in at most 8–22% of inputs (`f6.py`). Rotating seed 0's max clock moves the suppressed slot only weakly (7–14%, `f5.py`).

### 7.4 Seed 2's switch, checkpoint by checkpoint (`g2.py`)

| checkpoint | best 2nd-max-hand gain (mode angle) | gradient of max's own logit | usually-active neurons |
|---|---|---|---|
| 2²³ | 0.19 (105°) | −1.1 | 15 |
| 2²⁴ | 0.21 (119°) | −4.8 | 15 |
| **2²⁵** | **0.35 (41°)** | −13.4 | 13 |
| 2²⁶ | 0.36 (38°) | −18.5 | 13 |
| 2²⁷ | 0.25 (75°) | −26.1 | 12 |
| 2³⁰ | 0.12 (65°) | −92.8 | 12 |

At the switch, seed 2 briefly grows a ~40° 2nd-max clock, at about the speed of the mystery model's 2nd-max clock. At the same time, more neurons start switching on and off, and the max's negative write grows sharply. Later the ~40° clock fades as the 2nd-max storage moves into switching neurons, which is why the final seed 2 has no linearly decodable 2nd-max clock.

In seed 0 the 2nd-max-hand gain stays flat at 0.18–0.23 throughout. In seed 3 it peaks at 0.42 at 2²⁵ and settles around 0.30.

### 7.5 Using the mechanism to estimate accuracy (`g1.py`)

The mechanism says where errors happen: when the 2nd max is negative, and when gaps are small relative to M. Strata can be assigned **without running the model**, just by sorting the input values. Their weights come from cheap Gaussian sampling. Sampling within strata in proportion to their variability (Neyman allocation) gives these mean squared errors at 2,000 model evaluations, against plain random sampling:

| model | plain sampling MSE | stratified MSE | gain |
|---|---|---|---|
| mystery | 2.19e-5 | 8.79e-6 | 2.5× |
| seed 2 | 2.46e-5 | 9.69e-6 | 2.5× |
| seed 0 | 3.85e-5 | 2.47e-5 | 1.6× |

**Caveat:** the allocation used a pilot run of 240k model evaluations that is not counted in the budget. A fair version would set the allocation from the mechanism alone, or reuse one allocation across models. The gain is smaller for the memory-based seed because its errors are less concentrated in near-ties.

### 7.6 Seed 2 decoded: a write-pointer neuron plus a delay trajectory (`h1.py`–`h4.py`)

**Firing history carries the age.** A least-squares probe on the on/off history of all neurons predicts the 2nd max's age 86% of the time, more than the final values do (71%). Neuron 10's firing history alone gets 67% (chance is about 11%).

**What each switching neuron detects.** Each fires on a threshold computed from the previous state:

| neuron | input weight | fires when | threshold fit |
|---|---|---|---|
| n15 | +26.9 | a new max arrives | ≈ 0.99·M (R² 1.00) |
| n10 | +8.4 | a new 2nd max arrives | ≈ 0.71·S (R² 0.92) |
| n0, n3, n6 | −7.7 to −8.3 | the input is below something | not linear in M and S |

n10 fires for 53–99% of inputs between S and M, depending on the step. Its threshold is loose, so it also fires for about 9% of inputs a little below S.

**Causal test.** Force an n10 pulse at step t, in a run where nothing new happened at t. In runs where no later input beats the 2nd max, the answer becomes t in **59–87%** of cases, against about 0.3% at baseline.

**The pulse becomes a delay trajectory.** The average final-state effect of the pulse, passed through the readout, peaks at **exactly the right slot for every pulse age from 0 to 8**. The trajectory does not rotate: its angle steps are irregular and its size decays. It runs through switching neurons (n2, n4, n7, n12, n13), and the readout is matched to it.

So seed 2 = a new-max detector (n15) + a new-2nd-max write pointer (n10) + a non-rotating delay trajectory. The mystery model uses rotating clocks instead.

### 7.7 A near-mechanistic accuracy estimate fails (`h5.py`, `h6.py`)

Model: accuracy ≈ P(S>0)·E[(1 − e(gap₂₃/M))·(1 − e(gap₁₂/M))] + P(S<0)·(1 − e_neg). The error-versus-gap curve e(·) comes from a 20k-evaluation probe that places the last input near S. The expectation is computed over Gaussian order statistics, with no model runs.

| model | true accuracy | estimate |
|---|---|---|
| mystery | 0.953 | 0.858 |
| seed 2 | 0.948 | 0.847 |
| seed 3 | 0.932 | 0.763 |
| seed 0 | 0.910 | 0.736 |
| seed 1 | 0.880 | 0.707 |
| seed 4 | 0.889 | 0.730 |

Every estimate is 9.5–17.5 points too low; sampling at the same cost has a standard deviation of about 0.15 points. A logistic-resolution version (ε ≈ 0.04·M) is similar: 0.834 for the mystery model.

**Why it fails.**
- **The last step is special.** About half of all errors happen at the final input, with no later steps to settle them. Earlier comparisons are sharper.
- **The probe is confounded.** Moving x₉ away from S pushes it toward the 3rd value or toward M.

A single comparator with one resolution does not describe the model. An honest mechanistic estimate needs a separate comparator curve for each step, plus the leakage terms.

### 7.8 Other model sizes (`h7.py`–`h9.py`)

**4, 6 and 8 neurons** (best seeds) are too small, at 32%, 38% and 43% accuracy. The clock flags at 8 neurons are spurious: the "clock" barely rotates (about 1° per step).

**32 neurons** reach 99.4% accuracy.
- It still stores ages: 98.2% on the shifted-readout test at t=8.
- It has no slow rotating clock; only 13 of its 32 neurons are usually on.
- Its clean-case error is 0.01%.
- It largely fixes the negative-2nd-max blind spot: 22% error there, against 77%.

It uses **time cells**: sparse neurons that fire (rate 1.0) when the max *or* the 2nd max has one particular age:

| neuron | age it signals |
|---|---|
| n7, n16, n21, n23 | 0 (event detectors; n7 input weight +8.8) |
| n14 | 1 |
| n22 | 2 |
| n19 | 3 |
| n2 | 6 |
| n15 | 8 |

Only one link is a direct excitatory chain (n22 → n19, +1.19). The other time cells are gated by strong inhibition from always-on neurons (for example n12 → n14 at −1.70, n28 → n22 at −1.62, n30 → n2 at −2.09). They behave as thresholded readouts of a continuous trace.

**Across sizes and seeds, three codes for "how long ago" appear:**
- rotating clocks (the mystery model, and the max clock in every 16-neuron model);
- write-pointer pulses followed by a delay trajectory (seed 2);
- time cells (the 32-neuron model).

With more neurons, the code moves toward sparse, switching representations.

### 7.9 One shared principle: write pointer, impulse response, matched readout (`i1.py`–`i4.py`)

**Seed 2's trajectory follows from its weights.**
- Pushing the n10 pulse through the gated recurrence (Whh plus that run's on/off pattern) reproduces the measured final-state effect with cosine 0.99.
- Using the average on/off pattern at each step gives cosine 0.82–0.98.
- Even with no gating at all (every neuron on), the readout picks the right slot.

**Weights only, no data:** logits = Wo·Whh^a·e₁₀ give the correct slot 9−a for **all 10 ages**, with margins of 2.2–6.2. Seed 2's readout is close to an exact inverse of its recurrent weights' impulse response to the write-pointer neuron.

The new-max pulse (e₁₅) instead gives strongly negative logits at its own slot for ages 0–3 (−34, −17, −12, −13). That is the max suppression.

**The mystery model has the same structure.** Its write pointer is **n5**, which fires one step after an event:
- the step after a new max: 94–100% of the time;
- the step after a new 2nd max: 70–97%;
- otherwise: 5–8%.

From the weights alone, Wo·Whh^a·e₅ decodes to [8, 7, …, 0]: every age correct, offset by n5's one-step delay. Forcing an n5 pulse moves the answer to the marked position in **88–96%** of runs with no later updates, against 0.3% at baseline (`i4.py`).

So the "clocks" of section 2 are the eigenmode breakdown of n5's impulse response.

**Why n5 fires one step late: a gate opened by silent comparators (`n2.py`).** n5's input weight is near zero (−0.28). Its pre-activation, split by what happened on the previous step (step 6, per unit M):

| previous step | n5 pre-activation | main contributions |
|---|---|---|
| new max | **+9.6** | new-max detector n4 +6.0; clock neurons n0, n10, n11, n12, n14 about +0.9 to +1.4 each |
| new 2nd max | **+2.9** | comparator n7 −4.1 (partly on, since x < M); clock neurons +1.6 to +3.0 |
| neither | **−21.9** | comparators n8 −15.1 and n7 −13.9 (both strongly on) |

So n5 = ReLU(steady clock drive − comparator activity + new-max bonus). It fires exactly when the comparators were silent on the previous step. It is one step late because it reads the comparators' output, which is itself computed from the current input. It needs no input weight, being built purely from existing neurons through recurrent weights. (I had guessed this is why learning-rate kicks favour one-step-late pointers. The kick data do not support that: see 7.14.) No single comparator neuron's impulse response decodes cleanly (0–6 of 10 ages), but a least-squares write direction concentrated on n10–n12 decodes all 10.

**Unified account:** an event-detector neuron writes a pulse, the recurrent weights turn it into a time-dependent pattern, and the readout is (approximately) the inverse of that pattern. What differs between models is the shape of the pattern:
- rotating, damped clocks with harmonics (mystery model);
- a non-rotating decaying path (seed 2);
- a chain of sparse time cells (32 neurons, 7.10).

**The readout is more than an inverse of the impulse response (`i11.py`, `i12.py`).** The minimum-norm decoder of the impulse responses alone looks little like the trained Wo (cosine 0.36 for the mystery model, 0.25 for seed 2). Swapped in, it gives only 14–16% accuracy.

Only 21% (mystery) and 8% (seed 2) of the centred readout's norm lies in the impulse-response span. Meanwhile the readout almost completely ignores the three highest-variance state directions: gain 0.09 and 0.08, against 0.44 for a random readout. Projecting the readout onto the span, or onto its complement, gives 13–15% accuracy either way.

So the readout does two jobs that only work together: it **decodes the pointer's path**, and it **cancels the large M-scaled components** (value registers, max clock, memory of other inputs) that would otherwise swamp it. The "matched readout" in the unified account should be read in that sense: correct argmax on the impulse responses, with most of its weight spent on rejecting everything else.

**How the max-to-2nd handoff works around the pointer (`i9.py`, `i10.py`).** n5 fires after a new max as well, so why isn't the new max read out as the answer?

To find out, compare new-max events at step 4 with a matched counterfactual where x₄ is lowered to (S+M)/2, a new-2nd event. Restrict to runs with no later updates (n = 6,605). Then push the state difference just after the pointer fires through the *gated* recurrence, using the actual on/off patterns. Pure linear propagation blows up, because neurons that normally switch off have eigenvalues above 1. This reconstructs the model's answer in 94.5% of cases; the model itself gives the old max in 97.5%.

| part of the difference | Δlogit at event slot | Δlogit at old max's slot |
|---|---|---|
| n5 pointer | +0.9 | −3.0 |
| n4, n9 max markers | −10.5 | +5.3 |
| clock neurons 0, 10–15 | +30.4 | −18.0 |
| comparators 1, 6, 7, 8 | −37.1 | +24.4 |
| n2, n3 | −1.4 | +4.5 |
| **total** | **−17.9** | **+13.1** |

The pointer fires the same way for both kinds of event. The handoff is carried by the **comparators**, which are all silent when x > M, and the **max markers**. Together they suppress the event slot and promote the old max's slot, against partial opposition from the clock neurons.

**Pointer selectivity is what separates the strategies (`i13.py`, `i14.py`, `y1.py`).** During training, the weights-only pulse test (some neuron whose Wo·Whh^a pulse decodes every age) is passed or nearly passed by every seed tested (`i13.py`, seeds 2, 3 and 0 from 2²² on). Seed 2 passes at every checkpoint, seed 3 at three of eight, and memory-based seed 0 only at 2²³, scoring 0.8–0.9 elsewhere. Near-passes are expected, because a linear memory also stores each input by its age. On the final weights the test does separate the models. Of all 64 candidates (16 neurons × sign × 0- or 1-step delay), only the mystery model's n5 (one-step delay) and seed 2's n10 decode every age; seeds 0, 3 and 4 peak at 0.89–0.90 and seed 1 at 0.33 (`y1.py`). Seed 2 settles on n10 as its pointer from 2²⁴ on; seed 0's best neuron keeps changing (n7, n3, n13, n0, n11).

A property that separates them throughout training is **event selectivity**: how often the pointer neuron fires after a new max, after a new 2nd max, and otherwise.

| neuron | new max | new 2nd max | otherwise |
|---|---|---|---|
| mystery n5 | 0.97 | 0.84 | **0.06** |
| seed 2 n10 | 0.98 | 0.91 | **0.08** |
| seed 3 n9 | 0.98 | 0.90 | 0.19 |
| seed 0 n0, n7 | 1.00 | 1.00 | **1.00** (always on: memory channels, not pointers) |

Pointer selectivity follows the same order as clean-case error (mystery and seed 2 best, seed 3 next, seed 0 worst). That fits the picture: the online selectors write *only* events, while the memory seeds write every input.

### 7.10 How the 32-neuron model's time cells are gated (`i5.py`)

Each time cell is released from inhibition. For n14, n22 and n2, the strongest inhibitor drops to exactly zero (h/M = 0) one step before the cell's preferred age. n15 is the exception: its inhibitor n16 is instead positive only at age 0.

| time cell | fires at age | inhibitor (weight) | inhibitor is zero at age |
|---|---|---|---|
| n14 | 1 | n12 (−1.70) | 0 |
| n22 | 2 | n28 (−1.62), n25 (−1.16) | 1 |
| n2 | 6 | n30 (−2.09) | 5 |
| n15 | 8 | n16 (−0.80), n11 (−0.54) | n16 positive only at age 0 |

The inhibitors are themselves "off at one age" cells, so the delay line is a **chain of release from inhibition**.

From the weights alone, pulses in n16 (an age-0 event cell) decode 8/10 ages exactly, and pulses in n14 decode 9/9 ages with a one-step delay. So the 32-neuron model also follows the write-pointer, impulse-response, matched-readout principle.

### 7.11 A fair, mechanism-informed accuracy estimator (`i6.py`–`i8.py`)

**Uniform sampling with structure.** The probe samples inputs uniformly. It then estimates error separately for each combination of decisive pair (M~S or S~R3), normalised gap and decisive step, shrinking sparse combinations toward the average. The weight of each combination comes from model-free Gaussian sampling. It is nearly unbiased (bias −0.0002 to −0.0014) but beats plain sampling only on small budgets: 1.4–1.6× lower MSE at 2,000 evaluations and about 1× at 8,000.

**Pilot-free stratified sampling.** Strata are the blind spot plus 10 normalised-gap bins, all assigned without running the model. Samples are allocated using **one generic prior, shared by every model**: error ≈ 0.4·e^(−g/0.03) + 0.01, and 0.75 in the blind spot. Nothing is tuned per model, and no pilot evaluations are made.

| model | MSE gain at 2,000 evaluations | at 8,000 |
|---|---|---|
| mystery | 1.9× | 2.3× |
| seed 2 | 1.8× | 1.9× |
| seed 3 | 1.9× | 2.4× |
| seed 0 | 1.7× | 1.3× |
| seed 1 | 1.2× | 1.7× |

Bias stays within ±0.0006.

**Weight-noise test** (ARC's suggested protocol), on 4 mystery-model copies per noise level at 4,000 evaluations (`i8.py`). Noise is relative for W_ih and W_o; for W_hh it is additive, with scale equal to the noise level × mean |W_hh|.

| weight noise | model accuracy | MSE gain |
|---|---|---|
| 0.3% | 94–95% | 2.1–2.8× (geometric mean 2.4×) |
| 1% | 83–90% | 1.3–1.8× (1.5×) |
| 3% | 48–60% | 0.6–1.0× (0.75×) |

At 3% noise the errors stop clustering near ties. The prior helps exactly as long as the mechanism is intact. This is a modest but genuine step toward ARC's "beat sampling" target. It still uses sampling inside each stratum.

### 7.12 Why training lands on this readout, and what changed at the switch (`j1.py`–`j4.py`)

**The readout is the convex optimum for the recurrence's features.** Refit Wo from scratch by softmax regression (L-BFGS on 400k inputs), keeping the trained recurrence fixed:

| model | trained readout accuracy | refit readout accuracy | cosine to trained Wo | outputs agreeing |
|---|---|---|---|---|
| mystery | 0.9530 | 0.9560 | **0.998** | 98.5% |
| seed 2 | 0.9481 | 0.9487 | **0.997** | 99.3% |
| seed 0 | 0.9080 | 0.9120 | **0.993** | 97.9% |

So the readout's structure (7.9: decode the pointer's path, cancel the large M-scaled directions) is simply what cross-entropy produces for these features. **At convergence, all the interesting structure is in the recurrence.** The readout's early dynamics still matter, though: delaying readout learning changes which recurrence forms (7.14).

**Across seed 2's switch,** the gain sits in the recurrence:

| checkpoint | optimal readout on that recurrence | own trained readout |
|---|---|---|
| 2²³ | 0.592 | 0.581 |
| 2²⁴ | 0.747 | 0.724 |
| 2²⁵ | 0.830 | 0.792 |
| 2²⁶ | 0.846 | 0.835 |
| 2³⁰ | 0.949 | 0.948 |

The trained readout lags the optimum by up to 4 points during the switch.

**The switch is n10 becoming selective.** Firing rate of n10 on "neither" inputs (not a new max or new 2nd max): 0.95 at 2²², 0.84 at 2²³, 0.79 at 2²⁴, **0.135 at 2²⁵**. Before the switch n10 is effectively always on, a memory channel. After it, n10 is a write pointer.

**Transplant test** (2²⁵ weights into 2²⁴), measured by n10's firing rate on "neither" inputs:

| transplanted | n10 "neither" firing rate |
|---|---|
| nothing (2²⁴ as is) | 0.79 |
| only n10's incoming row | **0.20** |
| all other rows (n10's kept) | 0.65 |
| W_ih only | 0.77 |
| W_hh only | 0.14 |

So the change is in n10's own **recurrent** weights. Fitting n10's threshold shows it snapping, between 2²⁴ and 2²⁵, from an unstructured negative value (≈ −1.1·M + 0.3·S, R² 0.25) to about **0.5–0.7·S** (R² 0.83, later 0.92). Meanwhile its input weight grew steadily (0.6, 2.4, 5.1, 7.5, 8.4).

Single full-batch gradient steps on n10's weights were uninformative. Even 1% steps raise the loss (by 0.17–0.27 at 2²⁴–2²⁵), so the surface is very sharp, and the real optimiser (Adam, mini-batches) would not take them.

### 7.13 Accuracy estimates that never sample the model on random inputs (`j5.py`–`j7.py`)

**Method.** The top-3 values (r, g₁ = S − r, g₂ = M − S) are integrated by Gauss–Legendre quadrature, with g = 4u² to put more points near ties. All 720 arrangements of their positions are enumerated. The exact order-statistic density is φ(M)φ(S)φ(R)Φ(R)⁷, and the quadrature mass checks out at 1.0000. The other seven inputs are set to their exact conditional mean, −φ(R)/Φ(R) ("mean field"), optionally with leakage added from the model's Jacobian. The only randomness is a reference sample used to measure the true accuracy.

| model | true accuracy | mean field (fine grid, 3.7M evaluations) | + Gaussian linear leakage | + truncation-aware linear leakage (coarse grid) |
|---|---|---|---|---|
| mystery | 0.9529 | **0.9559 (+0.30 points)** | 0.927–0.929 | 0.937 |
| seed 2 | 0.9483 | **0.9498 (+0.15)** | 0.937–0.940 | 0.922 |
| seed 0 | 0.9100 | **0.9115 (+0.15)** | 0.61–0.67 | 0.762 |

**Findings:**
- **Mean field is accurate to 0.15–0.3 points with zero randomness.** So with the other inputs fixed at their mean, the network's decision as a function of the top-3 values and positions is almost the whole story. The small positive bias looked like the leakage it leaves out; it later turned out to be mostly quadrature error (see the CORRECTION below).
- **It is not compute-efficient.** The mystery model's error is −3.5, −0.65 and −0.55 points at 69k, 207k and 1M evaluations, against sampling standard deviations of 0.08, 0.05 and 0.02 points at the same cost. The 720-arrangement enumeration is the cost driver, and the +0.3 bias remains (mostly grid error, as it turned out).
- **Linear leakage models make it worse, catastrophically so for the memory seed.** The Gaussian version ignores that the lower inputs stay below the 3rd value. In the memory model each lower input drives its own slot's logit with weight about 23, so uncapped noise there looks like strong competition. The truncation-aware version helps (seed 0: 0.61 → 0.76) but still overstates the leakage. The real network clips it: comparators saturate and switch off, so small inputs pass through far less than the linear sensitivity at the mean point suggests.

**A nonlinear additive leakage model also fails (`k9.py`).** Each lower input's effect on every competitor margin was computed through the real network at 5 quantile nodes of its truncated distribution. The effects were summed across inputs with a Gaussian approximation, and competitors were treated as independent. On a coarse grid it shifts the estimate by −8.4 (mystery), −9.6 (seed 2) and −17.2 (seed 0) points. The true leakage effect is −0.15 to −0.30 points.

**Why leakage is small, and why it is hard to model (`k10.py`).** I ran Monte Carlo over the lower inputs on 3,000 random grid cells of the mystery model (top-3 values and positions):

| quantity | value |
|---|---|
| mean-field P(correct) | 0.9577 |
| Monte Carlo P(correct) | 0.9558 (net leakage effect −0.2 points, matching the fine-grid finding) |
| cells correct under mean field but wrong more often than not under Monte Carlo | 0.6% |
| cells wrong under mean field but right more often than not under Monte Carlo | 1.2% |
| average Monte Carlo accuracy in cells that mean field gets right | 0.984 (1.6% lost) |

So leakage both creates and removes errors, and the two nearly cancel. That is why plain mean field is accurate to 0.15–0.3 points. Any leakage model that is wrong in the tails, or that misses the rescues, overshoots badly; the additive-Gaussian and linear models both do.

**A two-sided model that works: first-order cut-HDMR (`l1.py`).** Vary each lower input *alone* through the real network, at 12 quadrature nodes of its truncated distribution, with the others at their mean. Then add the indicator changes: P ≈ I₀ + Σₖ(Eₖ[I] − I₀). This is deterministic and nonlinear, and it counts rescues as well as breaks. Results on the same 3,000 random cells, against Monte Carlo over the lower inputs (300 draws per cell):

| model | mean field | Monte Carlo | cut-HDMR, 1st order | 2nd order (5-point quadrature) | per-cell RMSE: mean field / 1st / 2nd |
|---|---|---|---|---|---|
| mystery | 0.9577 | 0.9558 (−0.19 points) | 0.9541 (−0.35) | 0.9537 (−0.40) | 0.114 / **0.082** / 0.108 |
| seed 2 | 0.9487 | 0.9476 (−0.10) | 0.9460 (−0.26) | 0.9398 (−0.89) | 0.135 / **0.098** / 0.127 |
| seed 0 | 0.9117 | 0.9161 (**+0.45**) | 0.9153 (+0.36) | 0.9100 (−0.16) | 0.162 / **0.118** / 0.152 |

First-order cut-HDMR is the first leakage model with the right sign and roughly the right size. It even catches seed 0's net *positive* leakage, where rescues outnumber breaks. It also cuts the per-cell error by about 30%. Its remaining error is 0.09–0.16 points. Adding second-order terms made things worse, and a finer pair quadrature does not fix it. With 8 nodes for both orders (`n5.py`), the error against the Monte Carlo values is −0.10 points at first order and −0.30 at second order for the mystery model, and −0.18 and −0.54 for seed 2. Per-cell RMSE also rises (0.081 → 0.092 and 0.100 → 0.105). So the cut-HDMR expansion of the correct/incorrect indicator does not converge: the pair terms overshoot, and first order is the best this family does.

**Combining mean field with a coarse-grid leakage correction fails (`l3.py`).** I computed the cut-HDMR correction on a coarse grid (r10 × g8 × g8, 4 nodes) and added it to the fine-grid mean-field estimate:

| model | coarse-grid correction | true leakage effect | combined error | mean-field-only error |
|---|---|---|---|---|
| mystery | +0.30 points | −0.19 | +0.60 | +0.30 |
| seed 2 | +0.17 | −0.10 | +0.32 | +0.15 |
| seed 0 | −0.43 | +0.45 | −0.28 | +0.15 |

**The correction has the wrong sign every time.** The leakage effect is concentrated in near-tie regions that the coarse grid and 4-node quadrature don't resolve. A working sampling-free estimate therefore needs the cut-HDMR correction on the *fine* grid, about 85× the mean-field cost (about 310M evaluations, several CPU-hours here). I did not run that; on cost alone it cannot compete with sampling.

**Fine-grid estimate with the cut-HDMR correction (`n1.py`).** The full grid (r20 × g16 × g16 × 720 position arrangements) with 4 quadrature nodes per lower input, about 107M network evaluations per model and about 9.5 minutes each:

| model | true accuracy | mean field | + cut-HDMR leakage | correction applied | true leakage effect |
|---|---|---|---|---|---|
| mystery | 0.9529 | 0.9559 (+0.30) | 0.9561 (+0.32) | +0.02 | −0.30 |
| seed 2 | 0.9483 | 0.9498 (+0.15) | 0.9497 (+0.14) | −0.01 | −0.15 |
| seed 0 | 0.9100 | 0.9115 (+0.15) | **0.9092 (−0.08)** | −0.23 | −0.15 |

The correction captures seed 0's leakage but contributes almost nothing for the two online selectors. On the 3,000-cell validation, 12 quadrature nodes gave −0.35 and −0.26 points; that suggests 4 nodes are too few to resolve the narrow region where a lower input flips a near-tie. Re-running the mystery model with **12 nodes** (`n4.py`, about 310M evaluations, split over 3 processes) gives a correction of **−0.13 points** (estimate 0.9547, error **+0.18**, against +0.30 for mean field alone). The correction now has the right sign and recovers about 40% of the true −0.30 leakage effect, so quadrature resolution was indeed the limiting factor. The resulting **sampling-free estimates are within 0.08–0.18 points** of the truth for the two models checked at adequate resolution (seed 0 at 4 nodes, mystery at 12), and within 0.14 for seed 2 at 4 nodes, using no random inputs. At about 107M evaluations, though, plain sampling would be accurate to about ±0.002 points, so this is not competitive on cost.

**CORRECTION: most of the "leakage" was quadrature error (`o1.py`).** I evaluated the mean-field indicator MF(x) on 2M *sampled* inputs: the model run with the 7 lower inputs replaced by their conditional mean, keeping the true top-3 values and positions. Its average is an unbiased estimate of the mean-field accuracy, free of quadrature error:

| model | true accuracy | sampled mean field | net leakage (true − sampled MF) | quadrature mean field | quadrature grid error | outputs where f ≠ MF |
|---|---|---|---|---|---|---|
| mystery | 0.9532 | 0.9535 | **−0.03 ± 0.01** | 0.9559 | **+0.24** | 2.9% |
| seed 2 | 0.9489 | 0.9483 | **+0.06** | 0.9498 | +0.15 | 4.1% |
| seed 0 | 0.9103 | 0.9110 | **−0.07** | 0.9115 | +0.05 | 6.6% |

So leakage changes 3–7% of *individual* outputs, but its breaks and rescues cancel almost exactly in accuracy: the net effect is under 0.1 points. The +0.15 to +0.30-point errors of the fine-grid "mean-field" estimates above are **mostly discretization error of the quadrature near ties**, not leakage. The earlier 3,000-cell validation's leakage numbers (−0.19, −0.10, +0.45) were within its sampling noise (standard deviation about 0.2 points per cell average). The cut-HDMR "corrections" were therefore correcting mostly noise and grid error; their apparent success for seed 0 was partly coincidental.

**Plain quadrature does not converge; boundary-aware quadrature does (`o2.py`, `o3.py`).** Refining the Gauss–Legendre grid makes the error *oscillate* rather than shrink, because the integrand is a piecewise-constant correct/incorrect indicator. Mean-field error relative to the sampled mean-field value:

| grid | mystery | seed 2 | seed 0 |
|---|---|---|---|
| 20 × 16 × 16, u² spacing | +0.05 | −0.09 | −0.01 |
| 24 × 24 × 24, u³ spacing | +0.22 | +0.21 | +0.03 |
| 28 × 32 × 32, u³ spacing | +0.12 | +0.21 | +0.09 |

The fix is to treat the decision boundaries explicitly. For each line (3rd value, max-to-2nd gap, positions), scan the 2nd-to-3rd gap at 48 points, locate every point where the decision flips by 14 bisection steps, and integrate the smooth order-statistic density exactly over the resulting intervals:

| model | true accuracy | boundary-aware mean-field estimate | error vs true | error vs sampled MF | evaluations |
|---|---|---|---|---|---|
| mystery | 0.9532 | 0.9538 | **+0.06** | +0.03 | 14.0M |
| seed 0 | 0.9103 | 0.9111 | **+0.08** | +0.01 | 14.1M |
| seed 2 | 0.9489 | 0.9476 | **−0.13** | −0.07 | 14.1M |

So **a sampling-free estimate of accuracy accurate to about 0.1 points**: exact order-statistic integration, the mean-field treatment of the lower inputs (justified by the near-perfect cancellation of leakage), and boundary-aware integration of the decision. The remaining error comes from the ordinary quadrature in the other two variables, which could be made boundary-aware the same way. It is still not cheaper than sampling: 14M samples would give about ±0.006 points.

**Can it be made competitive? Cost-accuracy curve and the caching lever (`o4.py`).** For the mystery model:

| lines × scan points | evaluations | quadrature error | sampling SD at equal cost | ratio |
|---|---|---|---|---|
| 8 × 6, scan 12 | 0.70M | 0.71 points | 0.025 | 28× |
| 12 × 8, scan 16 | 1.80M | 0.78 | 0.016 | 50× |
| 16 × 12, scan 24 | 4.75M | 0.42 | 0.010 | 43× |
| 20 × 16, scan 48 | 13.95M | 0.063 | 0.006 | 11× |

The obvious structural lever is time-invariance and prefix sharing: with the lower inputs at their mean, the 720 arrangements share prefixes, so hidden states can be cached. The trie has **2,815 distinct prefix states against 7,200 naive steps per line, a 2.56× saving**. Even with perfect caching, the best configuration's error would remain about 7× sampling's at equal cost, and coarser grids' about 17–31×. (Sampling's error at a given cost scales with the square root of the budget, so a 2.56× saving improves each ratio by only 1.6×.) Matching sampling would need the quadrature error to fall about 10× faster with resolution than it does. With a discontinuous integrand in three dimensions plus a 720-way enumeration, I don't see a route to that. That is consistent with ARC's description of matching sampling as "a major research breakthrough".

**Optimized: fewer scan points and scale-adapted coordinates (`r1.py`–`r3.py`).** A sweep over the three resolution knobs showed that **scan points barely matter**: 8 per line is as good as 32. The error was set almost entirely by the outer Gauss–Legendre grid over (3rd value, max-to-2nd gap).

Because the network has no biases and uses ReLU, its decisions depend on *ratios* of inputs. So I changed coordinates to M plus the normalized gaps a = (M−S)/|M| and b = (S−R)/|M| (Jacobian M²), with the all-negative region integrated separately. The decision boundaries then barely move with M. Results, with mean field for the lower inputs, boundary-aware integration in b, 8 scan points and 10 bisection steps:

| model | 0.43–0.46M evaluations | 0.70–0.77M | 1.06–1.15M | 1.57–1.68M | 2.75–2.96M |
|---|---|---|---|---|---|
| mystery: error (ratio to sampling SD) | 0.46 points (14×) | 0.42 (17×) | 0.19 (9×) | **0.005 (0.3×)** | 0.07 (5.5×) |
| seed 0 | 0.52 (12×) | 0.28 (8.5×) | **0.03 (1.0×)** | 0.14 (6×) | 0.04 (2.5×) |
| seed 2 | 0.22 (7×) | 0.20 (8×) | 0.16 (7.5×) | 0.05 (3×) | 0.16 (12×) |

So at about 1.5–3M evaluations the error is 0.005–0.16 points, typically **3–6× the sampling error at equal cost**. That is down from 11–50× before, and occasionally at parity. But it does not converge monotonically: errors still oscillate with resolution, because the integrand is also discontinuous in a (the max-versus-2nd ties). The parity points should be read as lucky alignments, not a reliable result. (The reference "true" accuracies also have about ±0.015 points of sampling uncertainty.) The remaining levers are boundary-aware integration in a as well, and prefix caching (2.56× fewer network steps, which would reduce every ratio by about 1.6×).

**With better references, convergence checks and prefix caching (`s1.py`–`s8.py`, `cache.py`, `r4.py`).** References from 20M samples: mystery 0.95313, seed 2 0.94851, seed 0 0.91009 (±0.005–0.006 points). For the mystery model, a 200M-sample reference gives **0.953119 ± 0.000015** (±0.0015 points).

- **The estimator was limited by the inner scan, not the outer dimensions.** Refining M or a alone left the estimate wandering by ±0.08 points. The inner scan converges (12×12 grid: 0.95325, 0.95279, 0.95267, 0.95264 at 8, 16, 32, 64 scan points), so 8 points had sometimes been lucky.
- **The mean-field model's own exact value** (20M samples) is 0.95323 ± 0.00005 (±0.005 points): bias +0.01 ± 0.007 points, essentially none, for the mystery model.
- **Prefix caching** (2,815 trie states per line against 7,200 steps) is implemented and verified: logits identical to within 4e−13. It is used in the scan stage; bisection stays uncached.

Mystery model, cached, against the 200M reference:

| grid (M × a, scan) | evaluations (sequence-equivalents) | estimate | error | sampling SD at equal cost | ratio |
|---|---|---|---|---|---|
| 8 × 8, 16 | 0.61M | 0.95733 | +0.42 | 0.027 | 15.6× |
| 12 × 12, 16 | 1.37M | 0.95279 | −0.033 | 0.018 | 1.8× |
| 12 × 12, 24 | 1.81M | 0.95270 | −0.042 | 0.016 | 2.7× |
| 16 × 16, 24 | 3.20M | 0.95207 | −0.105 | 0.012 | 8.9× |
| 20 × 20, 24 | 4.97M | 0.95301 | −0.011 | 0.0095 | 1.1× |
| 24 × 24, 24 | 7.27M | 0.95378 | +0.066 | 0.0078 | 8.5× |
| **32 × 32, 24** | **12.8M** | **0.95310** | **−0.002** | 0.0059 | **0.36×** |

The other models, uncached, against their 20M references:

| model | 12 × 12 | 20 × 20 | 32 × 32 (≈27M evaluations) |
|---|---|---|---|
| seed 2 | −0.031 | +0.029 | **−0.004** (sampling SD at that cost about 0.004) |
| seed 0 | +0.129 | +0.071 | +0.084 (floor) |

**Conclusions.**
- *(Superseded: the randomized replicates below show this came from a fortunate grid.)* **For the two online selectors (mystery, seed 2), the estimator reaches and slightly beats sampling at about 13–27M evaluations.** Mystery is at 0.36× the sampling error with caching (error 0.002 against SD 0.006; the 200M reference's own uncertainty is 0.0015). Seed 2 is at about 1×, but its −0.004-point error is smaller than its 20M reference's own uncertainty (±0.005 points), so that comparison cannot be resolved.
- **For the memory model (seed 0), it converges to a +0.08-point floor:** the mean-field shortcut's real bias, since that model's lower-input leakage does not cancel. So it cannot match sampling.
- **Convergence is erratic at intermediate grids** (1.1× → 8.5× → 0.36×), and comparing grids does not give a trustworthy error bar. A practical "mechanistic estimate" would need its own reliable error estimate, which this does not yet have.

So this is parity at the largest grid for the selector models, not a robust win over sampling.

**With honest error bars, the estimator's error is 6–25× sampling's (`t1.py`, `t2.py`).** To give the estimator its own uncertainty, I ran K = 4 independent randomized replicates and used the spread across them as the standard error (mystery model, against the 200M reference):

| design | evaluations | standard error | ratio to sampling SD | errors (z-scores) |
|---|---|---|---|---|
| stratified grid with random jitter in each cell, n = 12 | ~7.0M | 0.145–0.201 points | **18–25×** | −0.74, −1.25, −0.93 |
| Gauss–Legendre with randomized nuisance parameters (M range, a-mapping exponent, R cutoff), n = 16 | ~12.9M | 0.034–0.066 points | **6–11×** | −0.05, +0.46, +2.75 |

The error bars are roughly calibrated, though that rests on three z-scores, one of them 2.75. At about 13M evaluations, perturbing the grid's nuisance parameters moves the estimate by a few hundredths of a point. That is several times sampling's standard deviation, which suggests **the 0.36× result above came from one fortunate fixed grid**. (The replicates were run at 16 × 16, not at the 32 × 32 grid itself.) With an honest error bar, this mechanistic estimate's standard error is about 6–25× sampling's at equal cost. Matching sampling's error would take far more than 6–25× the compute, since sampling's error falls only as the square root of its budget. ARC's "match sampling" target is not met.

Two limits on these error bars:
- **K = 4 replicates give 3 degrees of freedom**, so each standard error is itself uncertain by a factor of about 0.6–3.7 (95%). For the better design, "6–11×" is consistent with anything from about 3× to 40×.
- **Randomizing the grid's nuisance parameters captures grid error, but not biases shared by every replicate.** These include the mean-field bias (seed 0's +0.08 points would be invisible) and decision flips that the scan misses.

The verdict holds either way. In the better design, one realized error is itself at least 0.09 points (z = 2.75 with a standard error of 0.034 or more), about 15× sampling's standard deviation at that cost.

**Fixing seed 0's bias by treating the 4th-largest input exactly fails (`r5.py`).** I extended the estimator to 4 exact order statistics: 5,040 position arrangements, a fifth symbol for the remaining six inputs at their conditional mean below R4, an extra Gauss–Legendre dimension for the 3rd-to-4th gap, density φ(M)φ(S)φ(R)φ(R4)Φ(R4)⁶·|M|³, and a 17,254-node prefix trie against 50,400 steps:

| model | grid | evaluations | mass | estimate | error |
|---|---|---|---|---|---|
| seed 0 | 8 × 8 × 4 | 15M | 0.936 | 0.90903 | −0.11 |
| seed 0 | 12 × 12 × 6 | 50M | 0.987 | 0.91291 | +0.28 |
| seed 0 | **16 × 16 × 10** | **147M** | **1.0000** | **0.91277** | **+0.27** |
| mystery | 12 × 12 × 6 | 48M | 0.987 | 0.95500 | +0.19 |

With the mass converged, seed 0's error is **+0.27 points, worse than the 3-input mean field's +0.08**. I first explained this through the memory model's linear memory channels, but that explanation does not hold: the selector gets worse too. A sampled check, free of quadrature error (4M inputs, `y6.py`), gives these mean-field biases:

| model | 3 exact order statistics | 4 exact order statistics |
|---|---|---|
| seed 0 | +0.08 ± 0.01 points | +0.31 ± 0.01 |
| mystery | +0.01 ± 0.01 | **+0.16 ± 0.01** |

Fixing the lower inputs at their conditional mean is not an approximation that improves as more order statistics are made exact. The 3-input version's near-zero bias for the selectors looks like a cancellation.

**A mechanism-based control variate does not help (`o1.py`).** The estimate mean(f − MF) + E[MF], with E[MF] from quadrature, has 0.4–0.7× the efficiency of plain sampling at equal evaluations. Each sample needs two network evaluations, the correlation between f and MF is only 0.58–0.67 (with the unit coefficient used here, at least 0.75 is needed to break even; 0.71 with an optimal coefficient), and the quadrature value of E[MF] carries the grid bias.

**Conclusion.** Leakage is nonlinear and *two-sided*. It changes 3–7% of individual outputs, but its breaks and rescues cancel to under 0.1 points net for these models. The larger 0.15–0.3-point errors earlier in this section were mostly quadrature error, not leakage. None of the explicit leakage models tried (Gaussian linear, truncated linear, additive nonlinear) gets both the losses and the rescues right.

### 7.14 Replaying training: when and why a run commits to a strategy (`k1.py`–`k4.py`)

I continued training from zoo checkpoints with a fresh AdamW and new data orders, using AlgZoo's schedule: peak learning rate 0.01 (0.04/√16), 16,384-step warmup, batch 1024, weight decay 0.001. The original optimizer state isn't saved, so this is a fork, not an exact replay. Score: the most selective pointer, meaning the lowest "neither" firing rate among neurons that fire on ≥80% of new-2nd events.

**Forking from 2²⁴, 2²⁴ more sequences, 3 data orders each:**

| seed | most selective pointer ("neither" rate) | accuracy |
|---|---|---|
| seed 2 | n10, 0.13–0.14, in **all 3** runs | 0.821–0.825 (original run at 2²⁵: 0.792) |
| seed 0 | n8, stays at 0.80, in **all 3** runs | 0.71–0.77 |

**By 2²⁴, each run's strategy is fixed, regardless of later data.**

**Forking earlier,** with a 500-step warmup to the full learning rate, 2²⁵ more sequences:

| fork point | seed 2 | seed 0 |
|---|---|---|
| 2²⁰ | 3/3 switch (n10, 0.14–0.22) | **3/3 switch** (n0 with one-step delay, 0.16–0.22) |
| 2²² | 3/3 switch | 3/3 switch, clean-case error ≈1.1–1.4%, below the original seed 0's 1.37% at 2³⁰ |

So memory-based seed 0 was *not* committed at 2²²; it committed between 2²² and 2²⁴.

**Learning-rate control.** In the original run, 2²⁰ sequences falls inside the 16,384-step warmup, at about 6% of the peak learning rate. Repeating the 2²⁰ forks with the *original* schedule:

| seed | runs that switch |
|---|---|
| seed 0 | **0/3** (pointer stays at 0.80) |
| seed 2 | **2/3** |

**From initial weights** (2²⁵ sequences, one data order):

| seed | original warmup | 500-step warmup |
|---|---|---|
| seed 0 | switches (0.20) | switches (0.19) |
| seed 1 | no (0.79) | **switches** (0.12) |
| seed 4 | no (0.80) | no (0.80) |

**More runs from initial weights** (`k5.py`; all 5 seeds × 2 warmups × 2 more data orders, 2²⁵ sequences, batch 1024 as before):

| warmup | runs that switch |
|---|---|
| 500 steps | 4/10 |
| 16,384 steps | 3/10 |

Including the three runs above, **6/13 (46%) switch with the fast warmup and 4/13 (31%) with the original**. That difference is not significant at this sample size.

The earlier lopsided result (8/9 against 3/9) came mainly from the **2²⁰ forks**. There, a fresh optimizer jumps to the full learning rate *in the middle of* training. That "kick" made seed 0 switch in 3/3 runs, against 0/3 with the original schedule. Warmup length from scratch does not clearly matter.

Per seed, from initial weights (both warmups). These counts sum to 9/24 rather than the 10/26 pooled above. The logs needed to reconcile them are not in the repository, and the larger per-seed table below supersedes them.

| seed | switch rate |
|---|---|
| 0 | 3/5 |
| 1 | 2/5 |
| 2 | 1/4 |
| 3 | 3/4 |
| 4 | **0/6** |

Switching does not pay off immediately. At 2²⁵ sequences, some memory runs are ahead of selector runs (for example 0.889 against 0.827).

**Per-seed switch rates (`k7.py`, `k11.py`, `l4.py`, `l8.py`).** 60 more runs from initial weights with the original schedule, pooled with the earlier original-schedule runs (14–15 runs per seed):

| seed | runs that switch | 95% interval (Clopper–Pearson) |
|---|---|---|
| 3 | **14/14** | 0.77–1.00 |
| 0 | 5/15 | 0.12–0.62 |
| 2 | 4/14 | 0.08–0.58 |
| 1 | 2/15 | 0.02–0.40 |
| 4 | 2/15 | 0.02–0.40 |
| **all** | **27/73 = 0.37** | |

**Initial weights matter.** Seed 3's starting point *always* produces a selector (14/14), while seeds 1 and 4 rarely do (2/15 each); the intervals do not overlap. The zoo's best seed (seed 2) switches in only 4 of 14 runs from scratch, so its original run was partly luck of the data order. Fresh initializations switch at 50% (24/48, below), within the range of the zoo seeds.

At 2²⁵ sequences, runs that switched are already ahead: mean accuracy 0.822 against 0.785, and clean-case error 6.8% against 10.7%.

**Fresh random initializations (`k12.py`, `k13.py`).** I trained 24 new initializations (PyTorch default init) with the original schedule for 2²⁵ sequences. **13/24 (54%) switch**, in line with the zoo seeds' pooled 37% (27/73). Switched and non-switched runs from these starts have the same mean accuracy at that point (0.838).

None of the simple initialization statistics predicts switching. I tested: number of positive input weights, largest and smallest input weight, mean |input weight|, spectral radius of W_hh, mean self-connection, self-connection of positive-input neurons, and the largest input-weight × self-connection product. All Mann–Whitney p-values are ≥ 0.08; the smallest is for the most negative input weight. Whatever makes seed 3 switch in 14/14 runs and seed 4 in 2/15 is a property of the specific configuration, not of these scalar summaries.

**How a pointer neuron gets recruited (`k14.py`).** I trained 10 fresh initializations, recording each neuron's input weight and self-connection at steps 0, 256, 1,024 and 4,096 (4,096 steps ≈ 2²² sequences). 9/10 switched.

The pointers come in the two kinds seen before:
- **7 with a strong positive input weight** at the end (4.9 to 7.1). They fire on the event itself, like seed 2's n10.
- **2 with almost no input weight** (−0.31 and −0.15). They fire one step late, like the mystery model's n5.

For the 7 positive-input pointers (rank 0 = largest of 16):

| quantity | rank |
|---|---|
| input weight at initialization | 0–5 (median 3): drawn from the neurons that start with positive input weights, but not reliably the largest |
| input weight at step 4,096 | 0–2 |
| input-weight growth over steps 0–4,096 | 0–3 |
| self-connection at initialization | 0–15 (no pattern) |

**Recruitment is a competition among the initially positive-input neurons, decided by early growth rather than by starting values.** It is settled by about 2²² sequences, matching the fork timing above. That explains why no scalar summary of the starting weights predicts the outcome.

**What the growth competition is about: threshold targets (`l2.py`).** For 10 more fresh initializations, I tracked the neuron with the largest input weight at step 4,096 (the "leader"): its "neither" firing rate and its threshold fit (≈ a·M + b·S) at steps 1k, 2k, 4k, 8k and 32k. Once its threshold structure fits well (R² 0.75–1.0), it lands in one of three places:

| where the leader's threshold ends up | role | inits |
|---|---|---|
| ≈ 0.87–0.93·M | **new-max detector** (the most common) | 1, 3, 7, 8 |
| ≈ 0.5–0.6·S | **new-2nd-max pointer** | 0, 9 |
| strongly negative (fires ≥ 96% on "neither") | **always-on memory channel** | 4, 6 |
| blends ≈ 0.3·S + 0.7·M | near-max detector | 2, 5 |

In early training (steps 1k–4k) the leader's threshold is unstructured (R² ≤ 0.4) and it fires on most inputs. Its eventual role is fixed between about 4k and 8k steps (2²²–2²³ sequences). That matches the fork timing.

Because the leader most often becomes the **new-max detector**, selection requires a *second* neuron to settle on the 2nd-max threshold. In inits 1, 3 and 7 the final pointer is a different neuron from the leader. In init 8 no second neuron did, and the run stayed memory-based. When the leader itself lands on S (inits 0, 9), it is the pointer.

Kick versus no kick, seed 0 forked from 2²² with the same data order:
- **No kick:** the leader (n5) goes to a near-max threshold (0.69·M + 0.27·S, R² 0.88). No pointer forms.
- **Kick:** the pointer that forms is n0, with an input weight near zero (−0.64). That is the **one-step-late** kind, the mystery model's n5 design. **But this does not generalize.** Across all 12 kicked fork runs (`k2.py`), every seed 0 fork produced a one-step-late pointer (n0, 6/6), and every seed 2 fork produced a pointer firing on the event itself (n10, 6/6). The kick makes switching likely; *which* pointer type forms depends on the seed, not the kick.

So the competition is over *which thresholds the positive-input neurons settle on*:
- M is the easiest target and gets taken first.
- A run becomes a selector if another neuron (positive-input, or a one-step-late relay) settles on the 2nd-max threshold before about 2²³ sequences.
- If neither happens, a neuron settles on the always-on target, and the run becomes memory-based.

**Which threshold the loss prefers, and when (`l5.py`).** At the zoo checkpoints, I rewired the leader's recurrent row by least squares so its threshold equals a target: λ·S + (1−λ)·M, or "always-on" (−2M). I then refit the readout and measured test loss. The table gives Δloss relative to the leader's original row, with the readout refit in both cases. All rewirings are worse than the co-adapted original row, as expected.

| checkpoint | always-on | M (λ=0) | λ=0.25 | λ=0.5 | λ=0.75 | S (λ=1) | row-fit R², M → S |
|---|---|---|---|---|---|---|---|
| seed 2, 2²² | +0.41 | +0.12 | +0.10 | +0.07 | **+0.06** | +0.06 | 0.77 → 0.56 |
| seed 0, 2²² | +0.41 | +0.18 | +0.17 | +0.15 | +0.14 | **+0.13** | 0.73 → 0.60 |
| seed 2, 2²³ | +0.76 | +0.08 | **+0.05** | +0.06 | +0.11 | +0.20 | 0.89 → 0.60 |
| seed 0, 2²³ | +1.09 | +0.26 | **+0.18** | +0.19 | +0.29 | +0.47 | 0.91 → 0.62 |

Two findings:
- **At 2²², the loss prefers S-ward thresholds, but they are harder to realize.** S is less linearly available in the state than M (row-fit R² 0.56–0.60 against 0.73–0.77).
- **By 2²³ the preference has flipped to M-ward thresholds**, because the rest of the network has co-adapted around the current design.

Always-on is the worst target throughout, so the memory channel is not something the loss pulls the leader toward directly. That gives a loss-landscape account of the fork: the selective (S) target is better early, but it is reachable only where the state already encodes S well enough, and the window closes by about 2²³.

**Does early S-encoding predict switching? No (`l6.py`, `l7.py`).** Across the 5 zoo seeds, the gap between M's and S's linear decodability at 2²² happened to order the switch rates almost perfectly, using the per-seed rates available at the time (gap 0.02 for seed 3, the most frequent switcher, up to 0.18 for seed 1, one of the two least frequent; rank correlation about −0.9). On 12 fresh runs it does not predict switching: at step 4,096 the gap is 0.126 for runs that switch and 0.116 for runs that don't (Mann–Whitney p = 0.87; p = 0.31 at step 2,048). The 5-seed pattern was a coincidence. Which feature of the early state decides the fork remains unidentified.

**Where seed 3's predisposition lives: initialization transplants (`l9.py`, `l10.py`).** I transplanted parts of seed 3's initial weights (switches 14/14) into seed 4's (switches 2/15), trained with the original schedule for 2²⁵ sequences, and used 4 data orders per condition unless noted:

| seed 3 part used | runs that switch |
|---|---|
| W_ih only | 1/4 |
| W_hh only | 0/4 |
| W_o only | 1/4 |
| W_ih + W_hh | 0/4 |
| W_ih + W_o | 0/4 |
| **W_hh + W_o** | **4/4** (pointer "neither" rate 0.09–0.13) |
| W_hh + W_o on all neurons (2 runs) | 2/2, pointer **n13, one-step-late**, both runs |
| W_hh + W_o, neurons 0–7 only (rows, columns and readout columns) | 2/4 |
| W_hh + W_o, neurons 8–15 only | 1/4 |
| W_hh + W_o, **n13 only** (its row, column and readout column) | 0/4 |
| W_hh + W_o, **all neurons except n13** | 1/4 |

The predisposition lies in the **joint configuration of the initial recurrent and readout weights**; seed 3's input weights are irrelevant. Neither part alone is enough. It is **spread across neurons**: each half transfers it only partly. **Neither n13's own initial connections nor the rest of the network without n13 is enough** (0/4 and 1/4, both near seed 4's own 2/15). The property is an *interaction* between n13 and its seed 3 surroundings. Its effect is to turn n13 into a **one-step-late pointer**, the mystery model's n5 design. Which pointer type forms is seed-specific: seed 0's kicked forks also produce one-step-late pointers, while seed 2's produce event-firing ones.

**The predisposition depends on the initial readout (`m1.py`–`m6.py`).**

*Same forward dynamics, different outcome.* "W_hh only" and "W_hh + W_o" have identical forward dynamics at initialization, since the readout doesn't affect the hidden state; n13 is always on (0.99) in both. Yet they switch 0/4 and 6/6. So the predisposition is not in what the network computes at the start. It is in the **gradient the initial readout sends back**.

*The initial gradient on n13's incoming recurrent weights* (200k inputs, float64):

| readout at initialization | n13's rank by gradient size (0 = largest of 16) | size / median | cosine to the full transplant's gradient | runs that switch |
|---|---|---|---|---|
| seed 3 W_o (full) | **0** | **2.64** | +1.00 | 6/6 |
| seed 4 W_o | 7 | 1.03 | **−0.88** | 0/4 |
| only n13's column from seed 3 | 1 | 1.71 | +0.99 | 0/4 |
| all columns from seed 3 except n13's | 7 | 1.14 | −0.86 | 1/4 |

- **n13's own readout column sets the direction it is pushed in.** With that column alone, the gradient direction is reproduced (cosine +0.99), including the sign of its input-weight gradient.
- **The other columns decide whether n13 wins the competition.** With only its own column, n13 ranks second and another neuron gets the larger push; the run does not switch. With the full readout, n13 dominates (2.64× the median).
- Neither column set alone transfers switching in training (0/4 and 1/4), matching the gradient picture.

Short training runs cannot stand in for full ones: n13's selectivity in a full transplant emerges only between steps 12,288 and 16,384, and in one of two runs a different neuron became the pointer.

*Is there a general predictor at initialization?* No robust one. On the 24 fresh initializations:

| statistic at initialization | runs that switch | runs that don't | Mann–Whitney p |
|---|---|---|---|
| winner's gradient / runner-up's | 1.32 | 1.21 | 0.056 (borderline) |
| winner's gradient / median | 2.68 | 2.73 | 0.39 |
| winner's input-weight gradient | — | — | 0.64 |

**A multivariate classifier also finds nothing (`n3.py`).** I used 28 initial-weight and initial-gradient features for each of the 48 fresh runs: the top-6 gradient-size profile; input-weight gradients, input weights and self-connections of the 4 most-pushed neurons; input-weight extremes; spectral radius; and readout norm. Leave-one-out logistic regression scores **46–50% accuracy** against 50% chance (permutation p = 0.51–0.57, C ∈ {0.03, 0.1, 1}).

The winner's-margin statistic **failed to replicate**. On 24 further fresh initializations (`m7.py`, switch rate 11/24) it points the *other* way: 1.19 for runs that switch against 1.24 for those that don't (p = 0.27). Over all 48 runs, p = 0.57. The first p = 0.056 was noise. It also does not order the 5 zoo seeds (Spearman −0.15). The overall fresh-initialization switch rate is **24/48 = 50%**. The winner's input-weight gradient orders the zoo seeds well (+0.87) but fails on fresh runs.

**A reduced model: a frozen random readout is enough (`p3.py`).** I trained the full seed 3 transplant (seed 3's recurrent weights and readout, seed 4's input weights; switches 6/6 normally) with parts frozen, 3 data orders each:

| what trains | runs that switch | pointer |
|---|---|---|
| input + recurrent weights, **readout frozen at seed 3's initialization** | **3/3** | n13, one-step-late, in all 3 |
| recurrent weights + readout, input weights frozen | 2/3 | n6 (event-firing), n13 (one-step-late) |
| **only n13's incoming row** (+ its input weight) and the readout | **0/3** | none (best "neither" rate 0.46) |

With the readout frozen, a selective n13 one-step-late neuron still forms every time. **But see the caveat below:** frozen-readout networks barely work, so this shows only that the *selective neuron* forms, not the full selection algorithm. But n13's own incoming weights cannot build it alone: the rest of the recurrence must learn too, consistent with n5-style pointers reading comparators and detectors that have to be learned first (7.9). Freezing further, with the readout frozen and the incoming rows of half the neurons held at initialization (never n13; `p4.py`):

| rows frozen | runs that switch | pointer |
|---|---|---|
| neurons 0–7 | **0/3** | none |
| neurons 8–15 (except n13) | **1/3** | n4, not n13 |

Neither half's learning can be dispensed with, and n13 never becomes the pointer when either half is frozen. **The n13 pointer is assembled by recurrent learning across the whole network, not by a local circuit.** So the competition can be studied in a simpler system: recurrent weights learning toward a fixed random readout. But even that system needs essentially all recurrent weights to learn.

**Caveat: a selective neuron is not the same as the selection algorithm (`q1.py`, `q2.py`).** I froze 8 fresh random readouts and paired each with seed 3's and with seed 4's initial recurrent weights. An event-selective neuron formed in **14/16** runs (6/8 with seed 3's recurrence, 8/8 with seed 4's, which switches only 2/15 under normal training). Seed 4's *own* readout frozen gives 4/4.

The selectivity criterion is lenient, though (`y5.py`). Untrained weights already score 0.26–0.30 on it, and a bare ReLU(x_t) neuron scores 0.30, both under the 0.4 cutoff. Under normal training the memory runs sit at 0.76–0.80, so the criterion separates trained runs. But in these poorly trained frozen-readout networks (here and in `p3.py`/`p4.py` above), some of the "selective" neurons may be no more than input-sign detectors.

But these networks are poor: on 8 fresh initializations with the readout frozen, accuracy is **44–58%** and clean-case error 31–49%. The same initializations with a trainable readout (`k12.py`) reach 77–92% and 0.7–12%. Selective neurons also don't become more common on matched runs: 5/8 frozen against 6/8 trainable.

**Delaying readout learning helps (`q3.py`).** The same 8 fresh initializations and data orders, with the readout frozen only for the first 8,192 steps (about 2²³ sequences) and trainable afterwards:

| | normal training (`k12.py`) | readout frozen for the first 8,192 steps |
|---|---|---|
| runs that switch | 6/8 | **7/8** |
| mean accuracy at 2²⁵ | 0.850 | **0.876** |
| mean clean-case error | 4.6% | **3.1%** |
| best accuracy | 0.921 | **0.944** |

Accuracy improves for 6/8 initializations. Initializations 0 and 5 go from memory to selection; initialization 1 goes the other way. At n = 8 this is suggestive, not significant (sign test p ≈ 0.29). **Extended to 24 matched pairs (`q4.py`):** selective pointers form in **20/24 against 13/24** runs. Mean accuracy at 2²⁵ is **0.864 against 0.838**, improved in **18/24 pairs** (sign test **p = 0.023**; Wilcoxon p = 0.087). Mean clean-case error is **3.8% against 5.6%**. Read the p-value with care. It is the better of the two tests, and the 24 pairs include the 8 pilot pairs whose result prompted the extension; the 16 new pairs alone improve in 12/16 (sign test p = 0.077). The difference in switch rates (20/24 against 13/24) was not tested; as unpaired counts it gives Fisher p = 0.06. So delaying readout learning by about 2²³ sequences probably makes the selection strategy more likely and improves accuracy at this stage of training, but the evidence is not yet decisive. **Longer training (`q5.py`, 2²⁷ sequences, the first 8 pairs):** mean accuracy **0.917 against 0.897**, improved in 6/8 pairs, with the same two initializations (0, 1) losing as at 2²⁵. Switch rate is 7/8 against 6/8. The best delayed run reaches **0.958**, above the zoo's best 16-neuron model (0.949, trained 8× longer). So the gain persists with 4× more training, but at n = 8 it is not significant by itself (sign test p ≈ 0.29). **Full training budget (`q6.py`, 2³⁰ sequences, the original schedule, 2 pairs):**

| initialization | normal training | readout frozen for the first 8,192 steps |
|---|---|---|
| 4 | 0.952 (clean-case error 0.29%) | **0.968** (0.12%) |
| 6 | 0.954 (0.06%) | **0.976** (0.21%) |

The normally trained runs match the zoo's best 16-neuron model and the mystery model (0.949–0.953). The delayed-readout runs reach **0.968–0.976**, the highest accuracy of any 16-neuron, length-10 model seen in this work, with the same architecture and budget. All four are online selectors, so at full budget the gain shows up as accuracy, not as a change of strategy. With n = 2 this is suggestive, though it agrees with the significant 24-pair result at 2²⁵ and the 2²⁷ result. It fits the picture that **early co-adaptation of the readout is what lets the memory solution win**: if the readout waits until event-selective neurons have formed, it more often builds the selection algorithm on them. Unlike the fully frozen readout, this gives working selectors.

So **forming an event-selective neuron is easy, and is necessary but not sufficient for the selection strategy.** Turning such a neuron into a working selector requires the readout to co-adapt. The frozen-readout "reduced model" above therefore captures the formation of the pointer neuron, not the full fork. In the normal-training statistics of this section, a selective pointer always coincided with good selection behaviour (lower clean-case error), so that metric remains valid *there*. The predisposition is a property of that readout's geometry relative to the initial recurrence.

**The initial-gradient account fails, at least by the lead-over-runner-up measure (`p1.py`, `p2.py`).** A greedy search found a minimal readout transplant: seed 3's readout columns for n13, n5, n3 and n14, with seed 3's recurrent weights and everything else from seed 4. It gives n13 an even *larger* lead in the initial gradient (1.46× the next neuron, against 1.33 for the full seed 3 readout and 0.56 for seed 4's). Yet it switches in **0/6** training runs; the pointer stays at 0.80. So winning the step-0 gradient competition is **not sufficient**. On the size-over-median measure used in the table above, though, the minimal transplant is weaker than the full one (1.97 against 2.64, `y7.py`). That leaves this statistic untested. The readout-dependent predisposition is real (full readout 6/6 against 0/4), but it acts through the *training trajectory*, not through the initial gradient alone.

**Interpretation.** Memory versus selection is a **stochastic fork early in training**, taken by roughly a third to a half of runs. It depends on data order and initial weights; seed 4 seems strongly biased against switching. It is settled by about 2²⁴ sequences. A sudden learning-rate increase before then (2²⁰–2²²) pushed all 12 fork runs into the selection basin. Those runs cover only two seeds, and seed 2 also switches without the kick (2/3). The kick's effect rests mainly on seed 0: 6/6 with the kick against 0/3 without (Fisher p = 0.012). That suggests the selector solution sits behind a barrier that a large enough update can cross, while ordinary training crosses it only sometimes.

**The barrier, measured (`k6.py`).** I re-created one switched fork of seed 0: from 2²², about 2²⁴ sequences in total, pointer "neither" rate 0.24. Along the straight line in weight space between it and the original memory model at 2²⁴ (rate 0.80):

| position | 0.0 (memory) | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 (switched) |
|---|---|---|---|---|---|---|---|
| loss | 0.617 | 1.283 | 1.953 | **2.057** | 1.972 | 1.291 | 0.502 |
| accuracy | 0.751 | 0.580 | 0.508 | 0.507 | 0.524 | 0.615 | 0.798 |

That is a linear barrier of about 1.4–1.6 nats. Pointer selectivity changes gradually along the path. Both models descend from the same 2²² checkpoint, so a relabelling of neurons is an unlikely explanation. At a matched amount of training, the selector basin is already lower in loss (0.502 against 0.617).

A straight-line barrier does not by itself show separate basins, though. The straight line from that 2²² parent to its own 2²⁴ descendant in the original run, two points on one training trajectory, peaks at 4.16 nats against endpoint losses of 1.30 and 0.62 (`y4.py`). Straight-line barriers this size are normal between distant points here, so the curved path below is the informative measurement. **A curved path is nearly flat (`k8.py`).** I optimized a quadratic Bézier curve between the same two endpoints, training only the control point with Adam (6,000 steps, t ~ U(0,1), batches of 2,048):

| t | 0.0 (memory) | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 (switched) |
|---|---|---|---|---|---|---|---|
| loss | 0.617 | 0.685 | **0.687** | 0.631 | 0.551 | 0.487 | 0.504 |
| pointer "neither" rate | 0.80 | 0.80 | 0.80 | 0.66 | 0.48 | 0.30 | 0.23 |

The barrier is only **about 0.07 nats**, against about 1.44 on the straight line. So the two basins are joined by a low-loss path. Along it, the pointer becomes selective gradually, after a small hump near the memory end (t ≈ 0.2–0.3).

The memory solution is therefore not walled off. It sits behind a shallow ridge that ordinary, slowly decaying training often fails to cross but a learning-rate kick crosses easily. That is consistent with the 12/12 kicked forks.

**Caveats.** There are 1–6 runs per cell. "Switching" is judged by the existence of a selective pointer neuron (a "neither" rate below 0.4). That criterion separates trained memory runs (0.76–0.80) from selectors (0.08–0.19), but untrained weights also pass it (0.26–0.30, `y5.py`). The training logs behind these counts are not in the repository (see README.md). The fork runs use a fresh optimizer and were trained for only 2²⁵ sequences.

### 7.15 Second round on accuracy estimation: task-only designs, held-out seeds, adaptive sampling, and explanation-predicted boundaries (`w1.py`–`w9.py`, `x1.py`–`x3.py`)

This round started from an analysis by ChatGPT Pro, given the question in `QUESTION_FOR_CHATGPT_PRO.md`. Its code was not recoverable, so I reimplemented its two main proposals from the description and checked them independently. Then, following the brief "use the explanation, and make finding it cheaper than brute force", I tested explanation-driven estimators across independently trained seeds.

**ARC's bar, for reference.** ARC would "consider it a major research breakthrough" to produce a *mechanistic* estimate, one that "reasons deductively based on the structure of the model" rather than from sampled examples, that "matches the performance of random sampling in terms of mean squared error versus compute". The comparison is to be averaged over random seeds, and weight noise is suggested only as a cheap proxy. By that definition, everything in this section except the last part still samples.

**A population of seeds (`w3.py`).** Weight-perturbed copies of one network are not independent draws, so I trained 16 fresh 16-neuron seeds (`pop/p0–15.npz`) with AlgZoo's schedule, scaled to 2²⁷ sequences to fit the compute available. They reach 0.73–0.88 accuracy, less than the zoo's 2³⁰ models. With the mystery model and seeds 0–4, that gives 22 independent networks.

**Task-only designs (`w1.py`, `w2.py`, `w5.py`).** Two unbiased estimators with honest error bars from independent scrambles:

- **Ordered Sobol.** An exact transport of N(0,1)¹⁰ into top-3 order statistics, an arrangement and 7 exact lower inputs, driven by scrambled Sobol points. It costs one forward pass per point. The design depends only on the task, so it is free prework.
- **Exact last-input integration.** Given x₁..x₉, the logits are piecewise linear in x₁₀, with at most 16 hinges, and the target changes only at the prefix's max and 2nd max. So P(correct | x₁..x₉) is a sum of at most 19 terms Φ(U) − Φ(L). This matches a dense numerical grid to about 1e-5. The fair cost, from matched numba kernels, is **2.2–2.5 forwards per prefix**. A NumPy version measured 5.3, an implementation artifact. ChatGPT reported 2.3 in C++.

Variance reduction against plain sampling at equal compute:

| network | ordered Sobol, 2¹² | 2¹⁶ | last input + Sobol per prefix, 2¹² | 2¹⁶ | net at 2.3 forwards per prefix, 2¹⁶ |
|---|---|---|---|---|---|
| mystery | 2.4× | 3.0× | 6.0× | 9.6× | 4.2× |
| seed 2 | 1.8× | 2.2× | 4.7× | 7.7× | 3.3× |
| seed 3 | 1.7× | 2.1× | 5.1× | 5.7× | 2.5× |
| seed 0 | 1.4× | 1.4× | 3.8× | 7.2× | 3.1× |
| seed 1 | 1.7× | 2.3× | 3.9× | 5.9× | 2.6× |
| seed 4 | 1.4× | 1.6× | 3.4× | 6.8× | 3.0× |

Gains are from 16–64 replicates, so each is uncertain by about ±50%; a plain-sampling control in the same runs reads 0.7–1.6×. **Integrating a second input does not pay (`w6.py`, `w7.py`).** With random prefixes, x₁₀ alone gives 1.9–2.1× per prefix; adding x₉ (exactly, or by 2–8 stratified points) raises that only to 2.5–3.5×. The cost rises much faster, to 2.8–13.5 forwards, so net efficiency falls from about 1.0× to 0.3–1.0×. On its own, conditioning on an input removes variance at about the rate it adds cost. The last-input method's net advantage above comes from pairing it with the Sobol design. None of this uses the mechanism: these are generic numerical tricks, capped at small constant factors.

**Most informative runs first (`w8.py`).** This is adaptive stratified sampling with a stopping rule:

- **Strata:** 44 strata from the failure modes (sign and size of S, and a grid in a = (M−S)/|M| and b = (S−R)/|M|), with exactly known masses.
- **Allocation:** each batch of 256 runs goes to strata in proportion to mass × √(e(1−e)), with error rates shrunk toward the other seeds (empirical Bayes, leave-one-seed-out).
- **Stopping:** stop when the estimated SE reaches the target.

Over the 22 held-out networks, 100 repetitions each:

| target SE | runs used (median) | brute force | saving | realized RMSE | bias | 95% coverage |
|---|---|---|---|---|---|---|
| 0.5 points | 4,340 | 5,725 | 1.3× | 0.50 | +0.16 | 0.94 |
| 0.2 points | 26,322 | 35,779 | 1.4× | 0.20 | +0.04 | 0.95 |
| 0.1 points | 104,826 | 143,117 | 1.4× | 0.10 | +0.01 | 0.94 |
| 0.05 points | 419,563 | 572,470 | 1.4× | 0.05 | +0.005 | 0.95 |

The error bars are honest. The small positive bias at loose targets is the usual optional-stopping effect, and an anytime-valid interval would remove it. **Why the saving is small (`w9.py`):** I computed the Neyman bound, the best any allocation over a given stratification can do even with *true* stratum error rates:

| network | 44 gap strata | 60 finer gap strata | + positions of max and 2nd (6,000 strata) |
|---|---|---|---|
| mystery | 2.9× | 2.9× | 5.0× |
| seed 2 | 2.4× | 2.4× | 3.8× |
| seed 3 | 2.3× | 2.3× | 3.7× |
| seed 0 | 1.8× | 1.8× | 2.6× |
| seeds 1, 4 | 1.3–1.5× | 1.3–1.5× | 2.4–2.5× |
| 16 new 2²⁷ seeds | 1.2–1.5× | 1.2–1.5× | 1.6–2.8× |

Finer gap bins add nothing: inside the near-tie strata the outcome stays close to a coin flip. A large saving needs strata where the outcome is nearly certain, which means knowing the network's own boundary, not the task's.

**Predicting the boundary from the explanation (`x1.py`, `x4.py`).**

- **A threshold-shifted tracker fails.** This is the online 2nd-max algorithm with the network's event thresholds for new max and new 2nd max, linear in the ages, with 6 parameters fitted by coordinate search. The best fit is zero shift, which agrees with the network on only 95.4% of inputs, no better than "always correct".
- **The errors are blur, not bias.** In near-ties, the error rate is the same whichever value came first: 25% when S precedes R against 23% when R precedes S, and 18% against 15% for M and S. It falls smoothly with the gap. For the mystery model it is 44% at b < 0.005, 21% at 0.01–0.02, 5% at 0.03–0.06 and 2% at 0.06–0.1. Seed 0's blur is about twice as wide. The network compares as if one value carried noise of about 0.02·|M|.
- **The ages explain part of the blur; the lower inputs explain none that a classifier can find.** Gradient-boosted classifiers predicting near-tie errors (residual variance, against 0.182 with no features): gap only 0.169; + positions of M, S and R 0.106; + all lower inputs and positions, no better. For seed 0 the same sequence is 0.217, 0.152, then no better. So the ages explain about 40% of which near-ties go wrong. That "blurred comparator with age-dependent width" explanation has a ceiling of about 5× (mystery) and 2.6× (seed 0), as above, and it needs fitting, that is, sampling.

**A weight-derived blur, with no sampling (`x2.py`, `x3.py`).** The mean-field boundary (lower inputs at μ(R)) plus a Gaussian blur, with width taken from the gradient of each logit margin with respect to the lower inputs and their truncated-normal variance: P(correct | M, S, R, positions) ≈ Πⱼ Φ(mⱼ/(σ(R)·‖∇ₗₒw mⱼ‖)). The integrand is smooth, so Gauss–Legendre in (M, a, b) converges, and it converges to the wrong answer:

| grid (points) | 0.23M | 0.51M | 1.35M | 2.95M | 6.05M |
|---|---|---|---|---|---|
| weight-derived blur | −4.31 points | −2.04 | −2.32 | −2.42 | −2.60 |
| mean field, sharp | −1.64 | +0.47 | +0.32 | +0.16 | +0.06 |

Pointwise checks at 3,000 random points, against 300 redraws of the lower inputs each, show two failures:

- **The ReLUs clip the lower inputs' influence**, so a linear spread predicts errors where there are none: clean region 0.980 predicted against 0.998 true.
- **The mean-field point is not a typical input.** In the worst cells the network errs at μ(R) but is right on 84–95% of real draws, and the linearization at that point gives 0–4%. The sharp mean field is off pointwise by 3.3 points on average, and is right overall only because breaks and rescues cancel.

The nonlinear one-input-at-a-time treatment (first-order cut-HDMR, 7.13) remains the best per-cell model of the lower inputs (about 30% lower per-cell error than mean field). Its net error, about 0.1–0.2 points, is within the noise of the 3,000-cell validation it was measured on.

**Bottom line.** Of the estimators that still sample, the best is last-input integration plus an ordered Sobol design, at about 2.5–4× sampling's efficiency for the zoo models. Explanation-driven allocation is honest but capped at about 1.4× in practice, and about 3–5× with perfect knowledge of the strata. Every mechanistic estimate tried needs the 720 arrangements × a 3D grid, about 10⁵–10⁶ network evaluations or more. At that compute sampling's SE is about 0.01–0.07 points, while the best mechanistic approximation errors are about 0.06–0.3 points (mean field) and worse with the derived blur. Matching sampling would need an estimate that avoids both the arrangement enumeration, using the time-translation structure, and network evaluation altogether. That means a closed-form state from the clock eigenmodes, with age-dependent comparator thresholds including ReLU clipping.

### 7.16 A population of 1,024 networks, and a lower-input leakage model that works (`batch_train.py`, `z1.py`–`z11.py`, `v1.py`, `lines.py`, `lw2.py`, `z12.py`–`z23.py`)

This round moved from the 22 seeds of 7.15 to 1,024 freshly trained networks, with two aims: learn which mechanisms set a network's accuracy, and build a mechanistic estimate that could meet ARC's bar. The working notes hold every number: `pop1k/notes_circuit_2e28.md` (first pass, 2²⁸-sequence snapshot) and `pop1k/notes_circuit_2e30.md` (converged). Printed outputs are in `pop1k/*.txt`. ChatGPT Pro was consulted twice: the briefs are `QUESTION2_FOR_CHATGPT_PRO.md` and `QUESTION3_FOR_CHATGPT_PRO.md`, and the answers are summarized in `chatgpt2/chatgpt_answer_{1,2}.md`.

**ARC's bar, as a budget.** An estimate with error ε matches sampling only if it costs at most p(1−p)/ε² forwards. Here p(1−p) ≈ 0.070 for the population and 0.0446 for the mystery model, so sampling reaches 1 point at 700 forwards, 0.3 at 7,800 and 0.1 at 70,000. Every error below is scored against that line.

**The population (`batch_train.py`).**
- `batch_train.py` trains N independent DistRNNs at once with AlgZoo's recipe. Its forward pass matches `nn.RNN` to 3e-7 after 200 steps.
- `pop1k/seed0.npz` holds 1,024 networks trained to 2³⁰ sequences, about 15.5 hours on one RTX 2060 SUPER, with snapshots along the way. Keys are `s<step>_{Wih,Whh,Wo}` and `init_*`; step 1,048,576 is 2³⁰ sequences.
- Accuracy is 0.924 ± 3.4 points across networks (5th/50th/95th percentiles 0.860, 0.932, 0.964). The six zoo models fall inside this range.

**Population-informed sampling gives about 2× (`z1.py`; ChatGPT round 1).**
- Strata built from the rest of the population's error rate on each input give a median 1.98× gain on the 2²⁶ snapshot (2.05× with the Neyman oracle), against 1.52× for task strata. A network's errors correlate only 0.51 with the population's.
- ChatGPT Pro's best scheme (`chatgpt2/population_estimator.py`) predicts the target's margins from 512 peers, then applies an importance-weighted residual correction. It is unbiased per network and gains 1.8× (95% interval 1.6–2.05×) on the population and about 2× on the zoo. The linear error basis saturates at R² ≈ 0.45 with all 1,023 peers. Three reruns on the mystery model came within 1 SE.
- This line is closed: it is statistics rather than mechanism, and the gain is a constant factor.

**Neuron roles predict accuracy to about 1 point (`z2.py`, `z3.py`; ChatGPT round 2).**
- **Buckets.** Rules taken from the zoo analyses put each neuron in a role. Per network on average: 6.0 always-on neurons, 3.1 comparators (96% of networks have at least one), 0.9 max detectors (78%), 5.3 "other". 38% of networks have an on-event pointer and 17% a one-step-late pointer, so 54% have a write pointer, in line with the 37–50% switch rate of 7.14.
- **The comparator count is the strongest single correlate.** Accuracy is 0.885–0.903 with 0–2 comparators and 0.941–0.949 with 4–6.
- **Predictions** (5-fold CV RMSE):

  | features | RMSE (points) |
  |---|---|
  | role counts | 1.8–2.1 |
  | role counts + per-role mean features | 1.2 |
  | each feature sorted over neurons | 1.08 |
  | weights only | 1.18 |
  | ChatGPT's hybrid GBM (weights + roles + gated features) | 1.00 |

  An RMSE of r is worth p(1−p)/r² samples, so the best is worth about 700 samples. Its features cost about 360 forward-equivalents, and it transfers worse to the zoo (1.3–1.5 points) than a weights-only ridge (0.83).
- **ChatGPT round 2's mechanistic findings** (`chatgpt2/roles_round2/roles_mechanism.py`, rerun on the converged population in `chatgpt2/roles_run_2e30/`):
  - Comparators are analog. A downstream gate sees Σ A·|w|·(θ − x)₊, so its root can sit between comparator hinges, and counting nearby thresholds misses amplitudes and routing.
  - Causal state swaps confirm the two pointer timings. Swapping comparator states at the event breaks late-pointer networks (8% correct); swapping pointer states breaks on-event-pointer networks (6%).
  - The z2 features are not invariant under the network's rescaling symmetry or under rotating the readout.
  - `z1.task_strata` yields 60 strata, not 44.
- **The strategy is fixed early and cheap to read** (scratch `strat3.py`). 98.8% of networks keep their pointer/memory label from 2²⁸ to 2³⁰, and 32 probes recover the label for 98.6% of them. But the two groups barely differ in error structure: mean-field leakage +0.23 ± 0.48 against +0.31 ± 0.56, and 58% of errors on ties in both.

**Where networks differ (`z4.py`–`z7.py`).**
- **The ties hold 37% of the across-network variance; the middle and clean regions hold 62%.**
- **Within middle and clean, the variance is handoffs**: the old max displaced into the 2nd slot by a new max.
  - A handoff at the last step errs 12.3% of the time on average (SD 13.5 across networks). Its error rate correlates −0.81 with accuracy.
  - It errs 38% of the time with no comparators and 3% with five or more. That is why the comparator count predicts accuracy.
  - For the zoo it is 0.7% (mystery), 2.4% (seed 2) and 23% (seed 1).
- **Inside each mechanism, errors are blur from many small contributions** (as in 4 and 7.15). A GBM on the top-3 values and ages explains only 30–44% of last-step-handoff deviance.
- **Greedy-selected informative inputs plateau at about 1 point** with a ridge fit (`z7.py`). 85% of the 500 most informative inputs are last-step handoffs.
- **Mean field leaks across the population.** MF3 (the top 3 exact, the 7 lower inputs at μ(R) = E[x | x < R], as in 7.13) correlates 0.989 with accuracy. But its leakage, accuracy(MF3) − accuracy, is +0.27 ± 0.52 points across networks (RMS 0.59). Network features predict the leakage only to 0.39 points.
  - 7.13's "leakage nets out" holds for the mystery model (+0.009) and seed 2 (−0.019), not for the population.
  - At that RMS, any MF3-based estimate is worth at most about 2,000 samples.

**Propagating the lower inputs through the network (`z8.py`).** Given the top 3, the 7 lower inputs are iid N(0,1) truncated below R. The aim was P(correct | top 3) from the weights alone.
- **Moment closures fail.** The logit margin (about 1–4) is a small difference of large activities (20–90). The network cancels lower-input noise in that difference, and closures with independent ReLU residuals destroy the cancellation: the margin SD comes out 2–11× too large. Hard gates fixed at the mean-field point keep the cancellation (SD ratio 0.85).
- **The delta method fails on gate flips**: −0.82 ± 1.06 points.
- **The first-order (one input at a time) model around MF3 was the best of this pass.** It gives −0.39 ± 0.44 points (Gaussian, q = 8 nodes) or −0.15 ± 0.45 (exact sum, q = 4). Adding all 21 pairs gives +0.21 ± 0.35, at about 150 forwards per configuration.
- **The top-3 integral is the larger obstacle.** 55% of each network's variance lies between top-3 configurations. So even an exact P(correct | top 3), averaged over sampled configurations, gains at most 1.8×. End to end on an ordered-Sobol design, the first-order model is 4–10× short of sampling.

**Option 1, a better sampler: exact integration along the 2nd-max value (`v1.py`).**
- Given everything except S's value, the network is piecewise affine along S ∈ (R, M), with a median of 6.8 pieces. So P(correct | rest) is a sum of exact Φ masses, and averaging it over samples is unbiased.
- It cuts the per-sample variance 2.1–4.6× on the zoo (mystery 4.6×), the best of four lines tried. But tracing costs 4–5 forwards, so the net gain over plain sampling is 1.2–3.9×. That is the same 2–6× band as last-input integration (7.15).
- This line is closed: a constant factor, and it still samples.

**Option 2, deterministic: the exact mean-field integral (`z9.py`).**
- **Method.** For each of the 720 top-3 arrangements and each R, the MF3 input is affine in (M, S), so the network is piecewise affine on the slice R < S < M. `z9` walks the slice's linear regions and integrates φ(M)φ(S) exactly over each region's incorrect polygon. The R integral is composite Gauss–Legendre with weight φ(R)Φ(R)⁷. Nothing is sampled. Single slices match Monte Carlo of the MF3 network (`z9.py check`).
- **It converges to the mean-field value.** For mystery, with R < 0 split finely, it gives 0.953213, converged to 2e-7. The sampled MF3 reference is 0.95323 ± 0.00005, and the truth is 0.953119. Two coarser runs looked closer to the truth only by luck: their R ∈ (−0.5, 0) panel was unconverged.
- **The mean-field bias sets a parity budget**, p(1−p)/bias². Above that cost, any MF3-based estimate loses to sampling, even one computed for free.

  | model | converged MF3 | truth | bias (points) | parity budget | cost of the converged integral |
  |---|---|---|---|---|---|
  | mystery | 0.953213 | 0.953119 | +0.009 | 5.0M | 15–20M |
  | seed 2 | 0.948323 | 0.948510 | −0.019 | 1.4M | about 45M |
  | seed 0 | 0.910885 | 0.910090 | +0.080 | 0.13M | about 5M |
  | population (RMS leakage 0.59) | | | 0.59 | about 2,000 | |

- **Where the cost sits.** R < 0 is 5.5% of inputs and 19% of the error, but 72–80% of the cost: about 2,000 forwards per slice against about 200 for R > 0, because the region count there is high. The error is spread over arrangements: 139 of the 720 hold half of it.
- **So the top-3 geometry under mean field is solved exactly and deterministically.** But it costs about as much as sampling the MF3 network to the same precision, and the MF3 bias caps its worth far below that cost.

**The lower-input leakage model: the top 4 jointly, the 6 smallest inputs independently (`z10.py`, `z11.py`, `lines.py`).**
- **Anatomy** (`z10.py`, 2²⁰ inputs per network).
  - No single lower position, zone, event subset or stand-in value localizes the leakage (as 7.13 found). MF3 and the truth disagree on about 5% of inputs, at every mean-field margin.
  - What was new is where it lives (regime split from `lw2.py`). S < 0 and R < 0 < S contribute only +0.04 to +0.05 each, because MF3 gets those inputs wrong too. All of the spread is in R > 0, where two mechanisms nearly cancel:
    - the true network answers at the 4th-largest's position, worth +0.45 points, which MF3 cannot do because the 4th sits at μ(R);
    - S-vs-R swaps flip 3% of inputs each way, netting −0.35.
  - Keeping the 4th exact (MF4) removes both mechanisms. What remains is a fairly uniform optimism, +0.30 ± 0.28.
- **The model.** P̂(correct | top 4) = I₀ + Σⱼ (Eⱼ[I] − I₀), with j over the 6 smallest inputs.
  - I₀ is the network's correctness at the MF4 point (6 smallest inputs at μ(x₍₄₎)).
  - Eⱼ[I] varies input j alone over its law, N(0,1) truncated below x₍₄₎, with the others at their mean.
  - Eⱼ is computed at q equal-probability nodes, or exactly by tracing the piecewise-affine line (`lines.trace_line`, which generalizes v1's tracer and matches a 20k-point grid to 4e-5).
  - **This scores the leakage model alone.** It is evaluated on the same 2¹⁸ inputs as the true accuracy, so the top-4 configurations here are a sample. Pool noise is about 0.03 points per network (0.066 for the exact rows, which use 2¹⁶ inputs on every 8th network plus the zoo).

| model | forwards per configuration | mean (points) | SD | RMS | corr. with true leakage |
|---|---|---|---|---|---|
| MF3 alone | 1 | +0.247 | 0.511 | 0.567 | |
| MF3 + first order, q = 8 | 57 | −0.000 | 0.156 | 0.156 | 0.96 |
| MF4 alone | 1 | +0.304 | 0.284 | 0.416 | |
| MF4 + first order, q = 4 | 25 | +0.124 | 0.092 | 0.154 | 0.96 |
| MF4 + first order, q = 8 | 49 | +0.032 | 0.061 | 0.069 | 0.98 |
| **MF4 + first order, q = 16** | 97 | **−0.013** | **0.055** | **0.057** | 0.98 (slope 0.99) |
| MF4 + first order, exact lines | about 65 | −0.051 | 0.078 | 0.093 (pool noise 0.066) | |
| MF5 + first order, q = 8 | 41 | +0.037 | 0.045 | 0.058 | 0.98 |

- **The q = 4 shortfall was node resolution, not interactions.** The flips caused by one small input are narrow and sit near its upper end, in near-ties with the 4th largest, where equal-probability midpoints under-sample them. The mean drifts from +0.12 to +0.03 to −0.01 as q goes 4, 8, 16, and the exact limit is −0.05.
- **After removing pool noise**, the exact first-order model is off by about −0.05 ± 0.04 points per network (RMS about 0.07). That compares with 0.57 for MF3 and 0.35–0.41 for z8's pairwise model.
- **This reconciles with 7.13 (`y6.py`).** MF4 *alone* is still worse in mean for the zoo selectors (mystery +0.15 against +0.02) and for the population (+0.30 against +0.25), although its spread is half. But MF4 is the base around which the remaining inputs are additive. Around MF3, the same first-order expansion has 3× the spread.
- **Mechanistic reading.** The decision depends jointly on the top-4 values and positions. Each of the 6 smallest inputs acts on it independently, through the real network, with no interactions among them to within about 0.05 points. It is a presumption of independence that holds once the 4th largest is excluded from it. (This holds for the integral. Configuration by configuration it holds only in the margin, not in the indicator; see the local check at the end of this section.)

**What the top-4 integrand looks like (`z12.py`; 64 population networks + the zoo, 2¹⁷ configurations).** The configurations are sampled here only to describe the integrand.
- **I₀ is a 0/1 indicator**, so its variance over configurations is 0.97·p(1−p). Adding the penalty smooths it only a little: Var(P̂) = 0.79·p(1−p). So sampling configurations gains about 1.3× per configuration, against a cost of about 65 forwards each.
- **The penalty is confined to a thin shell around the MF4 decision boundary.** It is nonzero on 12% of configurations (median; 7–19% across networks). Where the MF4 logit margin is below 0.5 (6% of configurations) sit 80% of its square and half of its mean. Within margin 1 (12%) sit 92% of its square.
- **Arrangements carry little of the variance.** The 5,040 arrangement means explain 6–8% of the variance of the truth, of I₀ and of P̂ (25% for seed 1), and 1% of the penalty's. The rest is the integral over the 4 values within each arrangement, that is, where each arrangement's decision boundary lies.

**Where this leaves ARC's bar.**

| piece | status | error (points) | worth p(1−p)/ε² | cost |
|---|---|---|---|---|
| lower inputs, MF3 alone | done | 0.57 RMS | about 2,000 | 1 forward per configuration |
| lower inputs, MF4 + first order, exact lines | done | about 0.07 | about 150k | about 65 forwards per configuration |
| lower inputs, MF4 + margin model with nearby pairs (`z18`, below) | done | about 0.05 (all 1,024), locally 40× better | about 300k | about 240–410 forwards per configuration |
| top-3 integral of MF3, region walk (`z9`) | done | converged | | 5–45M forwards |
| top-4 integral of the leakage model | **measured: off the line** (below) | MF4 anchor alone: 0.18 at 250k forwards, 0.03 at 2M (2.4–4.5× sampling's MSE from 1M up) | | 5,040 arrangements × outer nodes, plus about 4.8M per node for the correction |
| top-3 integral of MF3, exact S lines with shared prefixes (ChatGPT round 4) | done | 0.58 RMS (sampling 0.14 at the same cost) | | about 36k forwards |

- **The lower-input side is good enough for parity up to about 150k forwards.** Before this round it was good to about 2,000.
- **The obstacle is now the top-4 integral alone.** It must be done to error ε at a cost below p(1−p)/ε²: 0.3 points at 7,800 forwards, 0.1 at 70,000.
  - Enumerating linear regions already costs 100–1,000× too much for MF3.
  - Sampling top-4 configurations at about 65 forwards each is about 65× less efficient per forward than sampling the network.
  - A deterministic scheme has to locate each arrangement's decision boundary in the 4 values, and evaluate the penalty only in the shell around it.
- `QUESTION4_FOR_CHATGPT_PRO.md` poses this integral as the next question. The attempts that followed (ChatGPT's round 4 and `z20`–`z23`) are below. None reaches the line in the 10³–10⁵ range.

**Round 4: the integral attempted, and a local check of the leakage model (`z13.py`, `z14.py`; ChatGPT's answer summarized in `chatgpt2/chatgpt_answer_3.md`).**
- **No deterministic estimator reached the sampling line.** ChatGPT's best was MF3 with exact S lines. It shares recurrent prefixes across arrangements, applies sparse updates between adjacent pieces, and uses Gauss–Jacobi rules in the order-statistic CDFs.
  - On all 1,024 networks it has an RMS error of 0.58 points at a mean of 36k forwards, against 0.14 for sampling: 16× sampling's MSE.
  - Interpolating the logits along S instead is 8–76× worse in MSE between 700 and 97k forwards.
  - MF4 by the same quadrature is still 1.6 points off at 152k forwards.
  - Prefix sharing saves 2–3×. One MF4 value quadruple's full arrangement tree costs about 1,910 forwards, so 10⁵ forwards buys about 52 quadruples before any lower-input correction.
- **The clamped first-order model reverses the anchor at its boundary.** P̂ = clip(Σⱼ Eⱼ − 5 I₀, 0, 1). Where I₀ steps from 0 to 1 and the Eⱼ are continuous, the raw sum drops by 5, so whenever 1 < Σ Eⱼ < 5, P̂ jumps from 1 to 0. I reproduced its example on network 0: Σ Eⱼ = 3.168 on both sides, P̂ goes 1 → 0, and the true conditional accuracy is 0.72.
- **Local check (`z13.py`).** For 8,192 top-4 configurations c (every 16th network plus the zoo), I measured the true conditional accuracy p(c) with 256 draws of the 6 lower inputs. I then scored each model built from the same 97 rows.

| model at each configuration (q = 16) | local MSE / p(1−p), population median | integrated bias (points) |
|---|---|---|
| I₀ (MF4 anchor) | 0.18 | +0.27 |
| first order, unclamped | 0.54 | +0.16 |
| first order, clamped (P̂, z11) | 0.12 | −0.01 |
| independent break/rescue events (product) | 0.060 | −0.00 |
| **first order in the margin** | **0.010** | **−0.15** (−0.09 at q = 8, −0.13 at q = 12) |

  - P̂ reverses I₀ on 2.0% of configurations, where p(c) ≈ 0.55. Those carry 65% of its local squared error, so its small integrated bias is largely a cancellation across the boundary.
  - The margin model expands m = L_y − max_{o≠y} L_o to first order in the 6 lower inputs, and takes P(m > 0) exactly over the 16⁶ node combinations.
    - Its local error is 6% of I₀'s, and only 20% of it lies within |m₀| < 0.5, against 88% for P̂. Like the truth, its integrand is continuous across the anchor boundary, which is what a deterministic quadrature over the 4 values needs.
    - But it is biased low: −0.15 points on average at q = 16, negative in 61 of 64 networks (SE about 0.03 each), and −0.07 to −0.24 for the zoo. The bias grows with q (mean −0.09, −0.13, −0.15 at q = 8, 12, 16), so it is not node error that vanishes.
    - Making each rival difference additive instead (`z14.py`) is worse, at −0.42. So the bias does not come from two inputs pushing different rivals. Its source is found below (`z15`–`z19`).
- **Homogeneity (ChatGPT).** The network is positively homogeneous, so the overall scale integrates out exactly.
  - With the retained values v = ρu (|u| = 1) and the lower inputs xⱼ = ρzⱼ, the density per arrangement becomes (12/π⁵)(1 + |z|²)⁻⁵.
  - With u₁ = R cos θ and u₂ = R sin θ, every gate and logit within a region is A + B cos θ + C sin θ. So θ integrates exactly by elementary roots, leaving 2 outer coordinates for MF4 instead of 3.
  - A homogeneous MF4 + first-order model has about 0.07 points of noise-subtracted RMS bias (16 development and 8 held-out networks, plus the zoo). The arrangement cost is unchanged.
- **Two fixes from its audit.** `lines.trace_line` now resolves an exact tie on a piece by argmax's lowest-index rule (ties occur in fewer than 1 in 2 million draws), and `z11.py`'s exact mode now stops on tracer overflow (ChatGPT saw none in 3.3M traces).

**The margin model's bias: double counting at shared thresholds, removed by averaging over nearby inputs (`z15.py`–`z19.py`).**
- **Measured draw by draw (`z15.py`; 8,192 configurations × 64 lower-input draws, every 16th network plus the zoo).** Each input's effect is computed exactly at its drawn value, with no node grid.
  - The first-order margin model is −0.18 points low (p10 −0.30, p90 −0.09; all six zoo models −0.09 to −0.28). The drift with q in z13 was the midpoint nodes under-sampling the tails.
  - Its sign errors flow both ways: on 0.67% of draws the truth is right and the model wrong, on 0.50% the reverse.
  - The margin residual r = m − m₁ averages zero, but it opposes large predicted effects in both directions. Its mean is +0.16, +0.46 and +1.8 for Σ dⱼ in (−2, −1), (−4, −2) and below −4, and −0.21 and −1.0 for Σ dⱼ in (1, 2) and above 2. 39% of draws are exactly additive (no gate or winning rival differs from the anchor run), because within one linear region the margin is affine.
  - **The anchor does not matter.** Anchoring at the median instead of the mean gives −0.179.
- **Mechanism (`z16.py`, `z19.py`).** Classifying the gates (time step × neuron) that differ from the anchor run:
  - On draws carrying a sign error, gates flipped by two or more one-input runs (each run counts the flip, the joint run has it once) account for 79% of the net bias. A change in the winning rival accounts for 52% (the two overlap). Draws with neither carry 5%.
  - These shared flips sit in the analog channels: clock-like ("other") and always-on memory neurons hold 88% of them, against 72% of all neurons. Comparators, pointers and max detectors are under-represented (12% against 28%).
  - Among pair interactions, 62% of their squared size comes from inputs 1–2 positions apart, and 64% from pairs that both precede the 2nd max (a third of all pairs). One pair carries 72% of the interaction in a typical draw.
  - So nearby lower inputs push the same clock or memory neuron across its threshold, and a sum of one-input effects counts the crossing once per input.
- **Averaging out the double counting.** Measuring each input's effect averaged over the others' spread (first-order ANOVA, `z17.py`, sampled backgrounds) turns −0.18 into +0.06.
  - A deterministic version (`z18.py`) projects the second-order cut terms onto one-input functions: hⱼ(z) = dⱼ(z) + Σₗ Eₗ[Dⱼₗ(z, Zₗ)]. The model stays a sum of one-input terms, so P(m > 0) is still an exact convolution.
  - Keeping all 15 pairs over-corrects (+0.05). Keeping only pairs at most 2 positions apart (5.7 per configuration) is best, locally and overall.
  - Gauss–Legendre nodes in the CDF variable converge by q ≈ 6–8, where midpoints had not by q = 16.

| model (q = 8, Gauss–Legendre) | local MSE / p(1−p) | development: mean, RMS (noise removed) | held out: mean, RMS (noise removed) | all 1,024: mean, RMS (noise removed) | rows per configuration |
|---|---|---|---|---|---|
| I₀ (MF4 anchor) | 0.17 | +0.33, 0.42 | +0.34, 0.43 | +0.31, 0.40 | 1 |
| first order in the indicator, clamped (z11) | 0.12 | −0.03, (0.095) | −0.06, (0.10) | −0.04, 0.065 (`z20`) | 49 |
| first order in the margin | 0.011 | −0.17, 0.19 | −0.17, 0.19 | −0.17, 0.19 | 49 |
| margin + all pairs, projected | 0.005 | +0.07, 0.086 | +0.05, 0.052 | +0.05, 0.062 | 1,009 |
| **margin + pairs ≤ 2 apart, projected** | **0.003** | **+0.04, 0.073** | **+0.02, 0.042** | **+0.03, 0.048** | about 410 (about 240 at q = 6) |

  - The development set is every 16th population network; the held-out set is every 16th from 8, never used to choose the pair rule. Each has 4,096 configurations × 128 draws, and all 1,024 have 2,048 × 128.
  - Per-network measurement noise has been removed from the RMS. It is 0.03–0.07 points for the margin models, but 0.15–0.20 for the indicator model, whose local errors are large. So the indicator model's RMS (in parentheses) is barely resolved at these pool sizes. `z20` resolves it on all 1,024 networks (2¹⁷ configurations, noise 0.027): 0.065 for this q = 8 Gauss–Legendre version, and 0.052 for z11's q = 16 midpoints.
  - The zoo models are within −0.05 to +0.08 for the nearby-pair model.
  - **Third-order terms do not help (`z32.py`; 2¹⁷ configurations × 64 draws, per-network noise about 0.007 points).** Adding the projected third-order terms of every triple of lower inputs at three consecutive positions (1.3 per configuration) gives a noise-subtracted RMS across the 64 development networks of 0.081 (pairs gap ≤ 1) and 0.085 (gap ≤ 2). Without the triples it is 0.075 and 0.076.
    - The remaining error is spread between networks (SD about 0.07), not a common offset. Higher-order adjacent terms do not reduce it. Widening the triples to every triple spanning up to 3 positions (3.7 per configuration; 2¹⁶ × 64, noise 0.010) gives 0.074 (gap ≤ 2) and 0.101 (gap ≤ 1), against 0.075–0.077 without them. That is the ideal additive model's level (`z34`, 0.074).
    - **The additive family has a floor (`z34.py`)**, in the sense of the models that converge to the ideal projection below. It is not a floor for every additive model. Round 5's counterexample: m = |X+Y| − |X−Y| + 2|Z| has projection 2|Z|, so P(m_A > 0) = 1 while P(m > 0) = 5/6, and the additive g = X + √3|Z| gets 5/6 exactly. All these models approximate one ideal: the exact first-order ANOVA projection of the margin, the best possible sum of one-input effects. Measured with 32 shared backgrounds on 2¹⁵ configurations (noise 0.014), the ideal is +0.063 points on average, with SD 0.041 and noise-subtracted RMS 0.074. Every truncated scheme lands at the same level (0.075–0.085). Reaching 0.01 would need cancellation removing six-sevenths of the ideal's own error in every network.
    - **Joint, non-additive rules are worse at affordable cost (`z36.py`).** A product Gauss–Legendre rule over all six lower inputs at once has RMS 0.23 points at q = 2 (64 forwards per configuration) and 0.15 at q = 3 (729), against 0.075 for the additive pair model at 240–410. The six-dimensional indicator is discontinuous, so product rules converge slowly.
  - **In total, the nearby-pair margin model and the clamped indicator model are comparable, at about 0.05 points.** The indicator model gets there by cancelling reversed errors across the boundary. The margin model is right configuration by configuration (40× smaller local error) and changes smoothly across the boundary, at about 2.5–4× the cost. The top-4 integral above is still the obstacle.

**The top-4 integral, measured (`z20.py`–`z38.py`).**
- **The indicator model's total, resolved (`z20.py`).** On all 1,024 networks (2¹⁷ configurations, per-network noise 0.027 points), the q = 16 midpoint version is −0.015 points on average with RMS 0.052, after removing noise. Gauss–Legendre q = 8 gives −0.040 and 0.065. The unclamped sum gives +0.136 and 0.160. Every 4th network at 2¹⁸ configurations agrees (0.053, 0.067, 0.144). So the clamped indicator model and the nearby-pair margin model (0.048) are comparable in total.
- **The MF4 anchor integral by a deterministic outer rule (`z21.py`).** E[I₀] is a 3-D integral over the CDF coordinates of Q, R and M, with S integrated along each line and all 5,040 arrangements averaged. Tensor Gauss–Jacobi rules match the order-statistic laws exactly. I charged ChatGPT's exact-S tree cost of about 9,300 forwards per outer node; the S integral here is a brute-force grid, refined until it no longer matters.

| order | outer nodes | forwards | RMS error (points; 16 networks + zoo) | sampling at that cost | MSE / sampling |
|---|---|---|---|---|---|
| 1 | 1 | 9,300 | 3.6 | 0.28 | 160 |
| 2 | 8 | 74k | 1.27 | 0.10 | 160 |
| 3 | 27 | 251k | 0.18 | 0.055 | 11 |
| 4 | 64 | 595k | 0.18 | 0.035 | 25 |
| 5 | 125 | 1.16M | 0.039 | 0.025 | 2.4 |
| 6 | 216 | 2.0M | 0.031 | 0.019 | 2.7 |
| 7 | 343 | 3.2M | 0.032 | 0.015 | 4.5 |
| 8 | 512 | 4.8M | 0.026 | 0.013 | 4.0 |
| 9 | 729 | 6.8M | 0.022 | 0.011 | 4.0 |
| 10 | 1,000 | 9.3M | 0.011 | 0.009 | 1.5 |

  - The K = 64 S grid had left a +0.05 floor; K = 256 and 512 agree. Orders 6–8 use K = 512, orders 9–10 K = 256 (K = 256 and 512 agree at order 6), and all of them a 2²⁶-sample reference (noise 0.003).
  - So the anchor integral alone is 10–160× worse than sampling in MSE up to 600k forwards, and 2–3× worse at 1–2M. Beyond that the error plateaus through order 9 (4× sampling's MSE) and then drops at order 10 (1.5× at 9.3M).
  - **Rate.** Convergence is uneven: a plateau over orders 5–9, then a drop at order 10. Over orders 5–10 the error falls as nodes^(−0.5), the same rate as sampling. The MSE ratio is 2.5–4.7 through order 9 and 1.6 at order 10.
    - So for the anchor alone this rule runs parallel to sampling's line, a factor of about 1.5–5 above it, and touches it only around 10⁷ forwards, two orders of magnitude above ARC's range.
    - At that cost sampling's error is about 0.009 points. The anchor is 0.3 points optimistic, and the best lower-input models are good to only about 0.05, so it is still not an accuracy estimator there.
    - (An earlier fit over orders 5–8 alone gave nodes^(−0.25). That was the plateau, not the rate.)
  - **An adaptive rule (`z31.py`)** is worse still (0.063 points at 864 nodes, 40× sampling's MSE). It splits the cube into boxes with 2×2×2 Gauss–Legendre rules, refining where the box and centre estimates differ most. At 144, 216 and 432 nodes it is 1.46, 1.63 and 0.37 points off, against the tensor rule's 0.031 at 216 nodes and sampling's 0.014 at 432. Low-order local rules cannot keep up with a global high-order rule, and the kinks they chase are everywhere.
  - **Sparse grids (`z28.py`)** are worse at equal cost. A Smolyak combination of the same Gauss–Jacobi rules gives 0.83, 0.32, 0.38 and 0.46 points at levels 5–8 (260k–4M forwards). It does not converge: the combination's negative weights amplify the integrand's kinks.
- **Homogeneous coordinates do not help (`z23.py`).** Integrating the scale out leaves a 2-D outer integral and an exact angle. But where u₃ < −R the angular range splits into two arcs, which puts kinks into the outer integrand. Orders 1–12 give 5.3, 2.7, 2.8, 0.68, 0.22, 0.34, 0.12, 0.46, 0.35, 0.079, 0.13 and 0.28 points: erratic, and 3–13× sampling's RMS from 10⁵ to 1.3M forwards (best: 0.079 at 930k, where sampling gives 0.028). Removing those geometric kinks (`z27.py`) does not fix it. With the outer domain split into 9 smooth pieces along the curves where the arcs change shape, and 256 θ-points per arc, orders 4–7 give 0.176, 0.041, 0.052 and 0.035 points (1.3M–4.1M forwards; sampling 0.024–0.013). Over the last three orders the error falls as nodes^(−0.19), below sampling's −0.5. What remains is the network's own non-smoothness, which no change of coordinates removes.
- **The lower-input correction is not local to the MF4 boundary (`z22.py`, 1,024 sampled S-lines).** Of the first-order margin model's line penalty (−0.54 points), −0.36 (68%) sits on S-lines where the anchor never changes its answer: rare large lower-input effects flip answers the anchor gets confidently right.
  - A two-term boundary formula (the mean shift E[Σ] plus a smearing term in E[Σ²]) gets the sign wrong (+0.78). Most crossings sit at near-ties at the ends of the S-line, so the side of the crossing where rescues would occur is cut off.
  - Freezing the lower-input law at each crossing recovers only −0.19.
- **The enumeration cannot be cut by fixed subsets (`z24.py`; per-arrangement anchor accuracies by the order-5 rule).** Across the 5,040 arrangements the anchor accuracy has an SD of 8 points.
  - The positions of M and S explain 28% of that variance, and 41% once the order of R and Q relative to them is added. The exact positions of the 3rd and 4th largest carry the rest.
  - The dependence on positions does not factor. Additive effects of the four positions explain 18% of the variance across arrangements, and all six pairwise position interactions (463 parameters) explain 48%. The outcome depends on the joint pattern of all four positions, as a recurrent state would.
  - Keeping a fixed subset of the 56 (R, Q) placements for each of the 90 (M, S) classes is 4–10× worse in RMS than sampling at the reduced cost. With 1 placement (90 arrangements) the error is 1.2–2.1 points, against sampling's 0.19. With 8 placements (720) it is 0.27 against 0.067, and with 14 placements (1,260) 0.30 against 0.051.
- **A closed form from linear regions fails too (`z25.py`, `z26.py`).** Within one activation pattern the network is exactly linear, so each pattern gives a logit map Σₜ cₜxₜ for the cost of about one forward. If a few patterns covered each arrangement's value space, the value integral would need no further network evaluations, and the outer-node multiplier would disappear.
  - `z25` takes the patterns at K representative value points per arrangement (a Gauss–Jacobi grid in the CDF coordinates) and scores, on 2²⁰ sampled configurations, the accuracy given by each configuration's nearest pattern map. Covered configurations, those whose true pattern is one of the K, are exact.

| patterns per arrangement | forwards to build the maps | mass covered | error, nearest map (RMS, points) | error, exact where covered | sampling at that cost |
|---|---|---|---|---|---|
| 1 | 5k | 17% | 2.7 | 2.7 | 0.37 |
| 16 | 81k | 50% | 1.2 | 0.83 | 0.093 |
| 81 | 408k | 67% | 1.0 | 0.48 | 0.041 |
| 256 | 1.3M | 76% | 0.76 | 0.33 | 0.023 |

  - The reason is in `z26`: one arrangement's value space (lower inputs at μ(Q)) holds many linear regions. A median of 560 (190–870 across networks) appear in 2¹⁶ draws, and 460 in 2¹⁵, so the count is still rising. The most common carries 20%, the 10 most common 66% and the 100 most common 95%.
  - So an exact closed form would need hundreds of linear maps per arrangement, more than 10⁶ forwards to build, before any value integration or lower-input correction.
  - **The oracle bound, for any choice of patterns** (`z26`, 40 arrangements × 2¹⁶ draws per network). Suppose an oracle hands over each arrangement's K most probable patterns, at one forward each.
    - The uncovered mass is 80%, 34%, 15%, 5.0%, 1.3% and 0.29% at K = 1, 10, 30, 100, 300 and 1,000. A Good–Turing estimate puts the mass in patterns never seen at 0.23%.
    - Guaranteeing error ε needs uncovered mass ≤ ε, so K well above 1,000 (more than 5M forwards).
    - Allowing z25's rate of wrong fallback answers (about 1.4% of the uncovered mass), the oracle form is 0.47 points off at K = 10 (50k forwards; sampling 0.12) and 0.21 off at K = 30 (151k; sampling 0.068). It reaches parity only near K = 300, about 1.5M forwards.
    - So within 10⁵ forwards even an oracle-chosen linear-region closed form stays 3–4× sampling's RMS, before value integration or the lower-input correction.
  - **Measured with oracle selection (`z30.py`; 100 arrangements per network, 2¹⁴ training and 2¹⁴ held-out draws each).** The patterns are chosen on training draws, and the error is measured on held-out draws, with arrangement-sampling noise removed. Choosing the patterns greedily, for agreement with the network rather than for mass, helps a great deal:

| K | build cost (forwards) | error, K most massive patterns | error, greedy patterns | sampling at that cost |
|---|---|---|---|---|
| 3 | 15k | 1.22 | 0.33 | 0.215 |
| 10 | 50k | 0.93 | 0.17 | 0.118 |
| 30 | 151k | 0.54 | 0.11 | 0.068 |
| 100 | 504k | 0.21 | 0.04 (within noise) | 0.037 |

    - With the selection free, the greedy closed form for the anchor is 1.4–1.6× sampling's RMS in ARC's range and reaches it near 5×10⁵ forwards. This is the closest any deterministic form came.
    - But the oracle used 2¹⁴ network evaluations per arrangement, about 80M forwards per network. Choosing patterns deterministically and cheaply loses most of the gain: taking them at 16 grid points per arrangement (`z25`, 81k forwards) is 0.83 points off, against the oracle's 0.17 at 50k.
    - The figures also leave out the exact value integration of each region, and the anchor is not the accuracy: it is 0.3 points optimistic before the lower-input correction.
  - **Deterministic pattern discovery does not approach the oracle (`z33.py`).** The exact-S integrator's lines pass through every region they cross, so the patterns on the S-lines at a tensor rule's outer nodes come for about 1.84 forwards per line. Using all of them, with nearest-centroid fallback, on held-out draws:

| S-lines per arrangement | forwards | mass covered | disagreement | error (points) | sampling |
|---|---|---|---|---|---|
| 1 | 9k | 31% | 5.6% | 2.9 | 0.28 |
| 8 | 74k | 59% | 3.2% | 1.67 | 0.097 |
| 27 | 250k | 73% | 1.9% | 1.28 | 0.053 |

    - That is 13–24× sampling's RMS, and worse than integrating along the same lines directly (the tensor rule: 1.27 and 0.18 points at 8 and 27 nodes). Within ARC's range these are all the settings there are: 1 and 8 lines per arrangement (9k and 74k forwards) are the only tensor orders under 10⁵ forwards, and 27 lines already costs 250k. The fallback errs systematically in the uncovered mass, and only the oracle's choice avoided that.
  - **A more faithful fallback does not help (`z35.py`).** Answering an uncovered input with the selected region whose activation constraints it violates least, rather than the nearest centroid, raises the disagreement on the same greedy sets: 2.2%, 1.6% and 0.95% at K = 10, 30 and 100, against 1.3%, 0.9% and 0.6%. So z30's greedy oracle with centroid fallback stays this family's best case.
- **The clock-state simplification fails (`z29.py`).** The proposed closed form treats the usually-active neurons (clocks, memory) as a linear system and keeps ReLU only for the switching neurons. It would be exact if those neurons never switched off. They do: the most active neuron in a network is on for 94.8% of steps (population median).
  - Linearizing the 7 neurons (median) active on at least 90% of steps biases the anchor accuracy by 2.3 points RMS and changes 1.8% of answers, and still leaves about 230 linear regions per arrangement.
  - Linearizing the 10 neurons active on at least 80% gives a bias of 16 points, with about 104 regions left.
  - The rare switch-offs of the clock and memory channels are exactly what the accuracy depends on (`z19`: the double-counted flips sit in these neurons).
- **Cost of the correction inside the enumeration.** A per-configuration lower-input model costs about 240 forwards (nearby pairs) at each of about 4 S-points on each of 5,040 arrangement lines: about 4.8M forwards per outer node, against a whole budget of 10⁵. The only form that splits into separate line integrals is the unclamped sum, and it is biased by +0.14 points (RMS 0.16, worth about 30,000 samples). It would also cost several times the anchor integral, which already needs about 10⁶ forwards.
- **What the constraints leave open.** The brief that defines the task (`QUESTION4_FOR_CHATGPT_PRO.md`, Constraints) requires "a deterministic computation on that network's own weights".
  - It rules out "random or quasi-random point sets over the input or configuration space, including unscrambled Sobol or lattice rules used in place of sampling", "control variates, and importance or stratified sampling", and "any fitted prediction of accuracy or error from the population".
  - It allows "points the method chooses from the weights, such as region vertices, line pieces, roots, and quadrature nodes for a smooth reduced integrand", and design choices learned from other networks.
  - So quasi-Monte Carlo, randomized quadrature, control variates and fitted surrogates are excluded by definition, not by measurement. Bayesian quadrature on deterministic nodes is a weighted quadrature rule, limited by the integrand's smoothness like the rules above.
  - Within what is allowed, this section has tried the following, each measured against sampling at equal cost:
    - quadrature nodes: tensor, sparse, adaptive, homogeneous;
    - line pieces: exact S-lines;
    - region-based closed forms: grid, oracle, line-discovered;
    - population-learned design choices: the nearby-pair rule.
- **Other decompositions trade the integral against the correction (`z37.py`).** Keeping only the top 2 or top 3 inputs exact gives 90 or 720 arrangements and a 1-D or 2-D outer integral. Each outer node is charged pro rata to the exact S tree: 191 forwards for MF2 and 1,528 for MF3.

| exact inputs | anchor bias against the truth (RMS, points) | anchor integral, deterministic | sampling at that cost |
|---|---|---|---|
| top 2 (MF2) | 5.7 | 0.16 at 573 forwards, 0.03 at 1.7k | 1.16, 0.67 |
| top 3 (MF3) | 0.68 | 0.31 at 14k, 0.13 at 24k, 0.074 at 55k, 0.073 at 98k | 0.24, 0.18, 0.12, 0.089 |
| top 4 (MF4, `z21`) | 0.42 | 0.18 at 251k, 0.039 at 1.2M | 0.055, 0.025 |

  - So with fewer exact inputs the deterministic integral of the mean-field anchor beats sampling. MF3's is 1.5–2.6× better in MSE in 2×10⁴–10⁵ forwards, and MF2's far better at 10³. These are the only in-range wins in this section.
  - **Cost caveat (round 5).** These charges (1,528 forwards per MF3 node, 191 per MF2 node, and 9,300 per MF4 node in `z21`, `z23`, `z24`, `z27`, `z28`, `z31`) are ChatGPT round 4's measurements of an optimized prefix tree. That tree was never reproduced here. With the repo's exact tracer, `lines.trace_line`, an MF3 node costs 4,566 forwards (`z41`), 3× the charge. That turns MF3's 1.5–2.6× win into roughly 1.2–2× worse than sampling. MF2's node is a curved path, since its lower inputs move with μ(S), so the affine-line charge does not apply to it at all.
  - They are wins for the anchor, not the accuracy: MF2 is 5.7 points off and MF3 0.68.
  - The correction is the integral over the 7–8 smaller inputs. It is high-dimensional, discontinuous and mostly off the boundary (`z22`), and it has to be modelled per configuration inside the arrangement enumeration.
  - For MF3 that costs at least 720 arrangements × (S-points) × (rows) per outer node, about 41k forwards even with one S-point and the cheapest first-order model. That model is itself 0.156 points off (`z11`), already above sampling's 0.1 at 70k.
  - Using the cheap anchor as a base and sampling only the correction would be a control variate, which the constraints exclude.
  - **End to end (`z38.py`).** I built the full deterministic estimator: the MF3 outer rule (order n), K equal-mass S-points, all 720 arrangements, and at every one of those configurations the clamped first-order lower-input model over the 7 smaller inputs (q Gauss–Legendre nodes). There is no sampling anywhere. Its error against the true accuracy is 5.8, 3.2, 2.1, 0.63 and 0.34 points at 84k, 334k, 668k, 3.0M and 10.5M forwards; sampling gives 0.096, 0.048, 0.034, 0.016 and 0.009 at those costs. That is 3–4 orders of magnitude worse in MSE.
  - So for these networks the accuracy splits into a low-dimensional part that deterministic methods do better than sampling, and a high-dimensional correction that none of the allowed methods tried does as well.
- **Decomposing by events instead is worse (`z39.py`).** Each step's relative rank is independent and uniform, so the new-max / new-2nd / neither event sequence has a product law. That makes events a natural alternative frame.
  - Its mean-field anchor keeps every record input exact (4.9 per sequence) and puts each "neither" input at its conditional mean below the running 2nd max. It is +1.9 points off the true accuracy (RMS 2.0).
  - Keeping the last one or two "neither" inputs exact as well (5.9 and 6.9 exact) still leaves RMS 1.4 and 1.0. The top-4 anchor's RMS is 0.42 with only 4 exact.
  - So conditioning the smaller inputs on the final top 4 loses less than conditioning them on the running state, and the top-k frame is the better one.
- **Why this cannot be closed by a proof.**
  - Any computation on the weights can, given enough compute, find the accuracy exactly; `z9`'s region walk does it for MF3. So the question is only cost, at one fixed size (16 neurons, 10 steps). Complexity theory gives no per-instance lower bound at a fixed size.
  - The nearest theory is volume estimation in the black-box (membership-oracle) model. There, deterministic polynomial-time algorithms provably cannot approximate the volume of a convex body in n dimensions to better than a factor exponential in n (Elekes 1986; Bárány and Füredi 1987). Randomized algorithms can (Dyer, Frieze and Kannan 1991).
  - So in the black-box setting deterministic estimation is strictly weaker than sampling, and a mechanistic method could only win by using the weights' internal structure.
  - For white-box methods at a fixed size, a lower bound would have to say that no circuit computing the estimate from the 432 weights, within a given number of operations, reaches error ε over the population. That is a concrete circuit lower bound for an explicit function. Even superlinear circuit lower bounds for explicit functions are open in general, so no such proof is in reach. The most that can be shown is what is shown here: each nameable family has been measured, and where a best case could be defined (an ideal or an oracle version), that was measured too.
  - The two remaining frames the round-5 brief names are covered as well. "In ages" (of the max and 2nd max) is the top-2 frame, MF2 (`z37`). "Along the running max and 2nd max" is the event frame (`z39`).
  - That is what every route in this section tried: lines, regions, linear maps, clock states and additive lower-input effects. Each has now been bounded at the level of its family, by an ideal or oracle version: `z26`, `z30`, `z34`, the order-10 tensor rule. Even those best cases fall short of sampling in 10³–10⁵ forwards.
  - A successful method would need structure that none of these families uses. Nothing measured here points to where that structure would come from.
- **Verdict for ARC's bar.** Deterministic top-4 integration does not reach the sampling line anywhere in 10³–10⁵ forwards. The MF4 anchor alone needs about 10⁶ forwards to come within 3× of sampling's MSE, and it is not the accuracy (+0.3 points optimistic). The correction adds orders of magnitude more, because it is neither local to the boundary nor decomposable without bias. A closed form does not rescue it either: the position effects do not factor (`z24`), and each arrangement's value space holds hundreds of linear regions (`z25`, `z26`). So for these networks the accuracy integral has no low-cost deterministic description at the precision ARC asks for. That is a measured statement about these computations, not a proof about all algorithms. The population changes this at the low end: a correction learned from the 1,024 networks comes within 1.1× of sampling's MSE at about 1.5–1.8k forward-equivalents, all costs counted (`z47`, below).

**Round 5 (ChatGPT) and its checks (`z40`–`z44`).** The brief is `QUESTION5_FOR_CHATGPT_PRO.md`. The answer is summarized in `chatgpt2/chatgpt_answer_4.md`, with its full text, code, report and results in `chatgpt2/round5/`.

- **The scoring table (`z40.py`).** `pop1k/reference.npz` holds, for all 1,030 networks on the same 2²⁰ inputs, the true accuracy and the MF2, MF3 and MF4 anchors' accuracies. Because the inputs are paired, the leakages have SE 0.017–0.025 points.
- **Its best estimator: two levels (`z41.py`).**
  - The estimator is the exact-S MF3 anchor at outer order 6, plus a coarse MF4 correction Q(F₄ − I₃).
  - F₄ is the clamped first-order model. The correction uses 2 nodes per outer coordinate, 8 S midpoints and 720 balanced arrangements, 2.3M forwards in all.
  - Result: 0.225 / 0.249 points RMS (development / held out) at 2.5M forwards, 170 / 220× sampling's MSE. `z41` re-implements it and reproduces every digit.
  - The error is mostly the coarse integration of the correction (0.23–0.25), not the model (0.065) or the anchor (0.063).
- **Where the correction lives (`z42.py`).**
  - Evaluating the 48 lower-input rows only where the anchor's logit margin is smallest gives 104 / 104× at 0.92M.
  - Screening each lower input at its extreme nodes first gives 79 / 97× at 1.0M. Screening only the largest node is worse than no correction.
  - The correction is two opposing flows: lower inputs break right answers (−2.3 points) and fix wrong ones (+1.9). Very negative lower inputs almost only break (−0.50 against +0.08); inputs near Q do both.
- **The correction does not amortize across arrangements (`z43.py`).** Its SD across arrangement means is 2.0 points for true − MF4 and 3.9 for true − MF3. The positions of M and S explain 16–22% of that.
- **Gate histories proliferate (`z44.py`).** At step 10, 82% of test mass lies in histories absent from 32k discovery inputs, and 69% in ChatGPT's count.
- **Its lower bound is valid but narrow.**
  - Deterministic label-only membership queries on one polyhedral cone in R¹⁰ have worst-case error at least 0.0397·N^(−1/4). This is checked numerically in `chatgpt2/round5/check_lower_bound.py`.
  - It does not transfer to the RNN family, to weight access, or to the population average.
  - The measured deterministic frontier falls at about cost^−0.28, close to that label-only rate.
- **Corrections to this section:**
  - the additive floor (above);
  - `z26`'s uncovered mass is a certificate radius, not an error floor;
  - `z22`'s non-locality concerns its margin proxy and is not a lower bound;
  - the tree-cost caveat (above).

**Calibration against the population (`z45.py`).** The round-4 and round-5 briefs excluded "any fitted prediction of accuracy or error from the population". That exclusion was mine, and wrong: using the population is what it was trained for. Cross-validated linear fits of the true accuracy on deterministic per-network quantities use 8 folds over the 1,024 networks.

| deterministic part | + network features | forwards | CV RMS (points) | sampling | MSE / sampling |
|---|---|---|---|---|---|
| none | weight statistics (free) | 0 | 1.28 | | below sampling for budgets under about 420 |
| coarse MF4/MF3 anchors, 1 outer node | weight statistics | 11.5k | 0.65 | 0.25 | 7.0 |
| coarse anchors, 8 outer nodes | role features (4,096 probes) | 96k | 0.30 | 0.085 | 13 |
| exact MF3 + coarse anchors | role features | 265k | 0.29 | 0.051 | 32 |
| two-level, screened, components | role features | 1.0M | 0.18 | 0.026 | 46 |

- Calibration cuts the MSE 2–4× at every budget. The fits carry over to the zoo models with errors mostly under 0.4 points.

**A learned correction ("learned quadrature", `z46.py`, `z47.py`).**

- **Cells (`z46`).** The configuration space of the top 4 is cut into equally likely cells:
  - n pieces of each MF4 outer coordinate (n³ boxes);
  - K slices of S's conditional CDF;
  - the 720 MF3 arrangements, with Q's position uniform over the other 7.
- **Features.** Each cell gets 2 deterministic forward rows at its centre, the MF4 anchor and the MF3 anchor, giving their correctness bits and logit margins.
- **Training targets.** For training only, each cell's true accuracy is measured with 32–128 draws inside the cell. The mean of these targets matches the reference to 0.03–0.16 points.
- **The model (`z47`).**
  - A small MLP maps a cell's bits and margins, its position pattern and box and slice, plus the network's own context, to the cell's accuracy.
  - The context is the means of the cell features and, optionally, the free weight statistics.
  - It is trained with soft cross-entropy on networks outside the scored fold.
  - A network's estimate is the mean of the MLP over its cells, so the per-network work is only the 2 rows per cell, fully deterministic.

| cells (boxes × S slices × 720) | forwards | RMS, all 1,024 (8-fold cross-fitting, 2 training seeds) | MSE / sampling | zoo MSE / sampling |
|---|---|---|---|---|
| 1 × 1 | 1,440 | 0.64, 0.62 | **0.85, 0.80** | 1.9, 1.8 |
| 1 × 2 | 2,880 | 0.56, 0.56 | 1.29, 1.32 | 2.9, 2.8 |
| 1 × 4 | 5,760 | 0.45, 0.45 | 1.70, 1.71 | 3.2, 3.1 |
| 1 × 8 (dev/held split) | 11.5k | 0.36 / 0.35 | 2.1 | |
| 8 × 8 (dev/held split, no weight features) | 92k | 0.33 / 0.36 | 16 | |

- **Counting forward rows only, the learned model is below the sampling line at 1,440 rows.** With the free weight statistics in the context, its MSE is 0.80–0.85× sampling's on all 1,024, cross-fitted and robust to the training seed. Without the weight statistics it is 3.2× (dev/held split).
- **Counting its own arithmetic, it is not.** The cost unit counts all arithmetic on the target network, and the 128-128-64 MLP's inference is 25.8k network-dependent multiply-adds per cell. That is 6,452 forward-equivalents for 720 cells, which puts it at 4.68×. Smaller models keep most of the accuracy (1 × 1 cells, weight statistics, cross-fitted, seed 47):

  | MLP widths | inference (forward-equivalents) | total | RMS, all 1,024 | MSE / sampling, all costs | zoo |
  |---|---|---|---|---|---|
  | 128, 128, 64 | 6,452 | 7,892 | 0.640 | 4.68 | 10.3 |
  | 32, 32 | 340 | 1,780 | 0.654 | 1.10 | 2.3 |
  | 16, 16 | 108 | 1,548 | 0.710 | 1.13 | 1.8 |
  | 8 | 24 | 1,464 | 0.837 | 1.48 | 1.6 |

  So the closest honest point is about 1.1× sampling's MSE at 1.5–1.8k forward-equivalents: close to the line, not on it. The other rows of the table above count forward rows only.
- **The ratio worsens with budget.** It goes from 1.3× at 2.9k forwards to 16× at 92k.
- **ARC's zoo transfers worse,** at 1.8–3.2×. The mystery model is consistently 1.4–1.9 points high.
- **The error is missing information, not overfitting.**
  - In-sample RMS on the 896 training networks is 0.346, against 0.349 on the test networks.
  - The error sits in cells where both anchors are right, and in the extreme S slices (S next to R or next to M).
  - It is spread over every position pattern and correlated across them. So each network has its own rate at which lower inputs break right answers, and two anchor rows cannot reveal it.
  - Adding the 12 extreme-node screening rows per cell lowers the RMS a little (0.23 at 645k forwards), but their cost grows faster (51×). Using them only as network context on every 16th cell gives 16× at 127k.

### 7.17 A network-level readout, the mystery model's own training recipe, and ARC's earlier techniques (`gfeat.py`, `z49.py`–`z58.py`, `batch_train_unif.py`)

**ChatGPT round 6's global readout G(B, K), checked independently (`z49`, `z52`, `z53`).** Each network is run on fixed rows: 720 top-3 arrangements × K S-slices × B outer boxes, two rows per cell (the MF4 and MF3 anchors), through a prefix trie. Ten cell features (the two correctness bits, the two margins divided by the logit scale s = RMS(m4), their absolute values and two squashed versions of each) are averaged per network: globally, per S-slice and per box digit. The logit-scale context (s, log s and the two mean margins) and 20 raw weight statistics are added, and an RBF kernel ridge over the 1,024 networks maps the result to accuracy. All arithmetic is counted, at 2,880 multiply-adds per forward-equivalent (FE). Cross-fitted on the 1,024 with 8 folds (two values: folds by i mod 8 / by a random permutation; ranges span several fold schemes):

| design | cost (FE) | RMS (points) | MSE / sampling |
|---|---|---|---|
| G(1,1) | 619 | 0.58–0.61 | 0.30–0.33 |
| G(1,2) | 1,130 | 0.49–0.51 | 0.40–0.42 |
| G(1,4) | 2,152 | 0.43 | 0.58 |
| G(1,8) | 4,196 | 0.36–0.38 | 0.80–0.88 |
| G(8,2) | 8,792 | 0.353 / 0.375 | 1.6 |
| G(8,8) | 33,387 | 0.253 / 0.269 | 3.1 |
| G(27,8) | 112,333 | 0.222 / 0.233 | 8.0 |
| G(125,8) | 514,896 | 0.203 / 0.210 | 31 |

- **The readout beats sampling's MSE up to about 4k FE**, and falls further behind as the budget grows, as `z47` did.
- **The information is in the combination** (`z49`, K = 1, RMS): cells alone 1.36 points; cells + logit scale 1.12; cells + weights 0.87; all three 0.61. Weights alone give 1.17.
- **The logit scale and raw weight statistics are not invariant** under the maps that leave every decision unchanged (positive rescaling of hidden units, scaling of W_o). They carry a training fingerprint: with the weights put in a canonical gauge, cells + weights give 1.05 instead of 0.87.
- **The mystery model is the outlier:** it is predicted 1.5–2.3 points high at every budget, which is 7–11× sampling's MSE on it alone.

**The mystery model was trained differently (`z50`, `z51`, `z54`).** Zihao Chen's notebook, linked from ARC's post, trains it on Unif(−1, 1) inputs with Adam (lr 0.005, weight decay 10⁻¹⁴), learning rate × 0.995 per epoch, 1,000 epochs × 100,000 sequences. The batch size is not given. ARC's footnote says it predates their standardized codebase. The zoo uses N(0,1) inputs, AdamW with weight decay 10⁻³ and 2³⁰ sequences. The mystery model's logit margins exceed those of all 1,024 zoo-recipe networks. `batch_train_unif.py` reproduces the recipe in batch; after 300 Adam steps one batched model differs from `nn.RNN` by at most 2.4×10⁻⁷ in any weight. Accuracy on N(0,1) inputs (2²² inputs per network):

| batch | networks | mean accuracy (SD) | mystery percentile |
|---|---|---|---|
| 64 | 64 | 90.68% (3.53) | 96.9 |
| 128 | 1,024 | 90.56% (3.70) | 97.2 |
| 256 | 64 | 88.08% (5.75) | |
| 512 | 64 | 81.59% (6.08) | |
| 1,024 | 64 | 78.30% (6.77) | |

- **Batch 64–128 is the plausible setting.** At batch 256 and above the mystery model (95.30%) beats every network (`z54`); at batch 64–128 it is a top-3% network of its own recipe, not an anomaly.
- **The blog's subcircuit is more common under this recipe** (`z56`). The isolated max pair (neurons 2 and 4) and leave-one-out-max units (6, 7) are found from the weights alone by a rescaling-invariant template match. The mystery model matches at distance 0.066 with units 6 and 7. A pair within distance 1.0 appears in 69% of mystery-recipe networks against 50% of zoo-recipe ones; leave-one-out units average 0.27 against 0.14 per network. None of zoo seeds 0–4 has a clean copy.

**Training the readout on the right recipe fixes the mystery model's transfer (`z55`, `z58`).** The same G(1,K) is trained on each population. It is scored by 8-fold cross-fitting inside the training population, and on every other network by a model refitted on the whole training population. Errors are in accuracy points; MSE/s is the MSE divided by sampling's at the same cost. The noise sets are ARC's suggested protocol: 256 copies of the mystery model, with `i8.py`'s noise (relative on W_ih and W_o, additive on W_hh at the level × mean |W_hh|). At 0.3% the copies score 94.1% ± 0.7; at 1%, 84.0% ± 4.9.

| K (cost) | trained on | own-recipe cross-fit, RMS (MSE/s) | other recipe, mean error | mystery | zoo seeds 0–4 | noise 0.3%, RMS (MSE/s) | noise 1%, mean error |
|---|---|---|---|---|---|---|---|
| 1 (619) | zoo recipe | 0.58 (0.30) | +3.68 | +2.22 | −0.18 to +0.51 | 3.38 (12.8) | +10.5 |
| 1 (619) | mystery recipe | 0.95 (0.67) | −5.32 | −0.55 | −6.7 to −4.8 | 0.80 (0.72) | +6.9 |
| 2 (1,130) | mystery recipe | 0.65 (0.57) | −3.54 | −0.37 | −5.5 to −2.5 | 0.74 (1.12) | +5.3 |
| 4 (2,152) | zoo recipe | 0.43 (0.58) | +1.60 | +1.54 | −0.64 to +0.81 | 2.25 (19.7) | +6.6 |
| 4 (2,152) | mystery recipe | 0.56 (0.79) | −2.72 | −0.07 | −4.5 to −1.6 | 0.77 (2.31) | +5.1 |
| 4 (2,217) | both (2,048) | zoo 0.47 (0.70), mystery 0.61 (0.98) | | +0.33 | −0.67 to +0.93 | 1.00 (4.0) | +3.6 |

- **On the mystery model's own recipe the readout beats sampling's MSE** at 619–2,152 FE: cross-fitted over 1,024 training seeds (ARC's "average over random seeds") and on 64 further batch-128 seeds (0.56×, 0.42× and 0.71× at K = 1, 2 and 4). The mystery model itself is off by −0.55, −0.37 and −0.07 points.
- **On ARC's weight-noise protocol** it beats sampling at 0.3% noise for K = 1 (0.72×), but the copies are about 0.5–0.65 points overestimated at every K. At 1% noise it fails (+5 to +7 points). Those copies are damaged models far outside any trained population, and the cheap cells' own bias grows with the damage: the K = 1 cells' mean MF4 correctness bit is +8.2 points above accuracy on the population and +12.4 on the 1%-noise copies.
- **Recipes do not mix.** Each population's readout is 1.6–5.3 points off on average on the other recipe. Training on both is worse on each than training on its own (cross-fit 0.98 and 1.20 at K = 1, against 0.58 and 0.95), so the features cannot fully tell the recipes apart. Batch size matters a little: on 64 batch-64 networks the batch-128 readout is +0.80, +0.49 and +0.25 points biased at K = 1, 2 and 4 (1.04–1.20× sampling's MSE).

**ARC's earlier techniques as extra features (`z56`).** All are deterministic, and their cost is counted.

| added to G(1,K) | own-recipe cross-fit, K = 1 / 4 (zoo recipe; mystery recipe) | change |
|---|---|---|
| none (raw weight statistics) | 0.59 / 0.43; 0.95 / 0.56 | |
| balanced gauge (W_o centred, each unit's incoming norm = outgoing norm), replacing raw | 0.65 / 0.41; 0.96 / 0.57 | mixed (±0.05) |
| ARC's standardization (\|W_ih\| = 1), replacing raw | 0.78 / 0.46; 1.14 / 0.59 | worse |
| subcircuit template (10 features) | 0.65 / 0.44; 0.95 / 0.55 | none or worse |
| order-2 covariance propagation, exact bivariate ReLU closure (13 features, +167 FE) | 0.66 / 0.44; 0.93 / 0.56 | −0.02 at most; its cost makes MSE/sampling worse (0.66 → 0.81 at K = 1) |
| all three (+188 FE) | 0.60 / 0.40; 0.91 / 0.56 | −0.04 at most; worse per FE (0.79× at K = 1) |

- **Nothing improves the within-population fit at equal cost.** The changes are within ±0.05 points, near fold noise (0.02–0.03), except the mystery-recipe K = 1 fit with all three (0.95 → 0.91), which pays 188 FE for it.
- **ARC's standardization helps across recipes:** trained on the zoo recipe at K = 4, it cuts the mystery-recipe error from +1.60 to +0.95 points and the mystery model's from +1.54 to +0.74. But it costs 0.04 points of RMS inside the population.

**Hermite / low-degree polynomial propagation cannot work here (`z57`).** ARC's random-MLP method represents each unit as a degree-≤d polynomial in the Gaussian inputs, with n^d ≤ 1/ε². For n = 10 inputs that gives d = 2 at about 600 FE and d = 3 at about 4k. The oracle gives this method every advantage. At every step it replaces the hidden state by its exact least-squares projection onto degree ≤ d, fitted on 2¹⁸ inputs. It then evaluates the final argmax and the target exactly on 2¹⁹ fresh inputs. On 12 networks (the six zoo models and three from each population):

| degree | polynomials | oracle accuracy (truth 86–96%) | R² of the final hidden state | R² of the logit gaps |
|---|---|---|---|---|
| 1 | 11 | 7–24% | 0.33–0.57 | 0.19–0.84 |
| 2 | 66 | 7–24% | 0.65–0.81 | 0.46–0.90 |
| 3 | 286 | 9–33% | 0.81–0.92 | 0.63–0.93 |
| 4 | 1,001 | 12–40% | 0.86–0.94 | 0.73–0.95 |

- **Projecting the true final logits directly** (no recursion) is no better: 16–36% at degree 4.
- **The reason** is the same cancellation that broke the moment closures (`z8`). The margins (about 1–4) are small differences of activities of 20–90, so even 5–27% unexplained variance flips most decisions. ARC itself does not expect cumulant propagation to work even for its trained two-layer MLPs ("Competing with sampling", 2025). Order-2 propagation inside cells is `z8`'s closures, which already failed.

### 7.18 Cheaper probes, single-input pulses, and a test on new networks (`z59`–`z61`, ChatGPT round 7)

**The round.** `QUESTION7_FOR_CHATGPT_PRO.md` asked for a review of G(1, K) (7.17), a brainstorm of features that might carry its residual, and optimizations, all in the range where it already wins (10²–10⁴ FE). For the round, `z59` cached every mystery-recipe network's cell outputs so the fits run on a CPU (`pop_unif/cells_k{1,2,4}.npz`). The answer, its standalone estimator (`round7_estimator.py`), its PDF report and its result archive are in `chatgpt2/round7/`. Its script reproduces its own cross-fit numbers here to four decimals. `z60` rebuilds the new pieces from the answer's description and fits them with `gfeat`'s readout; it agrees within 0.01–0.02 points.

**Review.**
- **No leakage was found.** Standardization, centring and hyperparameter choice stay inside the folds. The labels of all networks come from one shared input bank, so their errors (0.014–0.026 points) are correlated across networks. That is small next to the residuals, but a final test should use a second bank.
- **The fold layout moves the numbers by up to 0.03 points.** `gfeat.pick` builds its inner folds from `index % len(y)`, so the inner folds, and the chosen hyperparameters, depend on which other networks sit in the array. G(1,4)'s cross-fit RMS on the mystery recipe is 0.557, 0.550 or 0.583 depending on the layout. The comparisons below use inner folds keyed on the 1,024 training networks only.
- **Two shortcuts that the cost ledger assumes are not in `gfeat`'s code.** It computes W_ih·x at every trie node rather than once per distinct input value (as written, +33, +63 and +122 FE at K = 1, 2, 4), and `fit_predict` evaluates all 63 readout candidates on each target rather than one compiled readout. `round7_estimator.py` implements both, so the charged costs are achievable and no reported number changes.

**What helps, on the mystery recipe** (cross-fitted over the 1,024 training networks; MSE / sampling at the total cost; external errors from a model refitted on all 1,024). "Full" is the RBF readout over all 1,024 networks, "128" a Nyström readout over 128 of them. Rows marked `z60` are the independent rebuild; the others are ChatGPT's script, rerun here.

| design | probe rows | total FE | RMS (points) | MSE / sampling | 64 more batch-128: RMS | batch 64: RMS, mean | mystery |
|---|---|---|---|---|---|---|---|
| G(1,1), 7.17 | 1,440 | 618.5 | 0.964 | 0.68 | 0.875 | 1.269, +0.80 | −0.55 |
| MF4 rows only, K = 2, full | 1,440 | 618.8 | 0.681 | 0.34 | 0.589 | 0.981, +0.51 | −0.26 |
| + pulses, full (`z60`) | 1,440 | 631.9 | 0.663 | 0.33 | 0.630 | 0.839, +0.37 | −0.36 |
| + pulses + temporal groups, 128 | 1,440 | 595.2 | 0.621 | 0.27 | 0.538 | 0.854, +0.43 | −0.25 |
| + pulses + temporal groups, full | 1,440 | 650.6 | 0.606 (`z60` 0.613) | 0.28 | 0.543 (0.565) | 0.831, +0.42 | −0.45 (−0.35) |
| the same at K = 4, full | 2,880 | 1,216.5 | 0.516 (`z60` 0.518) | 0.38 | 0.483 (0.488) | 0.585, +0.23 | +0.00 (+0.03) |
| compact, both rows, K = 1, + pulses, 128 | 360 | 176.0 | 0.951 | 0.19 | 0.851 | 1.095, +0.67 | −0.89 |
| compact, MF4 rows only, K = 1, + pulses, full (`z60`) | 180 | 132.3 | 0.949 | 0.14 | 0.895 | 1.065, +0.59 | −0.91 |
| compact, MF4 rows only, K = 2, + pulses, full (`z60`) | 360 | 207.1 | 0.797 | 0.16 | 0.827 | 0.810, +0.36 | −0.82 |
| G(1,4), 7.17 | 5,760 | 2,151.9 | 0.583 | 0.87 | 0.529 | 0.635, +0.25 | −0.07 |

- **MF4 rows at two values of S beat MF4 and MF3 rows at one.** The same 1,440 rows cost the same (566.9 against 563.4 FE), and the error falls from 0.96 to 0.68 points. How the margin changes as S moves between R and M says more about a network than the difference between the MF4 and MF3 anchors.
- **Two signed pulses** (6 FE). The network is run on the ten sequences that are zero except for one input of +1, and the ten with −1: h₀ = ReLU(±W_ih), then nine steps of h ← ReLU(W_hh·h) with logits W_o·h after each. For each sign, the features are the mean margin of the output at the pulse's position over the best other output, the same at ages 5–9 (both divided by the RMS of the centred logits), the share of ages where that output wins, and the mean and SD of the change in log logit scale with age. This is the impulse response of 3.3 and 7.9 read out through the real gates: whether an input's trace stays decodable as it ages. It is not task accuracy (a lone pulse has no well-defined 2nd max). The features do not change under the exact symmetries of 7.17. They improve G(1,1) from 0.964 to 0.841 points on the mystery recipe, but only from 0.597 to 0.579 on AlgZoo's.
- **Temporal groups** average each S slice's MF4 bit and squashed margin over six groups of arrangements: M last, S last, M right after S, M after S, R after S, and M and S at least 5 steps apart. They separate late handoffs, direct writes and long retention. On the mystery recipe they cut the error from 0.663 to 0.613 at K = 2. **On AlgZoo's recipe they make it worse** (0.526 → 0.684 at K = 2; 0.595 at K = 4 against G(1,4)'s 0.435). ChatGPT traced the K = 1 case to one network (number 994, 78.6% accurate), whose error rises from +4.2 to +13.7 points. Appending features to an RBF kernel multiplies the similarities, so an unusual temporal profile cuts a poor network off from its neighbours. An additive kernel (base plus mechanism term) is the suggested fix; it is untested.
- **The compact design** keeps all 90 placements of M and S but puts R only at the first or last free position (180 of the 720 arrangements). With MF4 rows only it costs 132 FE, against G(1,1)'s 619, at the same error.
- **A 128-network Nyström readout** cuts the readout's cost from 43–88 FE to 6–8 FE, at little loss on the mystery recipe. On AlgZoo's recipe it costs more accuracy for some designs (G(1,2): 0.515 → 0.567).

**AlgZoo recipe** (`z60`, cross-fitted; zoo seeds 0–4 from the refitted model): MF4 rows at K = 2 with pulses give 0.526 points at 632 FE (0.25× sampling's MSE; seeds −0.26, +0.75, −0.04, −0.42, −0.03). The compact MF4 design with pulses gives 0.653 at 132 FE (0.08×; seeds +1.08, +0.23, −0.22, −0.45, +0.38), and 0.587 at 207 FE with two slices (0.10×).

**What did not help, or not enough** (ChatGPT's measurements unless marked):
- **Margin gradients** with respect to the six lower inputs, on 68 MF4 rows at the extreme S slices and late-handoff arrangements (+151 FE as implemented, +78 with the forward gates kept): 0.583 → 0.516 at K = 4 on the mystery recipe, 0.435 → 0.406 on AlgZoo's; no gain per FE at K = 1–2.
- **Margin tails and MF3/MF4 disagreement** (counts of near-zero margins): 0.583 → 0.529 at K = 4, nothing at K = 1.
- **Impulses injected into single hidden units** (states inputs may not reach) and the bare W_hh spectrum: worse.
- **More S slices** do not move the crossover: MF4 with pulses is 0.83× sampling's MSE at K = 8 (2.2k FE) and 1.43–1.71× at K = 16 (4.4k FE). The crossover stays near 4k FE.
- **More training networks** may not help this representation. For MF4 K = 2 with pulses (591 FE), training on 224, 448, 672 and 896 networks gives 0.90, 0.77, 0.65 and 0.65 points.

**A test on new networks (`z61`).** The round tried many variants on the same 1,024 networks, and the zip held the labels of the 64-network groups. So 512 new batch-128 networks were trained after the answer came back (`pop_unif/b128_n512_seed6.npz`, seed 6; 90.22% ± 3.63, with the mystery model at their 98.6th percentile). They were labelled on an independent bank of 2²² inputs (seed 61). The estimators were fixed before the new networks finished training: ChatGPT's compiled models, fitted on the 1,024 and rerun here, and `z60`'s rebuilds, refitted on all 1,024. The intervals are bootstrap 95% intervals over the new networks.

| design | total FE | cross-fitted on the 1,024: RMS (MSE / sampling) | 512 new networks: RMS, mean error | MSE / sampling (95% interval) |
|---|---|---|---|---|
| G(1,1) | 618.5 | 0.964 (0.68) | 0.991, +0.08 | 0.70 (0.59–0.81) |
| G(1,2) | 1,129.7 | 0.678 (0.62) | 0.680, +0.10 | 0.60 (0.50–0.71) |
| G(1,4) | 2,151.9 | 0.583 (0.87) | 0.573, +0.10 | 0.81 (0.68–0.96) |
| MF4 rows only, K = 2 | 618.8 | 0.681 (0.34) | 0.676, +0.09 | 0.33 (0.27–0.38) |
| + pulses (`z60`) | 631.9 | 0.663 (0.33) | 0.618, +0.09 | 0.28 (0.24–0.32) |
| + pulses + temporal groups, 128 | 595.2 | 0.621 (0.27) | 0.681, +0.10 | 0.32 (0.24–0.42) |
| + pulses + temporal groups, full | 650.6 | 0.606 (0.28); `z60` 0.613 (0.29) | 0.599, +0.09; `z60` 0.588 | 0.27 (0.23–0.31); `z60` 0.26 (0.22–0.30) |
| the same at K = 4 | 1,216.5 | 0.516 (0.38); `z60` 0.518 (0.39) | 0.544, +0.11; `z60` 0.551 | 0.41 (0.35–0.48); `z60` 0.43 (0.35–0.50) |
| compact, both rows, K = 1, + pulses, 128 | 176.0 | 0.951 (0.19) | 0.950, +0.10 | 0.18 (0.16–0.21) |
| compact, MF4 rows only, K = 1, + pulses (`z60`) | 132.3 | 0.949 (0.14) | 0.963, +0.09 | 0.14 (0.12–0.16) |
| compact, MF4 rows only, K = 2, + pulses (`z60`) | 207.1 | 0.797 (0.16) | 0.785, +0.07 | 0.15 (0.13–0.17) |

- **The gains hold.** On new networks every design stays within about 0.06 points of its cross-fitted error, and every new design's interval stays well below 1 (the widest reaches 0.50). At 132 FE the compact MF4 design is 0.14× sampling's MSE, and at 651 FE the full K = 2 design is 0.26–0.27×. G(1,1) at the same cost is 0.70×.
- **Two variants slip a little:** the temporal groups with the 128-network readout (0.62 → 0.68 points), and the K = 4 designs (0.52 → 0.55). The full-kernel K = 2 version holds (0.61 → 0.59–0.60).
- **Every estimate is slightly high on the new networks** (+0.07 to +0.11 points). The new bank's labels are 0.030 points below the old bank's on the same networks, which accounts for +0.03; the rest is small next to the RMS.
- **The label bank barely matters.** Scored against the old bank, every RMS is within 0.006 points of the new bank's.

### 7.19 A readable estimator: what the readout uses (`z62`, `z63`)

**The question.** The readouts of 7.17–7.18 are kernels over 40–100 summaries. Can a few named quantities with a mechanistic meaning replace them, and what do those quantities measure? `z62` summarizes each mystery-recipe network by 44 named quantities from MF4 probe rows at two S slices (lower half of S's range, "S near R"; upper half, "S near M") and the two pulses of 7.18:
- **per slice:** the share of rows answered right (`bit`), the mean squashed margin a/(0.1+|a|) (`soft`), the mean |a|, and the share with |a| < 0.1, where a is the margin divided by the network's RMS margin s;
- **per slice and arrangement group** (7.18's six groups): `bit` and `soft`;
- **the ten pulse quantities** and the logit scale (log s, mean margin).

It fits ridge and additive (hinge-basis) models, choosing the quantities by greedy forward selection inside each training fold. Models are fitted on the 1,024 training networks, cross-fitted with 8 folds, and checked on 7.18's 512 new networks with their independent labels. `z63` tests the invariant quantities alone, the cheapest version, and links to neuron roles.

**Results on the mystery recipe** (RMS in points; MSE / sampling at the total cost):

| model | total FE | cross-fitted on the 1,024 | 512 new networks | mystery |
|---|---|---|---|---|
| kernel on the 44 named quantities + 20 weight statistics, 720 arrangements | 644.9 | 0.618 (0.29) | 0.593 (0.26) | −0.38 |
| ridge on the same | 587.3 | 0.683 (0.33) | 0.667 (0.30) | −0.94 |
| 3 selected quantities, ridge | 587.3 | 0.954 (0.63) | 0.979 (0.65) | −0.63 |
| 12 selected quantities, additive | 587.3 | 0.753 (0.40) | 0.747 (0.38) | −0.82 |
| compact design: ridge on the 44 + weight statistics | 161.3 | 0.887 (0.15) | 0.860 (0.14) | −1.55 |
| **2 quantities, 180 rows: S-vs-R separation + logit scale** | **83.6** | **1.394 (0.19)** | **1.446 (0.20)** | −0.53 |
| the same + pulse growth | 86.6 | 1.250 (0.16) | 1.304 (0.17) | −1.20 |
| the same + three pulse quantities | 89.6 | 1.202 (0.15) | 1.236 (0.16) | −1.28 |

- **A two-term formula beats sampling at 84 FE.** It runs the 180 compact MF4 rows at the lower S slice only. In units of each quantity's SD over the training networks: accuracy ≈ 90.56% + 2.18 × (S-vs-R separation) + 1.50 × (log logit scale), where both terms are measured from their population means.
- **More named quantities help, but slowly.** Forward selection on the full design picks, in order: S-vs-R separation, logit scale, the +1 pulse's growth, the SD of the −1 pulse's growth, `bit` with S near M, |a| with S near M. Three quantities reach 0.95 points, twelve 0.75–0.79; the kernel over everything reaches 0.62. A plain ridge on all 44 plus the weight statistics is within 0.07 points of the kernel, so most of what the kernel uses is linear in these quantities.
- **The mystery model is underestimated by every readable model**, by 0.5–1.6 points (the kernel: 0.4). It sits at the 97th–99th percentile of its recipe, where a model with few inputs pulls hardest toward the mean. At 84 FE sampling's own error on it is ±2.3 points.

**Against sampling that may also use the population** (`z63`, part D). The population mean alone (RMS 3.7 points) beats plain sampling below 65 FE, so at the low end the fair comparator is a sampler that also knows the population: the sampled accuracy shrunk toward the 1,024's mean, with weight var/(var + p(1−p)/C). On the new networks it has RMS 2.41 points at 84 FE, against plain sampling's 3.22. The two-term formula is 0.36× its MSE there, and 0.30× with the pulse growth. For 7.18's designs the same comparison gives 0.21× at 132 FE, 0.19× at 207, 0.29–0.30× at 651 and 0.44–0.45× at 1,216. So the gain is not just the population prior.

**What the main quantities measure.**
- **S-vs-R separation** (`soft` with S near R): how decisively the anchor rows pick the 2nd largest over a 3rd largest just below it. Across the 1,024 networks it correlates +0.92 with accuracy. It does not change under the exact symmetries. Of the neuron roles it tracks the comparators most (r = +0.43). Comparators test each new input against thresholds that blend M and S (3.1), so more of them separate S from a close R more sharply. On the mystery recipe, accuracy is 87.2–87.7% with 1–2 comparators, 90.4% with 3, 91.5% with 4 and 92.3–92.8% with 5–6.
- **Logit scale** (log s): r = +0.82. It is not a property of the computed function: rescaling W_o changes it and no answer. Within one recipe, networks that trained to larger margins are more accurate. Without it, and without the mean margin, the invariant quantities alone give 1.26 points with three quantities (1.11× sampling's MSE) and 1.02 with six (0.73×) on the full design, and 1.49 (0.42×) with three on the compact one. So the scale carries information the invariant probes miss at this budget.
- **Pulse growth**: the mean change per step of age in the log logit scale of the +1 pulse's response; r = −0.46. Networks in which a single input's trace fades faster are more accurate. Of the roles it links most to late markers (+0.31).

**What the readable model misses** (`z63`; neuron roles from 4,096 probes per network, used only to interpret, since they cost far more than the estimate).
- **Strategy:** 282 of the 1,024 networks have a late pointer (accuracy 91.17%), 153 an event pointer (90.56%), and 589 none (90.26%). Mean roles per network: 3.48 comparators, 3.18 always-on, 1.13 max detectors, 0.33 late markers, 7.43 other.
- **The roles explain part of the 6-quantity model's error.** Its cross-fitted RMS of 0.855 falls to 0.773 when a cross-fitted ridge on the role counts is added. The model underestimates networks with max detectors (residual correlation −0.31) and overestimates late-pointer networks (+0.24).
- **The worst-predicted networks:** over-estimated ones are late-pointer networks with 2–4 comparators (75–82% accurate, estimated 3–4 points too high). Under-estimated ones are mostly pointer-less networks with 1–2 max detectors.
- **So max detection is the clearest missing feature.** The arrangement groups of 7.18 (M last, M right after S) were offered to the selection and only weakly chosen. A dedicated probe fails (`z64`, below). 7.20 reads the max-detector count off the weights instead, which removes this part of the residual.

**A handoff probe does not help (`z64`).** The probe uses double pulses: sequences that are zero except x_i = 1 and a later x_j = r.
- **Handoff (r = 1.25 or 2):** the answer is i, so the old max must move into the 2nd-max slot, including when the new max arrives at the last step.
- **New 2nd max (r = 0.8 or 0.5):** the answer is j.

That is 180 rows, 68 FE. The networks answer them mostly right (91–99%), and the probe's quantities correlate at most +0.46 with accuracy. They track comparators and dead neurons rather than max detectors or late pointers.
- **Added to the minimal estimator,** they lower the error by 0.02–0.05 points but cost 68 FE: 0.24–0.27× sampling's MSE, against 0.16× without them.
- **In the full design,** none enters the first ten selected quantities, and the 6-quantity model's residual keeps the same links to max detectors (−0.31) and late pointers (+0.24).

In isolation almost every network hands over correctly; what separates them happens only with the other eight inputs present.

**Where the residual lives (`z65`; offline, 2²⁰ sampled inputs per network).** Each network's accuracy loss was measured in 88 regions, defined by the ages of M and S and by near ties M ≈ S and S ≈ R (within 0.05·|M|). Each region's loss was then predicted from a readable model's quantities by a cross-fitted ridge. The regions' unpredicted losses add up to the accuracy residual.
- **The residual is spread out, roughly in proportion to region size.** With the 44 named quantities and the weight statistics, the regions where M and S are both at least 4 steps old hold 33% of the inputs and carry 33% of the residual variance. Near ties S ≈ R are over-represented (31% of the residual at 16% of the inputs). For the two-quantity model, so is a new max at the last step (22% at 10%).
- **In each region that carries much of the residual, the loss falls with the number of max detectors** (r = −0.19 to −0.37). Their benefit is a network-wide rate, not a region, which is why no single probe family catches it.
- **This is 7.16's breaking rate again.** The lower inputs disturb old memories, and mean-field rows hold them at their mean. A probe that could see it would have to put real lower-input variation into rows with an old S. Candidates are near-threshold inputs between R and S arriving after S, or joint perturbations of adjacent lower inputs. Round 7's margin gradients point the same way: they helped only at K = 4, 0.58 → 0.52 points for +151 FE. 7.20 tries such probes; they turn out to be redundant with the anchors.

### 7.20 Cheaper still: memory probes, neuron roles from the weights, and learned probe rows (`z66`–`z71`)

All results in this section are on the mystery recipe. Everything is fitted on the 1,024 training networks, cross-fitted with 8 folds (all selection inside the folds) and applied unchanged to 7.18's 512 new networks with their independent labels. The new networks were not used for fitting, but their scores were watched while choosing what to try next, so they are no longer a blind test for this section. A fresh set of 1,024 networks (seed 7) was trained afterwards for a blind test of the frozen estimators (7.21).

**Probes that put lower-input variation into memory add little (`z66`).** All of these are fixed, structured rows:
- the compact design's 180 lower-slice MF4 rows with the six lower inputs moved together to 0.20 or to −1.52 (their conditional 90th and 10th percentiles below Q);
- the same rows with the lower inputs alternating between those values, in both phases;
- distractor trains: on a zero background, a max of 1.5 and a 2nd max of 1, with every later position filled at 0.95 or 0.75.

Each correlates +0.80 to +0.88 with accuracy, but at most 0.22 in size with the minimal estimator's residual: they show what S-vs-R separation and the logit scale already show. Added to the minimal estimator they lower its error from 1.20 to 1.12–1.16 points, for 64–160 FE, which is worse per FE (0.23–0.38× sampling's MSE against 0.15×).

**Neuron roles can be read off the weights (`z67`).** A logistic classifier, trained on the 1,024 networks' units, predicts each unit's role from 16 weight descriptors. They include the input weight, the self-loop, recurrent and readout norms, the readout's Fourier shares, input-normalized recurrent drives A_kj·w_j/w_k (invariant under the exact rescaling symmetry) and loop products. The classifier costs about 2 FE per network.
- **Unit AUC:** max detector 1.000, comparator 0.998, event pointer 0.993, late pointer 0.959. Network counts correlate with the true counts at +0.89 for max detectors and comparators, +0.67 for event pointers and only +0.13 for late pointers.
- **A boosted classifier** (about 19 FE) reaches +0.94, +0.92, +0.85 and +0.87, with late markers at +0.79.
- **As estimator quantities, the cheap counts help.** The minimal estimator improves from 1.20 to 1.13 points (1.24 → 1.14 on the new networks) for +1.8 FE, 0.14× sampling's MSE. Its residual loses its link to max detectors (−0.15 → −0.03). The compact ridge improves from 0.89 to 0.86 (0.86 → 0.83 on the new networks), 0.13× at 163 FE.
- **The boosted counts are no better as quantities** (1.12), for ten times the cost.

**Learned probe rows set a new low-budget frontier (`z68`–`z71`).** This is the approach ChatGPT's round 6 tried on AlgZoo's recipe. Rows are chosen by greedy orthogonal matching pursuit, on the ridge residual, from a pool of probe rows. The readout sees each chosen row's normalized margin and bit, log s over the chosen rows, the pulse quantities and the 20 weight statistics. The cost is the chosen rows' prefix trie plus 30 operations per row, the 5,000-operation allowance and 6 FE for the pulses.

| pool, extras | rows | total FE | RMS cross-fitted / new networks (points) | MSE / sampling (new) | against shrunk sampling | mystery |
|---|---|---|---|---|---|---|
| K = 4 MF4 rows (2,880), pulses | 16 / 64 | 22 / 53 | 1.36 / 1.25; 0.95 / 0.96 | 0.04×; 0.06× | | −0.66; −0.81 |
| the same + weight statistics | 16 / 64 | 22 / 55 | 1.08 / 1.11; 0.88 / 0.91 | 0.03×; 0.05× | | −0.52; −0.25 |
| K = 8 MF4 rows (5,760), + weight statistics + predicted roles | 8 | 17 | 1.11 / 1.15 | 0.03× | 0.12× | +0.08 |
| | 32 | 35 | 0.93 / 0.97 | 0.04× | 0.11× | −0.41 |
| | 64 | 57 | 0.88 / 0.90 | 0.05× | 0.12× | +0.04 |
| | 128 | 95 | 0.88 / 0.90 | 0.09× | 0.15× | +0.66 |
| | 256 | 164 | 0.98 / 0.99 | 0.18× | 0.26× | +0.78 |
| mixed pool (6,860), + weight statistics | 16 | 22 | 1.01 / 0.96 | 0.02× | 0.09× | −0.44 |
| | 32 | 35 | 0.91 / 0.93 | 0.04× | 0.10× | −0.36 |
| | 64 | 57 | 0.87 / 0.91 | 0.06× | 0.12× | −0.15 |
| | 128 | 99 | 0.87 / 0.86 | 0.09× | 0.14× | +0.05 |

- **More varied values help a little** (`z74`). This pool adds 8 value boxes (two pieces of each order-statistic coordinate) at 4 S values, so 23,040 MF4 candidates. On the new networks: 0.93 points at 38 FE (32 rows), 0.885 at 63 (64 rows) and 0.845 at 109 (128 rows), against 0.97, 0.90 and 0.90 for the one-box pool. The mystery model's errors there are −0.15 to −0.31 points.
- **The mixed pool** (`z71`) adds 7.18's single pulses, 7.19's double pulses and the memory probes above to the K = 8 MF4 rows. The selection still takes mostly MF4 rows (97 of 128), but the other kinds help at small sizes: 0.96 points at 22 FE against 1.06–1.11 for MF4 rows alone.
- **What the learned rows test** (`z76`, the first 32 rows chosen from the 8-box pool on all 1,024 networks). 23 have S within 0.25 of R, 20 have S at least 5 steps old, and 7 have M within 0.25 of S. Only 3 have M last and 4 have S last. Every row's normalized margin enters with a positive weight. They mostly measure how sharply a network keeps a long-held 2nd max apart from a close 3rd: the S-vs-R separation of 7.19, under long retention, which is where `z65` found the residual over-represented.
- **A four-term readable formula** (`z76`, the 180 compact lower-slice rows plus the logistic role counts, 85 FE): accuracy ≈ 90.56% + 1.89·z(S-vs-R separation) + 1.41·z(log s) + 0.58·z(predicted comparators) + 0.66·z(predicted max detectors). Here z(·) is a quantity in training-population SDs from its mean. It gives 1.29 points cross-fitted and 1.33 on the new networks (0.17× sampling's MSE), against 1.39 and 1.45 for the two-term version. With the three pulse quantities it reaches 1.16 and 1.18 at 91 FE (0.15×).
- **A mechanistic stand-in for the logit scale fails** (`z77`). The candidate is a signal-to-noise ratio: each row's margin divided by its change when the six lower inputs all move from μ(Q) to 0.20. It is invariant under the exact symmetries and correlates +0.85 with log s. But with S-vs-R separation it gives 1.59 points (1.65 on the new networks), against 1.39 / 1.45 with log s, and adds nothing alongside log s. Within one recipe, the logit scale seems to record how far training pushed the margins, beyond their robustness to the lower inputs.
- **The linear readout stops improving at about 64–128 rows, and overfits at 256.** The learning curve (`z70`; K = 8 pool with pulses and weight statistics, scored on the new networks) shows why. Trained on 128, 256, 512, 768 and 1,024 networks, 64 rows give 1.71, 1.27, 1.08, 0.95 and 0.95 points, and 128 rows give 1.48, 1.27, 1.17, 0.98 and 0.94. Doubling the training set does help (7.21).
- **Against sampling that may use the population** (shrunk toward the mean, 7.19), the learned rows are 0.09–0.12× its MSE from 17 to 57 FE. The compact ridge of 7.19 is 0.18× at 163 FE.
- **Nonlinear readouts do not break the plateau** (`z73`, the 64-row K = 8 model with the same 169 quantities).
  - A small network (32 ReLU units, 5 seeds averaged, 67 FE) matches the ridge: 0.89 / 0.90 points against 0.88 / 0.90. Its inner cross-validation always chose the strongest penalty tried, making it nearly linear.
  - Gradient boosting (58 FE) is worse: 1.09 / 1.05.
  - A 128-landmark RBF kernel (74 FE) is worse too: 1.05 / 1.01.
  - So the limit is the information in these quantities and the number of training networks, not the readout's shape.

### 7.21 A blind test, and more training networks (`z72`, `z75`, `z78`, `z79`)

**The blind test (`z72`).** `z72 --fit` fitted nine estimators of 7.19–7.20 on the 1,024 training networks and saved them (`pop_unif/frozen_720.pkl`, 08:08 on 2026-10-20). The 1,024 seed-7 test networks (`pop_unif/b128_n1024_seed7.npz`) finished training at 09:57. `z72 --score` then applied the saved estimators unchanged to them, with labels from an independent bank of 2²² inputs (seed 72). It also scored 7.18's compiled models. The test networks are 90.60% ± 3.78 accurate, with the mystery model at their 96.7th percentile. The intervals are bootstrap 95% intervals over the test networks.

| estimator | total FE | RMS (points) | mean error | MSE / sampling (95% interval) | against shrunk sampling |
|---|---|---|---|---|---|
| learned rows, K = 8 pool, 16 | 23.4 | 1.008 | −0.007 | 0.028 (0.025–0.032) | 0.10 |
| learned rows, K = 8 pool, 32 | 34.6 | 0.950 | +0.015 | 0.037 (0.033–0.042) | 0.10 |
| learned rows, K = 8 pool, 64 | 55.1 | 0.903 | +0.017 | 0.054 (0.047–0.061) | 0.11 |
| learned rows, mixed pool, 16 | 21.8 | 1.027 | −0.044 | 0.028 (0.024–0.031) | 0.10 |
| learned rows, mixed pool, 32 | 33.2 | 0.933 | −0.045 | 0.035 (0.031–0.039) | 0.10 |
| learned rows, mixed pool, 128 | 96.0 | 0.842 | −0.012 | 0.081 (0.073–0.090) | 0.13 |
| two-term formula (S-vs-R separation + log s) | 83.6 | 1.440 | +0.019 | 0.207 (0.180–0.235) | 0.35 |
| the five quantities + predicted roles | 91.4 | 1.170 | +0.022 | 0.149 (0.130–0.171) | 0.25 |
| compact 44 named + weight statistics + roles | 163.1 | 0.863 | −0.007 | 0.145 (0.126–0.164) | 0.20 |
| 7.18: compact, both rows, + pulses, 128 | 176.0 | 0.960 | −0.004 | 0.194 (0.170–0.219) | 0.26 |
| 7.18: MF4 K = 2 + pulses + groups, 128 | 595.2 | 0.608 | +0.008 | 0.263 (0.233–0.294) | 0.29 |
| 7.18: the same, full kernel | 650.6 | 0.587 | +0.013 | 0.268 (0.239–0.298) | 0.29 |
| 7.18: MF4 K = 4 + pulses + groups, full | 1,216.5 | 0.495 | +0.028 | 0.356 (0.314–0.400) | 0.37 |

- **Every estimator holds on networks nobody had looked at.** The errors match the 512 networks of 7.18–7.20 within about 0.05 points, and the mean errors are within ±0.05 points.

**Twice the training networks (`z75`).** The seed-7 networks then joined the training set: 2,048 networks, with their roles computed offline as before. Both training sets are scored on the 512 seed-6 networks. Learned rows here come from the 8-box pool, with pulses, weight statistics and predicted roles.

| training networks | 16 rows (24 FE) | 32 (37 FE) | 64 (61 FE) | 128 (104–108 FE) | 256 (191 FE) | compact ridge + roles (163 FE) |
|---|---|---|---|---|---|---|
| 1,024 | 1.063 | 0.929 | 0.885 | 0.845 | 0.938 | 0.831 |
| 2,048 | 0.956 | 0.853 | 0.771 | 0.727 | **0.693** | 0.828 |

- **The learned rows keep improving with more training networks:** −0.08 to −0.12 points at 16–128 rows. 256 rows no longer overfit, and reach 0.69 points at 191 FE (0.11× sampling's MSE).
- **The compact ridge does not improve.** It has few quantities, and is limited by them rather than by data.
- **With the mixed pool and up to 512 rows** (`z78`, 2,048 training networks, the 8-box pool plus 7.20's other families: 24,140 candidates), the errors on the 512 are:
  - 0.85 points at 38 FE (32 rows);
  - 0.733 at 63 (64 rows; 0.039× sampling's MSE);
  - 0.706 at 109 (128 rows; 0.063×);
  - 0.695 at 192 (256 rows; 0.107×);
  - 0.725 and 0.734 at 270 and 348 FE (384 and 512 rows).

  The selection overfits beyond 256 rows at this training size. The other families make up 8–10% of the rows chosen at 64–256, about twice their share of the pool.
- **Four times the training networks** (`z79`, the same estimator and pool). The 2,048 seed-8 networks (90.38% ± 3.72) were labelled on another independent bank (seed 79) and joined the training set. RMS on the 512 (points):

| training networks | 32 rows (37–38 FE) | 64 (61–63) | 128 (107–109) | 256 (191–193) | 384 (269–272) | 512 (344–350) |
|---|---|---|---|---|---|---|
| 1,024 | 0.927 | 0.883 | 0.820 | 0.842 | 0.885 | 0.704 |
| 2,048 | 0.850 | 0.733 | 0.706 | 0.695 | 0.725 | 0.734 |
| 4,096 | 0.883 | 0.744 | 0.668 | 0.605 | **0.562** | 0.589 |

  - **More data pays off mainly for the larger models.** At 4,096 training networks, 384 rows reach 0.56 points at 272 FE (0.099× sampling's MSE). That beats 7.18's MF4 K = 2 kernel (0.59 at 651 FE) at under half the cost. Sampling needs about 2,800 FE to match 0.56 points.
  - **The small models (32–64 rows) no longer improve,** and the 1,024 column is noisy at large sizes. The OMP selection itself varies.
  - **A second blind test is set up.** The 64–384-row models trained on 4,096 networks were frozen in `pop_unif/frozen_721.pkl` (`z80 --fit`, 13:57 on 2026-10-20, SHA-256 0e2828c2…). That was before the 2,048 seed-9 test networks finished training. They are scored by `z80 --score`.

## 8. Open problems and caveats

- **The clock analysis is linear.** It uses the eigenmodes of the usually-active neurons, but those neurons do switch off at certain ages. The causal interventions and patching are exact operations on the final state, but they act in that approximate basis.
- **Seed 2's delay trajectory is now derived from the weights** (7.9): it is the impulse response Whh^a·e₁₀, and the readout decodes it. The new-max handoff around the pointer is now attributed to the comparators and the max markers (7.9). What remains unexplained is how training made Wo approximately invert the impulse response. Section 6 shows when this happens (seed 2 at 2²⁴–2²⁵), not why.
- **The memory-based seeds' storage is only partly explained.** The linear-memory fit has R² of about 0.85. The max's opposite-sign write is identified, but the full nonlinear circuit is not.
- **A sampling-free accuracy estimate now exists, but it is not cheaper than sampling.** Section 7.13 gives a boundary-aware quadrature over the top-3 order statistics and their positions, with the lower inputs at their conditional mean. It is accurate to about 0.1 points for three models (+0.06, +0.08, −0.13) using no random inputs. It is justified by the finding that leakage from the lower inputs changes 3–7% of outputs but cancels to under 0.1 points net. After optimization (scale-adapted coordinates, a converged inner scan, prefix caching), single fixed grids sometimes hit sampling-level accuracy (mystery 0.36× at 12.8M evaluations). But with an honest error bar from randomized replicates, its standard error is **6–25× sampling's** at equal cost (with only 4 replicates, a loose range). The memory model (seed 0) also has a +0.08-point mean-field floor, and making a 4th order statistic exact worsens the bias for both model types. ARC's target is not met (7.13).
- **On the 1,024-network population (7.16), the lower inputs are now modelled, and the top-k integral is the open problem.** MF3's leakage is +0.27 ± 0.52 points there, which caps every MF3-based estimate at about 2,000 samples' worth. Keeping the 4th largest exact and adding each of the 6 smallest inputs' one-at-a-time effect (first-order cut-HDMR around MF4) cuts that to about −0.05 ± 0.04 points, worth about 150k samples. That model was scored on sampled top-4 configurations. Integrating it deterministically over the 5,040 arrangements × 4 values at a cost below p(1−p)/ε² is unsolved. The measurements in 7.16 (`z20`–`z26`) put it out of reach both by enumeration and by a linear-region closed form. The MF4 anchor integral alone is 10–160× sampling's MSE up to 600k forwards. The lower-input correction is mostly not local to the decision boundary (68% of it on lines where the anchor never changes its answer), so it would cost about 4.8M forwards per outer node. The exact region walk (`z9`) already costs 5–45M forwards for MF3 alone. The −0.05 residual is also unexplained. ChatGPT's round 4 got no closer than 16× sampling's MSE, and showed that the clamped model reverses the anchor at its boundary. `z13` confirms that the model's small integrated bias is a cancellation. A first-order model in the margin rather than the indicator is locally 12× better, but biased by −0.18 points: inputs that push the same clock or memory neuron across its threshold are counted once each (`z15`–`z19`). Averaging each input's effect over its neighbours (pair terms at most 2 positions apart, projected onto one-input terms) removes it: +0.03 points, with RMS 0.048 over all 1,024 networks. That matches the indicator model's total, but with 40× smaller local error (0.003·p(1−p)), at about 240–410 forwards per configuration. Round 5 (7.16) left the deterministic two-level estimator at 170–220× sampling's MSE. A correction learned from the population (`z47`) comes within 1.1× of sampling's MSE at about 1.5–1.8k forward-equivalents with all costs counted (0.80–0.85× counting forward rows only; the zoo is worse), but falls further behind at higher budgets: 16× at 92k in rows alone. The open problem is now the network-specific breaking rate of the lower inputs, the part of the correction the learned model cannot see from anchor rows. Sections 7.17–7.18 get under the line at low budgets with a network-level readout fitted on the population. On 512 networks not used anywhere in its design it is 0.14× sampling's MSE at 132 forward-equivalents and 0.26× at 651. Its error falls slowly with more probes, so sampling overtakes it above about 4k forward-equivalents. Most of it is readable (7.19): two named quantities, how sharply the network separates the 2nd largest from a close 3rd and its logit scale, already give 0.20× at 84 forward-equivalents. Probe rows learned on the population (7.20–7.21) reach 0.03× at 23 forward-equivalents on networks frozen out of the fit, and 0.10× at 272 when trained on 4,096 networks.
- **The readout side of "why training lands here" is resolved at convergence** (7.12): the final readout is the convex optimum for the recurrence's features, though early readout learning affects which recurrence forms (7.14). For the recurrence, the switch is located in one neuron's recurrent threshold (n10 snapping onto ≈0.5–0.7·S). Forked training (7.14) shows the fork is stochastic: 37% of zoo-seed runs (27/73) and 50% of fresh initializations (24/48) switch, the outcome depends on data order and initial weights, and it is settled by about 2²⁴ sequences. A mid-training learning-rate "kick" before then induced switching in all 12 kicked forks (two seeds). The barrier is shallow along a curved path (about 0.07 nats) (7.14). No simple scalar statistic of the initial weights predicts it (all p ≥ 0.08). Tracking recruitment shows the pointer is chosen among the initially positive-input neurons by early growth, by about 2²² sequences. Section 7.14 shows that the competition is over threshold targets. M (the new-max detector) is taken first. A run selects only if another neuron settles on the 2nd-max threshold by about 2²³ sequences; otherwise a neuron goes always-on. The loss-landscape scan (7.14) gives a partial reason. At 2²² the loss already prefers S-ward thresholds, but S is less linearly available in the state, so they are harder to reach. By 2²³ co-adaptation flips the preference to M. Always-on is never preferred directly. Per-seed rates (14–15 runs each) range from 2/15 to 14/14. Initialization transplants locate seed 3's predisposition in the joint recurrent-plus-readout configuration, spread across neurons, which consistently makes n13 a one-step-late pointer. Simple weight statistics (p ≥ 0.08) and the early S-decodability gap (p = 0.87) do not predict switching. Transplanting n13 alone (0/4) or everything except n13 (1/4) fails, so the property is an interaction between n13 and its seed 3 surroundings, not a single-neuron trait. The initial *readout* carries it: with forward dynamics identical, seed 3's readout gives 6/6 against 0/4. Apparently not through the step-0 gradient lead: a minimal readout transplant that gives n13 an even larger lead over the next neuron switches 0/6, though it is weaker by size over the median. So the predisposition probably acts through the training trajectory. A *frozen* seed 3 readout is enough for a selective pointer to form in 3/3 runs (by a lenient criterion, in poorly performing networks), while training only n13's own row fails 0/3, so the decisive dynamics are the whole recurrence learning toward a fixed readout geometry. **No predictor at initialization was found.** Six single statistics and a 28-feature cross-validated classifier (46–50% accuracy against 50% chance) all fail on 48 fresh runs. The best single candidate, the winner's gradient margin, looked borderline at first (p = 0.056) but failed to replicate (p = 0.27 held-out, 0.57 overall). The one-step-late pointer itself is now explained: it is a gate opened by silent comparators (7.9). It would also need a quantitative theory of how that initial competition plays out over the roughly 12k steps before the pointer forms. Narrowing it further would need many more transplant conditions, each needing several training runs, plus a theory of why that path is attractive under gradient descent.
- **Explanation-driven and task-only estimators were re-examined on 22 independent seeds (7.15).** The best sampling-based combination is about 2.5–4× sampling's efficiency. Mechanism-defined adaptive sampling saves 1.4× in practice and at most about 3–5× even with perfect stratum knowledge, because near-tie outcomes are "blurred" rather than shifted. A weight-derived Gaussian blur converges to −2.6 points because ReLU clipping makes the lower inputs' effect nonlinear. A competitive mechanistic estimate would need a closed-form state model that avoids both the 720-arrangement enumeration and network evaluation.
- **The 16 extra seeds are under-trained** (2²⁷ sequences, accuracy 0.73–0.88), so the held-out population is more heterogeneous than the zoo's 2³⁰ models.
- **The mystery model has no training checkpoints.** That it is a "tidier" online selector is inferred from the zoo seeds, especially seed 2's transient ~40° 2nd-max clock.

## 9. Ledger of ideas: tried, outcome, and why each line is closed

Each line of investigation is listed with what was tried, the result, and why I consider it closed (✔ = resolved; ✘ = tried and found unproductive; ◐ = partly resolved, and the remaining step would be a new research programme rather than a variation of what was tried).

**Mechanism of the mystery model**
- ✔ **Age code, clocks, comparators, Fourier readout** (sections 2.1–2.5), confirmed causally (2.4, 7.1).
- ✔ **Write pointer n5 and how it is gated** (7.9): forcing an n5 pulse moves the answer to the marked position in 88–96% of runs with no later updates. Of 64 candidate pulses, only n5 decodes every age from the weights alone (`y1.py`).
- ✔ **Max-to-2nd handoff** (7.9): carried by the comparators and the max markers.
- ✔ **Failure modes** (section 3).
- ✘ **Short program that reproduces each output, mistakes included** (section 4): the mistakes need nearly all 432 weights.

**Other models**
- ✔ **Seeds 0–4** (sections 5, 7.2–7.3, 7.6): memory versus selection; seed 2's write pointer.
- ✔ **4-, 6-, 8- and 32-neuron models** (7.8, 7.10): time cells and their release-from-inhibition chain.

**Training dynamics**
- ✔ **When the strategy is fixed** (7.14): a fork settled by 2²³–2²⁴ sequences.
- ✔ **Per-seed switch rates** (14–15 runs per seed) and 48 fresh initializations.
- ✔ **Loss-landscape barrier**: 0.07 nats on an optimized curve. The 1.4-nat straight-line barrier is not diagnostic, since two points on one training trajectory show a larger one (`y4.py`).
- ✔ **Learning-rate kick**: induces switching, but which pointer type forms is seed-specific.
- ✔ **Readout is the convex optimum for the recurrence's features** (7.12).
- ✔ **n10's threshold snap at seed 2's switch** (7.12).
- ✘ **Predicting the outcome at initialization**: 6 single statistics plus a 28-feature cross-validated classifier, all at chance; the most promising statistic failed to replicate.
- ✘ **Initial-gradient account of seed 3's predisposition**: fails for the lead-over-runner-up statistic (0/6 switch despite a larger lead); the size-over-median statistic is untested (`y7.py`).
- ◐ **Where seed 3's predisposition lives**: located by transplants in the joint initial recurrent and readout weights, as an interaction centred on n13. **Leave-one-neuron-out search (`u1.py`):** starting from the full seed 3 transplant (2/2 switch with these data orders), I reverted each other neuron's row, column and readout column to seed 4's values, with 2 data orders each:
  - reverting **n5, n8, n12 or n14** stops the switch (**0/2** each), so these are candidate necessary partners of n13;
  - reverting **n3 or n9** has no effect (2/2);
  - reverting any of the other 9 neurons weakens it (1/2).

  So the predisposition is a distributed configuration of n13 plus several partner neurons, with most others contributing marginally. With 2 runs per condition the labels are fragile: if every reversion left the switch rate at 0.5–0.7, about 1.4–3.8 of the 15 conditions would show 0/2 by chance. Confirming specific partners, or an edge-level search, would need far more training runs.
- ✔ **Reduced models**: a frozen readout lets selective neurons form (by a criterion that untrained weights also pass, `y5.py`), but networks stay at 44–58% accuracy; freezing either half of the recurrent rows blocks the pointer.
- ✔ **Practical result**: delaying readout learning for 8,192 steps raises the selector rate from 13/24 to 20/24 (untested; Fisher p = 0.06) and accuracy at 2²⁵ by 2.6 points (sign test p = 0.023, Wilcoxon 0.087; the 16 pairs added after the pilot alone give p = 0.077). It persists at 2²⁷ (0.917 against 0.897) and at 2³⁰ (0.968–0.976 against 0.952–0.954, n = 2).
- ◐ **Quantitative theory of the early competition**: everything above characterizes it empirically. A predictive theory would need a mathematical model of coupled readout and recurrent learning across about 12k steps; no further experiment of the kinds already used would supply it.

**Mechanistic accuracy estimation (ARC's challenge)**
- ✘ Rule-based programs, decision trees, single comparator-resolution models, linear and additive leakage models, and second-order cut-HDMR all failed (7.7, 7.13). The additive model fails around MF3; around MF4 it works (7.16).
- ✔ **Leakage nets out**: under 0.1 points for the selector models (sampled check), so the earlier "leakage" was quadrature error.
- ◐ **Boundary-aware quadrature in scale-adapted coordinates, with prefix caching**: single fixed grids reach 0.002–0.004 points for the selector models at the largest sizes, but convergence is erratic, and randomized replicates put its standard error at 6–25× sampling's.
- ✘ **Beating sampling with an honest error bar**: randomized replicates give standard errors 6–25× sampling's (4 replicates, so a loose range). A control variate is worse than plain sampling. Stratified sampling gives a 1.2–2.8× gain, but it still samples.
- ✘ **Removing the memory model's bias** by treating the 4th input exactly: worse (+0.27 points), and also worse for the mystery model (+0.16, `y6.py`). So the 3-input mean field's small bias for the selectors is partly a cancellation, not a converging approximation. With the first-order correction, MF4 is the right base (7.16).
- ✔ **Task-only designs (7.15)**: ordered Sobol gives 1.4–3.0×; exact last-input integration gives 4–10× per prefix and about 2.5–4× net at a matched cost of 2.2–2.5 forwards. Both still sample.
- ✘ **Integrating a second input exactly (7.15)**: cost rises faster than variance falls (net 0.3–1.0×).
- ✘ **Most-informative-first adaptive sampling over failure-mode strata (7.15)**: honest, 1.4× on 22 held-out seeds, Neyman ceiling 1.2–5×.
- ✘ **Explanation-predicted boundary (7.15)**: the threshold-shift tracker gets zero shift (no information); errors are symmetric blur; the ages explain about 40% of near-tie outcomes.
- ✘ **Weight-derived Gaussian blur, no sampling (7.15)**: converges to −2.6 points because of ReLU clipping and an unrepresentative mean-field point.
- ✘ **Closed-form clock-state estimator**: both forms tried. Compact rule-based descriptions of the state (short programs, comparator-resolution models) cannot reproduce the mistakes, which need nearly all 432 weights (section 4, 7.7). The exact form, one linear map per activation pattern, needs hundreds of patterns per arrangement (`z25`, `z26`, 7.16).
- ✘ **Population-informed sampling (7.16, `z1`, ChatGPT round 1)**: about 2×, unbiased, but statistics and a constant factor.
- ✘ **Role-mix predictors of accuracy (7.16, `z2`, `z3`, ChatGPT round 2)**: 1.0 point CV RMSE at best, worth about 700 samples, from features costing about 360 forward-equivalents.
- ✘ **Moment closures and the delta method for the lower inputs (7.16, `z8`)**: independent residuals destroy the network's cancellation; gate flips break the delta method.
- ✘ **Exact integration along the 2nd-max value (7.16, `v1`)**: 2.1–4.6× per sample, 1.2–3.9× net; a constant factor, and it samples.
- ✔ **Exact mean-field integral by linear-region walking (7.16, `z9`)**: deterministic and converged, but 5–45M forwards, and the MF3 bias caps its worth.
- ✔ **Lower-input leakage model (7.16, `z10`, `z11`, `lines`)**: MF4 plus first-order cut-HDMR on the 6 smallest inputs, about −0.05 ± 0.04 points per network.
- ✘ **The top-4 integral at sampling's cost by enumeration (7.16, `z20`–`z24`, ChatGPT round 4)**: the MF4 anchor integral reaches 2–3× sampling's MSE only at 1–2M forwards (tensor Gauss–Jacobi in CDF coordinates); homogeneous coordinates converge erratically; fixed subsets of arrangements are 4–10× sampling's RMS. The lower-input correction is non-local (68% off the boundary), and costs about 4.8M forwards per outer node; its only decomposable form is +0.14 points biased.
- ✘ **A closed form for the recurrent state that removes the enumeration (7.16, `z24`–`z26`)**: position effects do not factor (18% additive, 48% with all pairwise position interactions). A linear-map-per-activation-pattern closed form covers 17–76% of the mass with 1–256 patterns per arrangement and is 0.3–2.7 points off (7–20× sampling's RMS at its build cost), because each arrangement's value space holds about 560 linear regions. Even an oracle choosing the best patterns stays 3–4× sampling's RMS within 10⁵ forwards and reaches parity only near 1.5M.
- ✘ **Other outer rules for the anchor integral (7.16, `z21`, `z27`, `z28`, `z31`)**: tensor Gauss–Jacobi plateaus over orders 5–9 and reaches 1.5× sampling's MSE only at order 10 (9.3M forwards), for the anchor alone. Sparse grids do not converge, and an adaptive box rule is worse than the tensor rule. Homogeneous coordinates, with or without their geometric kinks removed, are no better.
- ✘ **Linear-region closed form with oracle-chosen patterns (7.16, `z30`, `z33`)**: choosing patterns greedily for agreement brings the anchor within 1.4–1.6× sampling's RMS in ARC's range, but only with a selection oracle costing about 80M forwards per network, and before the lower-input correction. Discovering the patterns deterministically along S-lines is 13–24× sampling's RMS.
- ✘ **Fewer exact inputs (top 2 or top 3) (7.16, `z37`)**: the mean-field anchor's integral then beats sampling (MF3: 1.5–2.6× better MSE at 2×10⁴–10⁵, at round 4's unreproduced tree cost; at the repo tracer's measured cost it is about 1.2–2× worse, `z41`), but the anchor is 5.7 (MF2) or 0.68 (MF3) points off. The correction is a high-dimensional per-configuration integral that costs at least about 41k forwards per outer node for MF3, and sampling it would be a control variate.
- ✘ **Round 5's two-level estimator (7.16, `z41`, `z42`)**: the exact-S MF3 anchor plus a coarse MF4 correction is 170–220× sampling's MSE at 2.5M forwards. Margin gating or extreme-node screening of the correction brings it to 79–104× at about 1M. The correction does not amortize across arrangements (`z43`), and gate histories proliferate (`z44`).
- ✔ **Calibration against the population (7.16, `z45`)**: cross-validated linear fits on deterministic per-network quantities and free weight statistics cut the MSE 2–4× at every budget. A free weights-only fit (RMS 1.28) beats sampling below about 420 forwards. Above that, calibration alone does not reach parity (7× at 11.5k, 46× at 1M).
- ✔ **A learned cell correction (7.16, `z46`, `z47`)**: an MLP trained on the population maps 2 deterministic anchor rows per cell, plus free weight statistics, to each cell's accuracy. Counting forward rows only, it is 0.80–0.85× sampling's MSE on the 1,024 networks at 1,440 rows (cross-fitted, two seeds), 1.3× at 2.9k, 1.7× at 5.8k and 16× at 92k. With its own inference counted, the best is 1.10–1.13× at 1.5–1.8k forward-equivalents, with a 16–32-unit model. The zoo transfers worse (1.8–3.2×). The remaining error is each network's own rate of lower inputs breaking right answers, which the anchor rows cannot see.
- ✔ **Network-level readout G(B, K) (7.17, `z49`, `z52`, ChatGPT round 6)**: cells, logit scale and raw weight statistics with an RBF kernel ridge: 0.30–0.33× sampling's MSE at 619 FE, 0.58× at 2.2k, 0.80–0.88× at 4.2k, then behind (31× at 515k). The mystery model is 1.5–2.3 points high, because it was trained with a different recipe.
- ✔ **A population with the mystery model's recipe (7.17, `batch_train_unif`, `z54`, `z55`, `z58`)**: trained on it, G(1,K) beats sampling's MSE cross-fitted over training seeds (0.57–0.79×), on 64 further seeds (0.42–0.71×) and on 0.3%-noise copies at K = 1 (0.72×). The mystery model is −0.55 to −0.07 points off. Readouts do not transfer across recipes, and 1%-noise copies (84% accuracy) fail at +5 to +7 points.
- ✘ **Gauge-fixed weight statistics, subcircuit templates and covariance-propagation features (7.17, `z56`)**: none improves the within-population fit; ARC's standardization helps cross-recipe transfer but costs 0.04 points within.
- ✘ **Hermite / low-degree polynomial propagation (7.17, `z57`)**: even exact degree-4 projections give 12–40% accuracy against 86–96%.
- ✔ **Cheaper probes and single-input pulses (7.18, `z59`–`z61`, ChatGPT round 7)**: MF4 rows at two S values instead of MF4 and MF3 rows at one, two signed single-input pulses (6 FE), arrangement-group averages and a 180-arrangement design. Fixed in advance and tested on 512 new networks with an independent label bank, they reach 0.14× sampling's MSE at 132 FE, 0.15× at 207, 0.26–0.27× at 651 and 0.41–0.43× at 1.2k, against 0.70× for G(1,1) at 619 FE. The crossover stays near 4k FE.
- ✘ **Arrangement-group features on AlgZoo's recipe, more S slices, margin gradients, hidden-unit impulses (7.18)**: the groups hurt on AlgZoo's recipe (0.53 → 0.68 points, from one outlier network); K = 16 is 1.4–1.7× sampling's MSE; gradients help only at K = 4, for +151 FE; impulses into single hidden units are worse.
- ✔ **A readable estimator (7.19, `z62`, `z63`)**: two named quantities from 180 probe rows, the S-vs-R separation and the logit scale, give 1.45 points on 512 new networks at 84 FE. That is 0.20× sampling's MSE, and 0.36× that of sampling shrunk toward the population mean. A ridge on 44 named quantities is within 0.07 points of the kernel. The S-vs-R separation correlates +0.92 with accuracy and tracks the comparator count. The logit scale is a training fingerprint: without it, three invariant quantities only match sampling. The remaining error is largest for networks with max detectors (underestimated) and with late pointers (overestimated).
- ✘ **A double-pulse handoff probe (7.19, `z64`, `z65`)**: 180 rows of handoffs and new 2nd maxima on zero backgrounds are answered 91–99% right. They add 0.02–0.05 points for 68 FE and leave the residual's links to max detectors and late pointers unchanged. The residual is spread over input space roughly in proportion to region size, with near ties S ≈ R over-represented. Max detectors lower the loss in every major region, so what is missing is a network-wide breaking rate that needs real lower-input variation to show.
- ✘ **Probes that perturb the lower inputs (7.20, `z66`)**: lower inputs moved together, alternating, or replaced by distractor trains. They correlate +0.80 to +0.88 with accuracy, but are redundant with S-vs-R separation and the logit scale, and worse per FE.
- ✔ **Neuron roles from the weights (7.20, `z67`)**: a logistic classifier on 16 weight descriptors per unit finds max detectors and comparators (unit AUC 1.000 and 0.998, network counts r = +0.89) for about 2 FE. As quantities the counts lower the minimal estimator's error from 1.20 to 1.13 points and remove its max-detector residual.
- ✔ **Learned probe rows (7.20, `z68`–`z71`)**: rows chosen by greedy pursuit from a pool of MF4 rows and other probes, with pulses, weight statistics and predicted roles. On the new networks they give 1.15 points at 17 FE, 0.96 at 22, 0.91 at 57 and 0.86 at 99. That is 0.02–0.09× sampling's MSE, and 0.09–0.15× that of sampling shrunk toward the population mean. The linear readout saturates at 64–128 rows, and the learning curve is still falling slowly at 1,024 training networks.
- ✔ **A blind test and doubled training (7.21, `z72`, `z75`)**: the estimators were frozen before 1,024 test networks finished training, and those networks were labelled on an independent bank. Every estimator holds on them: 0.028× sampling's MSE at 23 FE, 0.054× at 55 FE and 0.081× at 96 FE for the learned rows; 0.27× at 651 FE and 0.36× at 1,216 FE for 7.18's kernels. Trained on 2,048 networks, learned rows reach 0.77 points at 61 FE and 0.69 at 191 FE. On 4,096 they reach 0.56 points at 272 FE (0.10×).
- ✘ **Clock-state simplification (7.16, `z29`)**: making the clock and memory channels linear biases the accuracy by 2.3 points and leaves hundreds of regions.
- ✘ **Deterministic MF3/MF4 integrals with shared-prefix arrangement trees, exact S lines and interval/affine certificates (7.16, ChatGPT round 4)**: 16× sampling's MSE at 36k forwards; MF4 is over budget; certificates save about 1%.
- ✔ **Lower-input model additive in the margin (7.16, `z13`–`z19`, `z32`, `z34`, `z36`)**: the first-order version is −0.18 points low because shared gate flips (in clock and memory neurons) and rival take-overs are counted once per input. Projecting the pair terms of inputs at most 2 positions apart onto one-input terms fixes it: +0.03, RMS 0.048 on all 1,024 networks. That is comparable in total to the indicator model, and 40× better locally, at 2.5–4× its cost. Adding projected adjacent-triple terms does not lower it further (`z32`: 0.081–0.085 against 0.075–0.076 on the development networks). The ideal additive model is itself 0.074 off (`z34`), and joint product rules are worse at affordable cost (`z36`). So about 0.05–0.07 points is where this family of lower-input models stops.

**Conclusion.** Every line above is either resolved or has been tried and found unproductive. The partly-open items each need a new theoretical framework rather than another experiment of the kinds already run. For ARC's challenge specifically, the framework proposed after 7.15 was a closed-form model of the selector's state: clock eigenmodes plus age-dependent comparator thresholds, including ReLU clipping, to replace both the arrangement enumeration and network evaluation. Section 7.16 tests it (`z29`: making the clock and memory channels linear biases the accuracy by 2.3 points and leaves hundreds of regions), along with exact linear-region closed forms (`z25`, `z26`, `z30`). Section 7.16 narrows this: the lower inputs are handled to about 0.05 points, by MF4 plus either the clamped first-order indicator model (by cancellation) or a margin model with nearby pairs (faithful configuration by configuration). What remains is the integral over the top-4 configurations. Enumeration-based integration of it is 1–2 orders of magnitude off the line in ARC's range, and so is a closed form from linear regions, because each arrangement's value space holds hundreds of them, even with an oracle choosing them (7.16, `z20`–`z28`). For these networks every deterministic integration route tried here falls short of sampling's error versus compute. The two parts of it that do work are a lower-input model good to about 0.05 points, and an anchor integral that comes within 2.4× of sampling's MSE at about 10⁶ forwards. The exception is the population-fitted readout of 7.17–7.18. Below about 4k forward-equivalents it beats sampling's MSE, but its error levels off near 0.2–0.5 points, so sampling overtakes it at larger budgets. It is also a learned predictor with mechanistic features, not a derivation from the circuit.

## 10. Corrections after an audit

Every numbered claim was checked against its script, and most were re-run. Most numbers reproduced. These changed:

- **2.1:** the 98.7% state surrogate is circular, since it is fitted per (age of max, age of 2nd max) cell and so given the answer. It agrees with the model on only 95.5% of inputs.
- **2.2:** the n4/n7 thresholds drift toward S at older max ages. n5 is the write pointer, not only a max marker.
- **2.4:**
  - Mode removal now reports mean-ablation next to zero-ablation; the 0.97 value mode's effect was mostly its mean (`y2.py`).
  - The young/old asymmetry is not caused by amplification.
- **2.6:** the length test scores only sequences whose 2nd max is in the last 10 positions, and memory-based seed 0 generalizes as well.
- **3:** S < 0 is 1.1% of inputs, not 0.9%. The "no errors" region actually has a 0.04% error rate (`y3.py`).
- **6:** jitter and accuracy ranges corrected, and the memory score is not a phase marker. `d2.py` does not cover seed 1.
- **7.1–7.3:** "exactly the predicted set" and "in every model" narrowed to what was tested. The write norm is unsigned.
- **7.9:** on final weights, the weights-only pulse test does separate the models. Only mystery n5 and seed 2 n10 pass, out of 64 candidates (`y1.py`).
- **7.11:** the 3% weight-noise row is now produced by `i8.py`: 48–60% accuracy, 0.75× gain.
- **7.13:**
  - Evaluation counts are about 25% lower than stated.
  - The caching ratio is about 7×, not 4×.
  - Reference uncertainties now have correct units.
  - "Beats sampling" is marked superseded, and the replicate error bars are qualified (K = 4, and they cannot see shared bias). "6–20×" corrected to 6–25×.
  - The 4-input mean field is worse for the selector too, so the memory-channel explanation was dropped (`y6.py`).
  - The control-variate break-even is 0.75 with a unit coefficient.
  - The orphaned second-order cut-HDMR table (`n5.py`) moved into 7.13.
- **7.14:**
  - The learning rate is AlgZoo's peak of 0.01, not 0.01/√16.
  - Stale counts (8/8, 1/9, 42%, 54%, 31–46%) updated; `n3.py` uses 28 features.
  - The switching criterion is lenient: untrained weights pass it (`y5.py`). This weakens the frozen-readout counts.
  - The q4 p-value is qualified.
  - "Falsified" narrowed to the statistic tested (`y7.py`).
  - The kick evidence rests on seed 0.
  - The straight-line barrier is not diagnostic (`y4.py`).
  - The `u1` partner neurons are fragile at 2 runs per condition.
- **Code:**
  - `common.py` and the exporters no longer use absolute or `work/` paths.
  - `j1.py` no longer exits before its second table.
  - `f7.py` and `j2.py` no longer run their analyses when imported.
  - `q4.py` filters logs by K.
  - `i8.py` adds the 3% level.
