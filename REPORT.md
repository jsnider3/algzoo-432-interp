# Interpreting AlgZoo's 432-parameter 2nd-argmax RNN

Model: `alg_zoo.example_2nd_argmax()`. It is a one-layer ReLU RNN with 16 hidden neurons, no biases, and sequence length 10. Inputs are N(0,1), and it outputs the position of the second-largest value. Parameters: W_ih is 16, W_hh is 256 and W_out is 160, for 432 in total. Accuracy is 95.3%.

Prior work in ARC's blog post covered five neurons: n2 is the running max excluding the latest input, n4 is the increment of the running max, and n6, n7 and n1 are "leave-one-out max minus latest" features. This report covers the rest of the mechanism, its failure modes, and how it compares with the other zoo seeds and their training checkpoints.

All scripts are in `work/`. `common.py` is a NumPy copy of the RNN, exact to about 1e-12 against torch.

---

## 1. The algorithm, in one paragraph

The network runs an **online, time-invariant 2nd-max tracker**. It never stores *where* the max and 2nd max are. It stores *how long ago* they appeared, as the **phases of two damped rotating clocks that turn at the same speed (about 43° per step)**. Each clock's amplitude scales with the value it tracks.

At each step, ReLU comparator neurons test the new input against thresholds that blend M (the max) and S (the 2nd max). Depending on the result:
- if the input is a new max, it restarts the max clock and hands the old max clock's phase to the 2nd-max clock;
- if it is a new 2nd max, it restarts the 2nd-max clock;
- otherwise both clocks just keep rotating.

Because both clocks turn at the same speed, the handoff from max to 2nd max is a fixed linear map. The readout decodes the 2nd-max clock's phase, along with harmonics at about 2×, 3× and 4× the base speed, into a position.

**A more fundamental view (section 7.9).** A sparse **write-pointer neuron (n5)** fires one step after each new max or new 2nd max. Forcing it moves the answer to the marked position in 88–96% of runs. The "clocks" are the eigenmode breakdown of the recurrent weights' impulse response to that pulse. From the weights alone, Wo·Whh^a·e₅ decodes every age correctly.

The other designs follow the same principle (pointer, impulse response, readout that decodes it), with a different shape of response: a non-rotating path in seed 2 (pointer n10) and time cells in the 32-neuron model. The readout is just the softmax-regression optimum for these features (7.12).

The comparators carry the max-to-2nd handoff. Whether training reaches this design at all, or a "remember every input" design, is an early, stochastic fork across a loss barrier, settled by about 2²⁴ sequences (7.14).

## 2. Evidence

### 2.1 The state stores ages, not positions (`a6.py`, `a7.py`)

Apply the final readout to the hidden state after step t and shift the answer by 9−t slots. This gives the 2nd argmax of the prefix with accuracy 0.66, 0.78, 0.86, 0.91, 0.94 and 0.95 for t = 3 to 8. Readout slot i therefore means "the 2nd max appeared 9−i steps ago". Linear probes on the final state recover the age of the max with 98% accuracy, the age of the 2nd max with 90%, and the age of the 3rd max with only 24%.

Fitting the final state from just (age of max, age of 2nd max, M, S, x_9) and reading out from that fit gives **98.7%** accuracy. So these five quantities carry nearly all the information the model uses.

### 2.2 Comparators with age-dependent thresholds (`a10.py`–`a12.py`)

Neurons n1, n4, n6, n7 and n8 have large input weights (about ±10–13). Each fires when x_t is on one side of a threshold. The threshold comes from the previous state. Within each group of inputs with the same max age, it is almost exactly a·M + b·S with a + b ≈ 1 (fit quality R² 0.94–0.999):

| neuron | threshold ≈ S when … | otherwise ≈ M |
|---|---|---|
| n6 | the max arrived on the previous step | yes |
| n1 | the max is 1 step old (partly at 2 steps) | yes |
| n8 | the max is 0–3 steps old | ≈M from 5 steps old |
| n7, n4 | never: they test against M (they detect a new max) | — |

Neurons n4, n5 and n9 are sparse markers that the max is 0 or 1 steps old.

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

### 2.4 Causal tests (`e1.py`)

**Ageing the clock moves the answer.** In the final state, multiply the 2nd-max clock and its harmonics by λ^k, which is k steps of their own dynamics:

| intervention | answer moves by | fraction of inputs |
|---|---|---|
| 2 steps older | −2 positions | 78% |
| 1 step older | −1 position | 65% |
| 1 step younger | +1 position | 48% |
| 2 steps younger | +2 positions | 38% |

Making the clock younger works less well, because undoing the decay amplifies noise.

**Rotating the max clock alone changes only 5% of outputs.** So the max clock drives the updates, and the readout relies on the 2nd-max clock.

**Removing single modes from the readout:**

| mode removed | accuracy |
|---|---|
| 2nd-max clock | 50% |
| 0.97 value mode | 54% |
| 83° harmonic | 69% |
| 125° harmonic | 68% |
| max clock | 91% |

### 2.5 The readout is a Fourier decoder (`a23.py`)

Each readout slot i has a preferred phase on the 2nd-max clock. For slots 0–5 it matches the clock's phase at age 9−i to within about 10°: 6°/15°, 320°/316°, 273°/267°, 216°/227°, 175°/186° and 129°/139°. Readout weights grow for older ages, which makes up for the clock's decay (0.735 per step). Recent 2nd maxes (0–3 steps old) are decoded mostly from the sparse marker neurons and fast-decaying modes.

### 2.6 Length generalization (`e5.py`, `e6.py`)

On longer sequences, the unchanged readout still finds the 2nd max by its age (L is sequence length):

| L | accuracy |
|---|---|
| 12 | 92% |
| 17 | 85% |
| 30 | 89% |

Guessing would give 10%. This supports the time-invariant account. It is not perfect: when the max is also within the last 10 inputs, accuracy falls with L, reaching 81% at L=30. A likely cause is the mode with eigenvalue 1.01, which makes the max register drift.

## 3. Where and why it fails (`a13.py`, `a14.py`, `a17.py`, `e8.py`–`e10.py`)

The model is close to an exact algorithm, with comparisons whose **resolution scales with M** plus one blind spot:

| region | share of inputs | error rate there | share of all errors |
|---|---|---|---|
| S < 0 (2nd max is negative) | 0.9% | **77%** | 18% |
| M − S < 0.05·M (max and 2nd nearly tied) | — | — | 20% |
| S − R3 < 0.05·M (2nd and 3rd nearly tied) | — | — | 53% |
| everything else | — | — | 8% |
| "clean": S > 0 and both gaps > 0.1·M | 55% | **0.23%** | — |

With S/M in [0.2, 0.8) and (S − R3)/M ≥ 0.2, I found no errors in about 1.5M samples, so the error rate there is below 0.05%.

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
| round weights to 2 decimals | 90.5% |

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

1. **Up to 2²² training sequences:** no stable clock (±30–55° jitter per step), and accuracy is at most 53%.
2. **2²³–2²⁴:** a max clock forms in every seed, with jitter falling to ±10–20°. Accuracy jumps to 70–77%, and the clock speed settles.
3. **Every seed first goes through a memory-like phase.** The memory score, meaning agreement of the "linear memory + max suppression" fit on clear cases, peaks at 0.73–0.83 around 2²³–2²⁴.
4. **Seed 2 switches to online selection between 2²⁴ and 2²⁵.** Its readout's diagonal-to-off-diagonal ratio falls from 1.02 to 0.63 and later to 0.53. After the switch, its error at large gaps falls from 15% to 0.8%. Seed 3 switches earlier (ratio 1.16 → 0.92 → 0.67), and seed 1 drops slightly. Seeds 0 and 4 never switch, and their memory score keeps rising.

Interpretation: remembering everything is the easy solution that training always finds first. Online selection is better but has to be discovered. The mystery model looks like a run that found it, with an especially tidy implementation using two equal-speed clocks.

## 7. Follow-up experiments

### 7.1 Patching modes confirms the mystery model's 2nd-max clock (`f1.py`, `f2.py`)

Take pairs of inputs that differ only in where the 2nd max sits (it is swapped with a lower value).

- **Copying single neurons** from one run into the other carries over at most 8% of answers, in every model. The code is not organised neuron by neuron.
- **Copying eigenmodes** instead gives a clear answer for the mystery model. The most important single mode is the 2nd-max clock, which carries over 15% of answers. A greedy search then picks the 2nd-max clock, the 83° harmonic, the 125° harmonic and the fastest-decaying mode, in that order, with cumulative carry-over 15%, 27%, 47% and 56%. That is exactly the predicted set.
- **Seed 2, seed 3 and seed 0 have no dominant mode.**

### 7.2 How each model stores the 2nd max's age (`f4.py`)

Each rotating mode is fitted as z ≈ A[age of max]·M + B[age of 2nd max]·S, with a free complex coefficient for each age.

| model | storage |
|---|---|
| mystery | its own clock (0.73 mode): a 2nd-max hand turning 35–73° per step, gain +0.39. The 83° and 125° harmonics carry both hands. |
| seed 3 | **two hands on shared clocks.** On the 100°, 59° and 143° modes, the max and 2nd-max hands turn at the same speed. For example, on the 100° mode the max hand turns 98–109° per step and the 2nd-max hand 94–124°. Gain +0.27 to +0.29. |
| seed 2 (final) | a weak 2nd-max hand on the max clock (104°, gain +0.09). The rest lives in neurons that switch on and off. |
| seed 0 | no rotating 2nd-max hand. The information sits in non-rotating, memory-like modes. |

### 7.3 How the max gets suppressed (`f5.py`–`f7.py`)

In every model, raising the max's value **lowers its own logit**. The gradient of the max's own logit with respect to its own value is −12 (seed 0), −9 (seed 4), −20 (mystery) and −96 (seed 2). For the 2nd and 3rd values it is +12 to +41.

**Mechanism:**
- Ordinary values are written through the comparators as ReLU(threshold − x), which carries −x.
- A new max switches those comparators off and is written instead through the new-max detector, as ReLU(x − old max), which has the opposite sign.
- So the max is written into the state with the opposite sign as it arrives, and suppression is built into the writing. A new max is written with input-write strength 15.8 against 22.6 for ordinary values in seed 0, and 13.5 against 22.7 in the mystery model. Seed 2 instead has a dedicated new-max neuron (input weight +26.9).

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

**Pointer selectivity is what separates the strategies (`i13.py`, `i14.py`).** The weights-only pulse test (some neuron whose Wo·Whh^a pulse decodes every age) passes at almost every checkpoint of every seed from 2²² on, memory-based seed 0 included. That is expected, because a linear memory also stores each input by its age. So that test alone does not separate the strategies. Seed 2 settles on n10 as its pointer from 2²⁴ on; seed 0's best neuron keeps changing (n7, n3, n13, n0, n11).

The separating property is **event selectivity**: how often the pointer neuron fires after a new max, after a new 2nd max, and otherwise.

| neuron | new max | new 2nd max | otherwise |
|---|---|---|---|
| mystery n5 | 0.97 | 0.84 | **0.06** |
| seed 2 n10 | 0.98 | 0.91 | **0.08** |
| seed 3 n9 | 0.98 | 0.90 | 0.19 |
| seed 0 n0, n7 | 1.00 | 1.00 | **1.00** (always on: memory channels, not pointers) |

Pointer selectivity follows the same order as clean-case error (mystery and seed 2 best, seed 3 next, seed 0 worst). That fits the picture: the online selectors write *only* events, while the memory seeds write every input.

### 7.10 How the 32-neuron model's time cells are gated (`i5.py`)

Each time cell is released from inhibition: its strongest inhibitor drops to exactly zero (h/M = 0) one step before the cell's preferred age.

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

**Weight-noise test** (ARC's suggested protocol), on mystery-model copies at 4,000 evaluations:

| weight noise | model accuracy | MSE gain |
|---|---|---|
| 0.3% | 94–95% | 2.1–2.8× |
| 1% | 83–90% | 1.3–1.8× |
| 3% | 55–72% | 0.5–1.2× (geometric mean 0.87×) |

At 3% noise the errors stop clustering near ties. The prior helps exactly as long as the mechanism is intact. This is a modest but genuine step toward ARC's "beat sampling" target. It still uses sampling inside each stratum.

### 7.12 Why training lands on this readout, and what changed at the switch (`j1.py`–`j4.py`)

**The readout is the convex optimum for the recurrence's features.** Refit Wo from scratch by softmax regression (L-BFGS on 400k inputs), keeping the trained recurrence fixed:

| model | trained readout accuracy | refit readout accuracy | cosine to trained Wo | outputs agreeing |
|---|---|---|---|---|
| mystery | 0.9530 | 0.9560 | **0.998** | 98.5% |
| seed 2 | 0.9481 | 0.9487 | **0.997** | 99.3% |
| seed 0 | 0.9080 | 0.9120 | **0.993** | 97.9% |

So the readout's structure (7.9: decode the pointer's path, cancel the large M-scaled directions) is simply what cross-entropy produces for these features. **All the interesting learning is in the recurrence.**

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

| model | true accuracy | mean field (fine grid, 4.6M evaluations) | + Gaussian linear leakage | + truncation-aware linear leakage (coarse grid) |
|---|---|---|---|---|
| mystery | 0.9529 | **0.9559 (+0.30 points)** | 0.927–0.929 | 0.937 |
| seed 2 | 0.9483 | **0.9498 (+0.15)** | 0.937–0.940 | 0.922 |
| seed 0 | 0.9100 | **0.9115 (+0.15)** | 0.61–0.67 | 0.762 |

**Findings:**
- **Mean field is accurate to 0.15–0.3 points with zero randomness.** So with the other inputs fixed at their mean, the network's decision as a function of the top-3 values and positions is almost the whole story. The small positive bias is the leakage errors it leaves out.
- **It is not compute-efficient.** The mystery model's error is −3.5, −0.65 and −0.55 points at 69k, 207k and 1M evaluations, against sampling standard deviations of 0.08, 0.05 and 0.02 points at the same cost. The 720-arrangement enumeration is the cost driver, and the +0.3 bias floor remains.
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

First-order cut-HDMR is the first leakage model with the right sign and roughly the right size. It even catches seed 0's net *positive* leakage, where rescues outnumber breaks. It also cuts the per-cell error by about 30%. Its remaining error is 0.09–0.16 points. Adding second-order terms made things worse, probably because of the coarse 5-point pair quadrature plus clipping.

**Combining mean field with a coarse-grid leakage correction fails (`l3.py`).** I computed the cut-HDMR correction on a coarse grid (r10 × g8 × g8, 4 nodes) and added it to the fine-grid mean-field estimate:

| model | coarse-grid correction | true leakage effect | combined error | mean-field-only error |
|---|---|---|---|---|
| mystery | +0.30 points | −0.19 | +0.60 | +0.30 |
| seed 2 | +0.17 | −0.10 | +0.32 | +0.15 |
| seed 0 | −0.43 | +0.45 | −0.28 | +0.15 |

**The correction has the wrong sign every time.** The leakage effect is concentrated in near-tie regions that the coarse grid and 4-node quadrature don't resolve. A working sampling-free estimate therefore needs the cut-HDMR correction on the *fine* grid, about 85× the mean-field cost (about 390M evaluations, several CPU-hours here). I did not run that; on cost alone it cannot compete with sampling.

**Fine-grid estimate with the cut-HDMR correction (`n1.py`).** The full grid (r20 × g16 × g16 × 720 position arrangements) with 4 quadrature nodes per lower input, about 133M network evaluations per model and about 9.5 minutes each:

| model | true accuracy | mean field | + cut-HDMR leakage | correction applied | true leakage effect |
|---|---|---|---|---|---|
| mystery | 0.9529 | 0.9559 (+0.30) | 0.9561 (+0.32) | +0.02 | −0.30 |
| seed 2 | 0.9483 | 0.9498 (+0.15) | 0.9497 (+0.14) | −0.01 | −0.15 |
| seed 0 | 0.9100 | 0.9115 (+0.15) | **0.9092 (−0.08)** | −0.23 | −0.15 |

The correction captures seed 0's leakage but contributes almost nothing for the two online selectors. On the 3,000-cell validation, 12 quadrature nodes gave −0.35 and −0.26 points; that suggests 4 nodes are too few to resolve the narrow region where a lower input flips a near-tie. Re-running the mystery model with **12 nodes** (`n4.py`, about 390M evaluations, split over 3 processes) gives a correction of **−0.13 points** (estimate 0.9547, error **+0.18**, against +0.30 for mean field alone). The correction now has the right sign and recovers about 40% of the true −0.30 leakage effect, so quadrature resolution was indeed the limiting factor. The resulting **sampling-free estimates are within 0.08–0.18 points** of the truth for the two models checked at adequate resolution (seed 0 at 4 nodes, mystery at 12), and within 0.14 for seed 2 at 4 nodes, using no random inputs. At 133M evaluations, though, plain sampling would be accurate to about ±0.002 points, so this is not competitive on cost.

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

The obvious structural lever is time-invariance and prefix sharing: with the lower inputs at their mean, the 720 arrangements share prefixes, so hidden states can be cached. The trie has **2,815 distinct prefix states against 7,200 naive steps per line, a 2.56× saving**. Even with perfect caching, the best configuration would remain about 4× worse than sampling at equal cost, and coarser grids are 10–20× worse. Matching sampling would need the quadrature error to fall about 10× faster with resolution than it does. With a discontinuous integrand in three dimensions plus a 720-way enumeration, I don't see a route to that. That is consistent with ARC's description of matching sampling as "a major research breakthrough".

**Optimized: fewer scan points and scale-adapted coordinates (`r1.py`–`r3.py`).** A sweep over the three resolution knobs showed that **scan points barely matter**: 8 per line is as good as 32. The error was set almost entirely by the outer Gauss–Legendre grid over (3rd value, max-to-2nd gap).

Because the network has no biases and uses ReLU, its decisions depend on *ratios* of inputs. So I changed coordinates to M plus the normalized gaps a = (M−S)/|M| and b = (S−R)/|M| (Jacobian M²), with the all-negative region integrated separately. The decision boundaries then barely move with M. Results, with mean field for the lower inputs, boundary-aware integration in b, 8 scan points and 10 bisection steps:

| model | 0.43–0.46M evaluations | 0.70–0.77M | 1.06–1.15M | 1.57–1.68M | 2.75–2.96M |
|---|---|---|---|---|---|
| mystery: error (ratio to sampling SD) | 0.46 points (14×) | 0.42 (17×) | 0.19 (9×) | **0.005 (0.3×)** | 0.07 (5.5×) |
| seed 0 | 0.52 (12×) | 0.28 (8.5×) | **0.03 (1.0×)** | 0.14 (6×) | 0.04 (2.5×) |
| seed 2 | 0.22 (7×) | 0.20 (8×) | 0.16 (7.5×) | 0.05 (3×) | 0.16 (12×) |

So at about 1.5–3M evaluations the error is 0.005–0.16 points, typically **3–6× the sampling error at equal cost**. That is down from 11–50× before, and occasionally at parity. But it does not converge monotonically: errors still oscillate with resolution, because the integrand is also discontinuous in a (the max-versus-2nd ties). The parity points should be read as lucky alignments, not a reliable result. (The reference "true" accuracies also have about ±0.015 points of sampling uncertainty.) The remaining levers are boundary-aware integration in a as well, and prefix caching (2.56× fewer network steps, which would reduce every ratio by about 1.6×).

**With better references, convergence checks and prefix caching (`s1.py`–`s8.py`, `cache.py`, `r4.py`).** References from 20M samples: mystery 0.95313, seed 2 0.94851, seed 0 0.91009 (±0.005–0.006 points). For the mystery model, a 200M-sample reference gives **0.953119 ± 0.0015**.

- **The estimator was limited by the inner scan, not the outer dimensions.** Refining M or a alone left the estimate wandering by ±0.08 points. The inner scan converges (12×12 grid: 0.95325, 0.95279, 0.95267, 0.95264 at 8, 16, 32, 64 scan points), so 8 points had sometimes been lucky.
- **The mean-field model's own exact value** (20M samples) is 0.95323 ± 0.005: bias +0.01 ± 0.007 points, essentially none, for the mystery model.
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
- **For the two online selectors (mystery, seed 2), the estimator reaches and slightly beats sampling at about 13–27M evaluations.** Mystery is at 0.36× the sampling error with caching (error 0.002 against SD 0.006; the 200M reference's own uncertainty is 0.0015). Seed 2 is at about 1×.
- **For the memory model (seed 0), it converges to a +0.08-point floor:** the mean-field shortcut's real bias, since that model's lower-input leakage does not cancel. So it cannot match sampling.
- **Convergence is erratic at intermediate grids** (1.1× → 8.5× → 0.36×), and comparing grids does not give a trustworthy error bar. A practical "mechanistic estimate" would need its own reliable error estimate, which this does not yet have.

So this is parity at the largest grid for the selector models, not a robust win over sampling.

**With honest error bars, the estimator is 6–20× worse than sampling (`t1.py`, `t2.py`).** To give the estimator its own uncertainty, I ran K = 4 independent randomized replicates and used the spread across them as the standard error (mystery model, against the 200M reference):

| design | evaluations | standard error | ratio to sampling SD | errors (z-scores) |
|---|---|---|---|---|
| stratified grid with random jitter in each cell, n = 12 | ~7.0M | 0.145–0.201 points | **18–25×** | −0.74, −1.25, −0.93 |
| Gauss–Legendre with randomized nuisance parameters (M range, a-mapping exponent, R cutoff), n = 16 | ~12.9M | 0.034–0.066 points | **6–11×** | −0.05, +0.46, +2.75 |

The error bars are roughly calibrated. But at about 13M evaluations, perturbing the grid's nuisance parameters moves the estimate by a few hundredths of a point. That is several times sampling's standard deviation, which means **the 0.36× result above came from one fortunate fixed grid**. With an honest error bar, this mechanistic estimate costs about 6–20× as much as sampling for the same reliable accuracy. ARC's "match sampling" target is not met.

**Fixing seed 0's bias by treating the 4th-largest input exactly fails (`r5.py`).** I extended the estimator to 4 exact order statistics: 5,040 position arrangements, a fifth symbol for the remaining six inputs at their conditional mean below R4, an extra Gauss–Legendre dimension for the 3rd-to-4th gap, density φ(M)φ(S)φ(R)φ(R4)Φ(R4)⁶·|M|³, and a 17,254-node prefix trie against 50,400 steps:

| model | grid | evaluations | mass | estimate | error |
|---|---|---|---|---|---|
| seed 0 | 8 × 8 × 4 | 15M | 0.936 | 0.90903 | −0.11 |
| seed 0 | 12 × 12 × 6 | 50M | 0.987 | 0.91291 | +0.28 |
| seed 0 | **16 × 16 × 10** | **147M** | **1.0000** | **0.91277** | **+0.27** |
| mystery | 12 × 12 × 6 | 48M | 0.987 | 0.95500 | +0.19 |

With the mass converged, seed 0's error is **+0.27 points, worse than the 3-input mean field's +0.08**. A plausible reason: in the memory model the lower inputs feed linear memory channels, so its errors depend on the *spread* of those inputs, not just their mean. Any mean-field treatment misses that, however many order statistics are made exact. So the mean-field family of estimators is fundamentally limited for memory-strategy models.

**A mechanism-based control variate does not help (`o1.py`).** The estimate mean(f − MF) + E[MF], with E[MF] from quadrature, has 0.4–0.7× the efficiency of plain sampling at equal evaluations. Each sample needs two network evaluations, the correlation between f and MF is only 0.58–0.67 (at least 0.71 is needed to break even), and the quadrature value of E[MF] carries the grid bias.

**Conclusion.** Leakage is nonlinear and *two-sided*. Closing the last 0.15–0.3 points without sampling would need a model that gets both the losses and the rescues right. None of the three approximations I tried (Gaussian linear, truncated linear, additive nonlinear) does.

### 7.14 Replaying training: when and why a run commits to a strategy (`k1.py`–`k4.py`)

I continued training from zoo checkpoints with a fresh AdamW (learning rate 0.01/√16 schedule, batch 1024, weight decay 0.001) and new data orders. The original optimizer state isn't saved, so this is a fork, not an exact replay. Score: the most selective pointer, meaning the lowest "neither" firing rate among neurons that fire on ≥80% of new-2nd events.

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

**Larger batch from initial weights** (`k5.py`; all 5 seeds × 2 warmups × 2 more data orders, 2²⁵ sequences):

| warmup | runs that switch |
|---|---|
| 500 steps | 4/10 |
| 16,384 steps | 3/10 |

Including the three runs above, **6/13 (46%) switch with the fast warmup and 4/13 (31%) with the original**. That difference is not significant at this sample size.

The earlier lopsided result (8/9 against 3/9) came mainly from the **2²⁰ forks**. There, a fresh optimizer jumps to the full learning rate *in the middle of* training. That "kick" made seed 0 switch in 3/3 runs, against 0/3 with the original schedule. Warmup length from scratch does not clearly matter.

Per seed, from initial weights:

| seed | switch rate |
|---|---|
| 0 | 3/5 |
| 1 | 2/5 |
| 2 | 1/4 |
| 3 | 3/4 |
| 4 | **0/6** |

Switching does not pay off immediately. At 2²⁵ sequences, some memory runs are ahead of selector runs (for example 0.889 against 0.827).

**Per-seed switch rates (`k7.py`, `k11.py`).** 30 more runs from initial weights with the original schedule (6 data orders per seed), pooled with the earlier original-schedule runs:

Updated with 30 further runs (`l4.py`, `l8.py`; 14–15 runs per seed):

| seed | runs that switch | 95% interval (Clopper–Pearson) |
|---|---|---|
| 3 | **14/14** | 0.77–1.00 |
| 0 | 5/15 | 0.12–0.62 |
| 2 | 4/14 | 0.08–0.58 |
| 1 | 2/15 | 0.02–0.40 |
| 4 | 2/15 | 0.02–0.40 |
| **all** | **27/73 = 0.37** | |

**Initial weights matter.** Seed 3's starting point *always* produces a selector (14/14), while seeds 1 and 4 rarely do (2/15 each); the intervals do not overlap. The zoo's best seed (seed 2) switches in only 4 of 14 runs from scratch, so its original run was partly luck of the data order. Fresh initializations switch at 54% (13/24), within the range of the zoo seeds.

At 2²⁵ sequences, runs that switched are already ahead: mean accuracy 0.822 against 0.785, and clean-case error 6.8% against 10.7%.

**Fresh random initializations (`k12.py`, `k13.py`).** I trained 24 new initializations (PyTorch default init) with the original schedule for 2²⁵ sequences. **13/24 (54%) switch**, in line with the zoo seeds' pooled 42%. Switched and non-switched runs from these starts have the same mean accuracy at that point (0.838).

None of the simple initialization statistics predicts switching. I tested: number of positive input weights, largest and smallest input weight, mean |input weight|, spectral radius of W_hh, mean self-connection, self-connection of positive-input neurons, and the largest input-weight × self-connection product. All Mann–Whitney p-values are ≥ 0.08; the smallest is for the most negative input weight. Whatever makes seed 3 switch in 8/8 runs and seed 4 in 1/9 is a property of the specific configuration, not of these scalar summaries.

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

**Does early S-encoding predict switching? No (`l6.py`, `l7.py`).** Across the 5 zoo seeds, the gap between M's and S's linear decodability at 2²² happened to order the switch rates almost perfectly (gap 0.02 for seed 3 at 100% up to 0.18 for seed 1 at 22%; rank correlation about −0.9). On 12 fresh runs it does not predict switching: at step 4,096 the gap is 0.126 for runs that switch and 0.116 for runs that don't (Mann–Whitney p = 0.87; p = 0.31 at step 2,048). The 5-seed pattern was a coincidence. Which feature of the early state decides the fork remains unidentified.

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

**The predisposition acts through the initial gradient, set by the readout (`m1.py`–`m6.py`).**

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
| sign of winner's input-weight gradient | — | — | 0.64 |

**A multivariate classifier also finds nothing (`n3.py`).** I used 26 initial-weight and initial-gradient features for each of the 48 fresh runs: the top-6 gradient-size profile; input-weight gradients, input weights and self-connections of the 4 most-pushed neurons; input-weight extremes; spectral radius; and readout norm. Leave-one-out logistic regression scores **46–50% accuracy** against 50% chance (permutation p = 0.51–0.57, C ∈ {0.03, 0.1, 1}).

The winner's-margin statistic **failed to replicate**. On 24 further fresh initializations (`m7.py`, switch rate 11/24) it points the *other* way: 1.19 for runs that switch against 1.24 for those that don't (p = 0.27). Over all 48 runs, p = 0.57. The first p = 0.056 was noise. It also does not order the 5 zoo seeds (Spearman −0.15). The overall fresh-initialization switch rate is **24/48 = 50%**. The input-gradient sign orders the zoo seeds well (+0.87) but fails on fresh runs. **A reduced model: a frozen random readout is enough (`p3.py`).** I trained the full seed 3 transplant (seed 3's recurrent weights and readout, seed 4's input weights; switches 6/6 normally) with parts frozen, 3 data orders each:

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

But these networks are poor: on 8 fresh initializations with the readout frozen, accuracy is **44–58%** and clean-case error 31–49%. The same initializations with a trainable readout (`k12.py`) reach 77–92% and 0.7–12%. Selective neurons also don't become more common on matched runs: 5/8 frozen against 6/8 trainable.

**Delaying readout learning helps (`q3.py`).** The same 8 fresh initializations and data orders, with the readout frozen only for the first 8,192 steps (about 2²³ sequences) and trainable afterwards:

| | normal training (`k12.py`) | readout frozen for the first 8,192 steps |
|---|---|---|
| runs that switch | 6/8 | **7/8** |
| mean accuracy at 2²⁵ | 0.850 | **0.876** |
| mean clean-case error | 4.6% | **3.1%** |
| best accuracy | 0.921 | **0.944** |

Accuracy improves for 6/8 initializations. Initializations 0 and 5 go from memory to selection; initialization 1 goes the other way. At n = 8 this is suggestive, not significant (sign test p ≈ 0.29). **Extended to 24 matched pairs (`q4.py`):** selective pointers form in **20/24 against 13/24** runs. Mean accuracy at 2²⁵ is **0.864 against 0.838**, improved in **18/24 pairs** (sign test **p = 0.023**; Wilcoxon p = 0.087). Mean clean-case error is **3.8% against 5.6%**. So delaying readout learning by about 2²³ sequences makes the selection strategy markedly more likely and improves accuracy at this stage of training. **Longer training (`q5.py`, 2²⁷ sequences, the first 8 pairs):** mean accuracy **0.917 against 0.897**, improved in 6/8 pairs, with the same two initializations (0, 1) losing as at 2²⁵. Switch rate is 7/8 against 6/8. The best delayed run reaches **0.958**, above the zoo's best 16-neuron model (0.949, trained 8× longer). So the gain persists with 4× more training, but at n = 8 it is not significant by itself (sign test p ≈ 0.29). **Full training budget (`q6.py`, 2³⁰ sequences, the original schedule, 2 pairs):**

| initialization | normal training | readout frozen for the first 8,192 steps |
|---|---|---|
| 4 | 0.952 (clean-case error 0.29%) | **0.968** (0.12%) |
| 6 | 0.954 (0.06%) | **0.976** (0.21%) |

The normally trained runs match the zoo's best 16-neuron model and the mystery model (0.949–0.953). The delayed-readout runs reach **0.968–0.976**, the highest accuracy of any 16-neuron, length-10 model seen in this work, with the same architecture and budget. All four are online selectors, so at full budget the gain shows up as accuracy, not as a change of strategy. With n = 2 this is suggestive, though it agrees with the significant 24-pair result at 2²⁵ and the 2²⁷ result. It fits the picture that **early co-adaptation of the readout is what lets the memory solution win**: if the readout waits until event-selective neurons have formed, it more often builds the selection algorithm on them. Unlike the fully frozen readout, this gives working selectors.

So **forming an event-selective neuron is easy, and is necessary but not sufficient for the selection strategy.** Turning such a neuron into a working selector requires the readout to co-adapt. The frozen-readout "reduced model" above therefore captures the formation of the pointer neuron, not the full fork. In the normal-training statistics of this section, a selective pointer always coincided with good selection behaviour (lower clean-case error), so that metric remains valid *there*. The predisposition is a property of that readout's geometry relative to the initial recurrence.

**The initial-gradient account is falsified (`p1.py`, `p2.py`).** A greedy search found a minimal readout transplant: seed 3's readout columns for n13, n5, n3 and n14, with seed 3's recurrent weights and everything else from seed 4. It gives n13 an even *larger* lead in the initial gradient (1.46× the next neuron, against 1.33 for the full seed 3 readout and 0.56 for seed 4's). Yet it switches in **0/6** training runs; the pointer stays at 0.80. So winning the step-0 gradient competition is **not sufficient**. The readout-dependent predisposition is real (full readout 6/6 against 0/4), but it acts through the *training trajectory*, not through the initial gradient alone.

**Interpretation.** Memory versus selection is a **stochastic fork early in training**, taken by roughly a third to a half of runs. It depends on data order and initial weights; seed 4 seems strongly biased against switching. It is settled by about 2²⁴ sequences. A sudden learning-rate increase before then (2²⁰–2²²) reliably pushes runs into the selection basin (12/12 fork runs). That suggests the selector solution sits behind a barrier that a large enough update can cross, while ordinary training crosses it only sometimes.

**The barrier, measured (`k6.py`).** I re-created one switched fork of seed 0: from 2²², about 2²⁴ sequences in total, pointer "neither" rate 0.24. Along the straight line in weight space between it and the original memory model at 2²⁴ (rate 0.80):

| position | 0.0 (memory) | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 (switched) |
|---|---|---|---|---|---|---|---|
| loss | 0.617 | 1.283 | 1.953 | **2.057** | 1.972 | 1.291 | 0.502 |
| accuracy | 0.751 | 0.580 | 0.508 | 0.507 | 0.524 | 0.615 | 0.798 |

That is a linear barrier of about 1.4–1.6 nats. Pointer selectivity changes gradually along the path. Both models descend from the same 2²² checkpoint, so a relabelling of neurons is an unlikely explanation. At a matched amount of training, the selector basin is already lower in loss (0.502 against 0.617). That fits the picture of two basins separated by a barrier, with ordinary training sometimes and a learning-rate kick reliably crossing it before about 2²⁴. **A curved path is nearly flat (`k8.py`).** I optimized a quadratic Bézier curve between the same two endpoints, training only the control point with Adam (6,000 steps, t ~ U(0,1), batches of 2,048):

| t | 0.0 (memory) | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 (switched) |
|---|---|---|---|---|---|---|---|
| loss | 0.617 | 0.685 | **0.687** | 0.631 | 0.551 | 0.487 | 0.504 |
| pointer "neither" rate | 0.80 | 0.80 | 0.80 | 0.66 | 0.48 | 0.30 | 0.23 |

The barrier is only **about 0.07 nats**, against about 1.44 on the straight line. So the two basins are joined by a low-loss path. Along it, the pointer becomes selective gradually, after a small hump near the memory end (t ≈ 0.2–0.3).

The memory solution is therefore not walled off. It sits behind a shallow ridge that ordinary, slowly decaying training often fails to cross but a learning-rate kick crosses easily. That is consistent with the 12/12 kicked forks.

**Caveats.** There are 1–6 runs per cell. "Switching" is judged by the existence of a selective pointer neuron (a "neither" rate below 0.4). The fork runs use a fresh optimizer and were trained for only 2²⁵ sequences.

## 8. Open problems and caveats

- **The clock analysis is linear.** It uses the eigenmodes of the usually-active neurons, but those neurons do switch off at certain ages. The causal interventions and patching are exact operations on the final state, but they act in that approximate basis.
- **Seed 2's delay trajectory is now derived from the weights** (7.9): it is the impulse response Whh^a·e₁₀, and the readout decodes it. The new-max handoff around the pointer is now attributed to the comparators and the max markers (7.9). What remains unexplained is how training made Wo approximately invert the impulse response. Section 6 shows when this happens (seed 2 at 2²⁴–2²⁵), not why.
- **The memory-based seeds' storage is only partly explained.** The linear-memory fit has R² of about 0.85. The max's opposite-sign write is identified, but the full nonlinear circuit is not.
- **A sampling-free accuracy estimate now exists, but it is not cheaper than sampling.** Section 7.13 gives a boundary-aware quadrature over the top-3 order statistics and their positions, with the lower inputs at their conditional mean. It is accurate to about 0.1 points for three models (+0.06, +0.08, −0.13) using no random inputs. It is justified by the finding that leakage from the lower inputs changes 3–7% of outputs but cancels to under 0.1 points net. After optimization (scale-adapted coordinates, a converged inner scan, prefix caching), single fixed grids sometimes hit sampling-level accuracy (mystery 0.36× at 12.8M evaluations). But with an honest error bar from randomized replicates, its standard error is **6–20× sampling's** at equal cost. The memory model (seed 0) also has a +0.08-point mean-field floor. ARC's target is not met (7.13).

| model | first order | second order |
|---|---|---|
| mystery | −0.10 points | −0.30 |
| seed 2 | −0.18 | −0.54 |

Per-cell RMSE also rises (0.081 → 0.092 and 0.100 → 0.105). So the cut-HDMR expansion of the correct/incorrect indicator does not converge: pair terms overshoot. First order is the best this family of approximations can do.
- **The readout side of "why training lands here" is resolved** (7.12): it is the convex optimum for the recurrence's features. For the recurrence, the switch is located in one neuron's recurrent threshold (n10 snapping onto ≈0.5–0.7·S). Forked training (7.14) shows the fork is stochastic: about 31–46% of runs from scratch switch, the outcome depends on data order and initial weights, and it is settled by about 2²⁴ sequences. A mid-training learning-rate "kick" before then reliably induces switching. The barrier is shallow along a curved path (about 0.07 nats), and per-seed rates range from 1/9 to 8/8 (7.14). 24 fresh initializations switch at 54%, and no simple scalar statistic of the initial weights predicts it (all p ≥ 0.08). Tracking recruitment shows the pointer is chosen among the initially positive-input neurons by early growth, by about 2²² sequences. Section 7.14 shows that the competition is over threshold targets. M (the new-max detector) is taken first. A run selects only if another neuron settles on the 2nd-max threshold by about 2²³ sequences; otherwise a neuron goes always-on. The loss-landscape scan (7.14) gives a partial reason. At 2²² the loss already prefers S-ward thresholds, but S is less linearly available in the state, so they are harder to reach. By 2²³ co-adaptation flips the preference to M. Always-on is never preferred directly. Per-seed rates (14–15 runs each) range from 2/15 to 14/14. Initialization transplants locate seed 3's predisposition in the joint recurrent-plus-readout configuration, spread across neurons, which consistently makes n13 a one-step-late pointer. Simple weight statistics (p ≥ 0.08) and the early S-decodability gap (p = 0.87) do not predict switching. Transplanting n13 alone (0/4) or everything except n13 (1/4) fails, so the property is an interaction between n13 and its seed 3 surroundings, not a single-neuron trait. The initial *readout* carries it: with forward dynamics identical, seed 3's readout gives 6/6 against 0/4. But not through the step-0 gradient: a minimal readout transplant that gives n13 an even larger initial-gradient lead switches 0/6. So the predisposition acts through the training trajectory. A *frozen* seed 3 readout is enough for 3/3 switching, while training only n13's own row fails 0/3, so the decisive dynamics are the whole recurrence learning toward a fixed readout geometry. **No predictor at initialization was found.** Six single statistics and a 26-feature cross-validated classifier (46–50% accuracy against 50% chance) all fail on 48 fresh runs. The best single candidate, the winner's gradient margin, looked borderline at first (p = 0.056) but failed to replicate (p = 0.27 held-out, 0.57 overall). The one-step-late pointer itself is now explained: it is a gate opened by silent comparators (7.9). It would also need a quantitative theory of how that initial competition plays out over the roughly 12k steps before the pointer forms. Narrowing it further would need many more transplant conditions, each needing several training runs, plus a theory of why that path is attractive under gradient descent.
- **The mystery model has no training checkpoints.** That it is a "tidier" online selector is inferred from the zoo seeds, especially seed 2's transient ~40° 2nd-max clock.

## 9. Ledger of ideas: tried, outcome, and why each line is closed

Each line of investigation is listed with what was tried, the result, and why I consider it closed (✔ = resolved; ✘ = tried and found unproductive; ◐ = partly resolved, and the remaining step would be a new research programme rather than a variation of what was tried).

**Mechanism of the mystery model**
- ✔ **Age code, clocks, comparators, Fourier readout** (sections 2.1–2.5), confirmed causally (2.4, 7.1).
- ✔ **Write pointer n5 and how it is gated** (7.9): forcing an n5 pulse moves the answer to the marked position in 88–96% of runs.
- ✔ **Max-to-2nd handoff** (7.9): carried by the comparators and the max markers.
- ✔ **Failure modes** (section 3).
- ✘ **Short program that reproduces each output, mistakes included** (section 4): the mistakes need nearly all 432 weights.

**Other models**
- ✔ **Seeds 0–4** (sections 5, 7.2–7.3, 7.6): memory versus selection; seed 2's write pointer.
- ✔ **4-, 6-, 8- and 32-neuron models** (7.8, 7.10): time cells and their release-from-inhibition chain.

**Training dynamics**
- ✔ **When the strategy is fixed** (7.14): a fork settled by 2²³–2²⁴ sequences.
- ✔ **Per-seed switch rates** (14–15 runs per seed) and 48 fresh initializations.
- ✔ **Loss-landscape barrier**: 1.4 nats on the straight line, 0.07 on an optimized curve.
- ✔ **Learning-rate kick**: induces switching, but which pointer type forms is seed-specific.
- ✔ **Readout is the convex optimum for the recurrence's features** (7.12).
- ✔ **n10's threshold snap at seed 2's switch** (7.12).
- ✘ **Predicting the outcome at initialization**: 6 single statistics plus a 26-feature cross-validated classifier, all at chance; the most promising statistic failed to replicate.
- ✘ **Initial-gradient account of seed 3's predisposition**: falsified (0/6 switch despite a larger initial-gradient lead).
- ◐ **Where seed 3's predisposition lives**: located by transplants in the joint initial recurrent and readout weights, as an interaction centred on n13. **Leave-one-neuron-out search (`u1.py`):** starting from the full seed 3 transplant (2/2 switch with these data orders), I reverted each other neuron's row, column and readout column to seed 4's values, with 2 data orders each:
  - reverting **n5, n8, n12 or n14** stops the switch (**0/2** each), so these are necessary partners of n13;
  - reverting **n3 or n9** has no effect (2/2);
  - reverting any of the other 9 neurons weakens it (1/2).

  So the predisposition is a distributed configuration of n13 plus at least four specific partner neurons, with most others contributing marginally. A finer, edge-level search would need far more training runs.
- ✔ **Reduced models**: a frozen readout lets selective neurons form, but networks stay at 44–58% accuracy; freezing either half of the recurrent rows blocks the pointer.
- ✔ **Practical result**: delaying readout learning for 8,192 steps raises the selector rate from 13/24 to 20/24 and accuracy at 2²⁵ by 2.6 points (p = 0.023). It persists at 2²⁷ (0.917 against 0.897) and at 2³⁰ (0.968–0.976 against 0.952–0.954, n = 2).
- ◐ **Quantitative theory of the early competition**: everything above characterizes it empirically. A predictive theory would need a mathematical model of coupled readout and recurrent learning across about 12k steps; no further experiment of the kinds already used would supply it.

**Mechanistic accuracy estimation (ARC's challenge)**
- ✘ Rule-based programs, decision trees, single comparator-resolution models, linear and additive leakage models, and second-order cut-HDMR all failed (7.7, 7.13).
- ✔ **Leakage nets out**: under 0.1 points for the selector models (sampled check), so the earlier "leakage" was quadrature error.
- ✔ **Boundary-aware quadrature in scale-adapted coordinates, with prefix caching**: converges to 0.002–0.004 points for the selector models at the largest grids.
- ✘ **Beating sampling with an honest error bar**: randomized replicates give standard errors 6–20× sampling's. A control variate is worse than plain sampling. Stratified sampling gives a 1.2–2.8× gain, but it still samples.
- ✘ **Removing the memory model's bias** by treating the 4th input exactly: worse (+0.27 points). The bias comes from the lower inputs' variance, which no mean-field variant captures.

**Conclusion.** I have run out of productive ideas within the scope of this investigation. Every line above is either resolved or has been tried and found unproductive. The two partly-open items each need a new theoretical framework, not another experiment of the kinds already run, and I don't have such a framework to propose.
