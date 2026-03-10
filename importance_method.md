````markdown
# Proposal: Model-Free Frame Importance Labeling for Token Communication / Piggyback FEC
**Scope:** keep only **model-free** methods  
**Goal:** compute a causal or near-causal frame importance score `I[n]` for each 20 ms frame, so the transport / modem layer can drive:
- piggyback redundancy,
- selective retransmission,
- packet dropping priority,
- congestion-aware protection.

This proposal keeps only methods that **do not require training an additional model**. It is arranged **from lowest implementation complexity to highest**. The original note already includes pure heuristic methods such as **A1/A2/A3**, token-based **B1/B3**, and the **no-training fallback of B2**. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## 0. Definition of “important”
For real-time voice, frame `n` is important if:

> **losing frame `n` is hard to conceal and more likely to be noticed by the user.**

In practice, importance tends to rise when the frame contains:
- talkspurt onsets,
- transients,
- energy jumps,
- voicing transitions,
- low predictability from past frames,
- recovery-sensitive context such as recent prior loss or stressed jitter buffer. :contentReference[oaicite:3]{index=3}

---

# Methods sorted by complexity

## 1) A1 — VAD + talkspurt onset boost
**Complexity:** very low  
**Type:** audio heuristic  
**Why first:** cheapest useful baseline

### Idea
Frames right after silence-to-speech transition are harder to hide if lost, because PLC often needs a moment to “lock on”. This is explicitly aligned with the note’s **talkspurt onset boost** idea. :contentReference[oaicite:4]{index=4}

### Inputs
- `vad[n]`
- `vad[n-1]`
- optional short silence-run counter

### Example rule
```text
onset[n] = 1 if vad[n]=1 and vad[n-1]=0 else 0

I_A1[n] =
    1.0  for first M voiced frames after onset
    0.2  otherwise
````

Typical: `M = 2~3` frames.

### Pros

* fully causal
* tiny compute
* very easy to implement and debug

### Cons

* only captures one failure mode
* misses important non-onset frames

---

## 2) B1 — Token change-rate / novelty

**Complexity:** very low
**Type:** token heuristic
**Why second:** almost as cheap as A1, but token-native

### Idea

If many token/codebook entries change abruptly from frame `n-1` to `n`, frame `n` likely carries new information and is harder to reconstruct from history. This is the note’s **token novelty** method. 

### Inputs

* token indices `t[n,c]` for `c = 1..C`
* previous frame token indices `t[n-1,c]`

### Example score

```text
chg[n] = (1/C) * Σ_c 1[t[n,c] != t[n-1,c]]
I_B1[n] = clamp(chg[n] / τ, 0, 1)
```

Typical: `τ ≈ 0.4 ~ 0.6`

### Pros

* causal
* codec-token friendly
* no waveform DSP needed if tokens are already available

### Cons

* can overrate noisy or unstable frames
* should be gated by VAD / energy for robustness

---

## 3) A2 — Transient / spectral flux importance

**Complexity:** low
**Type:** audio heuristic
**Why third:** still simple, but needs a bit more DSP plumbing

### Idea

High spectral change means low short-term predictability, so concealment tends to be harder. This comes directly from the note’s **spectral flux / transient** method. 

### Inputs

* short-time energy `E[n]`
* spectral flux `SF[n]`
* voicing probability `V[n]`
* optional high-band energy ratio

### Example score

```text
I_A2[n] = clamp(
    a * norm(SF[n]) +
    b * norm(|E[n]-E[n-1]|) +
    c * (1 - V[n]),
    0, 1
)
```

### Pros

* causal
* captures transients, fricatives, and abrupt changes
* complements A1 well

### Cons

* slightly more feature extraction than A1/B1
* thresholds need tuning per codec / content

---

## 4) A4 — EVS-style frame criticality scorer

**Complexity:** low to medium
**Type:** model-free composite criticality
**Why included:** this is the EVS-inspired importance path you explicitly asked to add

### Idea

Instead of using only one cue, score whether **loss of frame `n` would be hard to hide** using a set of **EVS-like frame criticality indicators**. This is still model-free because it relies on rule-based features and fixed weights, not a trained predictor.

### Inputs

Use the following features for frame `n`:

#### A. Frame criticality inputs

* **frame class**: voiced / unvoiced / generic / transient / transform-like
* **onset / transition flags**
* **pitch stability** or pitch-lag behavior
* **gain jump / energy discontinuity risk**
* **whether previous frame(s) were lost**
* **current network FER state / jitter-buffer state**

These inputs match the EVS-style criticality concept you provided.

### Suggested interpretation

* **voiced** frames are often sensitive when pitch is unstable
* **unvoiced / transient** frames can be perceptually sharp and hard to fake
* **onsets / transitions** deserve extra boost
* **large gain jumps** imply audible discontinuity risk
* **previous frame already lost** means current loss is even more dangerous
* **high FER / shallow jitter buffer** means protection budget should become more selective

### Example score

```text
I_EVS[n] = clamp(
    w_fc   * S_frameclass[n]   +
    w_tr   * S_transition[n]   +
    w_pitch* S_pitchinstab[n]  +
    w_gain * S_gainjump[n]     +
    w_hist * S_prevloss[n]     +
    w_net  * S_netstress[n],
    0, 1
)
```

Where for example:

* `S_frameclass[n]` gives higher value to transient / unstable voiced frames
* `S_transition[n]` boosts silence↔speech or voiced↔unvoiced boundaries
* `S_pitchinstab[n]` rises when pitch lag changes abruptly
* `S_gainjump[n]` rises when energy jumps sharply
* `S_prevloss[n]` rises if `n-1` or `n-2` was lost
* `S_netstress[n]` rises when FER is high or jitter buffer is tight

### Example frame-class prior

```text
transient        -> 1.0
unstable voiced  -> 0.8
unvoiced         -> 0.7
generic          -> 0.5
stable voiced    -> 0.4
```

### Example previous-loss boost

```text
S_prevloss[n] =
    1.0 if frame n-1 was lost
    0.6 if frame n-2 was lost
    0.0 otherwise
```

### Example network-state boost

```text
S_netstress[n] =
    α * norm(FER_est[n]) +
    β * JBM_tight[n]
```

### Pros

* very practical for modem / transport use
* uses exactly the kind of cues PLC and EVS-like concealment care about
* lets importance react to both **content** and **network state**

### Cons

* more tuning than A1/A2
* needs explicit definition of frame classes and score weights

### Recommended use

Use this as the main **rule-based criticality engine** once A1/A2 are working.

---

## 5) A3 — Edge frames around phonetic boundaries without ASR

**Complexity:** medium
**Type:** audio boundary heuristic
**Why fifth:** still model-free, but boundary logic is more fiddly

### Idea

Important frames often sit near phonetic boundaries, but you do not need ASR to detect them. The note already defines this as a causal boundary detector using energy, voicing, and centroid jumps.  

### Inputs

* `E[n]`
* `V[n]`
* spectral centroid
* optional zero-crossing change

### Example rule

```text
boundary[n] = 1 if
    |E[n]-E[n-1]| > T_E
    or V[n] != V[n-1]
    or |SC[n]-SC[n-1]| > T_SC
else 0

I_A3[n] = 1.0 if boundary[n]=1 else 0.2
```

### Pros

* still fully model-free
* useful for word/phoneme transition sensitivity

### Cons

* threshold tuning is annoying
* boundary heuristics can be twitchy

---

## 6) B3 — Cross-codebook disagreement / structure cues

**Complexity:** medium to high
**Type:** token-structure heuristic
**Why sixth:** no training, but more engineering fuss

### Idea

Some frames show coherent token structure across codebooks, while others look like “detail explosion” or abrupt structure break. The note describes this as **cross-codebook disagreement / structure cues**. 

### Inputs

* token dispersion across codebooks
* entropy of token histogram
* pairwise correlation of codebook change events

### Example score

```text
I_B3[n] = clamp(
    a * H_codebook[n] +
    b * Dispersion[n] +
    c * ChangeCorrBreak[n],
    0, 1
)
```

### Pros

* fully token-native
* can catch frames that simple change-rate misses

### Cons

* more complicated statistics
* harder to interpret than A1/A2/B1

---

## 7) B2-fallback — Online n-gram / hash-table token surprisal

**Complexity:** highest among model-free methods
**Type:** token predictability without training
**Why last:** still model-free only in the fallback form, but clearly the most elaborate

### Important scope note

The main B2 method in the note uses a GRU/Transformer predictor, which is **not** model-free.
Only the **no-training fallback** is kept here: online n-gram counts / hash-table language model on token histories.  

### Idea

Estimate how surprising current token frame `n` is given recent token history, but do it using online counts rather than a trained neural predictor.

### Inputs

* token history `t[n-1], t[n-2], ..., t[n-K]`
* online count table / hash table

### Example score

```text
P_hat(t[n] | history) = count(history, t[n]) / count(history)

I_B2fallback[n] = normalize( -log P_hat(t[n] | history) )
```

### Pros

* gives a more principled predictability-style signal
* still avoids model training

### Cons

* more state bookkeeping
* history sparsity and smoothing become annoying little goblins
* more complicated than the other model-free options

---

# Recommended combined scorer

## Minimal prototype

Use:

```text
I[n] = w1 * I_A1[n] + w2 * I_B1[n]
```

This is the cheapest useful baseline.

---

## Strong model-free baseline

Use:

```text
I[n] = w1 * I_A1[n] +
       w2 * I_A2[n] +
       w3 * I_B1[n] +
       w4 * I_EVS[n]
```

### Why this is the best practical starting point

* `A1` handles talkspurt onset
* `A2` handles transients / abrupt acoustic change
* `B1` handles token novelty
* `I_EVS` adds concealment difficulty and network/recovery awareness

This gives you a robust **content + token + recovery-state** importance score without training any extra model.

---

## Optional richer model-free version

Add:

```text
+ w5 * I_A3[n]
+ w6 * I_B3[n]
+ w7 * I_B2fallback[n]
```

Use this only after the simpler stack is stable.

---

# Final complexity order

From lowest to highest implementation complexity:

1. **A1 — VAD + talkspurt onset boost**
2. **B1 — Token change-rate / novelty**
3. **A2 — Transient / spectral flux importance**
4. **A4 — EVS-style frame criticality scorer**
5. **A3 — Edge frames around phonetic boundaries without ASR**
6. **B3 — Cross-codebook disagreement / structure cues**
7. **B2-fallback — Online n-gram / hash-table token surprisal**

---

# Implementation notes for transport / FEC

Once `I[n]` is available, map it to 2-bit or 3-level priority:

```text
0 = low
1 = medium
2 = high
3 = critical
```

Then use it to drive:

* piggyback redundancy selection,
* retransmission permission before playout deadline,
* drop-low-first under congestion,
* jitter-buffer-aware protection policy.

---

# Recommended first implementation order

## Phase 1

* A1
* B1

## Phase 2

* A2
* A4 (EVS-style frame criticality)

## Phase 3

* A3
* B3

## Phase 4

* B2 fallback

---

# Bottom line

If you want the cleanest model-free proposal for a paper / patent / internal spec, the most defensible core is:

```text
A1 + A2 + B1 + EVS-style frame criticality
```

That stack is:

* causal,
* training-free,
* explainable,
* easy to map to piggyback FEC / selective redundancy,
* and much less likely to trigger “why did you need another model?” objections.

```

我也可以下一步直接幫你把這份改成更像 **3GPP-style contribution / patent proposal format**，例如補上 **Problem Statement / Embodiment / Claim-like bullets / evaluation plan**。
```
