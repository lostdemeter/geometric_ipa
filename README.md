# Geometric IPA

**Learn English-to-IPA phonetic transcription using pure geometry — no neural network, no training, no gradient descent.**

## What Is This?

This system learns to convert English text to the International Phonetic Alphabet (IPA) using only geometric primitives. Each phonetic rule is a **RECT pair** — two gate functions that activate at exactly one codepoint and shift it to another.

Rules compose additively. The final "program" is just a stack of gate pairs. Feed in ASCII codepoints, get IPA codepoints out.

```
EN:  The quick brown fox jumps over the lazy dog.
IPA: θɛ kwɪk bɹɒwn fɒx ʒʌmps ɒvɛɹ θɛ læzi dɒg.

EN:  She thinks singing is the best thing in the world.
IPA: ʃɛ θɪŋks sɪŋɪŋ ɪs θɛ bɛst θɪŋ ɪn θɛ wɒɹld.

EN:  I hope to make a fine cake and ride home in time.
IPA: ɪ hoʊp tɒ meɪk æ faɪn keɪk ænd ɹaɪd hoʊm ɪn taɪm.
```

## How It Works

### The Gate Primitive

Every rule is built from `gate_step(x, t, s)` — a width-1 rectangular pulse centered at codepoint `t`:

```
gate_step(x, t, s) = [IdealGate(s·(x-(t-0.5))) - IdealGate(s·(x-(t+0.5)))] / s
```

Where `IdealGate(x) = x · σ(√(8/π) · x · (1 + [(4-π)/(6π)] · x²))` is the exact geometric form of GELU.

A character substitution `'a' → 'æ'` becomes:
- One RECT pair at codepoint 97 (ASCII 'a') with height +227 (to reach codepoint 324, 'æ')
- At integer evaluation (the natural resolution), this is exact

### Four-Phase Architecture

The system processes text in four phases:

| Phase | Purpose | Example |
|-------|---------|---------|
| **0: Feature Extract** | Detect non-local patterns (magic-e, igh trigraph, silent-e) | "bite" → vowel 'i' becomes long |
| **1: Digraph Collapse** | Merge multi-char patterns into IPA symbols | "sh" → ʃ, "th" → θ, "ng" → ŋ |
| **2: Context Channels** | Apply context-dependent rules via gear shift | "c" → k or s depending on next char |
| **3: Character RECTs** | Simple 1:1 substitutions | "a" → æ, "r" → ɹ, "j" → ʒ |

### Context-Dependent Rules: The Gear Shift

Some letters have multiple pronunciations. The letter "c" is /k/ before a,o,u but /s/ before e,i. The system **discovers this automatically** using information gain:

1. Collect all (input, output, context) observations
2. For each context variable (prev_char, next_char, is_start, ...), compute information gain
3. The variable with highest gain IS the selector

For "g" (the hardest case), a **two-gear** system emerges:
- **Coarse gear** on `next_char`: resolves most cases (g before a,o,u,l,r → hard; g before e,y → soft)
- **Fine gear** on `next_next_char`: engages only when coarse gear is ambiguous (g before i: gift=hard, gin=soft)

No hard-coding. The gear structure is **discovered from examples**.

## Running the Demo

```bash
pip install numpy
python geometric_ipa_demo.py
```

The demo teaches 22 lessons progressively:
1. Consonant digraphs (sh→ʃ, th→θ, ng→ŋ, ch→ʧ, etc.)
2. Vowel digraphs (ee→iː, oo→uː, ai→eɪ, oa→oʊ)
3. Magic-e with trained exceptions (make→meɪk but come→kʌm)
4. Short vowels (a→æ, e→ɛ, i→ɪ, o→ɒ, u→ʌ)
5. Consonant substitutions (j→ʒ, r→ɹ)
6. Context-dependent rules (soft/hard c, soft/hard g, consonant/vowel y)

Each lesson shows the geometric detection, then applies ALL learned rules to a demo sentence. Watch the text progressively transform as rules accumulate.

## Final Statistics

```
Simple char rules: 7
Digraph rules: 13 (4 frozen)
Context rules: 3 (1 geared)
Magic-e: 5 trained vowel rules (4 geared)
Total rules: 29
Geometric primitives: 159 gate_step calls
Gradient descent: none
Neural network: none
```

## The φ Connection

The gate sharpness parameter is `s = φ²` (golden ratio squared). The IdealGate itself is derived from the identity `GELU(x) ≈ x · σ(φ · x)` — the same golden ratio that appears throughout neural network architecture.

At integer codepoint evaluation, the gates become exact indicator functions. The geometric structure (RECT pairs) is the same at every scale — this is self-similarity in action.

## Files

| File | Description |
|------|-------------|
| `geometric_ipa_demo.py` | Main demo — progressive IPA lessons |
| `auto_context_detection.py` | Gear-shift discovery engine (information gain + selector) |
| `README.md` | This document |

## Related Work

- [phi_gate](https://github.com/lostdemeter/phi_gate) — The GELU ≈ x·σ(φ·x) discovery that provides the gate primitive
- [holographic_gate](https://github.com/lostdemeter/holographic_gate) — 4-state gate structure used in the classification

## License

GPLv3
