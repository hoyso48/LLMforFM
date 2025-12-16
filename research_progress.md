# Research Progress: LLM-Driven Sound Synthesis with Synthesizer Tool Learning

## 1. Research Topic

**Title**  
LLM-driven sound synthesis with synthesizer tool learning (DX7 FM synthesizer as the initial tool).

**Core Idea**  
Use an LLM as a *tool-using controller* that maps natural-language sound descriptions to **interpretable DX7 FM synthesizer patches** (128 parameters), instead of generating raw waveforms directly.

---

## 2. Motivation

Recent neural sound synthesis models achieve high audio quality but are hard to interpret:

- **DDSP-style models (DDSP, DDX7, DiffMoog, …)**  
  - Neural networks control differentiable oscillators, filters, and envelopes.  
  - Partially interpretable, but constrained by differentiable design and limited flexibility.

- **End-to-end generative models (WaveNet, GANSynth, AudioGen, AudioLDM, …)**  
  - Directly generate waveforms or latent codes from text.  
  - High fidelity and strong text-to-audio performance, but internal representations are black-box and not easily editable.

Traditional synthesizers (e.g., Yamaha DX7) expose **compact, human-readable parameters** (operator routing, envelopes, frequency ratios).  
**Goal:** bridge

> text → human-readable synthesis parameters → audio

so that sound design is both *LLM-controllable* and *human-editable*.

---

## 3. Scope

### 3.1 Current Scope (This Work)

- **Input**: text-only (natural-language description of the sound).  
- **Model**: text-only LLM (Qwen3-8B) fine-tuned to output full DX7 patch dictionaries.  
- **Tool**: DX7 FM synthesizer (128 parameters per patch).  
- **Task**: text-to-sound via
  - caption → patch (LLM),
  - patch → audio (DX7 tool).

### 3.2 Future Scope

- Audio queries via Audio-Language Models (ALMs) for **sound matching**.  
- Multi-turn, iterative refinement (e.g., “shorter decay”, “brighter”, “less modulation”).  
- Tool feedback using ALMs or captioning models.  
- Multi-tool setting (DX7 + other synths + FX chains).  
- Extension to more general DSP languages / modular synthesis environments.

---

## 4. Background & Related Work

### 4.1 Neural Sound Synthesis

- **DDSP / DDX7 / DiffMoog**  
  - Neural networks drive differentiable synthesis modules.  
  - Balance between fidelity, generality, and interpretability; still limited by architecture design.

- **Text-to-audio generative models (AudioGen, AudioLDM, …)**  
  - Direct text-to-audio generation with good semantic alignment.  
  - Lack a clear mapping to interpretable synthesis parameters.

### 4.2 LLMs with Tool Learning

LLMs can act as planners that call domain-specific tools:

- **Text2CAD**: natural language → parametric CAD construction programs.  
- **LLM2FX-Tools**: LLM generates effect chains and parameters for music post-production.  
- **WavCraft**: LLM orchestrates DSP modules and expert models for audio editing.

Most existing audio-tool work focuses on **high-level FX chains** or editing.  
This project targets **low-level synthesis control**, where the tool API is a complete DX7 patch.

---

## 5. Method

### 5.1 Overall Pipeline

1. **Input**: one-sentence caption describing instrument + timbre + envelope.  
2. **LLM (Qwen3-8B, fine-tuned)**: generates a **structured DX7 patch dictionary**, including:
   - modulation matrix  
   - output matrix  
   - operator envelopes  
   - frequency ratios  
   - sensitivity and other parameters  
3. **DX7 Tool**: a non-differentiable DX7 FM engine renders 10 s audio at a fixed MIDI note.

The LLM does not generate audio directly; it writes the “program” (patch), and the DX7 tool executes it.

### 5.2 Dataset Construction

1. **Patch Collection**
   - 1120 DX7 factory presets (100 held out for test).  
   - ~390k web-crawled DX7 patches.  
   - After deduplication and filtering: **18,298 unique patches**.

2. **Audio Rendering**
   - Implement DX7 synthesis (based on pydx7-like engine).  
   - Render ~10 s audio per patch at a fixed note, with patch-specific key transposition when needed.

3. **Caption Generation (ALM)**
   - Use an Audio-Language Model to caption each rendered audio clip.  
   - Prompt design:
     - Start with **instrument type** (e.g., “electric piano”, “brass lead”),  
     - Follow with **timbre** (bright/dark, warm/metallic, etc.),  
     - Add **envelope** description (attack, decay, sustain, release).  
   - Compare several ALMs and select the one that yields the best caption–audio alignment (via CLAP/CLAP_synth) for final captions.

Result: a dataset of **(caption, patch, audio)** triples.

### 5.3 Model & Training

- **Backbone**: Qwen3-8B (open-weight instruction-tuned LLM).  
- **Input**: the caption only (no extra formatting hints), to keep prompts short and avoid over-guiding the output structure.  
- **Output**: full DX7 patch in Python-dict style, for example:

    {
      "modmatrix": [...],
      "outmatrix": [...],
      "env": [...],
      "freq_ratios": [...],
      "sensitivity": [...],
      ...
    }

- **Objective**: next-token prediction (cross-entropy) on the patch tokens (supervised fine-tuning on caption → patch).  
- **Training setup**:
  - Full fine-tuning for one epoch.  
  - FP8 precision on H100 GPU.  
- **Reasoning**: current version uses direct mapping (caption → patch) without explicit chain-of-thought.

---

## 6. Experimental Setup

### 6.1 Task Definition

- Input: caption of a held-out DX7 preset (test set).  
- For each model:
  - Generate a DX7 patch (if applicable).  
  - Render 10 s of audio via the DX7 tool.  
  - Compare:
    - generated audio vs. ground-truth DX7 audio,  
    - generated audio vs. caption.

Test split: 100 held-out presets with ALM-generated captions.

### 6.2 Baselines

- **Text-to-audio models**
  - AudioGen  
  - AudioLDM  

- **Zero-shot LLMs (no fine-tuning)**
  - Gemini-2.0-flash  
  - Gemini-2.5-flash  
  - LLaMA3-8B-Instruct  
  - Gemma-2-9B-it  
  - Qwen3-8B (base)

- **Ours**
  - Qwen3-8B (fine-tuned on the DX7 dataset)

### 6.3 Metrics

- **CLAP (↑)**: text–audio similarity (caption vs. generated audio).  
- **CLAP_synth (↑)**: similarity between generated and ground-truth DX7 audio.  
- **KL (↓)**: KL divergence between embedding distributions (generated vs. target).  
- **FAD (↓)**: Fréchet Audio Distance with Patch-out Transformer embeddings.  
- **KAD (↓)**: Kernel Audio Distance.  
- **AC (↓)**: error between timbral attribute vectors (e.g., hardness, brightness).  
- **RMS (↑)**: cosine similarity between frame-wise RMS envelopes.

---

## 7. Results

### 7.1 Main Comparison (Test Set)

| Model                     | CLAP ↑ | CLAP_synth ↑ | KL ↓  | FAD ↓ | KAD ↓ | AC ↓  | RMS ↑ |
|---------------------------|--------|--------------|-------|-------|-------|-------|-------|
| Test-data                 | 0.26   | 0.30         | –     | –     | –     | –     | –     |
| **Text-to-Audio Models**  |        |              |       |       |       |       |       |
| AudioGen                  | 0.17   | 0.17         | 1.33  | 7.93  | 6.96  | 0.19  | 0.57  |
| AudioLDM                  | 0.28   | 0.29         | 1.28  | 6.29  | 8.06  | 0.18  | 0.64  |
| **Zero-shot LLMs**        |        |              |       |       |       |       |       |
| Gemini-2.0-flash          | 0.12   | 0.13         | 3.82  | 7.56  | 13.13 | 0.30  | 0.55  |
| Gemini-2.5-flash          | 0.18   | 0.21         | 4.02  | 6.20  | 12.44 | 0.26  | 0.40  |
| LLaMA3-8B-Instruct        | 0.12   | 0.12         | 7.55  | 19.87 | 59.59 | 0.48  | 0.01  |
| Gemma-2-9B-it             | 0.12   | 0.11         | 7.40  | 17.14 | 55.43 | 0.48  | 0.02  |
| Qwen3-8B (base)           | 0.06   | 0.08         | 5.10  | 7.25  | 13.73 | 0.36  | 0.56  |
| **Fine-tuned LLM**        |        |              |       |       |       |       |       |
| Qwen3-8B (fine-tuned)     | 0.20   | 0.24         | 1.04  | 1.14  | 0.51  | 0.19  | 0.75  |

### 7.2 Key Observations

- Zero-shot LLMs often fail to produce valid or meaningful patches → poor KL/FAD/KAD.  
- Fine-tuned Qwen3-8B:
  - Strongly improves KL, FAD, and KAD.  
  - Achieves the best FAD, KAD, and RMS among all methods.  
  - CLAP/CLAP_synth scores move closer to the test-data reference.

Compared to AudioLDM and AudioGen, the fine-tuned LLM emphasizes **accurate reconstruction of DX7-style timbre and envelope**, while maintaining reasonable text–audio alignment.

---

## 8. Limitations

- **Caption quality**  
  - ALM-generated captions can miss subtle timbral details or mislabel instruments.  
  - Puts an upper bound on supervision quality.

- **Memorization risk**  
  - Supervised fine-tuning on a finite set of patches may lead to memorization rather than generalizable synthesis knowledge.

- **Lack of explicit reasoning**  
  - The model directly maps caption → patch, without exposed intermediate design steps (e.g., operator topology choice, modulation plan).

- **Evaluation coverage**  
  - CLAP-based metrics do not fully capture detailed timbre or playability.  
  - Current evaluation is in-domain (DX7, single pitch); out-of-domain and multi-pitch behavior is not yet tested.

---

## 9. Future Work / TODO

- **Improve labels**
  - Manually verify and correct captions, at least for test samples.  
  - Consider LLM-as-a-judge pipelines to filter or refine pseudo captions.

- **More diverse evaluation**
  - Build an out-of-domain test set:
    - human-written prompts,  
    - non-DX7 target sounds approximated by DX7 patches,  
    - multiple pitches and dynamics.

- **Chain-of-thought sound design**
  - Collect data where intermediate reasoning is explicit:
    - structural choices (operator graph, carrier/modulator roles),  
    - envelope and timbre tuning steps.  
  - Use MIR features and metadata as additional context.

- **RL with tool feedback**
  - Treat the DX7 engine as a black-box environment and apply RL / preference optimization using:
    - CLAP / CLAP_synth,  
    - FAD / KAD,  
    - AC / RMS  
    as reward components.

- **Multi-tool, multi-turn system**
  - Combine DX7 with other synths and FX tools.  
  - Support conversational refinement of patches in an interactive, multi-turn setup.


## IMPORTANT CHANGE
- WE CHANGED EVERYHING RELATED PYTHON FORMATTING TO JSON(TO FOLLOW GENERAL TOOL CALLING SCHEMES)



## 10. Ongoing Work

### Chain-of-thought reasoning path generation w/ frontier LLM

**Goal.**  
Generate chain-of-thought (CoT) trajectories for DX7 patch design that are:
- tightly grounded in actual audio behaviour via operator ablations,
- formatted as **pseudo reasoning traces** that a text-only student LLM can imitate at train/inference time,
- scalable over many patches without exploding sequence length or requiring interactive audio feedback.

---

#### Key design choice: operator-level ablation, single-shot CoT

We avoid modelling fine-grained parameter edit sequences (e.g., ratio 2.0→3.0, attack 10→5, …).  
Instead, we:

- work at the **operator / branch level** (OP1–OP6, or small branches of the modulation graph), and  
- use a **single-shot frontier ALM prompt** that asks for:
  1. an **analysis section** (full-knowledge, audio-grounded),  
  2. a **pseudo CoT block** (`\cot{...}`) that *pretends* to derive the patch from the prompt + ablations, and  
  3. the **final DX7 patch JSON** (value-identical to the ground-truth).

The **analysis** is allowed to be explicitly diagnostic (“FULL vs OP2_OFF loses metallic attack”), while the **CoT** is written as if a student model is reasoning its way to the solution, using the same cues but without “knowing” the answer a priori.

---

#### Ablation semantics: “turning an operator off”

We treat the DX7 algorithm as a directed modulation graph over 6 operators (`modmatrix` + `outmatrix`).

For an operator OP\_k, **ablation (“OPk\_OFF”)** is defined as:

- remove its **audio contribution**:
  - the operator’s direct output is muted (effectively `ol[k] = 0`, so it no longer acts as a carrier),
  - its modulation outputs are removed so that it no longer modulates downstream operators,
- **no new connections** are created to compensate for removal.

Example:

- Original graph: `OP2 → OP1 → (audio out)`.
- Ablating OP2:
  - we do **not** create a new edge (e.g., `OP1 → OP3`);
  - OP1 simply becomes a pure carrier without OP2’s modulation.

Thus, the difference between:
- `Audio_FULL` (full patch), and  
- `Audio_OPk_OFF` (OPk ablated)  

is attributable only to OPk (and its branch), which we use as the core evidence for non-hallucinated analysis.

---

#### Data generation pipeline (teacher side w/ frontier ALM)

For each training example (prompt, DX7 patch JSON, audio):

1. **Ground-truth patch and audio rendering**
   - Start from a natural language prompt describing the target sound.
   - Select or design a ground-truth DX7 patch in our unified JSON format.
   - Render:
     - `Audio_FULL` from the complete patch and a fixed MIDI input (pitch, velocity, duration),
     - `Audio_OPk_OFF` for a subset of operators (typically all modulators and/or non-trivial branches) using the ablation semantics above.

2. **Single-shot ALM prompting**
   - Provide the frontier ALM with:
     - the prompt text,
     - the DX7 JSON schema (key descriptions),
     - the ground-truth patch JSON (`{DX7_SPECS_JSON}`),
     - `Audio_FULL`, and the list of available ablations `{OP_ABLATION_LIST}`.
   - The prompt **explicitly specifies the required output structure**:
     1. **Analysis section** (outside `\cot{}`):  
        - Full-access analysis of `Audio_FULL` and `Audio_OPk_OFF`, plus the JSON.  
        - For each operator with ablation:
          - characterize carrier vs modulator role (from `"outmatrix"` / `"modmatrix"`),
          - describe audible change FULL vs OPk\_OFF (brightness, metallicity, body, percussiveness, sustain, noise, etc.),
          - note when differences are small or ambiguous (“OP3 contribution is subtle.”).
        - This section is allowed to be “omniscient” and purely diagnostic.
     2. **Pseudo CoT block**:  
        - A single block of text wrapped as:
          ```text
          \cot{
          ...
          }
          ```
        - Inside this block, the model **pretends not to know** the ground-truth patch and instead:
          - interprets the prompt,
          - reasons about how many carriers/modulators are needed and why,
          - uses imagined experiments with ablations (e.g., “Comparing FULL to OP2_OFF, the attack loses brightness, so OP2 should be a high-ratio modulator adding metallic attack.”),
          - incrementally justifies a plausible design of the patch structure and parameter tendencies.
        - The CoT is **pseudo reasoning**: it uses the same evidence as the analysis section, but is written as if solving the design problem, not as a post-hoc explanation of a known answer.
        - The JSON object itself is **not** printed here; only references to fields (e.g., `"coarse[0] = 1"`) may appear.
     3. **Final DX7 JSON**:  
        - Immediately after `\cot{...}`, the ALM outputs a single JSON object:
          ```json
          {
              ...
          }
          ```
        - This JSON must be **value-identical** to the ground-truth patch (keys and values unchanged); comments are allowed but do not affect values.

3. **Data extraction for student training**
   - From the ALM output, we parse:
     - the **prompt**,
     - the **pseudo CoT** (the text inside `\cot{...}`),
     - and the **final JSON** (ground-truth patch).
   - The **analysis section is discarded** for student training; it is only used to enforce good behaviour in the teacher prompt.
   - Training pairs thus have the form:
     - input: prompt (+ optional schema description),
     - output: `\cot{...}` reasoning text + final DX7 patch JSON.

The student LLM is trained purely on text, without audio, to imitate the teacher’s pseudo reasoning format and to emit a complete patch JSON in one shot.

---

#### Advantages of the revised scheme

- **Audio-grounded teacher, text-only student**
  - The frontier ALM’s analysis is constrained by actual `Audio_FULL` vs `Audio_OPk_OFF` differences and the exact JSON structure, reducing hallucination at the teacher level.
  - The student only sees the pseudo CoT + JSON pairs, but inherits the frontier model’s structured, audio-aligned reasoning style.

- **Operator-level abstraction, bounded sequence length**
  - We operate at the operator/branch level, not at individual parameter edit sequences, avoiding combinatorial explosion and excessively long CoTs.
  - Each CoT focuses on a small number of carriers and modulators with clear roles.

- **Clear separation of analysis vs pseudo reasoning**
  - The analysis section can be as precise and diagnostic as needed to “pin down” each operator’s contribution.
  - The `\cot{...}` block is explicitly framed as a **pseudo reasoning path**, which the student LLM is trained to emulate (not as ground-truth audio analysis).

- **Inference-time compatibility**
  - At inference, the student model receives **only text** (prompt + schema) and must output:
    - a CoT-style reasoning trace in the same `\cot{...}` format, and
    - a final DX7 patch JSON.
  - No iterative edit–listen loops are required, and no audio feedback is assumed at inference.

- **Alignment with interpretable patch structure**
  - Roles like “adds metallic attack”, “thickens low mids”, “provides soft sustain body” are explicitly tied to operator indices and modulation topology.
  - This yields CoTs that are both semantically meaningful and directly actionable in terms of DX7 parameters.