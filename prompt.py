"""
Prompt templates used in LLMforFM.

Notes
- These templates are intended to be filled with Python's `.format(...)`.
- Any literal JSON braces inside the templates are escaped as `{{` / `}}` so `.format`
  can be used safely.
- Audio is provided to the model as an *audio input part* (e.g., Gemini API inline_data),
  not pasted into the text prompt itself.
"""

base_prompt = """Find the parameter values for the famous 6-OP(operator) FM synthesizer, the DX7, such that the resulting sound matches the given prompt.
"""

# Full schema version (kept for convenience when you want to heavily constrain output format).
zeroshot_schema_prompt = base_prompt + """
Make sure to follow the json dictionary format given below.
```json
{{
    # parameters are indexed from OP1(index 0) to OP6(index 5)
    "modmatrix": [[int ∈ {{0,1}}, ...] * 6] * 6, # 6x6 binary matrix. modmatrix[i][j] = 1 means OP(i+1) is modulated by OP(j+1). Diagonal 1 means feedback(at most one OP can have feedback).
    "outmatrix": [int ∈ {{0,1}}] * 6,            # Binary flags for which operators send output to the final mix (1=active carrier).
    "feedback": int ∈ [0, 7],                    # Feedback intensity for the operator with self-modulation (modmatrix[i][i] = 1).
    "fixed_freq": [int ∈ [0, 1]] * 6,            # Binary flags for which operators have a fixed frequency(not affected by key). when set to 1, operator's frequency is fixed to 10**(coarse%4 + fine/10). detune is ignored.
    "coarse": [int ∈ [0, 31]] * 6,               # Coarse frequency ratios. Integer values determining harmonic relationship. 0 means 0.5 particularly.
    "fine": [int ∈ [0, 99]] * 6,                 # Fine frequency offset. Adds fractional harmonic variation.
    "detune": [int ∈ [-7, 7]] * 6,               # Pitch offset in cents. -7 ~ +7 range to enrich timbre.
    "transpose": int ∈ [-24, 24],                # Global pitch shift in semitones.
    "ol": [int ∈ [0, 99]] * 6,                   # Output Level per OP. Controls amplitude of each operator.
    "eg_rate": [
        [int ∈ [0, 99]] * 6,  # Rate 1 (Attack Rate)
        [int ∈ [0, 99]] * 6,  # Rate 2 (Decay 1)
        [int ∈ [0, 99]] * 6,  # Rate 3 (Sustain rate or Decay 2)
        [int ∈ [0, 99]] * 6   # Rate 4 (Release rate)
    ],  # Envelope generator speeds, high rate means faster.
    "eg_level": [
        [int ∈ [0, 99]] * 6,  # Level 1 (Attack Level)
        [int ∈ [0, 99]] * 6,  # Level 2 (Decay Level)
        [int ∈ [0, 99]] * 6,  # Level 3 (Sustain Level)
        [int ∈ [0, 99]] * 6   # Level 4 (Release Level)
    ],  # Envelope generator target levels, high level means louder.
    "sensitivity": [int ∈ [0, 7]] * 6            # Velocity sensitivity per operator. 0 (none) to 7 (max)
}}
```
### Prompt: {prompt}
"""

# Minimal version (kept as the default, to avoid over-guiding the model).
zeroshot_prompt = base_prompt + "### Prompt: {prompt}"

# Teacher-side prompt for generating (analysis + pseudo-CoT + final JSON) given:
# - one concatenated audio clip (FULL + OPk_OFF segments),
# - the prompt text,
# - the ground-truth patch JSON,
# - and metadata describing the time spans of each ablation segment.
#
# This is adapted from `research_progress.md` (405–617), but updated to use a single audio input
# and explicit time spans instead of providing separate audio clips per ablation.
cot_generation_prompt = r"""You are an Audio-Language Model (ALM) and an expert in FM synthesis, especially the 6-operator Yamaha DX7.

Your job is to produce, in a SINGLE response, (1) an analysis section, (2) a pseudo Chain-of-Thought (CoT) reasoning trace, and (3) the final DX7 patch JSON.

- The analysis section may use full knowledge of the ground-truth patch and all audio.
- The CoT section must simulate how a student model would reason from PROMPT only (text-only; no audio at inference).
- The final JSON must exactly match the given DX7 patch values (keys and values must be identical) and must be valid JSON (no comments).

This data will be used to train a text-only model that designs DX7 patches in one shot (no iterative edits, no audio at inference).

======================================================================
INPUTS
======================================================================

1. PROMPT

This is the user’s text description of the target sound:

PROMPT:
{PROMPT_TEXT}


2. DX7 PATCH JSON FORMAT

A DX7 patch is represented as a JSON object with the following structure:

```json
{{
    # parameters are indexed from OP1 (index 0) to OP6 (index 5)
    "modmatrix": [[int ∈ {{0,1}}, ...] * 6] * 6,  # 6x6 binary matrix. modmatrix[i][j] = 1 means OP(i+1) is modulated by OP(j+1). A diagonal 1 means self-feedback (at most one OP can have feedback).
    "outmatrix": [int ∈ {{0,1}}] * 6,             # Binary flags for which operators send output directly to the final mix (1 = active carrier).
    "feedback": int ∈ [0, 7],                     # Feedback intensity for the operator with self-modulation (modmatrix[i][i] = 1).
    "fixed_freq": [int ∈ [0, 1]] * 6,             # Binary flags for fixed frequency (not affected by key). When set to 1, the operator's frequency is fixed to 10**(coarse%4 + fine/10). detune is ignored.
    "coarse": [int ∈ [0, 31]] * 6,                # Coarse frequency ratios. Integer values determining harmonic relationship. 0 means 0.5.
    "fine": [int ∈ [0, 99]] * 6,                  # Fine frequency offset. Adds fractional harmonic variation.
    "detune": [int ∈ [-7, 7]] * 6,                # Pitch offset in detune steps. -7 ~ +7 range to enrich timbre.
    "transpose": int ∈ [-24, 24],                 # Global pitch shift in semitones.
    "ol": [int ∈ [0, 99]] * 6,                    # Output Level per operator. Controls amplitude of each operator.
    "eg_rate": [
        [int ∈ [0, 99]] * 6,  # Rate 1 (Attack Rate)
        [int ∈ [0, 99]] * 6,  # Rate 2 (Decay 1)
        [int ∈ [0, 99]] * 6,  # Rate 3 (Sustain rate or Decay 2)
        [int ∈ [0, 99]] * 6   # Rate 4 (Release rate)
    ],                                            # Envelope generator speeds; high rate means faster changes.
    "eg_level": [
        [int ∈ [0, 99]] * 6,  # Level 1 (Attack Level)
        [int ∈ [0, 99]] * 6,  # Level 2 (Decay Level)
        [int ∈ [0, 99]] * 6,  # Level 3 (Sustain Level)
        [int ∈ [0, 99]] * 6   # Level 4 (Release Level)
    ],                                            # Envelope generator target levels; higher level means louder.
    "sensitivity": [int ∈ [0, 7]] * 6             # Velocity sensitivity per operator. 0 (none) to 7 (max).
}}
```


3. PATCH_JSON (GROUND-TRUTH)

For this training example, the ground-truth DX7 patch that matches the prompt is provided as a JSON object:

PATCH_JSON:
{DX7_SPECS_JSON}

This JSON object is the correct tool call that renders the target sound.

IMPORTANT:
- All numeric values (and all key–value pairs) are correct and must NOT be changed.
- In your final JSON output, the values MUST be exactly identical to the JSON object shown above.
- The final JSON must be valid JSON (no comments, no trailing commas).


4. AUDIO CLIP (ONE CONCATENATED FILE: FULL + OPERATOR ABLATIONS)

You are given ONE audio clip as an audio input part in the API call (it is not embedded in this text):

- Audio_COMBINED: a single audio clip containing multiple 10.0-second segments concatenated back-to-back.

Segment semantics:
- The first segment is always the full patch:
  - Audio_FULL is the time span from t=0.0s to t=10.0s in Audio_COMBINED.
- All other segments are operator ablations (OPk_OFF), each also exactly 10.0 seconds long.
- IMPORTANT: The OPk_OFF segments appear in a RANDOMIZED ORDER in Audio_COMBINED.
  Do NOT assume OP1_OFF comes before OP2_OFF, etc. Re-ordering by operator index can be important when writing a coherent analysis.

Operator ablation (“OPk_OFF”) is defined precisely as:
- The k-th operator’s direct audio output is muted (its output level is effectively 0 so it no longer acts as a carrier).
- The k-th operator’s modulation outputs are removed: it no longer modulates any downstream operator.
- No new modulation connections are created to replace it.
- All other operators and parameters are identical to the FULL patch.

Thus, the difference between Audio_FULL and Audio_OPk_OFF isolates the audible contribution of operator k (and its branch) to the final sound.

You are given a list of the available ablations as metadata. Each item includes the segment label (e.g., OP2_OFF) and the time span (in seconds) of that segment inside Audio_COMBINED:

[AVAILABLE_ABLATIONS]
SEGMENT_MAP:
{OP_ABLATION_LIST}

Notes:
- The order of items in the list above is not guaranteed to be sorted (it may be shuffled).
- The order of OPk_OFF segments in Audio_COMBINED is randomized.
- Use the provided time spans (and Audio_FULL at 0.0s–10.0s) as the ONLY evidence when you discuss operator contributions.


======================================================================
OUTPUT REQUIREMENTS
======================================================================

Your response MUST consist of THREE parts, in this exact order:

1) An analysis section (outside of \cot{{...}}).
2) A single Chain-of-Thought (CoT) reasoning block wrapped in a LaTeX-style macro:

   \cot{{
   ...
   }}

3) Immediately after the closing brace of \cot{{...}}, the COMPLETE DX7 PATCH JSON:

   {{
       ...
   }}

Additional constraints:

- The **analysis section** (Part 1) appears as plain text before the \cot{{...}} block.
- The **CoT block** (Part 2) must contain ONLY the pseudo reasoning, no JSON object.
- The **final JSON** (Part 3) must be a single JSON object whose keys and values are EXACTLY the same as the ground-truth DX7 patch JSON shown above.
  The final JSON must be valid JSON (no comments, no trailing commas).

Do NOT wrap the JSON in backticks or any other markup.
Do NOT add any extra text before or after the JSON object.


======================================================================
DETAILED INSTRUCTIONS
======================================================================

----------------------
1. Analysis section
----------------------

In the analysis section (before \cot{{...}}), you may:

- Use full knowledge of:
  - PROMPT,
  - PATCH_JSON,
  - Audio_FULL (0.0s–10.0s in Audio_COMBINED),
  - and the OPk_OFF segments specified by SEGMENT_MAP.
- Explicitly describe what you observe when comparing FULL vs OPk_OFF:
  - Which operators behave as carriers vs modulators (based on "outmatrix" and "modmatrix").
  - How each operator changes the sound when ablated (e.g., loss of brightness, metallicity, body, etc.).
  - How envelope and sensitivity settings shape the temporal and dynamic behaviour.
- If the difference between FULL and OPk_OFF is small or ambiguous, you should say so explicitly
  (e.g., “The difference between FULL and OP3_OFF is subtle; OP3 seems to make only a minor contribution.”).

This section is allowed to be directly “analytic”.
You do NOT need to pretend you do not know the ground-truth patch here.
Think of it as your own internal verification that the patch and ablations are consistent.


----------------------
2. CoT block (\cot{{...}})
----------------------

Inside the \cot{{...}} block, you must generate a **pseudo Chain-of-Thought** that simulates how a student model would reason to arrive at the correct patch from PROMPT only (text-only; no audio at inference):

- You should **pretend that you do not already know** the ground-truth JSON as the answer.
- You may still refer to:
  - PROMPT,
  - the general DX7 format (what parameters exist, what operators are),
  - general FM sound-design heuristics (carrier/modulator roles, ratios, envelopes, feedback, modulation depth).
- The CoT should show a plausible reasoning path such as:
  - interpreting the prompt (e.g., “I need a bright percussive electric piano tone”),
  - deciding how many carriers are needed and why,
  - deciding which operators should be modulators and what ratios they likely use,
  - choosing envelopes and sensitivity to match the described attack/decay/sustain/release,
  - refining levels and modulation depth to match the described timbre (bright/dull, metallic/soft, etc.).

Important constraints for the CoT block:

- ALL content for the reasoning must be inside a single block:

  \cot{{
  ...
  }}

- Do NOT explicitly mention that you already know the ground-truth patch or that you are simulating.
  Instead, write as if you are actually solving the design problem using PROMPT only.
- Do NOT reference any audio inputs or audio-derived evidence inside \cot{{...}} (Audio_FULL, Audio_COMBINED, OPk_OFF/OPk_ON segments, or SEGMENT_MAP).
  If you describe an imagined test, phrase it generically and purely text-based (e.g., “to increase brightness, I can raise a modulator ratio or modulation depth…”).
- If you are unsure about an operator's role, you may say it plays a minor supportive role, but keep it in the style of someone exploring the solution.
- Do NOT output the full JSON inside \cot{{...}}. You may refer to specific fields (e.g., "coarse[0] = 1", "outmatrix[2] = 1") in text form, but the structured JSON object must not appear there.
- The CoT should be reasonably detailed and multi-step, not a 1–2 sentence summary. We want diverse, rich reasoning traces that are aligned with your internal reasoning style.


----------------------
3. Final DX7 JSON
----------------------

After the closing brace of the \cot{{...}} block, output a single JSON object:

{{
    ...
}}

Requirements:

- This JSON must be **value-identical** to PATCH_JSON:
  - same keys,
  - same values (numbers, arrays, nested arrays, etc.).
- The final JSON must be valid JSON (no comments, no trailing commas).
- Do NOT wrap this JSON in backticks or any other markup.
- Do NOT add any extra explanation after the JSON.

======================================================================
SUMMARY
======================================================================

- Analysis section: honest, full-knowledge analysis of FULL and ablated segments vs the patch JSON.
- CoT section (\cot{{...}}): pseudo reasoning path, as if discovering the patch from PROMPT only (no audio/segment references), without admitting that the answer is known.
- Final JSON: exact ground-truth DX7 patch, with identical key–value content to PATCH_JSON.
"""

# Same teacher prompt, but using OPk_ON segments instead of OPk_OFF segments.
# OPk_ON keeps OPk and all operators that are *downstream* of OPk (transitive closure),
# i.e., OPk -> ... following modulation edges, disabling everything else.
cot_generation_prompt_opk_on = r"""You are an Audio-Language Model (ALM) and an expert in FM synthesis, especially the 6-operator Yamaha DX7.

Your job is to produce, in a SINGLE response, (1) an analysis section, (2) a pseudo Chain-of-Thought (CoT) reasoning trace, and (3) the final DX7 patch JSON.

- The analysis section may use full knowledge of the ground-truth patch and all audio.
- The CoT section must simulate how a student model would reason from PROMPT only (text-only; no audio at inference).
- The final JSON must exactly match the given DX7 patch values (keys and values must be identical) and must be valid JSON (no comments).

This data will be used to train a text-only model that designs DX7 patches in one shot (no iterative edits, no audio at inference).

======================================================================
INPUTS
======================================================================

1. PROMPT

This is the user’s text description of the target sound:

PROMPT:
{PROMPT_TEXT}


2. DX7 PATCH JSON FORMAT

A DX7 patch is represented as a JSON object with the following structure:

```json
{{
    # parameters are indexed from OP1 (index 0) to OP6 (index 5)
    "modmatrix": [[int ∈ {{0,1}}, ...] * 6] * 6,  # 6x6 binary matrix. modmatrix[i][j] = 1 means OP(i+1) is modulated by OP(j+1). A diagonal 1 means self-feedback (at most one OP can have feedback).
    "outmatrix": [int ∈ {{0,1}}] * 6,             # Binary flags for which operators send output directly to the final mix (1 = active carrier).
    "feedback": int ∈ [0, 7],                     # Feedback intensity for the operator with self-modulation (modmatrix[i][i] = 1).
    "fixed_freq": [int ∈ [0, 1]] * 6,             # Binary flags for fixed frequency (not affected by key). When set to 1, the operator's frequency is fixed to 10**(coarse%4 + fine/10). detune is ignored.
    "coarse": [int ∈ [0, 31]] * 6,                # Coarse frequency ratios. Integer values determining harmonic relationship. 0 means 0.5.
    "fine": [int ∈ [0, 99]] * 6,                  # Fine frequency offset. Adds fractional harmonic variation.
    "detune": [int ∈ [-7, 7]] * 6,                # Pitch offset in detune steps. -7 ~ +7 range to enrich timbre.
    "transpose": int ∈ [-24, 24],                 # Global pitch shift in semitones.
    "ol": [int ∈ [0, 99]] * 6,                    # Output Level per operator. Controls amplitude of each operator.
    "eg_rate": [
        [int ∈ [0, 99]] * 6,  # Rate 1 (Attack Rate)
        [int ∈ [0, 99]] * 6,  # Rate 2 (Decay 1)
        [int ∈ [0, 99]] * 6,  # Rate 3 (Sustain rate or Decay 2)
        [int ∈ [0, 99]] * 6   # Rate 4 (Release rate)
    ],                                            # Envelope generator speeds; high rate means faster changes.
    "eg_level": [
        [int ∈ [0, 99]] * 6,  # Level 1 (Attack Level)
        [int ∈ [0, 99]] * 6,  # Level 2 (Decay Level)
        [int ∈ [0, 99]] * 6,  # Level 3 (Sustain Level)
        [int ∈ [0, 99]] * 6   # Level 4 (Release Level)
    ],                                            # Envelope generator target levels; higher level means louder.
    "sensitivity": [int ∈ [0, 7]] * 6             # Velocity sensitivity per operator. 0 (none) to 7 (max).
}}
```


3. PATCH_JSON (GROUND-TRUTH)

For this training example, the ground-truth DX7 patch that matches the prompt is provided as a JSON object:

PATCH_JSON:
{DX7_SPECS_JSON}

This JSON object is the correct tool call that renders the target sound.

IMPORTANT:
- All numeric values (and all key–value pairs) are correct and must NOT be changed.
- In your final JSON output, the values MUST be exactly identical to the JSON object shown above.
- The final JSON must be valid JSON (no comments, no trailing commas).


4. AUDIO CLIP (ONE CONCATENATED FILE: FULL + OPERATOR ISOLATIONS)

You are given ONE audio clip as an audio input part in the API call (it is not embedded in this text):

- Audio_COMBINED: a single audio clip containing multiple 10.0-second segments concatenated back-to-back.

Segment semantics:
- The first segment is always the full patch:
  - Audio_FULL is the time span from t=0.0s to t=10.0s in Audio_COMBINED.
- All other segments are operator isolations (OPk_ON), each also exactly 10.0 seconds long.
- IMPORTANT: The OPk_ON segments appear in a RANDOMIZED ORDER in Audio_COMBINED.
  Do NOT assume OP1_ON comes before OP2_ON, etc. Re-ordering by operator index can be important when writing a coherent analysis.

Operator isolation (“OPk_ON”) is defined precisely as:
- Start from the full ground-truth patch JSON shown above.
- Keep ONLY operator k and all operators that are downstream of operator k, recursively, via modulation:
  - Edge rule: if "modmatrix[i][j] = 1", then OP(j+1) modulates OP(i+1). This is a directed edge OP(j+1) -> OP(i+1).
  - OPk_ON keeps the transitive closure of operators reachable from OPk by repeatedly following these edges forward (downstream).
    Example: if OP1 -> OP2 -> OP3, then:
      - OP3_ON keeps OP3 only
      - OP2_ON keeps OP2 and OP3
      - OP1_ON keeps OP1, OP2, and OP3
- Output routing (important):
  - Operators not kept are disconnected from the final mix (outmatrix=0).
  - For kept operators, their original outmatrix values are preserved (carriers stay carriers).
  - If the kept downstream subgraph contains no carriers (outmatrix sum becomes 0), then OPk_ON would be silent and OPk does not contribute to the final sound.
    In this data generation setup, such OPk_ON segments may be omitted from the available list instead of forcing an artificial output connection.
- All other operators are disabled:
  - their direct audio output is muted (they do not act as carriers),
  - their modulation outputs are removed (they do not modulate any other operator),
  - no new modulation connections are created to replace removed operators.

Thus, OPk_ON preserves the original modulation topology *within* the OPk dependency subgraph, and removes everything else.

You are given a list of the available OPk_ON segments as metadata. Each item includes the segment label (e.g., OP2_ON) and the time span (in seconds) of that segment inside Audio_COMBINED:

[AVAILABLE_ISOLATIONS]
SEGMENT_MAP:
{OP_ABLATION_LIST}

Notes:
- The order of items in the list above is not guaranteed to be sorted (it may be shuffled).
- The order of OPk_ON segments in Audio_COMBINED is randomized.
- Use the provided time spans (and Audio_FULL at 0.0s–10.0s) as the ONLY evidence when you discuss operator contributions.


======================================================================
OUTPUT REQUIREMENTS
======================================================================

Your response MUST consist of THREE parts, in this exact order:

1) An analysis section (outside of \cot{{...}}).
2) A single Chain-of-Thought (CoT) reasoning block wrapped in a LaTeX-style macro:

   \cot{{
   ...
   }}

3) Immediately after the closing brace of \cot{{...}}, the COMPLETE DX7 PATCH JSON:

   {{
       ...
   }}

Additional constraints:

- The **analysis section** (Part 1) appears as plain text before the \cot{{...}} block.
- The **CoT block** (Part 2) must contain ONLY the pseudo reasoning, no JSON object.
- The **final JSON** (Part 3) must be a single JSON object whose keys and values are EXACTLY the same as the ground-truth DX7 patch JSON shown above.
  The final JSON must be valid JSON (no comments, no trailing commas).

Do NOT wrap the JSON in backticks or any other markup.
Do NOT add any extra text before or after the JSON object.


======================================================================
DETAILED INSTRUCTIONS
======================================================================

----------------------
1. Analysis section
----------------------

In the analysis section (before \cot{{...}}), you may:

- Use full knowledge of:
  - PROMPT,
  - PATCH_JSON,
  - Audio_FULL (0.0s–10.0s in Audio_COMBINED),
  - and the OPk_ON segments specified by SEGMENT_MAP.
- Explicitly describe what you observe when comparing FULL vs OPk_ON:
  - Which operators behave as carriers vs modulators (based on "outmatrix" and "modmatrix").
  - What remains when isolating a given OPk downstream subgraph (e.g., which partial/branch of the timbre/envelope is preserved).
  - If an OPk_ON segment is very quiet or nearly silent, say so explicitly (this can happen if the remaining output levels/envelopes are very low/short, or if OPk_ON is nearly silent because the downstream subgraph has little audible output).

This section is allowed to be directly “analytic”.
You do NOT need to pretend you do not know the ground-truth patch here.
Think of it as your own internal verification that the patch and isolation semantics are consistent.


----------------------
2. CoT block (\cot{{...}})
----------------------

Inside the \cot{{...}} block, you must generate a **pseudo Chain-of-Thought** that simulates how a student model would reason to arrive at the correct patch from PROMPT only (text-only; no audio at inference):

- You should **pretend that you do not already know** the ground-truth JSON as the answer.
- You may still refer to:
  - PROMPT,
  - the general DX7 format (what parameters exist, what operators are),
  - general FM sound-design heuristics (carrier/modulator roles, ratios, envelopes, feedback, modulation depth).
- The CoT should show a plausible reasoning path such as:
  - interpreting the prompt,
  - deciding which operator branches are likely needed for the described timbre and envelope,
  - using FM synthesis principles to justify carrier/modulator roles and parameter tendencies (ratios, envelopes, levels).

Important constraints for the CoT block:

- ALL content for the reasoning must be inside a single block:

  \cot{{
  ...
  }}

- Do NOT explicitly mention that you already know the ground-truth patch or that you are simulating.
- Do NOT reference any audio inputs or audio-derived evidence inside \cot{{...}} (Audio_FULL, Audio_COMBINED, OPk_OFF/OPk_ON segments, or SEGMENT_MAP).
  If you describe an imagined test, phrase it generically and purely text-based (e.g., “to get a bell-like tone, I can use higher modulator ratios and a fast decay…”).
- Do NOT output the full JSON inside \cot{{...}}. You may refer to specific fields (e.g., "coarse[0] = 1", "outmatrix[2] = 1") in text form, but the structured JSON object must not appear there.
- The CoT should be reasonably detailed and multi-step, not a 1–2 sentence summary.


----------------------
3. Final DX7 JSON
----------------------

After the closing brace of the \cot{{...}} block, output a single JSON object:

{{
    ...
}}

Requirements:

- This JSON must be **value-identical** to PATCH_JSON:
  - same keys,
  - same values (numbers, arrays, nested arrays, etc.).
- The final JSON must be valid JSON (no comments, no trailing commas).
- Do NOT wrap this JSON in backticks or any other markup.
- Do NOT add any extra explanation after the JSON.

======================================================================
SUMMARY
======================================================================

- Analysis section: honest, full-knowledge analysis of FULL and OPk_ON segments vs the patch JSON.
- CoT section (\cot{{...}}): pseudo reasoning path, as if discovering the patch from PROMPT only (no audio/segment references), without admitting that the answer is known.
- Final JSON: exact ground-truth DX7 patch, with identical key–value content to PATCH_JSON.
"""