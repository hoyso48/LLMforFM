from dx7.pydx7 import dx7_synth, midi_note
import numpy as np
import ast
import json
import re

def render_from_specs(specs: dict, sr=48000, n=60, v=100, out_scale=1.0):
    synth = dx7_synth(specs, sr=sr, block_size=64) # block_size can be adjusted

    # Create a note sequence: one note, velocity 100, on for 1.0 seconds, off for 0.5 seconds
    # Ensure ton and toff are integers for frame counts
    ton_frames = int(sr * 0.1) 
    toff_frames = int(sr * 0.05)
    note = midi_note(n=n, v=v, ton=ton_frames, toff=toff_frames, silence=0)
    audio = synth.render_from_midi_sequence([note])

    final_audio = (audio * 32767 * out_scale).astype(np.int16)
    return final_audio

def serialize_specs(specs: dict) -> str:
    """Serialize specs into JSON (handles numpy types)."""

    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported type for JSON serialization: {type(value)}")

    return json.dumps(specs, default=_json_default, indent=2)

def validate_specs(specs, syx_file='', patch_number=-1):
    valid = True
    #it's ok if name is not present or empty
    if 'name' not in specs:
        # print(f"[WARNING] {syx_file}: patch {patch_number}: 'name' is not present.")
        patch_name = 'NaN'
    elif specs['name'] == '':
        # print(f"[WARNING] {syx_file}: patch {patch_number}: 'name' is empty.")
        patch_name = 'NaN'
    else:
        patch_name = specs['name']
    
    if not isinstance(patch_name, str):
        print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: 'name' is not a string.")
        patch_name = 'NaN'

    def check_range(name, value, lo, hi):
        if not lo <= value <= hi:
            print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}' = {value} is out of range [{lo}, {hi}]")
            valid = False

    def check_list_range(name, lst, lo, hi):
        for idx, v in enumerate(lst):
            if not lo <= v <= hi:
                print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}[{idx}]' = {v} is out of range [{lo}, {hi}]")
                valid = False

    def check_matrix_range(name, matrix, lo, hi):
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if not lo <= v <= hi:
                    print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}[{i}][{j}]' = {v} is out of range [{lo}, {hi}]")
                    valid = False

    def check_list_shape(name, lst, shape):
        if len(lst) != shape[0]:
            print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}' has incorrect shape. Expected {shape}, got {len(lst)}")
            valid = False
        if len(shape) == 2:
            for sub_list in lst:
                if len(sub_list) != shape[1]:
                    print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}' has incorrect shape. Expected {shape}, got {len(sub_list)}")
                    valid = False
    
    # Validate all fields
    check_matrix_range("modmatrix", specs['modmatrix'], 0, 1)

    feedback_count = sum([specs['modmatrix'][i][i] for i in range(6)])
    if feedback_count > 1:
        print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: multiple operators have feedback.")
        valid = False

    check_list_range("outmatrix", specs['outmatrix'], 0, 1)
    check_range("feedback", specs['feedback'], 0, 7)
    check_list_range("coarse", specs['coarse'], 0, 31)
    check_list_range("fine", specs['fine'], 0, 99)
    check_list_range("detune", specs['detune'], -7, 7)
    check_range("transpose", specs['transpose'], -24, 24)
    check_list_range("ol", specs['ol'], 0, 99)
    check_list_shape("eg_rate", specs['eg_rate'], (4, 6))
    check_list_shape("eg_level", specs['eg_level'], (4, 6))
    check_list_shape("sensitivity", specs['sensitivity'], (6,))
    check_list_shape("modmatrix", specs['modmatrix'], (6, 6))
    check_list_shape("outmatrix", specs['outmatrix'], (6,))
    check_list_shape("coarse", specs['coarse'], (6,))
    check_list_shape("fine", specs['fine'], (6,))
    check_list_shape("detune", specs['detune'], (6,))
    check_list_shape("ol", specs['ol'], (6,))

    for r in range(4):
        check_list_range(f"eg_rate[{r}]", specs['eg_rate'][r], 0, 99)
        check_list_range(f"eg_level[{r}]", specs['eg_level'][r], 0, 99)

    check_list_range("sensitivity", specs['sensitivity'], 0, 7)

    #it's ok if has_fixed_freqs is not present
    if 'has_fixed_freqs' not in specs:
        # print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: 'has_fixed_freqs' is not present.")
        pass
    elif not isinstance(specs['has_fixed_freqs'], bool):
        print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: 'has_fixed_freqs' is not a boolean.")
        valid = False

    return valid


def parse_last_specs(text: str) -> dict:
    """
    Extract the last specs object from text, preferring JSON code blocks.

    Supports:
    - ```json { ... } ```
    - ``` { ... } ```
    - Any trailing JSON object starting with '{'
    Falls back to the legacy `specs = {...}` Python literal parser for
    backward compatibility.
    """

    def _try_json_block(block: str):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return None

    decoder = json.JSONDecoder()

    def _try_raw_decode(snippet: str):
        try:
            obj, _ = decoder.raw_decode(snippet)
            return obj
        except json.JSONDecodeError:
            return None

    # 1) Prefer fenced JSON blocks
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not fenced:
        fenced = re.findall(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in reversed(fenced):
        parsed = _try_json_block(block.strip())
        if parsed is not None:
            return parsed

    # 2) Scan from the last '{' backward and attempt a raw JSON decode
    brace_positions = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(brace_positions):
        candidate = text[start:].strip()
        parsed = _try_raw_decode(candidate)
        if parsed is not None:
            return parsed

    # 3) Legacy fallback: specs = {...} in Python literal form
    matches = re.findall(r"specs\s*=\s*({.*?})", text, re.DOTALL)
    if matches:
        try:
            return ast.literal_eval(matches[-1])
        except Exception as exc:
            raise ValueError(f"Failed to parse legacy specs block: {exc}") from exc

    raise ValueError("No parsable specs JSON found in text.")

def valid_char(c, invalid_chars=['/', '\\']):
    if (ord(c) < 32 or ord(c) == 127 or c in invalid_chars):
        return False
    return True

def is_invalid_name(name):
    if not isinstance(name, str):
        return True
    if any(not valid_char(c) for c in name):
        return True
    return False

def clean_name(name, replace='_NULLNAME_', nan_names = ['NULL', 'N/A', 'NaN'], length=10):
    if not isinstance(name, str):
        return replace
    elif name in nan_names:
        return replace
    # 출력 가능한 문자만 남김 (null byte, \x1c 등 제거)
    cleaned = ''.join(c for c in name if valid_char(c))
    cleaned = cleaned if cleaned else replace
    if len(cleaned) < length:
        cleaned = cleaned + '_' * (length - len(cleaned))
    return cleaned[:length]
