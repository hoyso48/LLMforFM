#!/usr/bin/env python3
"""
GRPO training for DX7 tool learning (caption -> DX7 patch JSON).

This script implements two reward modes:

1) toolrl_exact:
   - ToolRL-style reward decomposition with NO annealing:
     R_final = R_format + R_correct
   - R_format ∈ {0, 1}: 1 iff the completion contains a parsable DX7 patch JSON and passes schema/range checks.
   - R_correct ∈ [-3, 3]: ToolRL correctness reward with (tool-name, key-name, value) matching.

2) dx7_dist (our extension):
   - Dense, weighted parameter-distance reward with NO annealing:
     R = alpha * R_valid + beta * R_keys + gamma * R_dist
   - R_valid ∈ {0, 1}: 1 iff completion is a valid DX7 patch (schema + ranges), else 0.
   - R_keys ∈ [0, 1]: fraction of required DX7 keys present (partial credit).
   - R_dist ∈ [0, 1]: weighted, continuous similarity over parameters (off-by-one gets partial credit).

Notes
-----
- We intentionally avoid any time-dependent annealing / schedules (ToolRL ablations show abrupt schedules can hurt).
- We default to `beta=0.0` (no KL / no reference model) to mirror ToolRL’s GRPO setting.
- vLLM is optional. TRL warns if your vLLM version is not the supported one; to keep runs robust, default is
  `--use_vllm false`.
- LoRA is optional. By default we do **full fine-tuning**; enable LoRA explicitly with `--use_lora`.
- Qwen3 hybrid-thinking is optional:
  - `--enable_thinking` toggles thinking mode in the tokenizer chat template (if supported by your tokenizer).
  - `--append_think_control auto` appends `/think` iff `--enable_thinking` is set, else appends `/no_think`.

Run
---
Before running, activate the user's conda env:
  conda activate hoyso_ml

Example (recommended one-shot default):
  python grpo.py --model_name_or_path /path/to/your/sft_checkpoint --reward_mode dx7_dist --output_dir outputs/grpo_dx7
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import random
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

# Allow running from anywhere (not only repo root).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt import zeroshot_prompt, zeroshot_schema_prompt  # noqa: E402
from dx7.utils import parse_last_specs, validate_specs  # noqa: E402


KEY_ORDER: list[str] = [
    "modmatrix",
    "outmatrix",
    "feedback",
    "fixed_freq",
    "coarse",
    "fine",
    "detune",
    "transpose",
    "ol",
    "eg_rate",
    "eg_level",
    "sensitivity",
]


DX7_SCHEMA: dict[str, dict[str, Any]] = {
    "modmatrix": {"shape": (6, 6), "lo": 0.0, "hi": 1.0},
    "outmatrix": {"shape": (6,), "lo": 0.0, "hi": 1.0},
    "feedback": {"shape": None, "lo": 0.0, "hi": 7.0},
    "fixed_freq": {"shape": (6,), "lo": 0.0, "hi": 1.0},
    "coarse": {"shape": (6,), "lo": 0.0, "hi": 31.0},
    "fine": {"shape": (6,), "lo": 0.0, "hi": 99.0},
    "detune": {"shape": (6,), "lo": -7.0, "hi": 7.0},
    "transpose": {"shape": None, "lo": -24.0, "hi": 24.0},
    "ol": {"shape": (6,), "lo": 0.0, "hi": 99.0},
    "eg_rate": {"shape": (4, 6), "lo": 0.0, "hi": 99.0},
    "eg_level": {"shape": (4, 6), "lo": 0.0, "hi": 99.0},
    "sensitivity": {"shape": (6,), "lo": 0.0, "hi": 7.0},
}


DEFAULT_KEY_WEIGHTS: dict[str, float] = {
    # Topology (highest)
    "modmatrix": 4.0,
    "outmatrix": 4.0,
    "feedback": 3.0,
    "fixed_freq": 3.0,
    # Operator amplitude + envelopes (high)
    "ol": 2.0,
    "eg_rate": 2.0,
    "eg_level": 2.0,
    # Frequency ratios (medium)
    "coarse": 1.5,
    "fine": 1.0,
    "detune": 1.0,
    # Global / misc (low–medium)
    "transpose": 0.5,
    "sensitivity": 0.5,
}


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _setup_wandb(args: argparse.Namespace) -> None:
    """
    Non-interactive W&B configuration.

    - If `--report_to wandb` is used, we optionally set WANDB_* env vars from CLI args.
    - If `--wandb_api_key` is provided, we also call `wandb.login(key=..., relogin=True)` to
      avoid any interactive prompts.
    """
    if str(args.report_to) != "wandb":
        return

    if _is_non_empty_str(getattr(args, "wandb_mode", None)):
        os.environ["WANDB_MODE"] = str(args.wandb_mode)
    if _is_non_empty_str(getattr(args, "wandb_project", None)):
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
    if _is_non_empty_str(getattr(args, "wandb_entity", None)):
        os.environ["WANDB_ENTITY"] = str(args.wandb_entity)
    if _is_non_empty_str(getattr(args, "wandb_api_key", None)):
        os.environ["WANDB_API_KEY"] = str(args.wandb_api_key)

    # Store W&B artifacts under the run output dir by default.
    wandb_dir = Path(str(args.output_dir)) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))

    # If user provided an explicit key, login now to avoid interactive prompts later.
    if _is_non_empty_str(getattr(args, "wandb_api_key", None)):
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "W&B logging requested but `wandb` is not installed. Install `wandb` or run with --report_to none."
            ) from exc
        wandb.login(key=str(args.wandb_api_key), relogin=True)
        return

    # Otherwise, rely on existing W&B config (env, netrc, wandb settings). Warn if nothing obvious is set.
    mode = os.environ.get("WANDB_MODE", "online").lower()
    if mode == "online" and not _is_non_empty_str(os.environ.get("WANDB_API_KEY")):
        print(
            "[WARN] --report_to wandb is enabled but no --wandb_api_key / WANDB_API_KEY was provided. "
            "If W&B prompts for login, rerun with --wandb_api_key <KEY> or set --wandb_mode offline.",
            file=sys.stderr,
        )


def _is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _load_csv(path: Path, *, index_col: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, index_col=index_col)


def _parse_patch_data(patch_data: Any, *, row_id: Any) -> dict[str, Any]:
    """
    Parse `patch_data` which may be:
    - a dict,
    - a JSON object string,
    - (legacy) a Python literal dict string.
    """
    if isinstance(patch_data, dict):
        return patch_data
    if patch_data is None:
        raise ValueError(f"Row {row_id}: patch_data is missing")
    if not isinstance(patch_data, str):
        raise TypeError(f"Row {row_id}: patch_data has unsupported type {type(patch_data)}")

    candidate = patch_data.strip()
    if candidate == "":
        raise ValueError(f"Row {row_id}: patch_data is empty")

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Row {row_id}: patch_data JSON must be an object, got {type(parsed)}")
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Row {row_id}: failed to parse patch_data as JSON or Python literal: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Row {row_id}: patch_data literal must be a dict, got {type(parsed)}")
        return parsed


def _strip_and_keep_schema_keys(specs: dict[str, Any], *, keys_to_remove: list[str]) -> dict[str, Any]:
    out = dict(specs)
    for k in keys_to_remove:
        out.pop(k, None)
    # Keep only schema keys (ignore extras).
    return {k: out[k] for k in KEY_ORDER if k in out}


def _canonicalize_ground_truth(specs: dict[str, Any], *, row_id: Any, keys_to_remove: list[str]) -> dict[str, Any]:
    if not isinstance(specs, dict):
        raise TypeError(f"Row {row_id}: specs must be a dict, got {type(specs)}")
    stripped = _strip_and_keep_schema_keys(specs, keys_to_remove=keys_to_remove)
    missing = [k for k in KEY_ORDER if k not in stripped]
    if missing:
        raise ValueError(f"Row {row_id}: ground-truth patch is missing required keys: {missing}")
    canonical = {k: stripped[k] for k in KEY_ORDER}
    # Validate shapes/types first so validate_specs() can't crash on malformed arrays.
    if not _validate_discrete_dx7_schema(canonical):
        raise ValueError(f"Row {row_id}: ground-truth patch failed discrete DX7 schema validation")
    try:
        ok = bool(validate_specs(canonical, syx_file="", patch_number=-1, verbose=False))
    except (KeyError, TypeError, ValueError, IndexError):
        ok = False
    if not ok:
        raise ValueError(f"Row {row_id}: ground-truth patch failed validate_specs()")
    return canonical


def _completion_to_text(completion: Any) -> str:
    """
    TRL GRPOTrainer passes:
    - standard data: completion is a string
    - conversational data: completion is a list of messages, usually [{"role":"assistant","content": "..."}]
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg.get("content", "")))
        return "\n".join(parts)
    return str(completion)


def _try_parse_pred_patch(text: str, *, keys_to_remove: list[str]) -> dict[str, Any] | None:
    if not _is_non_empty_str(text):
        return None
    try:
        parsed = parse_last_specs(text)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _strip_and_keep_schema_keys(parsed, keys_to_remove=keys_to_remove)


def _answer_to_patch(answer_item: Any, *, keys_to_remove: list[str]) -> dict[str, Any] | None:
    """
    Convert a dataset `answer` item into a canonical DX7 patch dict.

    We support:
    - dict objects (already-parsed patches),
    - JSON strings (preferred for HF Dataset collation stability),
    - legacy Python-literal dict strings.
    """
    if isinstance(answer_item, dict):
        parsed = answer_item
    elif isinstance(answer_item, str):
        s = answer_item.strip()
        if s == "":
            return None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
        if not isinstance(parsed, dict):
            return None
    else:
        return None

    stripped = _strip_and_keep_schema_keys(parsed, keys_to_remove=keys_to_remove)
    if not _has_all_required_keys(stripped):
        return None
    return {k: stripped[k] for k in KEY_ORDER}


def _has_all_required_keys(specs: dict[str, Any]) -> bool:
    return all(k in specs for k in KEY_ORDER)


def _is_int_like(x: Any, *, tol: float = 1e-6) -> bool:
    v = _to_float(x)
    if v is None:
        return False
    return abs(v - round(v)) <= float(tol)


def _is_binary_like(x: Any, *, tol: float = 1e-6) -> bool:
    v = _to_float(x)
    if v is None:
        return False
    return abs(v - 0.0) <= float(tol) or abs(v - 1.0) <= float(tol)


def _validate_discrete_dx7_schema(specs: dict[str, Any]) -> bool:
    """
    Stricter DX7 schema check than `dx7.utils.validate_specs`.

    Rationale:
    The renderer (`dx7/pydx7.py`) treats key topology fields as discrete
    (e.g., checks `modmatrix[i,j] == 1`), so values like 0.5 must be treated as invalid.

    This function accepts int-like floats (e.g., 1.0) but rejects non-integral values.
    """
    # Shapes + numeric ranges are still checked by validate_specs(). Here we enforce discreteness.
    binary_keys = {"modmatrix", "outmatrix", "fixed_freq"}

    for k in KEY_ORDER:
        if k not in specs:
            return False
        v = specs[k]
        spec = DX7_SCHEMA[k]
        shape = spec["shape"]
        lo = float(spec["lo"])
        hi = float(spec["hi"])

        def _in_int_range(x: Any) -> bool:
            if not _is_int_like(x):
                return False
            xv = int(round(float(_to_float(x) or 0.0)))
            return lo <= float(xv) <= hi

        def _in_binary(x: Any) -> bool:
            return _is_binary_like(x)

        checker = _in_binary if k in binary_keys else _in_int_range

        if shape is None:
            if not checker(v):
                return False
            continue

        if len(shape) == 1:
            n = int(shape[0])
            if not isinstance(v, list) or len(v) != n:
                return False
            if not all(checker(elem) for elem in v):
                return False
            continue

        if len(shape) == 2:
            r, c = int(shape[0]), int(shape[1])
            if not isinstance(v, list) or len(v) != r:
                return False
            for row in v:
                if not isinstance(row, list) or len(row) != c:
                    return False
                if not all(checker(elem) for elem in row):
                    return False

            # Extra DX7-specific constraint: at most 1 diagonal feedback edge.
            if k == "modmatrix":
                diag = [v[i][i] for i in range(min(r, c))]
                fb_count = sum(int(round(float(_to_float(x) or 0.0))) for x in diag)
                if fb_count > 1:
                    return False
            continue

        return False

    return True


def _is_valid_dx7_patch(specs: dict[str, Any]) -> bool:
    if not _has_all_required_keys(specs):
        return False
    canonical = {k: specs[k] for k in KEY_ORDER}
    # Guard first: validate_specs() is not shape-safe (it can IndexError on malformed modmatrix).
    if not _validate_discrete_dx7_schema(canonical):
        return False
    try:
        return bool(validate_specs(canonical, syx_file="", patch_number=-1, verbose=False))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    denom = float(len(a.union(b)))
    if denom == 0.0:
        return 0.0
    return float(len(a.intersection(b))) / denom


def _to_float(x: Any) -> float | None:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _elem_similarity(pred: Any, true: Any, *, lo: float, hi: float) -> float:
    p = _to_float(pred)
    t = _to_float(true)
    if p is None or t is None:
        return 0.0
    denom = float(hi - lo)
    if denom <= 0.0:
        return 1.0 if p == t else 0.0
    return _clip01(1.0 - abs(p - t) / denom)


def _elem_match(pred: Any, true: Any, *, tol: float = 1e-6) -> float:
    """
    Exact(-ish) match for categorical / discrete fields.
    We treat numeric strings and int-like floats as equal (e.g., 1 == 1.0 == "1").
    """
    p = _to_float(pred)
    t = _to_float(true)
    if p is not None and t is not None:
        return 1.0 if abs(p - t) <= float(tol) else 0.0
    return 1.0 if pred == true else 0.0


def _score_with_shape(
    pred: Any,
    true: Any,
    *,
    shape: tuple[int, ...] | None,
    lo: float,
    hi: float,
    mode: str = "continuous",
) -> float:
    if mode not in {"continuous", "categorical"}:
        raise ValueError(f"Unexpected mode: {mode}")

    elem_fn = _elem_similarity if mode == "continuous" else _elem_match

    if shape is None:
        if mode == "continuous":
            return _elem_similarity(pred, true, lo=lo, hi=hi)
        return _elem_match(pred, true)

    # Vector
    if len(shape) == 1:
        n = int(shape[0])
        if not isinstance(true, list) or len(true) != n:
            # Ground-truth should always be well-formed; if not, treat as no reward.
            return 0.0
        if not isinstance(pred, list):
            return 0.0
        if mode == "continuous":
            scores = [_elem_similarity(pred[i] if i < len(pred) else None, true[i], lo=lo, hi=hi) for i in range(n)]
        else:
            scores = [_elem_match(pred[i] if i < len(pred) else None, true[i]) for i in range(n)]
        return float(sum(scores) / n) if n > 0 else 0.0

    # Matrix
    if len(shape) == 2:
        r, c = int(shape[0]), int(shape[1])
        if not isinstance(true, list) or len(true) != r:
            return 0.0
        for row in true:
            if not isinstance(row, list) or len(row) != c:
                return 0.0
        if not isinstance(pred, list):
            return 0.0

        scores: list[float] = []
        for i in range(r):
            pred_row = pred[i] if i < len(pred) and isinstance(pred[i], list) else None
            true_row = true[i]
            for j in range(c):
                pv = pred_row[j] if isinstance(pred_row, list) and j < len(pred_row) else None
                if mode == "continuous":
                    scores.append(_elem_similarity(pv, true_row[j], lo=lo, hi=hi))
                else:
                    scores.append(_elem_match(pv, true_row[j]))
        denom = float(r * c)
        return float(sum(scores) / denom) if denom > 0 else 0.0

    # Unexpected higher dims
    return 0.0


def _dx7_distance_reward(pred: dict[str, Any], true: dict[str, Any], *, key_weights: dict[str, float]) -> float:
    categorical_keys = {"modmatrix", "outmatrix", "fixed_freq"}
    total_w = 0.0
    acc = 0.0
    for k in KEY_ORDER:
        w = float(key_weights.get(k, 1.0))
        spec = DX7_SCHEMA[k]
        score_k = 0.0
        if k in pred:
            mode = "categorical" if k in categorical_keys else "continuous"
            score_k = _score_with_shape(
                pred[k],
                true[k],
                shape=spec["shape"],
                lo=float(spec["lo"]),
                hi=float(spec["hi"]),
                mode=mode,
            )
        acc += w * score_k
        total_w += w
    return (acc / total_w) if total_w > 0.0 else 0.0


# -----------------------
# Reward functions (ToolRL)
# -----------------------


def reward_toolrl_format(
    prompts: list[Any],
    completions: list[Any],
    answer: list[Any],
    *,
    keys_to_remove: list[str],
    **_: Any,
) -> list[float]:
    scores: list[float] = []
    for comp in completions:
        pred = _try_parse_pred_patch(_completion_to_text(comp), keys_to_remove=keys_to_remove)
        scores.append(1.0 if (pred is not None and _is_valid_dx7_patch(pred)) else 0.0)
    return scores


def reward_toolrl_correctness(
    prompts: list[Any],
    completions: list[Any],
    answer: list[Any],
    *,
    keys_to_remove: list[str],
    **_: Any,
) -> list[float]:
    """
    ToolRL correctness reward:
      R_correct = 6 * (Rmax / Smax) - 3  ∈ [-3, 3]
    where for our single-tool case:
      rname  = Jaccard({DX7}, {DX7} or empty)
      rparam = Jaccard(keys_true, keys_pred) over *flattened* parameter names
      rvalue = sum_k 1[pred[k] == true[k]] over flattened parameter names
      Smax   = 1 + |G| + sum |keys(G)| = 1 + 1 + 128 = 130
    """
    scores: list[float] = []
    tool_name = "dx7_patch"
    ng = {tool_name}

    def _flatten_params(specs: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten the DX7 patch into a dict of scalar parameters.

        Examples:
        - coarse[0], coarse[1], ...
        - eg_rate[0][0], ..., eg_rate[3][5]
        """
        out: dict[str, Any] = {}
        for key in KEY_ORDER:
            if key not in specs:
                continue
            v = specs[key]
            shape = DX7_SCHEMA[key]["shape"]
            if shape is None:
                out[key] = v
                continue
            if len(shape) == 1:
                if not isinstance(v, list):
                    continue
                for i, elem in enumerate(v):
                    out[f"{key}[{i}]"] = elem
                continue
            if len(shape) == 2:
                if not isinstance(v, list):
                    continue
                for i, row in enumerate(v):
                    if not isinstance(row, list):
                        continue
                    for j, elem in enumerate(row):
                        out[f"{key}[{i}][{j}]"] = elem
                continue
        return out

    for comp, gt_item in zip(completions, answer):
        gt = _answer_to_patch(gt_item, keys_to_remove=keys_to_remove)
        if gt is None:
            scores.append(-3.0)
            continue

        pred = _try_parse_pred_patch(_completion_to_text(comp), keys_to_remove=keys_to_remove)
        npred = {tool_name} if pred is not None else set()

        rname = _jaccard(ng, npred)

        gt_flat = _flatten_params(gt)
        pred_flat = _flatten_params(pred) if pred is not None else {}
        keys_true = set(gt_flat.keys())
        keys_pred = set(pred_flat.keys())
        rparam = _jaccard(keys_true, keys_pred)

        rvalue = 0.0
        for k in keys_true:
            if k in pred_flat and _elem_match(pred_flat[k], gt_flat[k]) > 0.5:
                rvalue += 1.0

        rmax = rname + rparam + rvalue
        smax = 1.0 + 1.0 + float(len(keys_true))
        rcorrect = 6.0 * (rmax / smax) - 3.0
        scores.append(float(rcorrect))
    return scores


# ----------------------------
# Reward functions (DX7 dense)
# ----------------------------


def reward_dx7_valid(
    prompts: list[Any],
    completions: list[Any],
    answer: list[Any],
    *,
    keys_to_remove: list[str],
    alpha: float,
    invalid_reward: float,
    **_: Any,
) -> list[float]:
    scores: list[float] = []
    for comp in completions:
        pred = _try_parse_pred_patch(_completion_to_text(comp), keys_to_remove=keys_to_remove)
        if pred is None:
            scores.append(float(invalid_reward))
        else:
            scores.append(float(alpha) if _is_valid_dx7_patch(pred) else 0.0)
    return scores


def reward_dx7_key_coverage(
    prompts: list[Any],
    completions: list[Any],
    answer: list[Any],
    *,
    keys_to_remove: list[str],
    beta: float,
    **_: Any,
) -> list[float]:
    denom = float(len(KEY_ORDER))
    scores: list[float] = []
    for comp in completions:
        pred = _try_parse_pred_patch(_completion_to_text(comp), keys_to_remove=keys_to_remove)
        if pred is None:
            scores.append(0.0)
            continue
        present = float(sum(1 for k in KEY_ORDER if k in pred))
        scores.append(float(beta) * (present / denom))
    return scores


def reward_dx7_distance(
    prompts: list[Any],
    completions: list[Any],
    answer: list[Any],
    *,
    keys_to_remove: list[str],
    gamma: float,
    key_weights: dict[str, float],
    **_: Any,
) -> list[float]:
    scores: list[float] = []
    for comp, gt_item in zip(completions, answer):
        gt = _answer_to_patch(gt_item, keys_to_remove=keys_to_remove)
        if gt is None:
            scores.append(0.0)
            continue
        pred = _try_parse_pred_patch(_completion_to_text(comp), keys_to_remove=keys_to_remove)
        if pred is None:
            scores.append(0.0)
            continue
        dist = _dx7_distance_reward(pred, gt, key_weights=key_weights)
        scores.append(float(gamma) * float(dist))
    return scores


def _build_train_df(args: argparse.Namespace) -> pd.DataFrame:
    root = PROJECT_ROOT
    yamaha_train_csv = root / args.yamaha_train_csv
    yamaha_train_caps_csv = root / args.yamaha_train_captions_csv
    alltheweb_train_csv = root / args.alltheweb_train_csv
    alltheweb_train_caps_csv = root / args.alltheweb_train_captions_csv

    train_caption = _load_csv(yamaha_train_caps_csv)
    train_data = _load_csv(yamaha_train_csv)
    train_df = pd.merge(train_data, train_caption[["id", "caption"]], on="id", how="left")

    if not args.no_alltheweb:
        train_caption_add = _load_csv(alltheweb_train_caps_csv)
        train_data_add = _load_csv(alltheweb_train_csv)
        train_df_add = pd.merge(train_data_add, train_caption_add[["id", "caption"]], on="id", how="left")
        train_df = pd.concat([train_df, train_df_add], ignore_index=True)

    if args.filter_train:
        if "inaudible" in train_df.columns:
            inaudible = train_df["inaudible"].fillna(False).astype(bool)
        else:
            inaudible = pd.Series([False] * len(train_df))

        if "name" in train_df.columns:
            name = train_df["name"].astype(str)
        else:
            name = pd.Series([""] * len(train_df))

        f1 = ~inaudible
        f2 = ~name.str.contains("NULL", na=False)
        train_df = train_df[f1 & f2].copy()

    train_df = train_df.reset_index(drop=True)
    return train_df


def _build_grpo_dataset(
    df: pd.DataFrame,
    *,
    tokenizer: Any,
    args: argparse.Namespace,
    keys_to_remove: list[str],
) -> tuple[Dataset, int | None]:
    prompt_template = zeroshot_schema_prompt if args.use_schema_prompt else zeroshot_prompt

    prompts: list[list[dict[str, str]]] = []
    answers: list[str] = []
    ids: list[Any] = []

    skipped_missing_caption = 0
    skipped_invalid_gt = 0
    for _, row in df.iterrows():
        row_id = row.get("id")
        caption = row.get("caption")
        patch_data = row.get("patch_data")

        if not _is_non_empty_str(caption):
            skipped_missing_caption += 1
            continue

        # Hybrid thinking control (Qwen3 soft-switch tags in the caption):
        cap = str(caption).rstrip()
        suffix: str | None = None
        if args.append_think_control == "no_think":
            suffix = "/no_think"
        elif args.append_think_control == "think":
            suffix = "/think"
        elif args.append_think_control == "auto":
            suffix = "/think" if bool(args.enable_thinking) else "/no_think"

        if suffix is not None:
            cap = cap + " " + suffix

        user_text = prompt_template.format(prompt=cap)

        try:
            gt_raw = _parse_patch_data(patch_data, row_id=row_id)
            gt = _canonicalize_ground_truth(gt_raw, row_id=row_id, keys_to_remove=keys_to_remove)
        except (ValueError, TypeError):
            # Skip rows with invalid ground-truth patches.
            skipped_invalid_gt += 1
            continue

        prompts.append([{"role": "user", "content": user_text}])
        # Store as JSON string for HF Dataset collation stability.
        answers.append(json.dumps(gt, ensure_ascii=False, separators=(",", ":")))
        ids.append(row_id)

    ds = Dataset.from_dict({"prompt": prompts, "answer": answers, "id": ids})
    print(
        f"[INFO] Built GRPO dataset: n={len(ds)} (skipped missing_caption={skipped_missing_caption}, invalid_gt={skipped_invalid_gt})",
        file=sys.stderr,
    )

    # Token-length filtering (ToolRL / Unsloth notebook style):
    max_len: int | None = None
    if args.prompt_length_quantile is not None:
        q = float(args.prompt_length_quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError("--prompt_length_quantile must be in (0,1].")

        tokenized = ds.map(
            lambda x: {"_L": len(tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True))},
        )
        max_len = int(np.quantile(np.asarray(tokenized["_L"], dtype=np.int64), q))
        ds = ds.select(np.where(np.asarray(tokenized["_L"], dtype=np.int64) <= max_len)[0])

    return ds, max_len


def main() -> None:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--yamaha_train_csv", type=str, default="data/DX7_YAMAHA_train.csv")
    parser.add_argument("--alltheweb_train_csv", type=str, default="data/DX7_AllTheWeb_train.csv")
    parser.add_argument("--yamaha_train_captions_csv", type=str, default="data/DX7_YAMAHA_train_captions.csv")
    parser.add_argument("--alltheweb_train_captions_csv", type=str, default="data/DX7_AllTheWeb_train_captions.csv")
    parser.add_argument("--no_alltheweb", action="store_true", help="Train only on Yamaha train split (skip AllTheWeb).")
    parser.add_argument("--filter_train", action="store_true", help="Filter out inaudible / NULL-name rows (recommended).")
    parser.add_argument(
        "--prompt_pool_size",
        "--max_train_samples",
        dest="max_train_samples",
        type=int,
        default=4096,
        help="Number of prompts in the GRPO prompt pool (sampled from the train split). Alias: --max_train_samples.",
    )
    parser.add_argument(
        "--prompt_length_quantile",
        type=float,
        default=0.90,
        help="Keep prompts up to this length quantile (avoids truncation). Set to 1.0 to disable filtering.",
    )

    # Prompt formatting
    parser.add_argument("--use_schema_prompt", action="store_true", help="Use the long schema prompt (slower, longer).")
    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to the tokenizer chat template (if supported).")
    parser.add_argument(
        "--append_think_control",
        type=str,
        default="auto",
        choices=["none", "auto", "think", "no_think"],
        help=(
            "Optionally append a Qwen3 soft-switch tag to the caption. "
            "'think' appends '/think', 'no_think' appends '/no_think'. "
            "'auto' appends '/think' when --enable_thinking is set, else appends '/no_think'."
        ),
    )
    parser.add_argument(
        "--keys_to_remove",
        type=str,
        default='["name","has_fixed_freq","has_fixed_freqs"]',
        help="JSON list of extra keys to remove from patch dicts.",
    )

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True, help="SFT checkpoint path or HF model id.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Optional tokenizer override.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA (parameter-efficient tuning). Default is full fine-tuning.")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]',
        help="JSON list of target module names for LoRA.",
    )

    # GRPO
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_dx7")
    parser.add_argument("--reward_mode", type=str, default="dx7_dist", choices=["toolrl_exact", "dx7_dist"])
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to keep in output_dir.")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Total context window budget (prompt + completion).")
    parser.add_argument("--max_prompt_length", type=int, default=None, help="Max prompt tokens. Default: inferred from prompt_length_quantile.")
    parser.add_argument("--max_completion_length", type=int, default=None, help="Max completion tokens. Default: max_seq_length - max_prompt_length.")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Number of prompts per device per step.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=-1, help="Set -1 to disable top-k (use all tokens).")
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0.0 disables reference model).")
    parser.add_argument("--loss_type", type=str, default="dapo", choices=["dapo", "dr_grpo", "grpo", "bnpo"])
    parser.add_argument("--mask_truncated_completions", action="store_true", help="Exclude truncated completions from loss.")
    parser.add_argument("--scale_rewards", type=str, default="group", choices=["group", "batch", "none"])
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for generation (requires compatible vLLM).")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)

    # Rewards
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for validity reward (dx7_dist mode).")
    parser.add_argument("--beta_keys", type=float, default=0.5, help="Weight for key coverage reward (dx7_dist mode).")
    parser.add_argument("--gamma", type=float, default=3.0, help="Weight for dense distance reward (dx7_dist mode).")
    parser.add_argument("--invalid_reward", type=float, default=-3.0, help="Reward for unparsable outputs (dx7_dist mode).")
    parser.add_argument(
        "--key_weights_json",
        type=str,
        default=json.dumps(DEFAULT_KEY_WEIGHTS),
        help="JSON dict mapping DX7 top-level keys to weights (dx7_dist mode).",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Optional. If set, exports WANDB_API_KEY and logs in non-interactively.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Optional. If set, exports WANDB_PROJECT.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional. If set, exports WANDB_ENTITY.")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional. If set, exports WANDB_MODE (use 'offline' to avoid login).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional. Path to a checkpoint dir, or 'latest' to auto-resume from the newest checkpoint in output_dir.",
    )

    args = parser.parse_args()

    _set_seed(int(args.seed))

    _setup_wandb(args)

    keys_to_remove = json.loads(args.keys_to_remove)
    if not isinstance(keys_to_remove, list) or not all(isinstance(x, str) for x in keys_to_remove):
        raise ValueError("--keys_to_remove must be a JSON list of strings")

    # Tokenizer
    tok_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If the tokenizer supports Qwen-style `enable_thinking`, set the default from CLI for all call sites,
    # including TRL internals that may call apply_chat_template without passing enable_thinking.
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        sig = None
    if sig is not None and "enable_thinking" in sig.parameters:
        original_apply_chat_template = tokenizer.apply_chat_template

        def _apply_chat_template_with_default_thinking(conversation: Any, *a: Any, **kw: Any):  # type: ignore[no-untyped-def]
            kw.setdefault("enable_thinking", bool(args.enable_thinking))
            return original_apply_chat_template(conversation, *a, **kw)

        tokenizer.apply_chat_template = _apply_chat_template_with_default_thinking  # type: ignore[method-assign]

    # Build dataset
    train_df = _build_train_df(args)
    if args.max_train_samples is not None and args.max_train_samples > 0 and len(train_df) > int(args.max_train_samples):
        train_df = train_df.sample(n=int(args.max_train_samples), random_state=int(args.seed)).reset_index(drop=True)
    else:
        train_df = train_df.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    train_ds, inferred_prompt_len = _build_grpo_dataset(train_df, tokenizer=tokenizer, args=args, keys_to_remove=keys_to_remove)

    # Derive prompt/completion budgets (matches the GRPO notebook best-practice).
    max_prompt_length: int
    if args.max_prompt_length is not None and int(args.max_prompt_length) > 0:
        max_prompt_length = int(args.max_prompt_length)
    elif inferred_prompt_len is not None and inferred_prompt_len > 0:
        max_prompt_length = int(inferred_prompt_len) + 1
    else:
        max_prompt_length = 1024

    if args.max_completion_length is not None and int(args.max_completion_length) > 0:
        max_completion_length = int(args.max_completion_length)
    else:
        max_seq = int(args.max_seq_length) if args.max_seq_length is not None else 0
        if max_seq <= 0:
            max_completion_length = 1024
        else:
            max_completion_length = int(max_seq) - int(max_prompt_length)

    if max_completion_length <= 0:
        raise ValueError(
            f"Computed max_completion_length={max_completion_length} is invalid. "
            f"Increase --max_seq_length or decrease --max_prompt_length."
        )
    if args.max_seq_length is not None and int(args.max_seq_length) > 0:
        if int(max_prompt_length) + int(max_completion_length) > int(args.max_seq_length):
            raise ValueError(
                f"max_prompt_length + max_completion_length must be <= max_seq_length "
                f"({max_prompt_length} + {max_completion_length} > {int(args.max_seq_length)})."
            )

    # Model init kwargs (when passing `model` as string to GRPOTrainer)
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": None,
    }
    torch_dtype = dtype_map[args.dtype]
    model_init_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_init_kwargs["torch_dtype"] = torch_dtype

    # LoRA config (optional; default is full fine-tuning)
    peft_config: LoraConfig | None = None
    if bool(args.use_lora):
        target_modules = json.loads(args.lora_target_modules)
        if not isinstance(target_modules, list) or not all(isinstance(x, str) for x in target_modules):
            raise ValueError("--lora_target_modules must be a JSON list of strings")
        peft_config = LoraConfig(
            r=int(args.lora_rank),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

    # Rewards wiring
    def _parse_key_weights_json(s: str) -> dict[str, float]:
        try:
            raw = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError("--key_weights_json must be valid JSON (object mapping DX7 keys to weights).") from exc
        if not isinstance(raw, dict):
            raise ValueError("--key_weights_json must be a JSON object mapping DX7 keys to weights.")
        merged = dict(DEFAULT_KEY_WEIGHTS)
        for k, v in raw.items():
            merged[str(k)] = float(v)
        return merged

    if args.reward_mode == "toolrl_exact":
        reward_funcs = [
            lambda **kw: reward_toolrl_format(keys_to_remove=keys_to_remove, **kw),
            lambda **kw: reward_toolrl_correctness(keys_to_remove=keys_to_remove, **kw),
        ]
        reward_names = ["toolrl_format", "toolrl_correctness"]
    else:
        key_weights = _parse_key_weights_json(str(args.key_weights_json))
        reward_funcs = [
            lambda **kw: reward_dx7_valid(keys_to_remove=keys_to_remove, alpha=float(args.alpha), invalid_reward=float(args.invalid_reward), **kw),
            lambda **kw: reward_dx7_key_coverage(keys_to_remove=keys_to_remove, beta=float(args.beta_keys), **kw),
            lambda **kw: reward_dx7_distance(keys_to_remove=keys_to_remove, gamma=float(args.gamma), key_weights=key_weights, **kw),
        ]
        reward_names = ["dx7_valid", "dx7_keys", "dx7_dist"]

    # Make function names stable in logs (TRL uses __name__):
    for fn, name in zip(reward_funcs, reward_names):
        if hasattr(fn, "__name__"):
            fn.__name__ = name  # type: ignore[attr-defined]

    # GRPO config
    top_k = None if int(args.top_k) < 0 else int(args.top_k)
    config_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "report_to": ("none" if args.report_to == "none" else "wandb"),
        "run_name": (args.run_name if _is_non_empty_str(args.run_name) else None),
        "remove_unused_columns": False,
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit) if args.save_total_limit is not None else None,
        "max_steps": int(args.max_steps),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_prompt_length": int(max_prompt_length),
        "num_generations": int(args.num_generations),
        "max_completion_length": int(max_completion_length),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": top_k,
        "min_p": float(args.min_p),
        "beta": float(args.beta),
        "loss_type": (str(args.loss_type) if _is_non_empty_str(args.loss_type) else None),
        "scale_rewards": (str(args.scale_rewards) if _is_non_empty_str(args.scale_rewards) else None),
        "mask_truncated_completions": bool(args.mask_truncated_completions),
        "use_vllm": bool(args.use_vllm),
        "vllm_mode": str(args.vllm_mode),
        "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "model_init_kwargs": model_init_kwargs,
    }

    # If the installed TRL has vLLM SamplingParams integration (as in the reference notebook),
    # populate it when --use_vllm is enabled.
    if bool(args.use_vllm):
        try:
            from vllm import SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed but --use_vllm was set. Install vLLM or disable --use_vllm.") from exc

        try:
            config_kwargs["vllm_sampling_params"] = SamplingParams(
                min_p=float(args.min_p),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                seed=int(args.seed),
                stop=[tokenizer.eos_token],
                include_stop_str_in_output=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Your vLLM SamplingParams signature looks incompatible with this configuration. "
                "Try a ToolRL/TRL-compatible vLLM version or run without --use_vllm."
            ) from exc

    # Filter unsupported GRPOConfig fields for compatibility across TRL versions.
    grpo_fields = {f.name for f in fields(GRPOConfig)}
    unsupported = sorted([k for k in config_kwargs.keys() if k not in grpo_fields])
    if unsupported:
        print(f"[INFO] Ignoring unsupported GRPOConfig args: {unsupported}", file=sys.stderr)
    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in grpo_fields and v is not None}
    training_args = GRPOConfig(**filtered_kwargs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grpo_config.json").write_text(json.dumps(asdict(training_args), indent=2, default=str), encoding="utf-8")

    # Some TRL versions only honor trust_remote_code/torch_dtype via explicit model loading.
    if "model_init_kwargs" in grpo_fields:
        model: Any = str(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(args.model_name_or_path), **model_init_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Optional resume
    resume_arg = args.resume_from_checkpoint
    resume_path: str | None = None
    if _is_non_empty_str(resume_arg):
        s = str(resume_arg).strip()
        if s.lower() == "latest":
            last = get_last_checkpoint(str(out_dir))
            if last is None:
                raise ValueError(f"--resume_from_checkpoint latest was set, but no checkpoint was found in: {out_dir}")
            resume_path = str(last)
        else:
            resume_path = s
        print(f"[INFO] Resuming training from checkpoint: {resume_path}", file=sys.stderr)

    trainer.train(resume_from_checkpoint=resume_path)

    # Save final model for inference.
    # - LoRA: save adapter weights under final_lora/
    # - Full FT: save full model under final_model/
    final_dir = out_dir / ("final_lora" if bool(args.use_lora) else "final_model")
    final_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.use_lora):
        trainer.model.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()


