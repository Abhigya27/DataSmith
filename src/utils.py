"""Small helpers shared by every module.

# DECISION: this file is not in the §4.2 layout. dtype predicates, JSON sanitizing and error
# helpers are needed by almost every module, and putting them in any one of them would create
# import cycles. Kept tiny on purpose.

The dtype predicates behave the same on pandas 2.x and 3.x. Pandas 3 makes string columns the
new ``str`` dtype instead of ``object``, so ``dtype == object`` checks silently stop matching:
always go through these predicates.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api import types as pdt


class DsagentError(Exception):
    """A user-facing error: the message is meant to be printed as-is by the CLI."""


# ------------------------------------------------------------------ dtype predicates
def is_bool(s: pd.Series) -> bool:
    return bool(pdt.is_bool_dtype(s))


def is_numeric(s: pd.Series) -> bool:
    """True numeric dtype, excluding booleans."""
    return bool(pdt.is_numeric_dtype(s)) and not is_bool(s)


def is_datetime(s: pd.Series) -> bool:
    return bool(pdt.is_datetime64_any_dtype(s))


def is_texty(s: pd.Series) -> bool:
    """object / str / category columns: anything we treat as strings."""
    if is_numeric(s) or is_bool(s) or is_datetime(s):
        return False
    return bool(
        pdt.is_object_dtype(s) or pdt.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    )


def as_object(s: pd.Series) -> pd.Series:
    """Plain object-dtype copy with np.nan for missing values: safe for .map / .where."""
    return s.astype(object).where(s.notna(), np.nan)


def to_str_keep_nan(s: pd.Series) -> pd.Series:
    """Row-wise string conversion that keeps missing values missing (object dtype)."""
    obj = as_object(s)
    return obj.map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)


# ------------------------------------------------------------------ JSON sanitizing
def _key(k: Any) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, tuple):
        return " | ".join(str(x) for x in k)
    return str(k)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas / datetime values to plain JSON types.
    NaN and +-inf become None; tuple dict-keys become 'a | b' strings."""
    if obj is None or obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if hasattr(obj, "model_dump"):  # pydantic model
        return to_jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if hasattr(obj, "__fspath__"):
        return str(obj)
    return str(obj)


def compact_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), separators=(",", ":"), ensure_ascii=False)


def round_or_none(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else round(f, nd)


# ------------------------------------------------------------------ errors and names
def suggest(name: str, options: Iterable[str]) -> str:
    match = difflib.get_close_matches(str(name), [str(o) for o in options], n=1, cutoff=0.6)
    return f"; did you mean '{match[0]}'?" if match else ""


def missing_column_message(name: str, columns: Iterable[str]) -> str:
    cols = list(columns)
    return f"Column '{name}' not found{suggest(name, cols)} (available: {', '.join(map(str, cols[:15]))}" + (", ..." if len(cols) > 15 else "") + ")"


def safe_name(s: str, max_len: int = 60) -> str:
    """Sanitize an id that will become a file name. Column names are data, never paths."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._")
    return (out or "x")[:max_len]
