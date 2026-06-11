"""Sign resolution: load paleocode database and sample/filter sign codes."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import GenerationConfig
from paleocodage import ParsedSign, load_paleocodes, load_paleocodes_json


def _normalize_code(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def resolve_sign_codes(
    sign_source: str,
    sign_list: Optional[List[str]],
    paleocodes_path: str,
    rng: np.random.Generator,
    config: GenerationConfig,
) -> List[str]:
    """Return a list of paleocode strings to use for one tablet."""
    if sign_source not in {"paleocode", "transliteration"}:
        raise ValueError("sign_source must be 'paleocode' or 'transliteration'")

    max_len = config.max_paleocode_length

    if sign_list:
        if sign_source == "paleocode":
            cleaned: List[str] = []
            dropped_empty = 0
            dropped_long = 0
            for sign in sign_list:
                code = _normalize_code(sign)
                if code is None:
                    dropped_empty += 1
                    continue
                if max_len is not None and len(code) > max_len:
                    dropped_long += 1
                    continue
                cleaned.append(code)
            if dropped_empty:
                print(f"Warning: dropped {dropped_empty} empty paleocode entries from sign_list")
            if dropped_long:
                print(
                    f"Warning: dropped {dropped_long} sign_list paleocodes longer than "
                    f"{max_len} chars"
                )
            return cleaned

        # transliteration source — resolve via mapping
        mapping = load_paleocodes(paleocodes_path)
        resolved: List[str] = []
        dropped_empty = 0
        dropped_long = 0
        for sign in sign_list:
            key = _normalize_code(sign)
            if key is None:
                dropped_empty += 1
                continue
            code = _normalize_code(mapping.get(key))
            if code is not None and (max_len is None or len(code) <= max_len):
                resolved.append(code)
            elif code is not None:
                dropped_long += 1
            else:
                print(f"Warning: transliteration '{key}' not found in database")
        if dropped_empty:
            print(f"Warning: dropped {dropped_empty} empty transliteration entries")
        if dropped_long:
            print(f"Warning: dropped {dropped_long} resolved paleocodes longer than {max_len}")
        return resolved

    # Random sampling from the database
    if paleocodes_path.lower().endswith(".json"):
        pool_raw = load_paleocodes_json(paleocodes_path)
    else:
        pool_raw = list(load_paleocodes(paleocodes_path).values())

    pool = [code for item in pool_raw if (code := _normalize_code(item)) is not None]
    if max_len is not None:
        pool = [code for code in pool if len(code) <= max_len]
    if not pool:
        return []

    lo, hi = config.num_signs_range
    num_signs = int(rng.integers(lo, hi))
    return list(rng.choice(pool, size=num_signs, replace=True))


def drop_signs_without_wedges(parsed_signs: List[ParsedSign]) -> Tuple[List[ParsedSign], int]:
    """Return (kept_signs, n_dropped) filtering out signs that have no wedges."""
    kept: List[ParsedSign] = []
    dropped = 0
    for sign in parsed_signs:
        if getattr(sign, "wedges", None):
            kept.append(sign)
        else:
            dropped += 1
    return kept, dropped


def load_sign_reference_metadata(paleocodes_path: str) -> Dict[str, Dict[str, str]]:
    """Build a mapping of paleocode → {transliteration, sign} for annotation export."""
    reference: Dict[str, Dict[str, str]] = {}

    if paleocodes_path.lower().endswith(".json"):
        try:
            import json
            with open(paleocodes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return reference

        if isinstance(data, dict):
            for transliteration, code in data.items():
                if not code:
                    continue
                entry = reference.setdefault(str(code), {})
                if transliteration and "transliteration" not in entry:
                    entry["transliteration"] = str(transliteration)
            return reference

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                code = item.get("PaleoCode") or item.get("paleocode") or item.get("code")
                if not code:
                    continue
                entry = reference.setdefault(str(code), {})
                transliteration = item.get("Transliteration") or item.get("transliteration")
                if transliteration and "transliteration" not in entry:
                    entry["transliteration"] = str(transliteration)
                sign_char = item.get("Sign") or item.get("sign")
                if sign_char and "sign" not in entry:
                    entry["sign"] = str(sign_char)
            return reference

        return reference

    for transliteration, code in load_paleocodes(paleocodes_path).items():
        if not code:
            continue
        entry = reference.setdefault(str(code), {})
        if transliteration and "transliteration" not in entry:
            entry["transliteration"] = str(transliteration)
    return reference
