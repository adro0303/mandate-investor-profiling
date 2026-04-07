import re
import numpy as np
import pandas as pd

def parse_gdelt_datetime_int(x: str) -> pd.Timestamp:
    """
    GDELT GKG DATE looks like YYYYMMDDHHMMSS (string/int).
    """
    s = str(x)
    s = re.sub(r"\D", "", s)
    if len(s) < 8:
        return pd.NaT
    s = s.ljust(14, "0")
    return pd.to_datetime(s[:14], format="%Y%m%d%H%M%S", errors="coerce")

def split_v2tone(v2tone: str) -> dict:
    """
    V2Tone is typically 7 comma-separated floats.
    """
    if v2tone is None or (isinstance(v2tone, float) and np.isnan(v2tone)):
        vals = [np.nan] * 7
    else:
        parts = [p.strip() for p in str(v2tone).split(",")]
        vals = []
        for p in parts[:7]:
            try:
                vals.append(float(p))
            except Exception:
                vals.append(np.nan)
        while len(vals) < 7:
            vals.append(np.nan)

    keys = ["tone", "pos", "neg", "polarity", "activity_density", "self_group_ref", "word_count"]
    return dict(zip(keys, vals))

def extract_theme_codes(v2themes: str) -> list[str]:
    """
    V2Themes: "CODE,score;CODE,score;..."
    Returns list of CODEs.
    """
    if v2themes is None or (isinstance(v2themes, float) and np.isnan(v2themes)):
        return []
    items = [t for t in str(v2themes).split(";") if t]
    codes = []
    for it in items:
        code = it.split(",")[0].strip()
        if code:
            codes.append(code)
    return codes

def extract_org_names(v2orgs: str) -> list[str]:
    """
    V2Organizations: "Name,id;Name,id;..."
    Returns list of lowercased names.
    """
    if v2orgs is None or (isinstance(v2orgs, float) and np.isnan(v2orgs)):
        return []
    items = [t for t in str(v2orgs).split(";") if t]
    names = []
    for it in items:
        name = it.split(",")[0].strip()
        if name:
            names.append(name.lower())
    return names
