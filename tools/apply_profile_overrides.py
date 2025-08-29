# tools/apply_profile_overrides.py (or paste into your script)
import json
from pathlib import Path
from transitbench import recover

def apply_profile_overrides(json_path: str):
    data = json.loads(Path(json_path).read_text())
    sugg = data.get("suggest_tau_base", {})

    for method, prof_map in sugg.items():
        for prof_name, tau in prof_map.items():
            prof = recover.PROFILES.setdefault(prof_name, {})
            # Coerce tau_base into a dict if it's missing or a float
            if not isinstance(prof.get("tau_base"), dict):
                prof["tau_base"] = {}
            prof["tau_base"][method] = float(tau)

    return recover.PROFILES

# Example usage:
# apply_profile_overrides("/Users/eshaantripathi/Documents/transitbench/profile_overrides.json")
# print("Applied profile overrides.")