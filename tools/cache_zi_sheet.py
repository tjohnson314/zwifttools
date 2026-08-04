"""Cache the ZwiftInsider speed-test spreadsheets into the repo and normalize
inconsistent make spellings so matching is stable.

The ZI sheet is exported to %TEMP%\\zi_sheet.csv and %TEMP%\\zi_Wheels.csv. Those
are volatile, and the sheet mixes spellings of the same make (e.g. "Van Rysel"
vs "VanRysel"), which splits one bike into two rows and breaks matching. This
copies them into zwiftdata/ (the source of truth the solver/compare/audit tools
read) and applies the canonical make-name fixes below.

Run whenever you re-export the sheet:
    python tools/cache_zi_sheet.py
"""
import os, re

ROOT = r"C:\Users\timjo\Documents\Coding\Zwift\zwifttools"
TEMP = os.environ["TEMP"]
DEST = os.path.join(ROOT, "zwiftdata")

# canonical make-name fixes applied to the raw CSV text: (pattern, replacement)
_FIXES = [
    (re.compile(r"Van\s+Rysel", re.I), "VanRysel"),
    (re.compile(r"WilierFilante"), "Wilier"),   # "WilierFilante Filante ..." -> "Wilier Filante ..."
]

# (source name in TEMP, destination name in zwiftdata)
_FILES = [("zi_sheet.csv", "zi_sheet.csv"), ("zi_Wheels.csv", "zi_wheels.csv")]


def main():
    for src, dst in _FILES:
        sp = os.path.join(TEMP, src)
        if not os.path.exists(sp):
            print(f"skip {src}: not found in {TEMP}")
            continue
        text = open(sp, encoding="utf-8-sig", newline="").read()
        fixes = 0
        for pat, rep in _FIXES:
            text, k = pat.subn(rep, text)
            fixes += k
        dp = os.path.join(DEST, dst)
        with open(dp, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(text)
        print(f"cached {src} -> {dp}  ({fixes} name fix(es))")


if __name__ == "__main__":
    main()
