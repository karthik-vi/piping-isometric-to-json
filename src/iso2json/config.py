"""Domain rules used by the rule-based stage. Edit here, not in the parser.

The piping-class table maps spec code -> material. On a real project this
comes from the piping material specification; here it matches the
synthetic generator's table.
"""
import re

PIPING_CLASS_TABLE = {
    "A1A": "CS A106-B",
    "A3B": "CS A333-6",
    "B1A": "SS A312-TP316L",
    "C2A": "CS GALV A53-B",
    "D1B": "DUPLEX A790-S31803",
    "S4A": "SS A312-TP304",
}
SERVICE_CODES = {"P", "CW", "ST", "N", "IA", "HC"}
TAG_PREFIXES = {"HV", "PI", "TI", "FT", "PSV", "XV", "LT", "TW"}
BOM_UNITS = {"EA", "M", "SET"}
NPS_SIZES = {0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24}

LINE_NO_RE = re.compile(r"\b(\d{1,2})-([A-Z]{1,2})-(\d{4})-([A-Z]\d[A-Z])\b")
DWG_NO_RE = re.compile(r"\bISO-\d{3}-\d{2}\b")
TAG_RE = re.compile(r"^(" + "|".join(sorted(TAG_PREFIXES, key=len, reverse=True)) + r")-(\d{3,4})$")
DIM_RE = re.compile(r"^\d{3,5}$")
DIM_RANGE_MM = (100, 20000)
PRESSURE_RANGE_BARG = (0.0, 420.0)
TEMP_RANGE_C = (-196, 600)

# Common OCR confusions applied only inside fields whose grammar is known.
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "L": "1", "T": "7"})
ALPHA_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"})

# Material catalog: description stem -> unit. On a real project this is the
# piping material catalog / item codes; here it matches the generator.
BOM_CATALOG = {
    "PIPE SMLS BE": "M",
    "ELBOW 90 LR": "EA",
    "FLANGE WN RF": "EA",
    "GASKET SPW": "EA",
    "GATE VALVE": "EA",
    "TEE EQUAL": "EA",
    "STUD BOLT": "SET",
    "REDUCER CONC": "EA",
}
