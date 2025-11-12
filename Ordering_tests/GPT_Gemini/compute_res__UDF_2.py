import os
import pandas as pd
import ast
import sqlite3
import datetime
from datetime import datetime

from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator

# ==========================
# Helper functions
# ==========================

def convert_result_column(result):
    """Converte stringhe di liste in vere liste, se necessario."""
    if isinstance(result, list):
        return result
    try:
        return ast.literal_eval(result)
    except Exception:
        return result

def detect_format_and_normalize(df):
    """
    Detect the input format and normalize column names.
    Returns a normalized dataframe with standard column names.
    """
    df_normalized = df.copy()
    
    # Map of possible column name variations to standard names
    column_mappings = {
        'SQL_query': 'sql_query',
        'AI_answer': 'result',
        'table_name': 'table'  # if needed
    }
    
    # Apply column mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df_normalized.columns:
            df_normalized.rename(columns={old_name: new_name}, inplace=True)
    
    # Ensure required columns exist
    required_columns = ['sql_query', 'result', 'db_path']
    missing_columns = [col for col in required_columns if col not in df_normalized.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df_normalized

# ==========================
# UDFs
# ==========================

# Date functions (format "dd-mm-yyyy")
def extract_day(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[0])
    except Exception:
        return None

def extract_month(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[1])
    except Exception:
        return None

def extract_year(date_str):
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[2])
    except Exception:
        return None

# Name functions
def extract_first_name(name):
    if not name:
        return None
    return " ".join(name.strip().split()[:-1]) or name.strip().split()[0]

def extract_last_name(name):
    if not name:
        return None
    parts = name.strip().split()
    return parts[-1] if len(parts) > 1 else None

# Extra
def extract_initials(name):
    if not name:
        return None
    return ''.join([p[0].upper() for p in name.strip().split() if p])

def reverse_string(s):
    if not s:
        return None
    return s[::-1]

def word_count(s):
    if not s:
        return 0
    return len(s.strip().split())

import math
import re
import unicodedata
from typing import Any, Dict, Iterable, Optional

# ---------- Helpers ----------

def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    )

def _only_alnum_and_space(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ]+", "", str(s))

def safe_divide(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    default: Optional[float] = 0
    try:
        if numer is None or denom in (None, 0, 0.0):
            return default
        return float(numer) / float(denom)
    except Exception:
        return default

# ---------- Domain UDFs ----------

def weight_gap(weight_kg: Optional[float], height_m: Optional[float]) -> Optional[float]:
    """
    Difference between current weight and target weight for a given BMI:
    gap = weight_kg - target_bmi * height_m^2
    """
    target_bmi: float = 21.0
    if weight_kg is None or height_m in (None, 0, 0.0):
        return None
    return float(weight_kg) - float(target_bmi) * (float(height_m) ** 2)

def normalize_name(s: Optional[str]) -> str:
    """
    Lowercase, remove accents, strip non-alphanumerics (keep spaces), collapse spaces, trim.
    """
    if s is None:
        return ""
    s = _strip_accents(str(s)).lower()
    s = _only_alnum_and_space(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_text(s: Optional[str]) -> str:
    """Alias of normalize_name for generic text."""
    return normalize_name(s)

def carbon_footprint(kwh_used: Optional[float]) -> Optional[float]:
    """
    Estimate CO2 kg = kWh * emission factor (kg CO2 per kWh).
    Set ef_kg_per_kwh to the desired factor for the grid; default is 0.4 (configurable).
    """
    ef_kg_per_kwh: Optional[float] = 0.3499
    if kwh_used is None or ef_kg_per_kwh is None:
        return None
    return float(kwh_used) * float(ef_kg_per_kwh)

def pass_flag(status: Any) -> int:
    """
    Map pass-like values to 1, otherwise 0.
    Accepted positives: 'pass','passed','true','1','yes','y','ok','passed ✅' (case-insensitive).
    """
    if status is None:
        return 0
    s = str(status).strip().lower()
    positives = {"pass", "passed", "true", "1", "yes", "y", "ok", "passed ✅"}
    return 1 if s in positives else 0

def extract_numeric_suffix(s: Optional[str]) -> Optional[int]:
    """
    Extract trailing digits as integer; return None if no trailing digits.
    """
    if s is None:
        return None
    m = re.search(r"([0-9]+)$", str(s))
    return int(m.group(1)) if m else None

def revenue_per_unit(total_revenue: Optional[float], units_sold: Optional[float]) -> Optional[float]:
    """
    total_revenue / units_sold with safe divide.
    """
    return safe_divide(total_revenue, units_sold)

def demand_elasticity(units_sold: Optional[float], price: Optional[float]) -> Optional[float]:
    """
    Simple cross-sectional proxy: units_sold / price (not a true elasticity; requires panel/time data for rigor).
    """
    return safe_divide(units_sold, price)

def category_popularity(category: Optional[str], counts: Optional[Dict[str, int]] = None, default_count: int = 0) -> int:
    """
    Popularity via precomputed frequency mapping: counts[category].
    Provide `counts` from a prior aggregation (e.g., SELECT category, COUNT(*) GROUP BY category).
    """
    if counts is None or category is None:
        return default_count
    return int(counts.get(str(category), default_count))

def parse_percent(s: Any) -> Optional[float]:
    """
    Convert percent-like strings to decimal:
    - '12%' -> 0.12
    - '12.5 %' -> 0.125
    - '0.12' -> 0.12
    - 12 -> 0.12 (assume percent)
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        v = float(s)
        # Heuristic: values > 1 are treated as percent points
        return v / 100.0 if abs(v) > 1 else v
    st = str(s).strip().replace(",", ".")
    m = re.match(r"^([+-]?[0-9]*\\.?[0-9]+)\\s*%?$", st)
    if not m:
        return None
    v = float(m.group(1))
    return v / 100.0 if "%" in st or abs(v) > 1 else v

def burn_rate(expenses: Optional[float], income: Optional[float]) -> Optional[float]:
    """
    expenses / income with safe divide; None if income is zero or null.
    """
    return safe_divide(expenses, income)

def efficiency_score(distance_km: Optional[float], fuel_used_l: Optional[float]) -> Optional[float]:
    """
    distance_km / fuel_used_l (km per liter), safe divide.
    """
    return safe_divide(distance_km, fuel_used_l)

def delivery_density(packages: Optional[float], distance_km: Optional[float]) -> Optional[float]:
    """
    packages / distance_km (packages per km), safe divide.
    """
    return safe_divide(packages, distance_km)

def route_stress(packages: Optional[float], time_hrs: Optional[float]) -> Optional[float]:
    """
    packages / time_hrs (packages per hour), safe divide.
    """
    return safe_divide(packages, time_hrs)

# ---------- Generic text/numeric UDFs ----------

def upper_case(s: Any) -> str:
    return "" if s is None else str(s).upper()

def lower_case(s: Any) -> str:
    return "" if s is None else str(s).lower()

def title_case(s: Any) -> str:
    return "" if s is None else str(s).title()

def trim_whitespace(s: Any) -> str:
    return "" if s is None else str(s).strip()

def normalize_spaces(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\\s+", " ", str(s)).strip()

def remove_accents(s: Any) -> str:
    return _strip_accents("" if s is None else str(s))

def to_number(s: Any) -> Optional[float]:
    """
    Parse strings with optional thousands separators and decimal points/commas.
    """
    if s is None or s == "":
        return None
    st = str(s).strip()
    st = st.replace(" ", "").replace(",", ".")
    try:
        return float(st)
    except Exception:
        m = re.search(r"([+-]?[0-9]*\\.?[0-9]+)", st)
        return float(m.group(1)) if m else None

def to_int(s: Any) -> Optional[int]:
    """
    Parse integer from string; returns None if not found.
    """
    if s is None or s == "":
        return None
    try:
        return int(str(s))
    except Exception:
        m = re.search(r"([+-]?[0-9]+)", str(s))
        return int(m.group(1)) if m else None

def coalesce_any(*args: Any) -> Any:
    """
    Return first argument that is not None and not empty string.
    """
    for a in args:
        if a is not None and str(a) != "":
            return a
    return None

def last_name(full_name: str) -> str:
    """Return the last whitespace-separated token in a name."""
    return (full_name or "").strip().split()[-1]

def duration_category(minutes: str | int) -> str:
    """Bucket runtimes: <90 Short, 90–150 Medium, otherwise Long."""
    try:
        m = int(minutes)
    except Exception:
        return "Unknown"
    if m < 90:
        return "Short"
    if m <= 150:
        return "Medium"
    return "Long"

def oscars_to_int(oscars: str) -> int:
    """Convert textual Oscar count to integer, defensively."""
    try:
        return int(re.sub(r"[^\d]", "", oscars))
    except Exception:
        return 0

def height_to_feet(meters: str | float) -> float:
    """Convert metres to feet using 1 m = 3.28084 ft."""
    try:
        return float(meters) * 3.28084
    except Exception:
        return math.nan

def year_to_era(year_val: str | int) -> str:
    """Render negative years as ‘BCE’ and positives as ‘CE’."""
    try:
        y = int(year_val)
    except Exception:
        return "Unknown"
    if y < 0:
        return f"{abs(y)} BCE"
    return f"{y} CE"

def age_from_year(year_val: str | int,
                  reference: int = datetime.now().year) -> int:
    """Compute age in whole years relative to reference year."""
    try:
        y = int(year_val)
    except Exception:
        return -1
    return reference - y

def consumption_category(l_per_100km: str | float) -> str:
    """
    Classify average fuel consumption:
    0 → Electric
    <5 → Low
    <6 → Medium
    else High
    """
    try:
        v = float(l_per_100km)
    except Exception:
        return "Unknown"
    if v == 0:
        return "Electric"
    if v < 5:
        return "Low"
    if v < 6:
        return "Medium"
    return "High"

_slug_rx = re.compile(r"[^a-z0-9\-]+")
def slugify(text: str) -> str:
    """Lower-case, replace spaces with ‘-’, strip non-alphanumerics."""
    t = (text or "").lower().strip().replace(" ", "-")
    return _slug_rx.sub("", t)

def weight_category(kg: str | float) -> str:
    """Light <10 kg, Medium <100 kg, Heavy <1000 kg, else Massive."""
    try:
        w = float(kg)
    except Exception:
        return "Unknown"
    if w < 10:
        return "Light"
    if w < 100:
        return "Medium"
    if w < 1000:
        return "Heavy"
    return "Massive"

def life_weight_index(lifespan: str | int, weight: str | float) -> float:
    """Compute kg per year of life expectancy."""
    try:
        ls = int(lifespan)
        wt = float(weight)
        return wt / ls if ls else math.nan
    except Exception:
        return math.nan
    
import re
import math
import unicodedata
from typing import Optional

def _safe_str(x) -> str:
    return "" if x is None else str(x).strip()

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        # Try to strip non-numeric chars
        try:
            cleaned = re.sub(r"[^0-9.+-]", "", str(x))
            return float(cleaned) if cleaned not in ("", "+", "-") else None
        except Exception:
            return None

def _safe_int(x) -> Optional[int]:
    f = _safe_float(x)
    return int(round(f)) if f is not None and not math.isnan(f) else None

def remove_diacritics(text: str) -> str:
    t = _safe_str(text)
    nfkd = unicodedata.normalize("NFKD", t)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_spaces(text: str) -> str:
    t = _safe_str(text)
    # Collapse multiple spaces, trim, and also normalize internal whitespace
    return re.sub(r"\s+", " ", t).strip()

def strip_articles(title):
    """
    Remove leading English articles A/An/The from the beginning of a title.
    """
    if title is None:
        return ""
    
    t = str(title).strip()
    t = re.sub(r"\s+", " ", t)
    
    return re.sub(r"^(?i)(the|a|an)\s+", "", t)

def director_lastname(name: str) -> str:
    """
    Extract the last family name from a director string.
    Examples: "Alejandro G. Iñárritu" -> "Iñárritu", "Ethan & Joel Coen" -> "Coen"
    """
    t = normalize_spaces(name)
    # Remove common connectors like '&' and 'and'
    t = re.sub(r"\s*(?:&|and)\s*", " ", t, flags=re.IGNORECASE)
    # Split on whitespace; take last token that contains a letter
    tokens = [tok for tok in re.split(r"\s+", t) if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", tok)]
    return tokens[-1] if tokens else ""

def genre_primary(genre: str) -> str:
    """
    Take the primary genre before hyphens (e.g., 'Comedy-Drama' -> 'Comedy').
    """
    t = normalize_spaces(genre)
    return t.split("-")[0].strip() if t else ""

def oscars_bucket(oscars) -> str:
    """
    Bucketize Oscars count: 0-3, 4-6, 7-9, 10+
    """
    n = _safe_int(oscars)
    if n is None:
        return "unknown"
    if n <= 3:
        return "0-3"
    if n <= 6:
        return "4-6"
    if n <= 9:
        return "7-9"
    return "10+"

def year_decade(year) -> str:
    """
    Convert a year like 1994 to '1990s'.
    """
    y = _safe_int(year)
    if y is None:
        return "unknown"
    decade = (y // 10) * 10
    return f"{decade}s"

def weight_class(weight_kg) -> str:
    """
    Classify animal average weight:
    - '<10kg', '10-99kg', '100-999kg', '>=1000kg'
    """
    w = _safe_float(weight_kg)
    if w is None:
        return "unknown"
    if w < 10:
        return "<10kg"
    if w < 100:
        return "10-99kg"
    if w < 1000:
        return "100-999kg"
    return ">=1000kg"

def lifespan_bucket(years) -> str:
    """
    Bucketize average lifespan:
    - '<10y', '10-29y', '30-59y', '>=60y'
    """
    y = _safe_int(years)
    if y is None:
        return "unknown"
    if y < 10:
        return "<10y"
    if y < 30:
        return "10-29y"
    if y < 60:
        return "30-59y"
    return ">=60y"

def habitat_norm(habitat: str) -> str:
    """
    Normalize habitat to a canonical label set.
    """
    t = remove_diacritics(normalize_spaces(habitat)).lower()
    # Simple keyword-based mapping
    mapping = [
        ("ocean", "ocean"),
        ("oceans", "ocean"),
        ("reef", "reef"),
        ("coral", "reef"),
        ("river", "river/lake"),
        ("lake", "river/lake"),
        ("savanna", "savanna"),
        ("savannas", "savanna"),
        ("forest", "forest"),
        ("rainforest", "rainforest"),
        ("tropical", "forest"),
        ("mountain", "mountain"),
        ("arctic", "arctic"),
        ("antarctica", "arctic"),
        ("desert", "desert"),
        ("coastal", "coastal"),
        ("island", "islands"),
        ("islands", "islands"),
        ("tundra", "tundra"),
        ("urban", "urban"),
        ("cliff", "mountain"),
    ]
    for key, label in mapping:
        if key in t:
            return label
    return "other"

def efficiency_label(horsepower, avg_consumption_l_per_100km) -> str:
    """
    Label power/consumption efficiency:
    - 'EV' if consumption == 0.0
    - else compute hp_per_l = horsepower / consumption and map to A/B/C/D
    """
    hp = _safe_float(horsepower)
    cons = _safe_float(avg_consumption_l_per_100km)
    if cons is not None and cons == 0.0:
        return "EV"
    if hp is None or cons is None or cons <= 0:
        return "unknown"
    ratio = hp / cons
    if ratio >= 40:
        return "A"
    if ratio >= 30:
        return "B"
    if ratio >= 20:
        return "C"
    return "D"

_COUNTRY_TO_ALPHA2 = {
    # Common in cars.csv
    "germany": "DE",
    "italy": "IT",
    "usa": "US",
    "united states": "US",
    "united kingdom": "GB",
    "france": "FR",
    "japan": "JP",
    "sweden": "SE",
    "south korea": "KR",
    "romania": "RO",
    "czech republic": "CZ",
    "spain": "ES",
    # Also appear in monuments.csv (not used here, but harmless)
    "china": "CN",
    "brazil": "BR",
    "egypt": "EG",
    "jordan": "JO",
    "peru": "PE",
    "cambodia": "KH",
    "chile": "CL",
    "greece": "GR",
    "russia": "RU",
    "india": "IN",
    "belgium": "BE",
    "poland": "PL",
    "canada": "CA",
    "australia": "AU",
    "united arab emirates": "AE",
}

def country_code(country: str) -> str:
    """
    Map country name to ISO-like alpha-2 code for the set present in the data.
    """
    t = normalize_spaces(country).lower()
    return _COUNTRY_TO_ALPHA2.get(t, "UN")  # UN = unknown/unsupported name

def _ordinal(n: int) -> str:
    """
    Return English ordinal like 1st, 2nd, 3rd, 4th...
    """
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def year_century(year) -> str:
    """
    Convert year (supports BCE as negative) to a century label:
    e.g., 1889 -> '19th c.', -447 -> '5th c. BCE'
    """
    y = _safe_int(year)
    if y is None:
        return "unknown"
    if y > 0:
        c = (y - 1) // 100 + 1
        return f"{_ordinal(c)} c."
    else:
        c = ((abs(y) - 1) // 100) + 1
        return f"{_ordinal(c)} c. BCE"

def name_canonical(name: str) -> str:
    """
    Canonicalize monument names:
    - remove parenthetical parts, collapse spaces, strip diacritics
    """
    t = _safe_str(name)
    # Remove parenthetical content
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)
    t = normalize_spaces(t)
    t = remove_diacritics(t)
    return t

def number_to_words_en(x) -> str:
    """
    Convert an integer into English words in lowercase, with spaces only
    (no hyphens, no 'and'), e.g.:
      3   -> "three"
      134 -> "one hundred thirty four"
    Supports negatives and up to trillions.
    """
    def _to_int(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if s == "":
            raise ValueError("empty")
        import re
        s2 = re.sub(r"[^\d+-]", "", s)
        if s2 in ("", "+", "-"):
            raise ValueError("not a number")
        return int(s2)

    units = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    teens = ["ten","eleven","twelve","thirteen","fourteen","fifteen",
             "sixteen","seventeen","eighteen","nineteen"]
    tens_words = ["", "", "twenty","thirty","forty","fifty",
                  "sixty","seventy","eighty","ninety"]
    scales = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000,     "billion"),
        (1_000_000,         "million"),
        (1_000,             "thousand"),
    ]

    def _under_1000(n: int) -> str:
        assert 0 <= n < 1000
        parts = []
        h = n // 100
        r = n % 100
        if h > 0:
            parts.append(f"{units[h]} hundred")
        if r > 0:
            if r < 10:
                parts.append(units[r])
            elif r < 20:
                parts.append(teens[r - 10])
            else:
                t = r // 10
                u = r % 10
                if u == 0:
                    parts.append(tens_words[t])
                else:
                    parts.append(f"{tens_words[t]} {units[u]}")
        if not parts:
            return "zero"
        return " ".join(parts)

    n = _to_int(x)
    if n == 0:
        return "zero"

    sign = "minus " if n < 0 else ""
    n = abs(n)

    parts = []
    for scale_value, scale_name in scales:
        if n >= scale_value:
            chunk = n // scale_value
            n %= scale_value
            parts.append(f"{_under_1000(chunk)} {scale_name}")
    if n > 0:
        parts.append(_under_1000(n))

    return sign + " ".join(parts)

import re
import unicodedata
from typing import Optional

# ---------- Helpers ----------
def _to_str(x) -> str:
    return "" if x is None else str(x)

def _to_number(s: str) -> Optional[float]:
    """
    Extract the first numeric token (supports thousands separators and decimals).
    Examples: '2,499' -> 2499, '6.1 inches' -> 6.1
    """
    if s is None:
        return None
    txt = str(s)
    m = re.findall(r"-?\d[\d,]*\.?\d*", txt)
    if not m:
        return None
    try:
        return float(m[0].replace(",", ""))
    except Exception:
        return None

# ---------- olympic_games ----------
def title_case(s: str) -> str:
    """
    Convert text to title case after lowercasing and normalizing spaces.
    """
    s = _to_str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.title()

def city_canonical(s: str) -> str:
    """
    Canonicalize city: remove punctuation, normalize spaces, lowercase.
    """
    s = _to_str(s)
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def year_decade(y) -> str:
    """
    Return decade label like '1890s' from integer year.
    """
    try:
        yi = int(str(y).strip())
    except Exception:
        return ""
    return f"{(yi // 10) * 10}s"

# ---------- mobiles ----------
def weight_grams(s: str) -> Optional[int]:
    """
    Parse weight like '174g' -> 174 (grams).
    """
    val = _to_number(s)
    return None if val is None else int(round(val))

def battery_mah(s: str) -> Optional[int]:
    """
    Parse capacity like '3,600mAh' -> 3600 (mAh).
    """
    val = _to_number(s)
    return None if val is None else int(round(val))

def screen_inches(s: str) -> Optional[float]:
    """
    Parse screen like '6.1 inches' -> 6.1 (inches).
    """
    val = _to_number(s)
    return None if val is None else float(val)

def ram_gb(s: str) -> Optional[int]:
    """
    Parse RAM like '6GB' -> 6 (GB).
    """
    val = _to_number(s)
    return None if val is None else int(round(val))

def price_usd(s: str) -> Optional[int]:
    """
    Parse 'USD 899' -> 899; fallback to first number if 'USD' not present.
    """
    if s is None:
        return None
    txt = str(s)
    m = re.search(r"USD\s*([\d,]+)", txt, flags=re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))
    val = _to_number(txt)
    return None if val is None else int(round(val))

# ---------- breast_cancer ----------
def tumor_size_band(size) -> str:
    """
    Bucket tumor size:
    'A_<=20', 'B_21_30', 'C_31_50', 'D_50_plus'
    """
    try:
        s = int(float(str(size).strip()))
    except Exception:
        return ""
    if s <= 20:
        return "A_<=20"
    if s <= 30:
        return "B_21_30"
    if s <= 50:
        return "C_31_50"
    return "D_50_plus"

# ---------- fitness_trakers ----------
def price_int(s: str) -> Optional[int]:
    """
    Parse price like '2,499' -> 2499 (integer).
    """
    val = _to_number(s)
    return None if val is None else int(round(val))

def rating_label(x) -> str:
    """
    Map numeric rating to labels: 'low' (<4.0), 'mid' (4.0–4.49), 'high' (>=4.5).
    """
    try:
        r = float(str(x).strip())
    except Exception:
        return ""
    if r < 4.0:
        return "low"
    if r < 4.5:
        return "mid"
    return "high"

def color_canonical(s: str) -> str:
    """
    Take the first comma-separated color, trim, normalize spaces, and title-case it.
    """
    if s is None:
        return ""
    first = str(s).split(",")[0].strip()
    first = re.sub(r"\s+", " ", first)
    return first.title()

# ---------- heart_attack ----------
def age_band(age) -> str:
    """
    Return age buckets '30s', '40s', ... using floor to decade.
    """
    try:
        a = int(float(str(age).strip()))
    except Exception:
        return ""
    return f"{(a // 10) * 10}s"

def sex_label(sex) -> str:
    """
    Map 0 -> 'female', 1 -> 'male'; otherwise ''.
    """
    try:
        v = int(float(str(sex).strip()))
    except Exception:
        return ""
    if v == 0:
        return "female"
    if v == 1:
        return "male"
    return ""

# ---------- accounts_receivable ----------
def settle_speed_band(days) -> str:
    """
    Bucket DaysToSettle into ordered labels:
    'A_00_07', 'B_08_30', 'C_31_60', 'D_61_plus'
    """
    try:
        d = int(float(str(days).strip()))
    except Exception:
        return ""
    if d <= 7:
        return "A_00_07"
    if d <= 30:
        return "B_08_30"
    if d <= 60:
        return "C_31_60"
    return "D_61_plus"

def register_udfs(conn):
    conn.create_function("extract_day", 1, extract_day)
    conn.create_function("extract_month", 1, extract_month)
    conn.create_function("extract_year", 1, extract_year)
    conn.create_function("extract_first_name", 1, extract_first_name)
    conn.create_function("extract_last_name", 1, extract_last_name)
    conn.create_function("extract_initials", 1, extract_initials)
    conn.create_function("reverse_string", 1, reverse_string)
    conn.create_function("word_count", 1, word_count)
      # ========== Helper UDFs ==========
    conn.create_function("strip_accents", 1, _strip_accents)
    conn.create_function("only_alnum_and_space", 1, _only_alnum_and_space)
    conn.create_function("safe_divide", 2, safe_divide)
    
    # ========== Domain UDFs ==========
    conn.create_function("weight_gap", 2, weight_gap)
    conn.create_function("normalize_name", 1, normalize_name)
    conn.create_function("normalize_text", 1, normalize_text)
    conn.create_function("carbon_footprint", 1, carbon_footprint)
    conn.create_function("pass_flag", 1, pass_flag)
    conn.create_function("extract_numeric_suffix", 1, extract_numeric_suffix)
    conn.create_function("revenue_per_unit", 2, revenue_per_unit)
    conn.create_function("demand_elasticity", 2, demand_elasticity)
    conn.create_function("category_popularity", 3, category_popularity)
    conn.create_function("parse_percent", 1, parse_percent)
    conn.create_function("burn_rate", 2, burn_rate)
    conn.create_function("efficiency_score", 2, efficiency_score)
    conn.create_function("delivery_density", 2, delivery_density)
    conn.create_function("route_stress", 2, route_stress)
    
    # ========== Generic text/numeric UDFs ==========
    conn.create_function("upper_case", 1, upper_case)
    conn.create_function("lower_case", 1, lower_case)
    conn.create_function("title_case", 1, title_case)
    conn.create_function("trim_whitespace", 1, trim_whitespace)
    conn.create_function("normalize_spaces", 1, normalize_spaces)
    conn.create_function("remove_accents", 1, remove_accents)
    conn.create_function("to_number", 1, to_number)
    conn.create_function("to_int", 1, to_int)
    conn.create_function("last_name", 1, last_name)
    conn.create_function("duration_category", 1, duration_category)
    conn.create_function("oscars_to_int", 1, oscars_to_int)
    conn.create_function("height_to_feet", 1, height_to_feet)
    conn.create_function("year_to_era", 1, year_to_era)
    conn.create_function("age_from_year", 2, age_from_year)
    conn.create_function("consumption_category", 1, consumption_category)
    conn.create_function("slugify", 1, slugify)
    conn.create_function("weight_category", 1, weight_category)
    conn.create_function("life_weight_index", 2, life_weight_index)
    
    # ========== Movies Dataset UDFs ==========
    conn.create_function("remove_diacritics", 1, remove_diacritics)
    conn.create_function("strip_articles", 1, strip_articles)
    conn.create_function("director_lastname", 1, director_lastname)
    conn.create_function("genre_primary", 1, genre_primary)
    conn.create_function("oscars_bucket", 1, oscars_bucket)
    conn.create_function("year_decade", 1, year_decade)
    
    # ========== Animals Dataset UDFs ==========
    conn.create_function("weight_class", 1, weight_class)
    conn.create_function("lifespan_bucket", 1, lifespan_bucket)
    conn.create_function("habitat_norm", 1, habitat_norm)
    
    # ========== Cars Dataset UDFs ==========
    conn.create_function("efficiency_label", 2, efficiency_label)
    conn.create_function("country_code", 1, country_code)
    
    # ========== Monuments Dataset UDFs ==========
    conn.create_function("year_century", 1, year_century)
    conn.create_function("name_canonical", 1, name_canonical)
    conn.create_function("number_to_words_en", 1, number_to_words_en)
    
    # ========== Olympic Games Dataset UDFs ==========
    conn.create_function("city_canonical", 1, city_canonical)
    
    # ========== Mobile Phones Dataset UDFs ==========
    conn.create_function("weight_grams", 1, weight_grams)
    conn.create_function("battery_mah", 1, battery_mah)
    conn.create_function("screen_inches", 1, screen_inches)
    conn.create_function("ram_gb", 1, ram_gb)
    conn.create_function("price_usd", 1, price_usd)
    
    # ========== Breast Cancer Dataset UDFs ==========
    conn.create_function("tumor_size_band", 1, tumor_size_band)
    
    # ========== Fitness Trackers Dataset UDFs ==========
    conn.create_function("price_int", 1, price_int)
    conn.create_function("rating_label", 1, rating_label)
    conn.create_function("color_canonical", 1, color_canonical)
    
    # ========== Heart Attack Dataset UDFs ==========
    conn.create_function("age_band", 1, age_band)
    conn.create_function("sex_label", 1, sex_label)
    
    # ========== Accounts Receivable Dataset UDFs ==========
    conn.create_function("settle_speed_band", 1, settle_speed_band)

# ==========================
# SQL Execution functions
# ==========================

def execute_sql_query(query, db_path):
    """Execute SQL query and return results as list of lists."""
    try:
        conn = sqlite3.connect(db_path)
        register_udfs(conn)
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Get results and convert to list of lists (removing dictionary keys)
        results = cursor.fetchall()
        
        conn.close()
        return list(results)  # Convert tuples to lists
        
    except Exception as e:
        print(f"Error executing query: {query[:100]}... Error: {e}")
        return []

# ==========================
# Subclass OrchestratorEvaluator
# ==========================

class OrchestratorEvaluatorWithUDFs(OrchestratorEvaluator):
    """Subclass that registers custom SQLite UDFs on each DB connection."""
    def _connect(self, db_path):
        # Use original connection method if exists, otherwise sqlite3.connect
        conn = getattr(super(), "_connect", lambda path: sqlite3.connect(path))(db_path)
        register_udfs(conn)
        return conn

# ==========================
# Custom metric: TupleOrder
# ==========================

def compute_tuple_order(target, prediction) -> float:
    """
    Calcola tuple_order come rapporto tra la cardinalità di result (prediction)
    e sql_query (target), normalizzato in [0, 1].
    """
    len_target = len(target)
    len_pred = len(prediction)

    if len_target == len_pred == 0:
        return 1.0
    if len_target == 0:
        return 0.0
    if len_pred == 0:
        return 0.0

    if len_pred >= len_target:
        return round(len_target / len_pred, 3)
    return round(len_pred / len_target, 3)

# ==========================
# Main processing functions
# ==========================

def compute_avg_res(input_csv, output_csv):
    """Calcola medie raggruppate su colonne specifiche."""
    df = pd.read_csv(input_csv)

    # Determine grouping columns based on what's available
    possible_group_cols = ['table', 'model', 'table_name']
    group_cols = []
    
    for col in possible_group_cols:
        if col in df.columns:
            group_cols.append(col)
    
    # If we have both 'table' and 'table_name', prefer 'table'
    if 'table' in group_cols and 'table_name' in group_cols:
        group_cols.remove('table_name')
    
    # Ensure we have at least model for grouping
    if 'model' not in group_cols:
        print("Warning: 'model' column not found for grouping")
        return

    avg_cols = [
        'valid_efficiency_score',
        'cell_precision',
        'cell_recall',
        'execution_accuracy',
        'tuple_cardinality',
        'tuple_constraint',
        'tuple_order'
    ]
    
    # Only include columns that exist in the dataframe
    existing_avg_cols = [col for col in avg_cols if col in df.columns]

    if not existing_avg_cols:
        print("Warning: No metric columns found for averaging")
        return

    summary_df = df.groupby(group_cols)[existing_avg_cols].mean().reset_index()
    summary_df.rename(columns={col: f'avg_{col}' for col in existing_avg_cols}, inplace=True)
    summary_df.to_csv(output_csv, index=False)
    print(f'Average results saved to {output_csv}')

def process_file(input_path):
    """Execute SQL queries, evaluate results, and save to new location without modifying input."""
    print(f'Processing {input_path}...')
    
    # Read the input file
    df = pd.read_csv(input_path)
    
    # Detect format and normalize column names
    try:
        processed_df = detect_format_and_normalize(df)
    except ValueError as e:
        print(f"Error processing {input_path}: {e}")
        return None, None
    
    # Convert result column if needed (handle string representations of lists)
    if 'result' in processed_df.columns:
        processed_df['result'] = processed_df['result'].apply(convert_result_column)
    
    # Execute SQL queries and replace results
    for idx, row in processed_df.iterrows():
        if pd.notna(row['sql_query']) and pd.notna(row['db_path']):
            sql_result = execute_sql_query(row['sql_query'], row['db_path'])
            processed_df.at[idx, 'sql_query'] = sql_result
        else:
            processed_df.at[idx, 'sql_query'] = []

    # Create evaluator and evaluate
    evaluator = OrchestratorEvaluatorWithUDFs()
    res = evaluator.evaluate_df(
        df=processed_df,
        target_col_name='sql_query',  # Now contains executed results
        prediction_col_name='result',
        db_path_name='db_path',
    )

    # ==========================
    # Aggiungiamo la metrica tuple_order
    # ==========================
    tuple_orders = []
    for idx, row in processed_df.iterrows():
        target = row['sql_query']
        prediction = row['result']
        val = compute_tuple_order(target, prediction)
        tuple_orders.append(val)

    # Inseriamo la colonna nel punto giusto (dopo tuple_constraint se esiste)
    if "tuple_order" in res.columns:
        res["tuple_order"] = tuple_orders
    else:
        insert_at = res.columns.get_loc("tuple_constraint") + 1 if "tuple_constraint" in res.columns else len(res.columns)
        res.insert(insert_at, "tuple_order", tuple_orders)

    # Create output directory structure
    res_path = input_path.replace('output', 'results_UDF')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    
    # Save detailed results
    res.to_csv(res_path, index=False)
    print(f'Saved detailed results to {res_path}')

    # Calculate and save averages
    avg_path = res_path.replace('.csv', '_avg.csv')
    compute_avg_res(res_path, avg_path)

    return res_path, avg_path

def process_all_files(input_dir):
    """Scan recursively all CSV files and process them one by one."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_path = os.path.join(root, file)
                try:
                    result = process_file(input_path)
                    if result[0] is None:
                        print(f'Skipped {input_path} due to format issues')
                except Exception as e:
                    print(f'Error processing {input_path}: {e}')

# ==========================
# Entry point
# ==========================

if __name__ == '__main__':
    input_dir = 'output/UDF/classic_ds_lama_deep'  # Change this directory as needed
    process_all_files(input_dir)