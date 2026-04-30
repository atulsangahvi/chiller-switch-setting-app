# chiller_pressure_switch_app.py
# Streamlit app to estimate preliminary pressure-switch settings for Freon chillers.
# Features:
# - Manual entry of refrigerant, design temperatures, switch philosophy, and compressor limits.
# - PDF upload for compressor datasheet text extraction.
# - Automatic candidate extraction of compressor model, refrigerant, pressure/temperature limits.
# - Missing-data flagging.
# - Single circuit, two separate circuits, and tandem compressor logic support.
#
# Run with:
#   streamlit run chiller_pressure_switch_app.py
#
# Recommended install:
#   pip install streamlit CoolProp pandas PyMuPDF pypdf

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from CoolProp.CoolProp import PropsSI
except Exception:  # pragma: no cover
    PropsSI = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


# ----------------------------
# Refrigerant database
# ----------------------------


@dataclass
class RefrigerantInfo:
    coolprop_name: str
    display_name: str
    note: str = ""


REFRIGERANTS: Dict[str, RefrigerantInfo] = {
    "R134a": RefrigerantInfo("R134a", "R134a"),
    "R407C": RefrigerantInfo("R407C", "R407C", "Zeotropic blend: use dew/bubble values carefully."),
    "R410A": RefrigerantInfo("R410A", "R410A", "High-pressure refrigerant."),
    "R404A": RefrigerantInfo("R404A", "R404A"),
    "R507A": RefrigerantInfo("R507A", "R507A"),
    "R22": RefrigerantInfo("R22", "R22", "Legacy refrigerant; check local restrictions."),
    "R1234yf": RefrigerantInfo("R1234yf", "R1234yf"),
    "R1234ze(E)": RefrigerantInfo("R1234ze(E)", "R1234ze(E)"),
    "R513A": RefrigerantInfo("R513A", "R513A"),
    "R32": RefrigerantInfo("R32", "R32", "A2L refrigerant; apply relevant safety standards."),
    "R290": RefrigerantInfo("R290", "R290 / Propane", "A3 flammable refrigerant; specialist safety design required."),
}

REFRIGERANT_ALIASES = {
    "R-134A": "R134a",
    "R134A": "R134a",
    "R-407C": "R407C",
    "R407C": "R407C",
    "R-410A": "R410A",
    "R410A": "R410A",
    "R-404A": "R404A",
    "R404A": "R404A",
    "R-507A": "R507A",
    "R507A": "R507A",
    "R-22": "R22",
    "R22": "R22",
    "R-32": "R32",
    "R32": "R32",
    "R-513A": "R513A",
    "R513A": "R513A",
    "R-1234YF": "R1234yf",
    "R1234YF": "R1234yf",
    "R-1234ZE": "R1234ze(E)",
    "R1234ZE": "R1234ze(E)",
    "R290": "R290",
    "PROPANE": "R290",
}

BAR_PER_PA = 1e-5
PSI_PER_PA = 0.0001450377377
ATM_BAR = 1.01325


# ----------------------------
# Unit and pressure helpers
# ----------------------------


def c_to_k(temp_c: float) -> float:
    return temp_c + 273.15


def pa_to_bar_g(pa_abs: float) -> float:
    return pa_abs * BAR_PER_PA - ATM_BAR


def pa_to_bar_abs(pa_abs: float) -> float:
    return pa_abs * BAR_PER_PA


def pa_to_psig(pa_abs: float) -> float:
    return pa_abs * PSI_PER_PA - 14.6959


def barg_to_pa_abs(barg: float) -> float:
    return (barg + ATM_BAR) / BAR_PER_PA


def pa_to_display(pa_abs: float, units: str) -> str:
    if pa_abs is None or math.isnan(pa_abs):
        return "—"
    if units == "bar(g)":
        return f"{pa_to_bar_g(pa_abs):.2f} bar(g)"
    if units == "bar(abs)":
        return f"{pa_to_bar_abs(pa_abs):.2f} bar(abs)"
    if units == "psig":
        return f"{pa_to_psig(pa_abs):.0f} psig"
    return f"{pa_abs:.0f} Pa(abs)"


def sat_pressure_pa(ref: str, temp_c: float, quality: float) -> float:
    """Saturation pressure at temperature.

    quality = 1.0 = dew pressure, normally used for suction/vapor side.
    quality = 0.0 = bubble pressure, normally used for liquid/subcooling side.
    """
    if PropsSI is None:
        raise RuntimeError("CoolProp is not installed. Install it with: pip install CoolProp")
    return float(PropsSI("P", "T", c_to_k(temp_c), "Q", quality, ref))


# ----------------------------
# PDF extraction and parsing
# ----------------------------


@dataclass
class ExtractedField:
    value: Any
    confidence: str
    source: str


def extract_pdf_text(uploaded_file) -> Tuple[str, str, List[str]]:
    """Extract text from a PDF file using available local libraries.

    Returns: text, method_used, warnings
    """
    warnings: List[str] = []
    data = uploaded_file.getvalue()

    # Method 1: PyMuPDF
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            if len(text.strip()) > 100:
                return text, "PyMuPDF", warnings
            warnings.append("PyMuPDF extracted very little text; the PDF may be scanned or image-based.")
        except Exception as exc:
            warnings.append(f"PyMuPDF extraction failed: {exc}")

    # Method 2: pypdf fallback
    if PdfReader is not None:
        try:
            reader = PdfReader(uploaded_file)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if len(text.strip()) > 100:
                return text, "pypdf", warnings
            warnings.append("pypdf extracted very little text; OCR may be needed for scanned PDFs.")
        except Exception as exc:
            warnings.append(f"pypdf extraction failed: {exc}")

    return "", "none", warnings + ["No usable text could be extracted. Use a text-based PDF or add OCR."]


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _find_first(patterns: List[str], text: str, flags=re.IGNORECASE) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return None


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except Exception:
        return None


def _pressure_to_barg(number: float, unit: str) -> float:
    unit = unit.lower().replace(" ", "")
    if unit in ["bar", "barg", "bar(g)"]:
        return number
    if unit in ["bara", "barabs", "bar(a)"]:
        return number - ATM_BAR
    if unit in ["mpa", "mpag"]:
        return number * 10.0
    if unit in ["kpa", "kpag"]:
        return number / 100.0
    if unit in ["psi", "psig"]:
        return number * 0.0689476
    if unit in ["psia"]:
        return number * 0.0689476 - ATM_BAR
    return number


def _find_pressure_near_keywords(text: str, keywords: List[str]) -> Optional[ExtractedField]:
    # Search short windows around each keyword set.
    for kw in keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 220)
            window = text[start:end]
            p = re.search(
                r"([0-9]+(?:\.[0-9]+)?)\s*(bar\s*\(?g?\)?|bar\s*abs|bara|barg|MPa|kPa|psi|psig|psia)",
                window,
                re.IGNORECASE,
            )
            if p:
                value = _parse_float(p.group(1))
                if value is not None:
                    barg = _pressure_to_barg(value, p.group(2))
                    return ExtractedField(round(barg, 3), "medium", window.strip()[:500])
    return None


def _find_temperature_near_keywords(text: str, keywords: List[str]) -> Optional[ExtractedField]:
    for kw in keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 220)
            window = text[start:end]
            t = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:°\s*C|deg\s*C|C\b)", window, re.IGNORECASE)
            if t:
                value = _parse_float(t.group(1))
                if value is not None:
                    return ExtractedField(value, "medium", window.strip()[:500])
    return None


def parse_compressor_datasheet(text: str) -> Dict[str, ExtractedField]:
    """Best-effort parser for compressor datasheets.

    This parser is intentionally conservative. It fills candidates and the UI asks the user to verify them.
    Supplier PDFs vary widely, so vendor-specific templates can be added later.
    """
    text = _normalize_text(text)
    found: Dict[str, ExtractedField] = {}

    model = _find_first(
        [
            r"(?:Compressor\s+Model|Model\s+No\.?|Model|Type)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\. ]{2,40})",
            r"(?:Designation)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\. ]{2,40})",
        ],
        text,
    )
    if model:
        found["compressor_model"] = ExtractedField(model[:60], "low", "Matched model/type pattern. Verify manually.")

    make = _find_first(
        [
            r"\b(Copeland|Emerson|Danfoss|Maneurop|Bitzer|Frascold|RefComp|Hanbell|Carrier|Trane|Daikin|Mitsubishi)\b",
        ],
        text,
    )
    if make:
        found["compressor_make"] = ExtractedField(make, "medium", "Matched known compressor/company name.")

    # Refrigerant candidates: list all recognizable refrigerants in the text.
    upper = text.upper().replace(" ", "")
    refrigerants: List[str] = []
    for alias, canonical in REFRIGERANT_ALIASES.items():
        if alias.replace(" ", "") in upper and canonical not in refrigerants:
            refrigerants.append(canonical)
    if refrigerants:
        found["refrigerants"] = ExtractedField(", ".join(refrigerants), "medium", "Matched refrigerant names in datasheet text.")
        found["first_refrigerant"] = ExtractedField(refrigerants[0], "medium", "First matched refrigerant candidate.")

    max_hp = _find_pressure_near_keywords(
        text,
        [
            r"maximum\s+(?:allowable\s+)?(?:high|discharge|operating)\s+pressure",
            r"max\.?\s+(?:high|discharge)\s+pressure",
            r"high\s+pressure\s+cut\s*out",
            r"standstill\s+pressure",
            r"design\s+pressure",
        ],
    )
    if max_hp:
        found["max_high_pressure_barg"] = max_hp

    max_cond = _find_temperature_near_keywords(
        text,
        [
            r"maximum\s+condensing\s+temperature",
            r"max\.?\s+condensing\s+temperature",
            r"condensing\s+temperature\s+max",
        ],
    )
    if max_cond:
        found["max_condensing_temp_c"] = max_cond

    min_evap = _find_temperature_near_keywords(
        text,
        [
            r"minimum\s+evaporating\s+temperature",
            r"min\.?\s+evaporating\s+temperature",
            r"evaporating\s+temperature\s+min",
            r"operating\s+envelope",
        ],
    )
    if min_evap:
        found["min_evaporating_temp_c"] = min_evap

    rla = _find_first([r"(?:RLA|MCC|Rated\s+load\s+amps?|Max\.?\s+operating\s+current)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*A"], text)
    if rla:
        found["rated_current_a"] = ExtractedField(float(rla), "medium", "Matched RLA/MCC/current pattern.")

    lra = _find_first([r"(?:LRA|Locked\s+rotor\s+amps?)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*A"], text)
    if lra:
        found["locked_rotor_current_a"] = ExtractedField(float(lra), "medium", "Matched LRA pattern.")

    return found


# ----------------------------
# Chiller calculations
# ----------------------------


@dataclass
class CircuitInputs:
    ref_key: str
    evap_c: float
    cond_c: float
    subcool_k: float
    max_hp_cutout_barg: float
    max_condensing_temp_c: float
    min_evaporating_temp_c: float
    fan1_on_c: float
    fan1_off_c: float
    fan2_on_c: float
    fan2_off_c: float
    lps_cutout_c: float
    lps_cutin_c: float
    hgb_used: bool
    hgb_open_c: float
    hgb_close_c: float
    hps_margin_k: float
    pump_start_delay_s: int
    flow_proving_delay_s: int
    lp_bypass_delay_s: int
    anti_short_cycle_s: int
    min_on_time_s: int
    pumpdown_max_s: int
    pump_off_delay_s: int
    fan_stage_delay_s: int


def calc_settings(ci: CircuitInputs) -> Dict[str, float]:
    ref = REFRIGERANTS[ci.ref_key].coolprop_name
    values: Dict[str, float] = {}
    values["normal_suction_dew"] = sat_pressure_pa(ref, ci.evap_c, 1.0)
    values["normal_condensing_dew"] = sat_pressure_pa(ref, ci.cond_c, 1.0)
    values["liquid_after_subcool_bubble"] = sat_pressure_pa(ref, ci.cond_c - ci.subcool_k, 0.0)

    values["cps1_on"] = sat_pressure_pa(ref, ci.fan1_on_c, 1.0)
    values["cps1_off"] = sat_pressure_pa(ref, ci.fan1_off_c, 1.0)
    values["cps2_on"] = sat_pressure_pa(ref, ci.fan2_on_c, 1.0)
    values["cps2_off"] = sat_pressure_pa(ref, ci.fan2_off_c, 1.0)

    values["lps_cutout"] = sat_pressure_pa(ref, ci.lps_cutout_c, 1.0)
    values["lps_cutin"] = sat_pressure_pa(ref, ci.lps_cutin_c, 1.0)

    if ci.hgb_used:
        values["hgb_open"] = sat_pressure_pa(ref, ci.hgb_open_c, 1.0)
        values["hgb_close"] = sat_pressure_pa(ref, ci.hgb_close_c, 1.0)
    else:
        values["hgb_open"] = float("nan")
        values["hgb_close"] = float("nan")

    hps_by_temp = sat_pressure_pa(ref, ci.cond_c + ci.hps_margin_k, 1.0)
    hps_limit = barg_to_pa_abs(ci.max_hp_cutout_barg)
    values["hps_by_temp"] = hps_by_temp
    values["hps_limit"] = hps_limit
    values["hps_cutout"] = min(hps_by_temp, hps_limit)
    return values


def validate_inputs(ci: CircuitInputs, compressor_refrigerants: List[str]) -> List[str]:
    warnings: List[str] = []
    if compressor_refrigerants and ci.ref_key not in compressor_refrigerants:
        warnings.append(f"Selected refrigerant {ci.ref_key} was not found in the uploaded compressor datasheet refrigerant list: {', '.join(compressor_refrigerants)}.")
    if ci.cond_c > ci.max_condensing_temp_c:
        warnings.append("Design condensing temperature is above compressor maximum condensing temperature.")
    if ci.evap_c < ci.min_evaporating_temp_c:
        warnings.append("Design evaporating temperature is below compressor minimum evaporating temperature.")
    if ci.fan1_off_c >= ci.fan1_on_c:
        warnings.append("CPS1 OFF temperature should be lower than CPS1 ON temperature.")
    if ci.fan2_off_c >= ci.fan2_on_c:
        warnings.append("CPS2 OFF temperature should be lower than CPS2 ON temperature.")
    if ci.fan2_on_c <= ci.fan1_on_c:
        warnings.append("CPS2 ON should normally be higher than CPS1 ON.")
    if ci.lps_cutin_c <= ci.lps_cutout_c:
        warnings.append("LPS cut-in should be higher than LPS cut-out.")
    if ci.lps_cutout_c < ci.min_evaporating_temp_c:
        warnings.append("LPS cut-out equivalent evaporating temperature is below compressor minimum evaporating temperature.")
    if ci.hgb_used:
        if ci.hgb_open_c <= ci.lps_cutout_c:
            warnings.append("Hot gas bypass should normally open above LPS cut-out, otherwise the compressor may trip before bypass opens.")
        if ci.hgb_close_c <= ci.hgb_open_c:
            warnings.append("Hot gas bypass close temperature should be higher than open temperature.")
    if ci.pump_start_delay_s < ci.flow_proving_delay_s:
        warnings.append("Pump start delay should normally be equal to or longer than flow proving delay.")
    if ci.anti_short_cycle_s < 180:
        warnings.append("Anti-short-cycle delay below 180 seconds may be too short for many compressors. Check manufacturer guidance.")
    return warnings


def output_rows(ci: CircuitInputs, values: Dict[str, float], units: str) -> List[Dict[str, str]]:
    rows = [
        {
            "Device": "CPS1",
            "Purpose": "Fan 1 condenser pressure control",
            "ON / Cut-in": pa_to_display(values["cps1_on"], units),
            "OFF / Cut-out": pa_to_display(values["cps1_off"], units),
            "Reset": "Automatic differential",
            "Temperature basis": f"ON {ci.fan1_on_c:.1f} °C cond., OFF {ci.fan1_off_c:.1f} °C cond.",
            "Notes": "First condenser fan stage.",
        },
        {
            "Device": "CPS2",
            "Purpose": "Fan 2 condenser pressure control",
            "ON / Cut-in": pa_to_display(values["cps2_on"], units),
            "OFF / Cut-out": pa_to_display(values["cps2_off"], units),
            "Reset": "Automatic differential",
            "Temperature basis": f"ON {ci.fan2_on_c:.1f} °C cond., OFF {ci.fan2_off_c:.1f} °C cond.",
            "Notes": "Second condenser fan stage.",
        },
        {
            "Device": "LPS",
            "Purpose": "Pump-down stop and low suction protection",
            "ON / Cut-in": pa_to_display(values["lps_cutin"], units),
            "OFF / Cut-out": pa_to_display(values["lps_cutout"], units),
            "Reset": "Automatic for pump-down",
            "Temperature basis": f"Cut-out {ci.lps_cutout_c:.1f} °C evap., cut-in {ci.lps_cutin_c:.1f} °C evap.",
            "Notes": "Cut-in must be reached after YV1 opens before compressor restart.",
        },
        {
            "Device": "HPS",
            "Purpose": "High pressure safety",
            "ON / Cut-in": "Manual reset preferred",
            "OFF / Cut-out": pa_to_display(values["hps_cutout"], units),
            "Reset": "Manual reset",
            "Temperature basis": f"Design cond. + margin = {ci.cond_c + ci.hps_margin_k:.1f} °C, limited by max HP if needed",
            "Notes": "Must be below compressor/system design pressure.",
        },
    ]
    if ci.hgb_used:
        rows.insert(
            3,
            {
                "Device": "YV2 / HGBP",
                "Purpose": "Hot gas bypass / low-load support",
                "ON / Cut-in": pa_to_display(values["hgb_open"], units),
                "OFF / Cut-out": pa_to_display(values["hgb_close"], units),
                "Reset": "Automatic differential or modulating valve",
                "Temperature basis": f"Open {ci.hgb_open_c:.1f} °C evap., close {ci.hgb_close_c:.1f} °C evap.",
                "Notes": "Should open before LPS cut-out.",
            },
        )
    return rows


# ----------------------------
# Streamlit UI helpers
# ----------------------------


def default_from_parsed(parsed: Dict[str, ExtractedField], key: str, default: Any) -> Any:
    if key in parsed:
        return parsed[key].value
    return default


def required_data_report(parsed: Dict[str, ExtractedField]) -> pd.DataFrame:
    required = [
        ("compressor_model", "Compressor model", "For record and report"),
        ("refrigerants", "Approved refrigerant(s)", "To check selected refrigerant"),
        ("max_high_pressure_barg", "Maximum allowed high-side pressure", "To limit HPS cut-out"),
        ("max_condensing_temp_c", "Maximum condensing temperature", "To validate design Tcond"),
        ("min_evaporating_temp_c", "Minimum evaporating temperature", "To validate LPS/Tevap"),
    ]
    rows = []
    for key, label, use in required:
        if key in parsed:
            rows.append({"Data item": label, "Status": "Found - verify", "Value": parsed[key].value, "Why needed": use})
        else:
            rows.append({"Data item": label, "Status": "Missing", "Value": "", "Why needed": use})
    return pd.DataFrame(rows)


def circuit_input_form(prefix: str, parsed: Dict[str, ExtractedField], compressor_refrigerants: List[str]) -> CircuitInputs:
    # Refrigerant default from parsed if possible.
    parsed_ref = default_from_parsed(parsed, "first_refrigerant", "R407C")
    ref_options = list(REFRIGERANTS.keys())
    ref_index = ref_options.index(parsed_ref) if parsed_ref in ref_options else ref_options.index("R407C")

    col1, col2, col3 = st.columns(3)
    with col1:
        ref_key = st.selectbox("Refrigerant", ref_options, index=ref_index, key=f"{prefix}_ref")
        evap_c = st.number_input("Design evaporating temp, °C", value=3.0, step=0.5, key=f"{prefix}_evap")
        cond_c = st.number_input("Design condensing temp, °C", value=50.0, step=0.5, key=f"{prefix}_cond")
        subcool_k = st.number_input("Design subcooling, K", value=5.0, min_value=0.0, step=0.5, key=f"{prefix}_subcool")
    with col2:
        max_hp_cutout_barg = st.number_input(
            "Compressor/system max high pressure, bar(g)",
            value=float(default_from_parsed(parsed, "max_high_pressure_barg", 30.0)),
            step=0.5,
            key=f"{prefix}_maxhp",
        )
        max_condensing_temp_c = st.number_input(
            "Compressor max condensing temp, °C",
            value=float(default_from_parsed(parsed, "max_condensing_temp_c", 65.0)),
            step=0.5,
            key=f"{prefix}_maxcond",
        )
        min_evaporating_temp_c = st.number_input(
            "Compressor min evaporating temp, °C",
            value=float(default_from_parsed(parsed, "min_evaporating_temp_c", -10.0)),
            step=0.5,
            key=f"{prefix}_minevap",
        )
    with col3:
        hps_margin_k = st.number_input("HPS margin above design condensing temp, K", value=10.0, step=0.5, key=f"{prefix}_hps_margin")
        hgb_used = st.checkbox("Use hot gas bypass / YV2", value=True, key=f"{prefix}_hgb_used")

    st.subheader("Fan pressure switch temperature basis")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        fan1_on_c = st.number_input("CPS1 Fan 1 ON, °C cond.", value=42.0, step=0.5, key=f"{prefix}_fan1_on")
    with f2:
        fan1_off_c = st.number_input("CPS1 Fan 1 OFF, °C cond.", value=36.0, step=0.5, key=f"{prefix}_fan1_off")
    with f3:
        fan2_on_c = st.number_input("CPS2 Fan 2 ON, °C cond.", value=48.0, step=0.5, key=f"{prefix}_fan2_on")
    with f4:
        fan2_off_c = st.number_input("CPS2 Fan 2 OFF, °C cond.", value=42.0, step=0.5, key=f"{prefix}_fan2_off")

    st.subheader("Low pressure and hot gas bypass temperature basis")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        lps_cutout_c = st.number_input("LPS cut-out, °C evap.", value=-1.0, step=0.5, key=f"{prefix}_lps_out")
    with l2:
        lps_cutin_c = st.number_input("LPS cut-in, °C evap.", value=5.0, step=0.5, key=f"{prefix}_lps_in")
    with l3:
        hgb_open_c = st.number_input("YV2/HGBP open, °C evap.", value=1.0, step=0.5, disabled=not hgb_used, key=f"{prefix}_hgb_open")
    with l4:
        hgb_close_c = st.number_input("YV2/HGBP close, °C evap.", value=4.0, step=0.5, disabled=not hgb_used, key=f"{prefix}_hgb_close")

    with st.expander("Timer inputs"):
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            pump_start_delay_s = st.number_input("Pump start delay before compressor, sec", value=30, min_value=0, step=5, key=f"{prefix}_td1")
            flow_proving_delay_s = st.number_input("Flow proving delay, sec", value=10, min_value=0, step=5, key=f"{prefix}_flowdelay")
        with t2:
            lp_bypass_delay_s = st.number_input("LP bypass delay after start, sec", value=60, min_value=0, step=5, key=f"{prefix}_lpdelay")
            anti_short_cycle_s = st.number_input("Anti-short-cycle timer, sec", value=180, min_value=0, step=30, key=f"{prefix}_ast")
        with t3:
            min_on_time_s = st.number_input("Minimum compressor ON time, sec", value=120, min_value=0, step=30, key=f"{prefix}_minon")
            pumpdown_max_s = st.number_input("Maximum pump-down time, sec", value=90, min_value=0, step=10, key=f"{prefix}_pumpdown")
        with t4:
            pump_off_delay_s = st.number_input("Pump OFF delay after compressor stop, sec", value=120, min_value=0, step=10, key=f"{prefix}_pumpoff")
            fan_stage_delay_s = st.number_input("Fan stage delay, sec", value=10, min_value=0, step=5, key=f"{prefix}_fandelay")

    return CircuitInputs(
        ref_key=ref_key,
        evap_c=evap_c,
        cond_c=cond_c,
        subcool_k=subcool_k,
        max_hp_cutout_barg=max_hp_cutout_barg,
        max_condensing_temp_c=max_condensing_temp_c,
        min_evaporating_temp_c=min_evaporating_temp_c,
        fan1_on_c=fan1_on_c,
        fan1_off_c=fan1_off_c,
        fan2_on_c=fan2_on_c,
        fan2_off_c=fan2_off_c,
        lps_cutout_c=lps_cutout_c,
        lps_cutin_c=lps_cutin_c,
        hgb_used=hgb_used,
        hgb_open_c=hgb_open_c,
        hgb_close_c=hgb_close_c,
        hps_margin_k=hps_margin_k,
        pump_start_delay_s=int(pump_start_delay_s),
        flow_proving_delay_s=int(flow_proving_delay_s),
        lp_bypass_delay_s=int(lp_bypass_delay_s),
        anti_short_cycle_s=int(anti_short_cycle_s),
        min_on_time_s=int(min_on_time_s),
        pumpdown_max_s=int(pumpdown_max_s),
        pump_off_delay_s=int(pump_off_delay_s),
        fan_stage_delay_s=int(fan_stage_delay_s),
    )


def show_circuit_results(name: str, ci: CircuitInputs, units: str, compressor_refrigerants: List[str]) -> None:
    st.markdown(f"### {name} results")
    try:
        vals = calc_settings(ci)
    except Exception as exc:
        st.error(f"Could not calculate pressure settings for {name}: {exc}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Normal suction pressure", pa_to_display(vals["normal_suction_dew"], units))
    m2.metric("Normal condensing pressure", pa_to_display(vals["normal_condensing_dew"], units))
    m3.metric("Liquid pressure after subcooling reference", pa_to_display(vals["liquid_after_subcool_bubble"], units))

    warnings = validate_inputs(ci, compressor_refrigerants)
    if vals["hps_cutout"] < vals["hps_by_temp"]:
        warnings.append("HPS temperature-based setting was above the entered maximum high-pressure limit, so it has been limited to the entered maximum.")
    for w in warnings:
        st.warning(w)

    st.dataframe(pd.DataFrame(output_rows(ci, vals, units)), use_container_width=True, hide_index=True)

    with st.expander("Control sequence for this circuit"):
        st.code(
            f"""START:
K0 control ON → pump starts → wait {ci.pump_start_delay_s} sec pump delay → flow must prove within {ci.flow_proving_delay_s} sec → TC1 calls cooling → YV1 opens → suction rises → LPS closes → AST satisfied → compressor starts.

NORMAL STOP:
TC1 satisfied → YV1 closes → compressor pumps down → LPS opens → compressor stops → anti-short-cycle timer {ci.anti_short_cycle_s} sec starts → pump continues {ci.pump_off_delay_s} sec.

FAN STAGING:
Compressor running → CPS1 starts fan 1 → CPS2 starts fan 2 → optional fan stage delay {ci.fan_stage_delay_s} sec.

SAFETY STOP:
HPS / flow fail / freeze stat / overload / phase relay trip → compressor stops immediately.
""",
            language="text",
        )


# ----------------------------
# Main app
# ----------------------------


st.set_page_config(page_title="Chiller Pressure Switch Setting Tool", layout="wide")

st.title("Chiller Pressure Switch Setting Tool")
st.caption("Preliminary pressure switch and timer setting calculator for Freon chillers. Includes compressor datasheet PDF upload and multi-circuit logic.")

if PropsSI is None:
    st.error("CoolProp is not installed. Install dependencies first: pip install streamlit CoolProp pandas PyMuPDF pypdf")
    st.stop()

parsed: Dict[str, ExtractedField] = {}
compressor_refrigerants: List[str] = []

with st.sidebar:
    st.header("1) Upload compressor datasheet")
    uploaded = st.file_uploader("Upload compressor PDF datasheet", type=["pdf"])
    if uploaded is not None:
        text, method, extraction_warnings = extract_pdf_text(uploaded)
        for w in extraction_warnings:
            st.warning(w)
        if text:
            parsed = parse_compressor_datasheet(text)
            st.success(f"PDF text extracted using {method}. Candidate data found: {len(parsed)} fields.")
            if "refrigerants" in parsed:
                compressor_refrigerants = [r.strip() for r in str(parsed["refrigerants"].value).split(",") if r.strip()]
            with st.expander("Show extracted PDF text"):
                st.text_area("Extracted text", text[:20000], height=300)
        else:
            st.error("No usable text extracted. For scanned PDFs, add OCR or manually enter data.")

    st.header("2) App configuration")
    units = st.radio("Pressure display unit", ["bar(g)", "bar(abs)", "psig"], index=0)
    system_type = st.selectbox(
        "Chiller configuration",
        [
            "Single compressor / single refrigerant circuit",
            "Two compressors / two separate refrigerant circuits",
            "Tandem compressors / one common refrigerant circuit",
        ],
    )

st.markdown("## Uploaded Datasheet Data Check")
if parsed:
    st.dataframe(required_data_report(parsed), use_container_width=True, hide_index=True)
    with st.expander("All extracted candidate fields"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": k, "Value": v.value, "Confidence": v.confidence, "Source / comment": v.source}
                    for k, v in parsed.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Upload a text-based compressor PDF to auto-fill candidate compressor data, or enter values manually below.")

st.markdown("---")

if system_type == "Single compressor / single refrigerant circuit":
    st.header("Single Circuit Inputs")
    ci = circuit_input_form("c1", parsed, compressor_refrigerants)
    st.markdown("---")
    show_circuit_results("Circuit 1", ci, units, compressor_refrigerants)

elif system_type == "Two compressors / two separate refrigerant circuits":
    st.header("Two Independent Refrigerant Circuits")
    st.info(
        "Use this when compressor 1 and compressor 2 have separate refrigerant circuits, separate evaporator circuits/sections, separate condensers/sections, separate YV1 valves, and separate HP/LP/fan pressure controls. The app calculates settings circuit by circuit."
    )
    tab1, tab2 = st.tabs(["Circuit 1", "Circuit 2"])
    with tab1:
        ci1 = circuit_input_form("c1", parsed, compressor_refrigerants)
    with tab2:
        st.caption("If circuit 2 uses a different compressor datasheet, upload/read that datasheet in a future version or manually edit the values here.")
        ci2 = circuit_input_form("c2", parsed, compressor_refrigerants)
    st.markdown("---")
    show_circuit_results("Circuit 1", ci1, units, compressor_refrigerants)
    show_circuit_results("Circuit 2", ci2, units, compressor_refrigerants)

else:
    st.header("Tandem Compressors on One Common Refrigerant Circuit")
    st.info(
        "Use this when two compressors share one refrigerant circuit, common suction/discharge headers, common evaporator and condenser circuit, common liquid line solenoid YV1, and common pressure controls. Individual compressors still need their own contactors, overloads, anti-short-cycle timers, and fault indication."
    )
    ci = circuit_input_form("tandem", parsed, compressor_refrigerants)
    st.subheader("Tandem staging inputs")
    a, b, c = st.columns(3)
    with a:
        lead_lag_enabled = st.checkbox("Enable lead/lag rotation", value=True)
        lag_start_delay_s = st.number_input("Lag compressor start delay after lead, sec", value=120, min_value=0, step=30)
    with b:
        stage2_temp_offset_k = st.number_input("Stage 2 ON if water temp above setpoint by, K", value=2.0, min_value=0.0, step=0.5)
        stage2_off_offset_k = st.number_input("Stage 2 OFF if water temp above setpoint by, K", value=0.5, min_value=0.0, step=0.5)
    with c:
        comp2_anti_short_cycle_s = st.number_input("Compressor 2 anti-short-cycle timer, sec", value=180, min_value=0, step=30)
        stop_lag_first = st.checkbox("Stop lag compressor first on unloading", value=True)

    st.markdown("---")
    show_circuit_results("Common Tandem Refrigerant Circuit", ci, units, compressor_refrigerants)

    st.markdown("### Tandem compressor logic summary")
    st.code(
        f"""TANDEM START:
K0 ON → pump starts → flow proven → TC1 stage 1 cooling demand → common YV1 opens → suction rises → LPS closes → lead compressor starts if AST1 is satisfied.

LAG COMPRESSOR START:
If water temperature remains above stage-2 demand by {stage2_temp_offset_k:.1f} K and lead compressor has run for at least {lag_start_delay_s} sec, start lag compressor if its own overload and anti-short-cycle timer ({comp2_anti_short_cycle_s} sec) are healthy.

UNLOADING / STOP:
If load reduces, stop lag compressor first = {stop_lag_first}. Keep lead compressor running until TC1 is satisfied. Then close common YV1 and stop lead compressor by pump-down through common LPS.

COMMON SAFETIES:
Common HPS, LPS, FS1, FRZ1, and PR1 trip the common circuit and stop both compressors. Individual OL1/OL2 or discharge temperature trips stop the affected compressor and create an alarm.

LEAD/LAG ROTATION:
Lead/lag rotation enabled = {lead_lag_enabled}. Rotate lead compressor by run hours or each start to balance wear.
""",
        language="text",
    )

st.markdown("---")
st.warning(
    "This app gives preliminary engineering settings. Final settings must be verified against the compressor manufacturer operating envelope, approved refrigerant, system design pressure, actual pressure gauges, refrigerant safety standard, and commissioning test readings. PDF extraction is only a helper; always verify extracted values manually."
)
