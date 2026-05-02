# chiller_design_generator_app.py
# Full preliminary Streamlit app for chiller pressure settings, electrical schematic,
# refrigerant schematic, chilled-water schematic, component specifications, and BOM.
# Run: streamlit run chiller_design_generator_app.py

from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from CoolProp.CoolProp import PropsSI
except Exception:
    PropsSI = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

ATM_BAR = 1.01325
BAR_PER_PA = 1e-5
PSI_PER_PA = 0.0001450377377

REFS = {
    "R134a": "R134a", "R407C": "R407C", "R410A": "R410A", "R404A": "R404A",
    "R507A": "R507A", "R22": "R22", "R1234yf": "R1234yf",
    "R1234ze(E)": "R1234ze(E)", "R513A": "R513A", "R32": "R32", "R290": "R290",
}
REF_ALIASES = {
    "R-134A": "R134a", "R134A": "R134a", "R-407C": "R407C", "R407C": "R407C",
    "R-410A": "R410A", "R410A": "R410A", "R-404A": "R404A", "R404A": "R404A",
    "R-507A": "R507A", "R507A": "R507A", "R-22": "R22", "R22": "R22",
    "R-32": "R32", "R32": "R32", "R-513A": "R513A", "R513A": "R513A",
    "R-1234YF": "R1234yf", "R1234YF": "R1234yf", "R-1234ZE": "R1234ze(E)",
    "R1234ZE": "R1234ze(E)", "R290": "R290", "PROPANE": "R290",
}
STANDARD_BREAKERS_A = [6, 10, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 320, 400, 500, 630]
STANDARD_CONTACTORS_A = [9, 12, 18, 25, 32, 40, 50, 65, 80, 95, 115, 150, 185, 225, 265, 330, 400]


@dataclass
class Project:
    project_name: str
    chiller_type: str
    configuration: str
    number_of_circuits: int
    number_of_compressors: int
    design_ambient_c: float
    standard: str
    tag_prefix: str


@dataclass
class Circuit:
    name: str
    refrigerant: str
    compressor_make: str
    compressor_model: str
    compressor_type: str
    approved_refrigerants: str
    compressor_kw: float
    compressor_flc_a: float
    compressor_lra_a: float
    max_high_pressure_barg: float
    max_condensing_temp_c: float
    min_evaporating_temp_c: float
    cooling_capacity_kw: float
    evap_temp_c: float
    cond_temp_c: float
    superheat_k: float
    subcooling_k: float
    liquid_line_mm: float
    suction_line_mm: float
    discharge_line_mm: float
    expansion_device: str
    receiver: bool
    suction_accumulator: bool
    oil_separator: bool
    liquid_solenoid_yv1: bool
    hot_gas_bypass_yv2: bool
    filter_drier: bool
    sight_glass: bool
    hps_margin_k: float
    lps_cutout_evap_c: float
    lps_cutin_evap_c: float
    cps1_on_cond_c: float
    cps1_off_cond_c: float
    cps2_on_cond_c: float
    cps2_off_cond_c: float
    hgb_open_evap_c: float
    hgb_close_evap_c: float


@dataclass
class Water:
    fluid: str
    glycol_percent: float
    entering_c: float
    leaving_c: float
    flow_lps: float
    evap_dp_kpa: float
    pump_qty: int
    pump_arrangement: str
    pump_head_m: float
    pump_kw: float
    pump_flc_a: float
    pipe_mm: float
    strainer: bool
    flow_switch_type: str
    expansion_tank: bool
    air_vent: bool
    drain_valves: bool
    bypass_line: bool


@dataclass
class Fan:
    qty: int
    motor_kw_each: float
    flc_a_each: float
    voltage_v: float
    phase: str
    control_type: str
    contactor_per_fan: bool
    overload_per_fan: bool
    stage_delay_s: int


@dataclass
class Electrical:
    main_voltage_v: float
    phase: str
    frequency_hz: float
    control_voltage: str
    compressor_starter: str
    pump_starter: str
    panel_ip: str
    panel_location: str
    control_method: str
    bms: bool
    remote_start_stop: bool
    common_fault: bool
    phase_relay: bool
    emergency_stop: bool
    door_interlock: bool
    control_transformer_va: float


@dataclass
class Logic:
    chw_setpoint_c: float
    temp_differential_k: float
    pumpdown: bool
    pump_start_delay_s: int
    flow_proving_delay_s: int
    lp_bypass_delay_s: int
    anti_short_cycle_s: int
    min_on_time_s: int
    pumpdown_max_s: int
    pump_off_delay_s: int
    freeze_stat_c: float
    crankcase_preheat_h: float
    lead_lag: bool
    stage2_on_offset_k: float
    stage2_off_offset_k: float
    lag_start_delay_s: int


# ---------------- utility calculations ----------------

def c_to_k(c: float) -> float:
    return c + 273.15


def sat_pa(ref: str, temp_c: float, quality: float) -> float:
    if PropsSI is None:
        raise RuntimeError("CoolProp is not installed")
    return float(PropsSI("P", "T", c_to_k(temp_c), "Q", quality, REFS[ref]))


def barg_to_pa_abs(barg: float) -> float:
    return (barg + ATM_BAR) / BAR_PER_PA


def pa_to_barg(pa: float) -> float:
    return pa * BAR_PER_PA - ATM_BAR


def pa_to_barabs(pa: float) -> float:
    return pa * BAR_PER_PA


def pa_to_psig(pa: float) -> float:
    return pa * PSI_PER_PA - 14.6959


def ptxt(pa: float, unit: str) -> str:
    if pa is None or (isinstance(pa, float) and math.isnan(pa)):
        return "—"
    if unit == "bar(g)":
        return f"{pa_to_barg(pa):.2f} bar(g)"
    if unit == "bar(abs)":
        return f"{pa_to_barabs(pa):.2f} bar(abs)"
    return f"{pa_to_psig(pa):.0f} psig"


def next_std(value: float, standards: List[int]) -> int:
    for x in standards:
        if value <= x:
            return x
    return standards[-1]


def flc_3ph(kw: float, v: float, pf: float = 0.85, eff: float = 0.90) -> float:
    if kw <= 0 or v <= 0:
        return 0.0
    return kw * 1000 / (math.sqrt(3) * v * pf * eff)


def water_flow_lps(cap_kw: float, ewt: float, lwt: float, glycol_pct: float) -> float:
    dt = max(0.1, ewt - lwt)
    cp = max(3.2, 4.186 * (1 - 0.006 * glycol_pct))
    density = 1.0 + 0.001 * glycol_pct
    return cap_kw / (cp * density * dt)


# ---------------- compressor PDF extraction ----------------

def extract_pdf_text(uploaded) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    data = uploaded.getvalue()
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            if len(text.strip()) > 100:
                return text, warnings
            warnings.append("PyMuPDF extracted little text; the PDF may be scanned/image-based.")
        except Exception as exc:
            warnings.append(f"PyMuPDF failed: {exc}")
    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if len(text.strip()) > 100:
                return text, warnings
            warnings.append("pypdf extracted little text; OCR may be required.")
        except Exception as exc:
            warnings.append(f"pypdf failed: {exc}")
    warnings.append("No usable text extracted. Enter data manually.")
    return "", warnings


def parse_float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def pressure_to_barg(value: float, unit: str) -> float:
    u = unit.lower().replace(" ", "")
    if u in ["bar", "barg", "bar(g)"]:
        return value
    if u in ["bara", "barabs", "bar(a)"]:
        return value - ATM_BAR
    if u in ["mpa", "mpag"]:
        return value * 10
    if u in ["kpa", "kpag"]:
        return value / 100
    if u in ["psi", "psig"]:
        return value * 0.0689476
    if u == "psia":
        return value * 0.0689476 - ATM_BAR
    return value


def find_one(patterns: List[str], text: str) -> str | None:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def find_pressure_near(text: str, keywords: List[str]) -> float | None:
    for kw in keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            win = text[max(0, m.start()-180): min(len(text), m.end()+260)]
            p = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(bar\s*\(?g?\)?|bar\s*abs|bara|barg|MPa|kPa|psi|psig|psia)", win, re.I)
            if p:
                val = parse_float(p.group(1))
                if val is not None:
                    return round(pressure_to_barg(val, p.group(2)), 3)
    return None


def find_temp_near(text: str, keywords: List[str]) -> float | None:
    for kw in keywords:
        for m in re.finditer(kw, text, re.IGNORECASE):
            win = text[max(0, m.start()-180): min(len(text), m.end()+260)]
            t = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:°\s*C|deg\s*C|C\b)", win, re.I)
            if t:
                val = parse_float(t.group(1))
                if val is not None:
                    return val
    return None


def parse_compressor_pdf(text: str) -> Dict[str, Any]:
    text = text.replace("\u00a0", " ")
    out: Dict[str, Any] = {}
    make = find_one([r"\b(Copeland|Emerson|Danfoss|Maneurop|Bitzer|Frascold|RefComp|Hanbell|Carrier|Trane|Daikin|Mitsubishi)\b"], text)
    if make:
        out["compressor_make"] = make
    model = find_one([r"(?:Compressor\s+Model|Model\s+No\.?|Model|Type|Designation)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_\/\. ]{2,55})"], text)
    if model:
        out["compressor_model"] = model[:70]
    upper = text.upper().replace(" ", "")
    refs: List[str] = []
    for alias, canon in REF_ALIASES.items():
        if alias.replace(" ", "") in upper and canon not in refs:
            refs.append(canon)
    if refs:
        out["approved_refrigerants"] = ", ".join(refs)
        out["first_refrigerant"] = refs[0]
    hp = find_pressure_near(text, [
        r"maximum\s+(?:allowable\s+)?(?:high|discharge|operating)\s+pressure",
        r"max\.?\s+(?:high|discharge)\s+pressure",
        r"standstill\s+pressure", r"design\s+pressure", r"high\s+pressure\s+cut\s*out",
    ])
    if hp is not None:
        out["max_high_pressure_barg"] = hp
    maxcond = find_temp_near(text, [r"maximum\s+condensing\s+temperature", r"max\.?\s+condensing\s+temperature", r"condensing\s+temperature\s+max"])
    if maxcond is not None:
        out["max_condensing_temp_c"] = maxcond
    minevap = find_temp_near(text, [r"minimum\s+evaporating\s+temperature", r"min\.?\s+evaporating\s+temperature", r"evaporating\s+temperature\s+min", r"operating\s+envelope"])
    if minevap is not None:
        out["min_evaporating_temp_c"] = minevap
    rla = find_one([r"(?:RLA|MCC|Rated\s+load\s+amps?|Max\.?\s+operating\s+current)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*A"], text)
    if rla:
        out["compressor_flc_a"] = float(rla)
    lra = find_one([r"(?:LRA|Locked\s+rotor\s+amps?)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*A"], text)
    if lra:
        out["compressor_lra_a"] = float(lra)
    return out


def pv(parsed: Dict[str, Any], key: str, default: Any) -> Any:
    return parsed.get(key, default)


# ---------------- calculation tables ----------------

def pressure_settings(c: Circuit, unit: str) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    warnings: List[str] = []
    values: Dict[str, float] = {}
    if PropsSI is None:
        return pd.DataFrame(), ["CoolProp not installed."], values
    try:
        values["normal_suction"] = sat_pa(c.refrigerant, c.evap_temp_c, 1.0)
        values["normal_condensing"] = sat_pa(c.refrigerant, c.cond_temp_c, 1.0)
        values["subcooled_liquid"] = sat_pa(c.refrigerant, c.cond_temp_c - c.subcooling_k, 0.0)
        values["hps_by_temp"] = sat_pa(c.refrigerant, c.cond_temp_c + c.hps_margin_k, 1.0)
        values["hps_limit"] = barg_to_pa_abs(c.max_high_pressure_barg)
        values["hps_cutout"] = min(values["hps_by_temp"], values["hps_limit"])
        values["lps_cutout"] = sat_pa(c.refrigerant, c.lps_cutout_evap_c, 1.0)
        values["lps_cutin"] = sat_pa(c.refrigerant, c.lps_cutin_evap_c, 1.0)
        values["cps1_on"] = sat_pa(c.refrigerant, c.cps1_on_cond_c, 1.0)
        values["cps1_off"] = sat_pa(c.refrigerant, c.cps1_off_cond_c, 1.0)
        values["cps2_on"] = sat_pa(c.refrigerant, c.cps2_on_cond_c, 1.0)
        values["cps2_off"] = sat_pa(c.refrigerant, c.cps2_off_cond_c, 1.0)
        values["hgb_open"] = sat_pa(c.refrigerant, c.hgb_open_evap_c, 1.0) if c.hot_gas_bypass_yv2 else float("nan")
        values["hgb_close"] = sat_pa(c.refrigerant, c.hgb_close_evap_c, 1.0) if c.hot_gas_bypass_yv2 else float("nan")
    except Exception as exc:
        return pd.DataFrame(), [f"Pressure calculation error: {exc}"], values

    if c.cond_temp_c > c.max_condensing_temp_c:
        warnings.append("Design condensing temperature is above compressor maximum condensing temperature.")
    if c.evap_temp_c < c.min_evaporating_temp_c:
        warnings.append("Design evaporating temperature is below compressor minimum evaporating temperature.")
    if c.lps_cutout_evap_c < c.min_evaporating_temp_c:
        warnings.append("LPS cut-out is below compressor minimum evaporating temperature.")
    if c.lps_cutin_evap_c <= c.lps_cutout_evap_c:
        warnings.append("LPS cut-in must be higher than LPS cut-out.")
    if c.cps1_off_cond_c >= c.cps1_on_cond_c:
        warnings.append("CPS1 OFF should be lower than CPS1 ON.")
    if c.cps2_off_cond_c >= c.cps2_on_cond_c:
        warnings.append("CPS2 OFF should be lower than CPS2 ON.")
    if c.cps2_on_cond_c <= c.cps1_on_cond_c:
        warnings.append("CPS2 ON should normally be higher than CPS1 ON.")
    if c.hot_gas_bypass_yv2 and c.hgb_open_evap_c <= c.lps_cutout_evap_c:
        warnings.append("YV2/HGBP should open above LPS cut-out.")
    if values["hps_cutout"] < values["hps_by_temp"]:
        warnings.append("HPS setting is limited by the compressor/system maximum high-side pressure.")

    rows = [
        [c.name, "HPS", "High pressure safety", "Manual reset", ptxt(values["hps_cutout"], unit), f"Tcond + margin = {c.cond_temp_c+c.hps_margin_k:.1f}°C; max limit {c.max_high_pressure_barg:.1f} bar(g)"],
        [c.name, "LPS", "Low pressure / pump-down", ptxt(values["lps_cutin"], unit), ptxt(values["lps_cutout"], unit), f"Cut-in {c.lps_cutin_evap_c:.1f}°C evap.; cut-out {c.lps_cutout_evap_c:.1f}°C evap."],
        [c.name, "CPS1", "Fan 1 pressure switch", ptxt(values["cps1_off"], unit), ptxt(values["cps1_on"], unit), f"ON {c.cps1_on_cond_c:.1f}°C cond.; OFF {c.cps1_off_cond_c:.1f}°C cond."],
        [c.name, "CPS2", "Fan 2 pressure switch", ptxt(values["cps2_off"], unit), ptxt(values["cps2_on"], unit), f"ON {c.cps2_on_cond_c:.1f}°C cond.; OFF {c.cps2_off_cond_c:.1f}°C cond."],
    ]
    if c.hot_gas_bypass_yv2:
        rows.append([c.name, "YV2/HGBP", "Hot gas bypass", ptxt(values["hgb_close"], unit), ptxt(values["hgb_open"], unit), f"Open {c.hgb_open_evap_c:.1f}°C evap.; close {c.hgb_close_evap_c:.1f}°C evap."])
    df = pd.DataFrame(rows, columns=["Circuit", "Device", "Function", "Cut-in / Reset", "Cut-out / ON", "Basis"])
    return df, warnings, values


def component_specs(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical, logic: Logic) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(circuits, 1):
        comp_qty = project.number_of_compressors if project.configuration.startswith("Tandem") and i == 1 else 1
        rows += [
            {"System":"Refrigerant", "Tag":f"COMP-{i}", "Component":"Compressor", "Qty":comp_qty, "Specification":f"{c.compressor_make} {c.compressor_model}, {c.compressor_type}, {c.compressor_kw:.1f} kW, {c.refrigerant}", "Remarks":"Verify with supplier datasheet"},
            {"System":"Refrigerant", "Tag":f"COND-{i}", "Component":"Air-cooled condenser", "Qty":1, "Specification":f"Tcond {c.cond_temp_c:.1f}°C, {fan.qty} fans", "Remarks":"Size by total heat rejection"},
            {"System":"Refrigerant", "Tag":f"EVAP-{i}", "Component":"Evaporator / cooler", "Qty":1, "Specification":f"Capacity {c.cooling_capacity_kw:.1f} kW, Tevap {c.evap_temp_c:.1f}°C", "Remarks":"Shell-and-tube/BPHE as selected"},
            {"System":"Refrigerant", "Tag":f"YV1-{i}", "Component":"Liquid line solenoid", "Qty":1 if c.liquid_solenoid_yv1 else 0, "Specification":f"{c.refrigerant}, liquid line {c.liquid_line_mm:.1f} mm", "Remarks":"Pump-down control"},
            {"System":"Refrigerant", "Tag":f"EXP-{i}", "Component":c.expansion_device, "Qty":1, "Specification":f"{c.refrigerant}, {c.cooling_capacity_kw:.1f} kW, SH {c.superheat_k:.1f} K", "Remarks":"Select from valve maker"},
            {"System":"Refrigerant", "Tag":f"FD-{i}", "Component":"Filter drier", "Qty":1 if c.filter_drier else 0, "Specification":f"Liquid line {c.liquid_line_mm:.1f} mm", "Remarks":""},
            {"System":"Refrigerant", "Tag":f"SG-{i}", "Component":"Sight glass", "Qty":1 if c.sight_glass else 0, "Specification":f"Liquid line {c.liquid_line_mm:.1f} mm", "Remarks":"Moisture indicator type preferred"},
            {"System":"Refrigerant", "Tag":f"HPS-{i}", "Component":"High pressure switch", "Qty":1, "Specification":"Manual reset, high side", "Remarks":"Set from pressure table"},
            {"System":"Refrigerant", "Tag":f"LPS-{i}", "Component":"Low pressure switch", "Qty":1, "Specification":"Auto reset / pump-down", "Remarks":"Set from pressure table"},
            {"System":"Refrigerant", "Tag":f"CPS1-{i}", "Component":"Fan pressure switch 1", "Qty":1 if fan.qty >= 1 else 0, "Specification":"High-side fan stage 1", "Remarks":"Set from pressure table"},
            {"System":"Refrigerant", "Tag":f"CPS2-{i}", "Component":"Fan pressure switch 2", "Qty":1 if fan.qty >= 2 else 0, "Specification":"High-side fan stage 2", "Remarks":"Set from pressure table"},
            {"System":"Refrigerant", "Tag":f"YV2-{i}", "Component":"Hot gas bypass solenoid", "Qty":1 if c.hot_gas_bypass_yv2 else 0, "Specification":"Discharge to evaporator inlet/distributor", "Remarks":"Optional low-load support"},
            {"System":"Refrigerant", "Tag":f"REC-{i}", "Component":"Liquid receiver", "Qty":1 if c.receiver else 0, "Specification":"High-side design pressure", "Remarks":"Size by charge and pump-down"},
            {"System":"Refrigerant", "Tag":f"ACC-{i}", "Component":"Suction accumulator", "Qty":1 if c.suction_accumulator else 0, "Specification":f"Suction {c.suction_line_mm:.1f} mm", "Remarks":"Use where liquid return risk exists"},
            {"System":"Refrigerant", "Tag":f"OS-{i}", "Component":"Oil separator", "Qty":1 if c.oil_separator else 0, "Specification":f"Discharge {c.discharge_line_mm:.1f} mm", "Remarks":"Optional/required for long lines or screw systems"},
        ]
    rows += [
        {"System":"Chilled Water", "Tag":"P-1", "Component":"Chilled water pump", "Qty":water.pump_qty, "Specification":f"{water.pump_kw:.1f} kW, {water.flow_lps:.2f} L/s, {water.pump_head_m:.1f} m head", "Remarks":water.pump_arrangement},
        {"System":"Chilled Water", "Tag":"STR-1", "Component":"Y-strainer", "Qty":1 if water.strainer else 0, "Specification":f"Pipe {water.pipe_mm:.0f} mm", "Remarks":"Before evaporator/pump"},
        {"System":"Chilled Water", "Tag":"FS-1", "Component":"Flow switch", "Qty":1, "Specification":f"{water.flow_switch_type}, pipe {water.pipe_mm:.0f} mm", "Remarks":"Compressor interlock"},
        {"System":"Chilled Water", "Tag":"TS-IN/OUT", "Component":"Temperature sensors", "Qty":2, "Specification":"Chilled water inlet/outlet", "Remarks":"Control and display"},
        {"System":"Chilled Water", "Tag":"PG/TG", "Component":"Pressure gauges and thermometers", "Qty":4, "Specification":"At evaporator inlet/outlet", "Remarks":"Commissioning"},
        {"System":"Chilled Water", "Tag":"ET-1", "Component":"Expansion tank", "Qty":1 if water.expansion_tank else 0, "Specification":"Closed chilled water system", "Remarks":"Size by system water volume"},
        {"System":"Electrical", "Tag":"PANEL-1", "Component":"Electrical panel", "Qty":1, "Specification":f"{elec.panel_ip}, {elec.panel_location}, {elec.main_voltage_v:.0f} V, control {elec.control_voltage}", "Remarks":elec.control_method},
        {"System":"Electrical", "Tag":"K0", "Component":"Master control relay", "Qty":1, "Specification":f"Coil {elec.control_voltage}", "Remarks":"Feeds controlled live bus"},
        {"System":"Electrical", "Tag":"TD/AST", "Component":"Control timers", "Qty":5, "Specification":f"Pump delay {logic.pump_start_delay_s}s, AST {logic.anti_short_cycle_s}s, pump off {logic.pump_off_delay_s}s", "Remarks":"Relay/PLC/controller logic"},
    ]
    return pd.DataFrame(rows).query("Qty != 0").reset_index(drop=True)


def electrical_selection(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_a = 0.0
    for i, c in enumerate(circuits, 1):
        qty = project.number_of_compressors if project.configuration.startswith("Tandem") and i == 1 else 1
        flc = c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)
        total_a += flc * qty
        rows += [
            {"Tag":f"KM-C{i}", "Item":f"Compressor contactor {i}", "Qty":qty, "Basis":f"FLC {flc:.1f} A", "Preliminary selection":f"AC-3 contactor ≥ {next_std(flc*1.15, STANDARD_CONTACTORS_A)} A, coil {elec.control_voltage}", "Notes":"Verify by compressor duty"},
            {"Tag":f"OL-C{i}", "Item":f"Compressor overload {i}", "Qty":qty, "Basis":f"FLC {flc:.1f} A", "Preliminary selection":f"Adjustable range covering {flc:.1f} A", "Notes":"Set to nameplate"},
        ]
    pump_flc = water.pump_flc_a if water.pump_flc_a > 0 else flc_3ph(water.pump_kw, elec.main_voltage_v)
    total_a += pump_flc * water.pump_qty
    fan_flc = fan.flc_a_each if fan.flc_a_each > 0 else flc_3ph(fan.motor_kw_each, fan.voltage_v)
    total_a += fan_flc * fan.qty
    rows += [
        {"Tag":"KM-P", "Item":"Pump contactor", "Qty":water.pump_qty, "Basis":f"FLC {pump_flc:.1f} A", "Preliminary selection":f"AC-3 contactor ≥ {next_std(pump_flc*1.15, STANDARD_CONTACTORS_A)} A, coil {elec.control_voltage}", "Notes":"One set per pump"},
        {"Tag":"OL-P", "Item":"Pump overload", "Qty":water.pump_qty, "Basis":f"FLC {pump_flc:.1f} A", "Preliminary selection":f"Adjustable range covering {pump_flc:.1f} A", "Notes":"Set to nameplate"},
        {"Tag":"KM-F", "Item":"Fan contactor(s)", "Qty":fan.qty if fan.contactor_per_fan else max(1, min(2, fan.qty)), "Basis":f"Fan FLC {fan_flc:.1f} A each", "Preliminary selection":f"AC-3 contactor ≥ {next_std(fan_flc*1.15, STANDARD_CONTACTORS_A)} A each", "Notes":"For grouped fans, size for group current"},
        {"Tag":"OL-F", "Item":"Fan overload(s)", "Qty":fan.qty if fan.overload_per_fan else max(1, min(2, fan.qty)), "Basis":f"Fan FLC {fan_flc:.1f} A each", "Preliminary selection":f"Overload range covering {fan_flc:.1f} A", "Notes":"Per motor overload preferred"},
        {"Tag":"Q1", "Item":"Main MCCB / isolator", "Qty":1, "Basis":f"Estimated running current {total_a:.1f} A", "Preliminary selection":f"MCCB/isolator ≥ {next_std(total_a*1.25, STANDARD_BREAKERS_A)} A", "Notes":"Breaking capacity per site fault level"},
        {"Tag":"T1", "Item":"Control transformer", "Qty":1, "Basis":elec.control_voltage, "Preliminary selection":f"{elec.control_transformer_va:.0f} VA primary {elec.main_voltage_v:.0f} V secondary {elec.control_voltage}", "Notes":"Increase VA for PLC/HMI/many relays"},
        {"Tag":"PR1", "Item":"Phase failure/sequence relay", "Qty":1 if elec.phase_relay else 0, "Basis":f"{elec.main_voltage_v:.0f} V", "Preliminary selection":"3-phase monitoring relay", "Notes":"Trips control on phase fault"},
    ]
    return pd.DataFrame(rows).query("Qty != 0").reset_index(drop=True)


def bom_from(specs: pd.DataFrame, elec_sel: pd.DataFrame) -> pd.DataFrame:
    """Build BOM from component specs and electrical selection tables.

    This avoids itertuples because columns such as "Preliminary selection"
    are renamed/mangled in namedtuples and can cause tuple indexing errors.
    """
    cols = ["System", "Tag", "Component", "Qty", "Specification", "Remarks"]
    frames = []

    if specs is not None and not specs.empty:
        s = specs.copy()
        for col in cols:
            if col not in s.columns:
                s[col] = ""
        frames.append(s[cols])

    if elec_sel is not None and not elec_sel.empty:
        e = elec_sel.copy()
        e["System"] = "Electrical"
        e = e.rename(
            columns={
                "Item": "Component",
                "Preliminary selection": "Specification",
                "Notes": "Remarks",
            }
        )
        for col in cols:
            if col not in e.columns:
                e[col] = ""
        frames.append(e[cols])

    if not frames:
        return pd.DataFrame(columns=cols)

    return pd.concat(frames, ignore_index=True)


# ---------------- SVG diagrams ----------------

def esc(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_start(w: int, h: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#111"/></marker>
<style>.title{{font:bold 22px Arial}} .head{{font:bold 16px Arial}} .txt{{font:13px Arial}} .small{{font:11px Arial}} .box{{fill:#fff;stroke:#111;stroke-width:1.5;rx:8;ry:8}} .dash{{fill:#fff;stroke:#111;stroke-width:1.3;stroke-dasharray:5 4;rx:8;ry:8}} .wire{{stroke:#111;stroke-width:1.4;fill:none}} .arrow{{stroke:#111;stroke-width:1.5;fill:none;marker-end:url(#arrow)}}</style></defs>
<rect width="{w}" height="{h}" fill="white"/><text x="{w/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>'''


def bx(x: int, y: int, w: int, h: int, label: str, cls: str = "box") -> str:
    parts = label.split("\n")
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="{cls}"/>'
    yy = y + h/2 - (len(parts)-1)*7
    for i, p in enumerate(parts):
        out += f'<text x="{x+w/2}" y="{yy+i*15}" text-anchor="middle" class="txt">{esc(p)}</text>'
    return out


def ar(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arrow"/>'


def ln(x1: int, y1: int, x2: int, y2: int, cls: str = "wire") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>'


def refrigerant_svg(project: Project, circuits: List[Circuit], fan: Fan) -> str:
    h = 350 + 225*len(circuits)
    s = svg_start(1320, h, f"{project.project_name} - Refrigerant / Freon Circuit")
    s += f'<text x="30" y="75" class="head">Configuration: {esc(project.configuration)}</text>'
    for idx, c in enumerate(circuits, 1):
        y = 110 + (idx-1)*225
        s += f'<text x="35" y="{y}" class="head">{esc(c.name)}: {c.refrigerant}, Tevap {c.evap_temp_c:.1f}°C, Tcond {c.cond_temp_c:.1f}°C</text>'
        yy = y + 35
        labels = [(40,"COMP\nCompressor"),(210,"HPS\nHP safety"),(360,f"COND\nAir condenser\n{fan.qty} fans"),(560,"RECEIVER" if c.receiver else "Liquid\nheader"),(720,"FD + SG"),(880,"YV1\nLiquid SV" if c.liquid_solenoid_yv1 else "No YV1"),(1030,c.expansion_device),(1160,"EVAP\nCooler")]
        for x, lab in labels: s += bx(x, yy, 120, 58, lab)
        for x1, x2 in [(160,210),(330,360),(480,560),(680,720),(840,880),(1000,1030),(1150,1160)]: s += ar(x1, yy+29, x2, yy+29)
        s += ar(1280, yy+29, 1280, yy+125) + ar(1280, yy+125, 105, yy+125) + ar(105, yy+125, 105, yy+58)
        s += bx(245, yy+98, 110, 44, "LPS\nSuction") + bx(410, yy+98, 110, 44, "FRZ\nFreeze") + bx(610, yy+98, 120, 44, "CPS1/2\nFan pressure")
        if c.hot_gas_bypass_yv2:
            s += f'<path d="M160 {yy+18} C260 {yy-22},500 {yy-22},1030 {yy+5}" class="arrow"/>' + bx(560, yy-40, 165, 42, "YV2 + HGBP\nHot gas bypass")
        if c.oil_separator: s += bx(185, yy-42, 125, 42, "Oil\nseparator")
        if c.suction_accumulator: s += bx(580, yy+150, 140, 42, "Suction\naccumulator")
        s += f'<text x="40" y="{yy+183}" class="small">Line sizes: Liquid {c.liquid_line_mm:.1f} mm | Suction {c.suction_line_mm:.1f} mm | Discharge {c.discharge_line_mm:.1f} mm | SH {c.superheat_k:.1f} K | SC {c.subcooling_k:.1f} K</text>'
    s += f'<text x="30" y="{h-30}" class="small">Schematic only. Final piping requires detailed refrigeration design: oil traps, slopes, reliefs, valves, risers and code compliance.</text></svg>'
    return s


def water_svg(project: Project, water: Water) -> str:
    s = svg_start(1320, 540, f"{project.project_name} - Chilled Water Circuit")
    y = 180
    labels = [(40,"Load/AHU\nProcess"),(210,"IV-1\nIsolation"),(360,"STR-1\nStrainer" if water.strainer else "Pipe"),(510,"P-1\nCHW pump"),(680,"NRV/BV\nCheck+balance"),(835,"EVAP\nCooler"),(1010,"FS-1\nFlow switch"),(1165,"IV-2\nIsolation")]
    for x, lab in labels: s += bx(x, y, 120, 58, lab)
    for x1, x2 in [(160,210),(330,360),(480,510),(630,680),(800,835),(955,1010),(1130,1165)]: s += ar(x1, y+29, x2, y+29)
    s += ar(1285, y+29, 1285, y+150) + ar(1285, y+150, 100, y+150) + ar(100, y+150, 100, y+58)
    s += bx(825, y-70, 80, 40, "TS-IN") + bx(1015, y-70, 80, 40, "TS-OUT") + ln(865,y-30,865,y) + ln(1055,y-30,1055,y)
    s += bx(260,y+110,105,45,"PG/TG\nInlet") + bx(980,y+110,105,45,"PG/TG\nOutlet")
    if water.expansion_tank: s += bx(560,y-95,130,45,"ET-1\nExpansion tank") + ln(625,y-50,625,y)
    if water.air_vent: s += bx(925,y-95,90,45,"AV\nAir vent") + ln(970,y-50,970,y)
    if water.drain_valves: s += bx(850,y+70,95,40,"DV\nDrain") + ln(900,y+58,900,y+70)
    if water.bypass_line: s += f'<path d="M510 {y+58} C455 {y+95},460 {y+125},835 {y+115}" class="dash"/><text x="555" y="{y+115}" class="small">Optional bypass</text>'
    s += f'<text x="40" y="430" class="head">Design data</text><text x="40" y="458" class="txt">Fluid {esc(water.fluid)}, glycol {water.glycol_percent:.1f}% | EWT {water.entering_c:.1f}°C | LWT {water.leaving_c:.1f}°C | Flow {water.flow_lps:.2f} L/s | Pipe {water.pipe_mm:.0f} mm</text>'
    s += f'<text x="40" y="485" class="txt">Pump: {esc(water.pump_arrangement)}, qty {water.pump_qty}, head {water.pump_head_m:.1f} m, motor {water.pump_kw:.1f} kW | Evap ΔP {water.evap_dp_kpa:.1f} kPa</text></svg>'
    return s


def electrical_svg(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical, logic: Logic) -> str:
    s = svg_start(1500, 980, f"{project.project_name} - Electrical Power and Control Circuit")
    s += ln(735,60,735,920) + '<text x="40" y="75" class="head">1) Power Circuit</text><text x="765" y="75" class="head">2) Control Circuit / Ladder Logic</text>'
    s += f'<text x="45" y="115" class="txt">Incoming: {elec.phase}, {elec.main_voltage_v:.0f} V, {elec.frequency_hz:.0f} Hz</text>'
    s += bx(45,140,120,55,"Q1\nMain MCCB") + ar(165,168,235,168) + bx(235,140,130,55,"PR1\nPhase relay" if elec.phase_relay else "No PR1") + ar(365,168,435,168) + bx(435,140,130,55,f"T1\n{elec.control_voltage}\nControl")
    for x, lab in zip([50,220,390,560], ["COMP\nKM-C/OL-C", "PUMP\nKM-P/OL-P", "FAN 1\nKM-F1/OL-F1", "FAN 2\nKM-F2/OL-F2"]):
        s += bx(x,280,120,60,lab) + ln(x+60,245,x+60,280) + bx(x,375,120,55,"MOTOR\n3~") + ln(x+60,340,x+60,375)
    s += ln(50,245,680,245)
    xL,xN=780,1450; y0,yend=105,900
    s += ln(xL,y0,xL,yend) + ln(xN,y0,xN,yend) + f'<text x="{xL}" y="98" text-anchor="middle" class="head">L raw</text><text x="{xN}" y="98" text-anchor="middle" class="head">N</text>'
    r=125
    for label,x in [("F1\nFuse",835),("Q2\nMCB",920),("E-STOP\nNC",1030),("S0 STOP\nNC",1135),("S2 ON\nNO",1240),("K0 Coil",1340)]:
        s += bx(x-35,r-22,70,44,label)
    for x1,x2 in [(xL,800),(870,885),(955,995),(1065,1100),(1170,1205),(1275,1305),(1375,xN)]: s += ln(x1,r,x2,r)
    s += bx(1210,r+45,90,35,"K0 NO\nholding","dash") + f'<path d="M1195 {r} L1195 {r+62} L1210 {r+62}" class="wire"/><path d="M1300 {r+62} L1325 {r+62} L1325 {r}" class="wire"/>' + bx(1375,r+45,60,35,"H0\nON")
    s += bx(790,188,85,44,"K0 NO\nMaster") + ln(xL,220,790,220) + ln(875,220,875,yend) + '<text x="885" y="208" class="head">LC controlled live bus</text><text x="885" y="226" class="small">All lower rungs are after F1, Q2, E-stop, Stop and K0.</text>'
    def rung(y:int, items:List[Tuple[str,int]], coil:str, lamp:str=""):
        out=ln(875,y,items[0][1]-45,y)
        for label,x in items:
            out += bx(x-45,y-20,90,40,label) + ln(x+45,y,x+55,y)
        out += ln(items[-1][1]+55,y,1275,y) + bx(1320,y-24,85,48,coil) + ln(1405,y,xN,y)
        if lamp: out += bx(1410,y+25,55,32,lamp)
        return out
    s += rung(285,[("SA1\nPump",950),("OL-P\nNC",1060),(f"TD2\n{logic.pump_off_delay_s}s",1170)],"KM-P\nPump","H2")
    s += rung(350,[("TC1\nCooling",970)],"YV1\nLiquid SV","H5")
    s += rung(420,[("KM-P\nNO",930),(f"TD1\n{logic.pump_start_delay_s}s",1015),("FS1\nNO",1100),("HPS\nNC",1185),("LPS\nNC",1270)],"KM-C\nComp","H3")
    s += rung(495,[("FRZ1\nNC",950),("PR1\nOK",1035),("OL-C\nNC",1120),(f"AST\n{logic.anti_short_cycle_s}s",1210)],"KM-C\nEnable","")
    s += rung(570,[("SA2\nFans",950),("KM-C\nNO",1055),("OL-F1\nNC",1165),("CPS1\nNO",1270)],"KM-F1\nFan1","H6")
    s += rung(635,[("SA2\nFans",950),("KM-C\nNO",1055),("OL-F2\nNC",1165),("CPS2\nNO",1270)],"KM-F2\nFan2","H7")
    s += rung(700,[("KM-C\nNC",980)],"HTR\nCrankcase","H4")
    s += '<text x="780" y="760" class="head">Fault indication</text>'
    for i,(lab,lamp) in enumerate([("HPS trip","H8"),("LPS trip","H9"),("Flow fail","H10"),("Comp O/L","H11"),("Pump O/L","H12")]):
        x=900+i*105; s += bx(x,780,90,40,lab)+bx(x+15,830,60,35,lamp)
    s += '<text x="45" y="905" class="small">Preliminary schematic. Final drawing must add terminal numbers, wire numbers, cable sizes and code-compliant protection.</text></svg>'
    return s


def show_svg(svg: str, height: int=650):
    components.html(f'<div style="width:100%; overflow:auto; border:1px solid #ddd; padding:8px">{svg}</div>', height=height)


def svg_link(svg: str, filename: str, label: str) -> str:
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<a download="{filename}" href="data:image/svg+xml;base64,{b64}">{label}</a>'


# ---------------- UI forms ----------------

def nfloat(label, value, key, step=0.1, min_value=None) -> float:
    kwargs = dict(label=label, value=float(value), step=step, key=key)
    if min_value is not None: kwargs["min_value"] = min_value
    return float(st.number_input(**kwargs))


def project_form() -> Project:
    c1,c2,c3 = st.columns(3)
    with c1:
        name = st.text_input("Project / chiller name", "Air Cooled Water Chiller")
        chiller_type = st.selectbox("Chiller type", ["Air-cooled", "Water-cooled"], index=0)
        tag = st.text_input("Tag prefix", "CH")
    with c2:
        config = st.selectbox("Configuration", ["Single compressor / single refrigerant circuit", "Two compressors / two separate refrigerant circuits", "Tandem compressors / one common refrigerant circuit"])
        nc, ncomp = (1,1) if config.startswith("Single") else ((2,2) if config.startswith("Two") else (1,2))
        st.write(f"Circuits: **{nc}**, compressors: **{ncomp}**")
    with c3:
        amb = nfloat("Design ambient, °C", 45.0, "amb", 0.5)
        std = st.selectbox("Drawing basis", ["IEC style", "ANSI/simplified"], index=0)
    return Project(name,chiller_type,config,nc,ncomp,amb,std,tag)


def circuit_form(prefix: str, name: str, parsed: Dict[str,Any], project: Project) -> Circuit:
    st.subheader(name)
    ref_default = pv(parsed,"first_refrigerant","R407C")
    refs = list(REFS.keys()); idx = refs.index(ref_default) if ref_default in refs else refs.index("R407C")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        ref = st.selectbox("Refrigerant", refs, idx, key=f"{prefix}_ref")
        make = st.text_input("Compressor make", str(pv(parsed,"compressor_make","")), key=f"{prefix}_make")
        model = st.text_input("Compressor model", str(pv(parsed,"compressor_model","")), key=f"{prefix}_model")
        ctype = st.selectbox("Compressor type", ["Scroll", "Semi-hermetic reciprocating", "Screw", "Hermetic reciprocating", "Other"], key=f"{prefix}_ctype")
    with c2:
        approved = st.text_input("Approved refrigerants", str(pv(parsed,"approved_refrigerants", ref)), key=f"{prefix}_approved")
        ckw = nfloat("Compressor motor kW", 15.0, f"{prefix}_kw", 0.5, 0.0)
        cflc = nfloat("Compressor FLC/RLA A", float(pv(parsed,"compressor_flc_a",0.0)), f"{prefix}_flc", 0.5, 0.0)
        clra = nfloat("Compressor LRA A", float(pv(parsed,"compressor_lra_a",0.0)), f"{prefix}_lra", 1.0, 0.0)
    with c3:
        maxhp = nfloat("Max high-side pressure bar(g)", float(pv(parsed,"max_high_pressure_barg",30.0)), f"{prefix}_maxhp", 0.5, 0.0)
        maxcond = nfloat("Max condensing temp °C", float(pv(parsed,"max_condensing_temp_c",65.0)), f"{prefix}_maxcond", 0.5)
        minevap = nfloat("Min evaporating temp °C", float(pv(parsed,"min_evaporating_temp_c",-10.0)), f"{prefix}_minevap", 0.5)
        cap = nfloat("Cooling capacity kW", 50.0, f"{prefix}_cap", 1.0, 0.0)
    with c4:
        evap = nfloat("Design evaporating temp °C", 3.0, f"{prefix}_evap", 0.5)
        cond = nfloat("Design condensing temp °C", max(project.design_ambient_c+10,50), f"{prefix}_cond", 0.5)
        sh = nfloat("Superheat K", 6.0, f"{prefix}_sh", 0.5, 0.0)
        sc = nfloat("Subcooling K", 5.0, f"{prefix}_sc", 0.5, 0.0)
    st.markdown("**Refrigerant components and line sizes**")
    p1,p2,p3,p4 = st.columns(4)
    with p1:
        liq = nfloat("Liquid line mm",16.0,f"{prefix}_liq",1.0,0.0)
        suc = nfloat("Suction line mm",35.0,f"{prefix}_suc",1.0,0.0)
        dis = nfloat("Discharge line mm",28.0,f"{prefix}_dis",1.0,0.0)
    with p2:
        exp = st.selectbox("Expansion device", ["TXV", "EEV", "Capillary/Other"], key=f"{prefix}_exp")
        receiver = st.checkbox("Liquid receiver", True, key=f"{prefix}_rec")
        accum = st.checkbox("Suction accumulator", False, key=f"{prefix}_acc")
    with p3:
        oil = st.checkbox("Oil separator", False, key=f"{prefix}_oil")
        yv1 = st.checkbox("Liquid solenoid YV1", True, key=f"{prefix}_yv1")
        yv2 = st.checkbox("Hot gas bypass YV2", False, key=f"{prefix}_yv2")
    with p4:
        fd = st.checkbox("Filter drier", True, key=f"{prefix}_fd")
        sg = st.checkbox("Sight glass", True, key=f"{prefix}_sg")
    st.markdown("**Pressure switch basis temperatures**")
    q1,q2,q3,q4 = st.columns(4)
    with q1:
        hpsm=nfloat("HPS margin K",10.0,f"{prefix}_hpsm",0.5)
        lpout=nfloat("LPS cut-out evap °C",-1.0,f"{prefix}_lpout",0.5)
    with q2:
        lpin=nfloat("LPS cut-in evap °C",5.0,f"{prefix}_lpin",0.5)
        c1on=nfloat("CPS1 ON cond °C",42.0,f"{prefix}_c1on",0.5)
    with q3:
        c1off=nfloat("CPS1 OFF cond °C",36.0,f"{prefix}_c1off",0.5)
        c2on=nfloat("CPS2 ON cond °C",48.0,f"{prefix}_c2on",0.5)
    with q4:
        c2off=nfloat("CPS2 OFF cond °C",42.0,f"{prefix}_c2off",0.5)
        hgbopen=nfloat("YV2 open evap °C",1.0,f"{prefix}_hgbopen",0.5)
        hgbclose=nfloat("YV2 close evap °C",4.0,f"{prefix}_hgbclose",0.5)
    return Circuit(name,ref,make,model,ctype,approved,ckw,cflc,clra,maxhp,maxcond,minevap,cap,evap,cond,sh,sc,liq,suc,dis,exp,receiver,accum,oil,yv1,yv2,fd,sg,hpsm,lpout,lpin,c1on,c1off,c2on,c2off,hgbopen,hgbclose)


def water_form(total_cap: float) -> Water:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        fluid=st.selectbox("Fluid", ["Water", "Ethylene glycol", "Propylene glycol"])
        glycol=nfloat("Glycol %", 0.0 if fluid=="Water" else 25.0, "glycol", 1.0, 0.0)
        ewt=nfloat("Entering water °C",12.0,"ewt",0.5)
        lwt=nfloat("Leaving water °C",7.0,"lwt",0.5)
    suggested=water_flow_lps(total_cap,ewt,lwt,glycol)
    with c2:
        flow=nfloat("Water flow L/s",suggested,"flow",0.1,0.0)
        dp=nfloat("Evaporator ΔP kPa",50.0,"evapdp",5.0,0.0)
        pipe=nfloat("Water pipe mm",65.0,"waterpipe",5.0,0.0)
    with c3:
        pumpqty=int(st.number_input("Pump qty", min_value=1, max_value=4, value=1, step=1, key="pumpqty"))
        pumparr=st.selectbox("Pump arrangement", ["Single duty", "1 duty + 1 standby", "2 duty parallel", "External pump only"])
        head=nfloat("Pump head m",20.0,"pumphead",1.0,0.0)
        pkw=nfloat("Pump motor kW",3.7,"pumpkw",0.1,0.0)
        pflc=nfloat("Pump FLC A (0=estimate)",0.0,"pumpflc",0.5,0.0)
    with c4:
        strainer=st.checkbox("Y-strainer",True)
        fstype=st.selectbox("Flow switch", ["Paddle flow switch", "Inline flow switch", "Differential pressure switch", "Flow sensor"])
        et=st.checkbox("Expansion tank",True)
        av=st.checkbox("Air vent",True)
        dv=st.checkbox("Drain valves",True)
        bp=st.checkbox("Bypass line",False)
    return Water(fluid,glycol,ewt,lwt,flow,dp,pumpqty,pumparr,head,pkw,pflc,pipe,strainer,fstype,et,av,dv,bp)


def fan_form() -> Fan:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        qty=int(st.number_input("No. of condenser fans", min_value=0, max_value=20, value=2, step=1, key="fan_qty"))
        kw=nfloat("Fan kW each",0.75,"fankw",0.05,0.0)
    with c2:
        flc=nfloat("Fan FLC A each (0=estimate)",0.0,"fanflc",0.1,0.0)
        volt=nfloat("Fan voltage V",415.0,"fanvolt",1.0,0.0)
        phase=st.selectbox("Fan phase", ["3-phase", "1-phase"])
    with c3:
        ctrl=st.selectbox("Fan control", ["Pressure switch staging", "Pressure transducer + controller", "VFD", "EC fan 0-10 V", "Always ON with compressor"])
        cont=st.checkbox("Contactor per fan", True)
    with c4:
        ol=st.checkbox("Overload per fan", True)
        delay=int(st.number_input("Fan stage delay sec", min_value=0, max_value=120, value=10, step=5, key="fan_stage_delay"))
    return Fan(qty,kw,flc,volt,phase,ctrl,cont,ol,delay)


def electrical_form() -> Electrical:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        v=nfloat("Main voltage V",415.0,"mainv",1.0,0.0)
        ph=st.selectbox("Main supply", ["3-phase", "1-phase"])
        hz=nfloat("Frequency Hz",50.0,"hz",1.0,0.0)
    with c2:
        cv=st.selectbox("Control voltage", ["230 VAC", "110 VAC", "24 VAC", "24 VDC"])
        cs=st.selectbox("Compressor starter", ["DOL", "Star-delta", "Soft starter", "VFD"])
        ps=st.selectbox("Pump starter", ["DOL", "Star-delta", "Soft starter", "VFD"])
    with c3:
        ip=st.selectbox("Panel IP", ["IP54", "IP55", "IP65", "IP66"])
        loc=st.selectbox("Panel location", ["Indoor", "Outdoor under canopy", "Outdoor exposed"])
        method=st.selectbox("Control method", ["Hardwired relay logic", "PLC", "Dedicated chiller controller"])
    with c4:
        bms=st.checkbox("BMS",True); remote=st.checkbox("Remote start/stop",True); fault=st.checkbox("Common fault",True)
        pr=st.checkbox("Phase relay",True); estop=st.checkbox("Emergency stop",True); door=st.checkbox("Door interlock",False)
        va=nfloat("Control transformer VA",250.0,"ctva",50.0,0.0)
    return Electrical(v,ph,hz,cv,cs,ps,ip,loc,method,bms,remote,fault,pr,estop,door,va)


def logic_form(project: Project) -> Logic:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        sp=nfloat("CHW setpoint °C",7.0,"sp",0.5)
        diff=nfloat("Temp differential K",2.0,"diff",0.5,0.1)
        pd=st.checkbox("Pump-down YV1+LPS",True)
        frz=nfloat("Freeze stat °C",3.0,"frz",0.5)
    with c2:
        pstart=int(st.number_input("Pump start delay sec", min_value=0, max_value=300, value=30, step=5, key="pump_start_delay"))
        flow=int(st.number_input("Flow proving delay sec", min_value=0, max_value=120, value=10, step=5, key="flow_proving_delay"))
        poff=int(st.number_input("Pump off delay sec", min_value=0, max_value=600, value=120, step=10, key="pump_off_delay"))
    with c3:
        lp=int(st.number_input("LP bypass delay sec", min_value=0, max_value=300, value=60, step=5, key="lp_bypass_delay"))
        ast=int(st.number_input("Anti-short-cycle sec", min_value=0, max_value=900, value=180, step=30, key="anti_short_cycle"))
        minon=int(st.number_input("Min compressor ON sec", min_value=0, max_value=900, value=120, step=30, key="min_comp_on"))
        pdmax=int(st.number_input("Max pumpdown sec", min_value=0, max_value=600, value=90, step=10, key="max_pumpdown"))
    with c4:
        pre=nfloat("Crankcase preheat h",8.0,"preheat",1.0,0.0)
        lead=st.checkbox("Lead/lag rotation",project.configuration.startswith("Tandem"))
        s2on=nfloat("Stage 2 ON offset K",2.0,"s2on",0.5,0.0)
        s2off=nfloat("Stage 2 OFF offset K",0.5,"s2off",0.5,0.0)
        lag=int(st.number_input("Lag start delay sec", min_value=0, max_value=900, value=120, step=30, key="lag_start_delay"))
    return Logic(sp,diff,pd,pstart,flow,lp,ast,minon,pdmax,poff,frz,pre,lead,s2on,s2off,lag)


def excel_report(project, circuits, water, fan, elec, logic, ps_dfs, specs, elec_sel, bom) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        rows=[]
        for section,obj in [("Project",project),("Water",water),("Fan",fan),("Electrical",elec),("Logic",logic)]:
            rows += [{"Section":section,"Input":k,"Value":v} for k,v in asdict(obj).items()]
        for c in circuits:
            rows += [{"Section":c.name,"Input":k,"Value":v} for k,v in asdict(c).items()]
        pd.DataFrame(rows).to_excel(writer,"Inputs",index=False)
        if ps_dfs: pd.concat(ps_dfs,ignore_index=True).to_excel(writer,"Pressure Settings",index=False)
        specs.to_excel(writer,"Component Specs",index=False)
        elec_sel.to_excel(writer,"Electrical Selection",index=False)
        bom.to_excel(writer,"BOM",index=False)
        pd.DataFrame([{"Note":"Preliminary app output only. Verify all component selections, wiring, pressure settings and code compliance before manufacturing."}]).to_excel(writer,"Notes",index=False)
    return out.getvalue()



# ---------------- password protection ----------------

def check_password() -> bool:
    """Password gate using Streamlit secrets.

    Required secret:
        APP_PASSWORD = "your-password"
    """
    try:
        expected_password = st.secrets["APP_PASSWORD"]
    except Exception:
        st.error("APP_PASSWORD is not set in Streamlit secrets.")
        st.info("Streamlit Cloud: Manage app → Settings → Secrets, then add: APP_PASSWORD = \"your-password\"")
        return False

    if st.session_state.get("password_ok", False):
        return True

    st.title("Chiller App Login")
    entered_password = st.text_input("Enter app password", type="password")
    if st.button("Login"):
        if entered_password == expected_password:
            st.session_state["password_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ---------------- main app ----------------

def main():
    st.set_page_config(page_title="Chiller Circuit + BOM Generator", layout="wide")
    if not check_password():
        st.stop()
    st.title("Chiller Electrical, Freon and Chilled Water Circuit Generator")
    st.caption("Full preliminary Streamlit app: inputs → pressure switch settings → schematics → component specifications → BOM.")
    with st.sidebar:
        st.header("Settings")
        pressure_unit = st.radio("Pressure unit", ["bar(g)", "bar(abs)", "psig"], index=0)
        st.warning("Preliminary engineering tool only. Final design must be verified.")
    if PropsSI is None:
        st.error("CoolProp is not installed. Install requirements.txt before running pressure calculations.")

    tabs = st.tabs(["1 Project", "2 Compressor PDF", "3 Refrigerant", "4 Water + Fans", "5 Electrical", "6 Logic", "7 Outputs"])
    with tabs[0]: project = project_form()
    with tabs[1]:
        st.subheader("Compressor PDF upload and missing data check")
        parsed: Dict[str,Any] = {}
        uploaded = st.file_uploader("Upload compressor supplier PDF", type=["pdf"])
        if uploaded:
            text, warnings = extract_pdf_text(uploaded)
            for w in warnings: st.warning(w)
            if text:
                parsed = parse_compressor_pdf(text)
                st.success(f"Candidate fields found: {len(parsed)}. Please verify before using.")
                required = [("compressor_make","Make"),("compressor_model","Model"),("approved_refrigerants","Approved refrigerants"),("max_high_pressure_barg","Max high-side pressure"),("max_condensing_temp_c","Max condensing temp"),("min_evaporating_temp_c","Min evaporating temp"),("compressor_flc_a","FLC/RLA"),("compressor_lra_a","LRA")]
                st.dataframe(pd.DataFrame([{"Data":lab,"Status":"Found - verify" if k in parsed else "Missing","Value":parsed.get(k,"")} for k,lab in required]),width="stretch",hide_index=True)
                with st.expander("Extracted text"):
                    st.text_area("Text", text[:30000], height=250)
    with tabs[2]:
        circuits=[]
        if project.number_of_circuits == 1:
            circuits.append(circuit_form("c1","Circuit 1",parsed,project))
        else:
            t1,t2=st.tabs(["Circuit 1","Circuit 2"])
            with t1: circuits.append(circuit_form("c1","Circuit 1",parsed,project))
            with t2: circuits.append(circuit_form("c2","Circuit 2",parsed,project))
    total_cap = sum(c.cooling_capacity_kw for c in circuits)
    if project.configuration.startswith("Tandem") and circuits: total_cap = circuits[0].cooling_capacity_kw
    with tabs[3]:
        st.subheader("Chilled water inputs")
        water = water_form(total_cap)
        st.markdown("---")
        st.subheader("Condenser fan inputs")
        fan = fan_form()
    with tabs[4]: elec = electrical_form()
    with tabs[5]: logic = logic_form(project)
    with tabs[6]:
        st.header("Outputs")
        ps_dfs=[]
        for c in circuits:
            df,warns,vals = pressure_settings(c, pressure_unit)
            st.subheader(f"{c.name} pressure switches")
            for w in warns: st.warning(w)
            if vals:
                m1,m2,m3=st.columns(3)
                m1.metric("Normal suction",ptxt(vals["normal_suction"],pressure_unit))
                m2.metric("Normal condensing",ptxt(vals["normal_condensing"],pressure_unit))
                m3.metric("Subcooled liquid reference",ptxt(vals["subcooled_liquid"],pressure_unit))
            if not df.empty:
                ps_dfs.append(df); st.dataframe(df,width="stretch",hide_index=True)
        st.markdown("---")
        st.subheader("Diagrams")
        esvg = electrical_svg(project,circuits,water,fan,elec,logic)
        rsvg = refrigerant_svg(project,circuits,fan)
        wsvg = water_svg(project,water)
        dt1,dt2,dt3 = st.tabs(["Electrical", "Freon / refrigerant", "Chilled water"])
        with dt1: show_svg(esvg,720); st.markdown(svg_link(esvg,"electrical_circuit.svg","Download electrical SVG"), unsafe_allow_html=True)
        with dt2: show_svg(rsvg,620); st.markdown(svg_link(rsvg,"freon_circuit.svg","Download Freon SVG"), unsafe_allow_html=True)
        with dt3: show_svg(wsvg,600); st.markdown(svg_link(wsvg,"chilled_water_circuit.svg","Download chilled-water SVG"), unsafe_allow_html=True)
        st.markdown("---")
        specs = component_specs(project,circuits,water,fan,elec,logic)
        elec_sel = electrical_selection(project,circuits,water,fan,elec)
        bom = bom_from(specs,elec_sel)
        st.subheader("Component specifications"); st.dataframe(specs,width="stretch",hide_index=True)
        st.subheader("Electrical selections"); st.dataframe(elec_sel,width="stretch",hide_index=True)
        st.subheader("Bill of material"); st.dataframe(bom,width="stretch",hide_index=True)
        st.subheader("Sequence logic")
        if project.configuration.startswith("Tandem"):
            st.code(f"""TANDEM COMMON CIRCUIT\nControl ON → pump → flow proven → TC1 stage 1 → common YV1 opens → LPS closes → lead compressor starts.\nIf load remains high by {logic.stage2_on_offset_k:.1f} K after {logic.lag_start_delay_s}s, lag compressor starts.\nOn unloading, lag compressor stops first. Lead stops by pump-down when TC1 is satisfied.\nCommon HPS/LPS/FS/FRZ/PR trips stop both compressors. Individual overload trips stop the affected compressor.""", language="text")
        else:
            st.code(f"""START\nControl ON → K0 ON → pump starts → wait {logic.pump_start_delay_s}s → flow proves → TC1 calls cooling → YV1 opens → suction rises → LPS closes → compressor starts.\n\nNORMAL STOP\nTC1 satisfied → YV1 closes → compressor pumps down → LPS opens → compressor stops → anti-short-cycle timer {logic.anti_short_cycle_s}s starts → pump off delay {logic.pump_off_delay_s}s.\n\nSAFETY STOP\nHPS / flow fail / freeze / overload / phase fault opens → compressor stops immediately.""", language="text")
        xlsx = excel_report(project,circuits,water,fan,elec,logic,ps_dfs,specs,elec_sel,bom)
        st.download_button("Download Excel report", xlsx, "chiller_design_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download electrical SVG", esvg, "electrical_circuit.svg", "image/svg+xml")
        st.download_button("Download Freon SVG", rsvg, "freon_circuit.svg", "image/svg+xml")
        st.download_button("Download chilled-water SVG", wsvg, "chilled_water_circuit.svg", "image/svg+xml")
        st.warning("Schematic-level output only. Manufacturing drawings require exact part numbers, terminal numbers, wire numbers, cable sizing, protection coordination, and code compliance.")

if __name__ == "__main__":
    main()
