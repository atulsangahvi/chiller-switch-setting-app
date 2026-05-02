# chiller_design_generator_app.py
# Full preliminary Streamlit app for chiller pressure settings, electrical schematic,
# refrigerant schematic, chilled-water schematic, component specifications, and BOM.
# Run: streamlit run chiller_design_generator_app.py

from __future__ import annotations

import base64
import io
import zipfile
from datetime import datetime
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_VERSION = "v9-clean-reference-diagram"

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
    preferred_manufacturer: str
    available_fault_ka: float

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



def flc_1ph(kw: float, v: float, pf: float = 0.85, eff: float = 0.90) -> float:
    if v <= 0 or pf <= 0 or eff <= 0:
        return 0.0
    return kw * 1000.0 / (v * pf * eff)


def motor_flc(kw: float, voltage_v: float, phase: str) -> float:
    return flc_3ph(kw, voltage_v) if phase == "3-phase" else flc_1ph(kw, voltage_v)


def cable_size_sqmm(current_a: float, factor: float = 1.25) -> float:
    amp_table = [
        (1.5, 14), (2.5, 18), (4, 25), (6, 32), (10, 45), (16, 61), (25, 80),
        (35, 99), (50, 119), (70, 151), (95, 182), (120, 210), (150, 240),
        (185, 273), (240, 321), (300, 367), (400, 443)
    ]
    required = max(0.1, current_a * factor)
    for size, amp in amp_table:
        if amp >= required:
            return float(size)
    return 400.0


def earth_size_sqmm(phase_size: float) -> float:
    if phase_size <= 16:
        return phase_size
    if phase_size <= 35:
        return 16.0
    return max(16.0, phase_size / 2.0)


def power_cable_desc(current_a: float, phase: str = "3-phase") -> str:
    size = cable_size_sqmm(current_a)
    pe = earth_size_sqmm(size)
    if phase == "3-phase":
        return f"3C x {size:g} sqmm Cu PVC/PVC + PE {pe:g} sqmm"
    return f"2C x {size:g} sqmm Cu PVC/PVC + PE {pe:g} sqmm"


def control_cable_desc(size: float = 1.5, cores: int = 2) -> str:
    return f"{cores}C x {size:g} sqmm Cu control cable"


def overload_range(current_a: float) -> str:
    ranges = [
        (0.63, 1.0), (1.0, 1.6), (1.6, 2.5), (2.5, 4.0), (4.0, 6.0),
        (5.5, 8.0), (7.0, 10.0), (9.0, 13.0), (12.0, 18.0), (17.0, 25.0),
        (23.0, 32.0), (30.0, 38.0), (37.0, 50.0), (48.0, 65.0), (55.0, 70.0),
        (63.0, 80.0), (80.0, 104.0), (95.0, 120.0), (110.0, 150.0)
    ]
    for lo, hi in ranges:
        if current_a <= hi:
            return f"{lo:g}-{hi:g} A"
    return f"{current_a*0.9:.0f}-{current_a*1.2:.0f} A"


def contactor_part_schneider(rating_a: int, control_voltage: str) -> str:
    base_map = {
        9:"LC1D09", 12:"LC1D12", 18:"LC1D18", 25:"LC1D25", 32:"LC1D32",
        40:"LC1D40A", 50:"LC1D50A", 65:"LC1D65A", 80:"LC1D80A", 95:"LC1D95A",
        115:"LC1D115", 150:"LC1D150", 185:"LC1D185", 225:"LC1D225", 265:"LC1D265",
        330:"LC1D330", 400:"LC1D400"
    }
    coil_map = {"230 VAC":"P7", "110 VAC":"F7", "24 VAC":"B7", "24 VDC":"BD"}
    base = base_map.get(rating_a, f"LC1D{rating_a}")
    coil = coil_map.get(control_voltage, "")
    return f"{base}{coil}" if coil else base


def overload_part_schneider(current_a: float) -> str:
    if current_a <= 1.0: return "LRD01"
    if current_a <= 1.6: return "LRD04"
    if current_a <= 2.5: return "LRD06"
    if current_a <= 4.0: return "LRD08"
    if current_a <= 6.0: return "LRD10"
    if current_a <= 8.0: return "LRD12"
    if current_a <= 10.0: return "LRD14"
    if current_a <= 13.0: return "LRD16"
    if current_a <= 18.0: return "LRD21"
    if current_a <= 25.0: return "LRD22"
    if current_a <= 32.0: return "LRD32"
    if current_a <= 38.0: return "LRD3353"
    if current_a <= 50.0: return "LRD3365"
    if current_a <= 65.0: return "LRD3367"
    return "Electronic overload relay - verify with vendor"


def mccb_part_schneider(frame_a: int, breaking_ka: int) -> str:
    series = "F" if breaking_ka <= 36 else ("H" if breaking_ka <= 70 else "L")
    frame = 100 if frame_a <= 100 else 160 if frame_a <= 160 else 250 if frame_a <= 250 else 400 if frame_a <= 400 else 630
    return f"Compact NSX{frame}{series}"


def mcb_part_schneider(poles: int, rating_a: int) -> str:
    pole_code = "1P" if poles == 1 else "2P" if poles == 2 else "3P" if poles == 3 else "4P"
    return f"Acti9 iC60N {pole_code} {rating_a}A"


def candidate_part(manufacturer: str, kind: str, rating_a: float = 0.0, control_voltage: str = "230 VAC", breaking_ka: int = 10) -> str:
    if manufacturer != "Schneider Electric":
        if kind == "phase_relay":
            return "3-phase monitoring relay - vendor select"
        if kind == "control_relay":
            return "Plug-in relay + socket - vendor select"
        if kind == "transformer":
            return "Control transformer - vendor select"
        return "Vendor-specific selection required"

    if kind == "contactor":
        return contactor_part_schneider(next_std(max(1.0, rating_a), STANDARD_CONTACTORS_A), control_voltage)
    if kind == "overload":
        return overload_part_schneider(rating_a)
    if kind == "mccb":
        return mccb_part_schneider(next_std(max(1.0, rating_a), STANDARD_BREAKERS_A), breaking_ka)
    if kind == "mcb_q2":
        return mcb_part_schneider(2, int(rating_a))
    if kind == "phase_relay":
        return "RM35TF30"
    if kind == "control_relay":
        return "RXM2AB2P7 + RXZE2S108M"
    if kind == "transformer":
        return "ABL6TS25U or equivalent"
    if kind == "terminal":
        return "Linergy terminal blocks"
    if kind == "pilot_lamp":
        return "XB5AV / XB5AD family"
    return "Verify part number"


def q1_breaking_capacity(available_fault_ka: float) -> int:
    for x in [10, 16, 25, 36, 50, 70]:
        if available_fault_ka <= x:
            return x
    return 70


def electrical_selection(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_a = 0.0
    manufacturer = elec.preferred_manufacturer
    q1_ka = q1_breaking_capacity(elec.available_fault_ka)

    for i, c in enumerate(circuits, 1):
        qty = project.number_of_compressors if project.configuration.startswith("Tandem") and i == 1 else 1
        flc = c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)
        total_a += flc * qty
        cont_rating = next_std(flc * 1.15, STANDARD_CONTACTORS_A)
        rows += [
            {
                "Tag":f"KM-C{i}", "Item":f"Compressor contactor {i}", "Qty":qty, "Duty / Basis":f"Compressor FLC {flc:.1f} A",
                "Selection":f"AC-3 contactor {cont_rating} A, coil {elec.control_voltage}",
                "Candidate part no.":candidate_part(manufacturer, "contactor", cont_rating, elec.control_voltage),
                "Recommended cable":power_cable_desc(flc), "Breaking capacity":"—",
                "Cross reference":"Aux NO to fan rung(s), Aux NC to crankcase heater", "Notes":"Select with 1NO+1NC auxiliaries"
            },
            {
                "Tag":f"OL-C{i}", "Item":f"Compressor overload {i}", "Qty":qty, "Duty / Basis":f"Compressor FLC {flc:.1f} A",
                "Selection":f"Adjustable overload range {overload_range(flc)}",
                "Candidate part no.":candidate_part(manufacturer, "overload", flc, elec.control_voltage),
                "Recommended cable":"—", "Breaking capacity":"—",
                "Cross reference":"NC contact in compressor safety chain", "Notes":"Set to compressor nameplate current"
            },
        ]

    pump_flc = water.pump_flc_a if water.pump_flc_a > 0 else flc_3ph(water.pump_kw, elec.main_voltage_v)
    total_a += pump_flc * water.pump_qty
    pump_cont = next_std(pump_flc * 1.15, STANDARD_CONTACTORS_A)
    rows += [
        {
            "Tag":"KM-P1", "Item":"Pump contactor", "Qty":water.pump_qty, "Duty / Basis":f"Pump FLC {pump_flc:.1f} A",
            "Selection":f"AC-3 contactor {pump_cont} A, coil {elec.control_voltage}",
            "Candidate part no.":candidate_part(manufacturer, "contactor", pump_cont, elec.control_voltage),
            "Recommended cable":power_cable_desc(pump_flc), "Breaking capacity":"—",
            "Cross reference":"Aux NO in compressor permissive rung", "Notes":"One starter set per pump"
        },
        {
            "Tag":"OL-P1", "Item":"Pump overload", "Qty":water.pump_qty, "Duty / Basis":f"Pump FLC {pump_flc:.1f} A",
            "Selection":f"Adjustable overload range {overload_range(pump_flc)}",
            "Candidate part no.":candidate_part(manufacturer, "overload", pump_flc, elec.control_voltage),
            "Recommended cable":"—", "Breaking capacity":"—",
            "Cross reference":"NC contact in pump rung and alarm", "Notes":"Set to motor nameplate current"
        },
    ]

    fan_flc = fan.flc_a_each if fan.flc_a_each > 0 else motor_flc(fan.motor_kw_each, fan.voltage_v, fan.phase)
    total_a += fan_flc * fan.qty
    fan_cont_qty = fan.qty if fan.contactor_per_fan else max(1, min(2, fan.qty))
    fan_ol_qty = fan.qty if fan.overload_per_fan else max(1, min(2, fan.qty))
    fan_cont = next_std(max(0.5, fan_flc * 1.15), STANDARD_CONTACTORS_A)
    rows += [
        {
            "Tag":"KM-F", "Item":"Fan contactor(s)", "Qty":fan_cont_qty, "Duty / Basis":f"Fan FLC {fan_flc:.1f} A each",
            "Selection":f"AC-3 contactor {fan_cont} A each, coil {elec.control_voltage}",
            "Candidate part no.":candidate_part(manufacturer, "contactor", fan_cont, elec.control_voltage),
            "Recommended cable":power_cable_desc(max(0.5, fan_flc), fan.phase), "Breaking capacity":"—",
            "Cross reference":"Coils controlled by CPS1/CPS2 rungs", "Notes":"Grouped fan contactors must be upsized for combined current"
        },
        {
            "Tag":"OL-F", "Item":"Fan overload(s)", "Qty":fan_ol_qty, "Duty / Basis":f"Fan FLC {fan_flc:.1f} A each",
            "Selection":f"Adjustable overload range {overload_range(max(0.5, fan_flc))}",
            "Candidate part no.":candidate_part(manufacturer, "overload", max(0.5, fan_flc), elec.control_voltage),
            "Recommended cable":"—", "Breaking capacity":"—",
            "Cross reference":"NC contact in fan rung / fault indication", "Notes":"Per motor overload preferred"
        },
    ]

    q1_rating = next_std(max(1.0, total_a * 1.25), STANDARD_BREAKERS_A)
    rows += [
        {
            "Tag":"Q1", "Item":"Main MCCB / isolator", "Qty":1, "Duty / Basis":f"Estimated running current {total_a:.1f} A",
            "Selection":f"TP MCCB {q1_rating} A thermal-magnetic",
            "Candidate part no.":candidate_part(manufacturer, "mccb", q1_rating, elec.control_voltage, q1_ka),
            "Recommended cable":power_cable_desc(total_a), "Breaking capacity":f"Icu ≥ {q1_ka} kA @ {elec.main_voltage_v:.0f} V",
            "Cross reference":"Main incomer feeding all power branches", "Notes":"Verify coordination with site fault level"
        },
        {
            "Tag":"Q2", "Item":"Control MCB", "Qty":1, "Duty / Basis":"Control supply branch",
            "Selection":"2P MCB 6 A",
            "Candidate part no.":candidate_part(manufacturer, "mcb_q2", 6, elec.control_voltage, 10),
            "Recommended cable":control_cable_desc(1.5, 2), "Breaking capacity":"10 kA",
            "Cross reference":"Feeds F1 / master control rung", "Notes":"Rating may be adjusted to transformer secondary current"
        },
        {
            "Tag":"F1", "Item":"Control fuse", "Qty":1, "Duty / Basis":"Transformer secondary protection",
            "Selection":"2 A gG fuse", "Candidate part no.":"gG 10x38 fuse + holder",
            "Recommended cable":"—", "Breaking capacity":"—",
            "Cross reference":"Ahead of Q2 / control circuit", "Notes":"Adjust to transformer design"
        },
        {
            "Tag":"T1", "Item":"Control transformer", "Qty":1, "Duty / Basis":elec.control_voltage,
            "Selection":f"{elec.control_transformer_va:.0f} VA, primary {elec.main_voltage_v:.0f} V, secondary {elec.control_voltage}",
            "Candidate part no.":candidate_part(manufacturer, "transformer"),
            "Recommended cable":control_cable_desc(2.5, 2), "Breaking capacity":"—",
            "Cross reference":"Provides control supply", "Notes":"Increase size if PLC/HMI or many relays are added"
        },
        {
            "Tag":"PR1", "Item":"Phase failure / sequence relay", "Qty":1 if elec.phase_relay else 0, "Duty / Basis":f"{elec.main_voltage_v:.0f} V 3-phase",
            "Selection":"3-phase monitoring relay", "Candidate part no.":candidate_part(manufacturer, "phase_relay"),
            "Recommended cable":control_cable_desc(1.5, 3), "Breaking capacity":"—",
            "Cross reference":"Healthy contact in compressor safety chain", "Notes":"Recommended for 3-phase motors"
        },
        {
            "Tag":"K0", "Item":"Master control relay", "Qty":1, "Duty / Basis":f"Coil {elec.control_voltage}",
            "Selection":"Plug-in control relay, 2CO", "Candidate part no.":candidate_part(manufacturer, "control_relay"),
            "Recommended cable":"—", "Breaking capacity":"—",
            "Cross reference":"Seal-in contact and LC bus feed contact", "Notes":"Use 2NO/2CO auxiliary relay or equivalent"
        },
        {
            "Tag":"TB1", "Item":"Field terminal strip", "Qty":1, "Duty / Basis":"Control field wiring",
            "Selection":"DIN rail terminal strip", "Candidate part no.":candidate_part(manufacturer, "terminal"),
            "Recommended cable":"Field dependent", "Breaking capacity":"—",
            "Cross reference":"FS1, HPS, LPS, CPS1/2, FRZ1, remote I/O", "Notes":"Provide terminal markers and end stops"
        },
    ]
    return pd.DataFrame(rows).query("Qty != 0").reset_index(drop=True)


def wire_schedule(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical, logic: Logic) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = 1

    def add(frm: str, to: str, service: str, cable: str, remarks: str = ""):
        nonlocal n
        rows.append({
            "Wire No.": f"W{n:03d}",
            "From": frm,
            "To": to,
            "Service": service,
            "Cable / wire": cable,
            "Remarks": remarks
        })
        n += 1

    # Power feeders
    comp_qty = project.number_of_compressors if project.configuration.startswith("Tandem") else len(circuits)
    for i in range(comp_qty):
        c = circuits[0] if project.configuration.startswith("Tandem") else circuits[i]
        flc = c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)
        cab = power_cable_desc(flc)
        add("Q1", f"KM-C{i+1}", f"Power feeder compressor {i+1}", cab, "3-phase + PE")
        add(f"KM-C{i+1}", f"OL-C{i+1}", f"Motor branch compressor {i+1}", cab, "")
        add(f"OL-C{i+1}", f"M-C{i+1}", f"Motor outgoing compressor {i+1}", cab, "To compressor terminals U/V/W")

    pump_flc = water.pump_flc_a if water.pump_flc_a > 0 else flc_3ph(water.pump_kw, elec.main_voltage_v)
    pump_cab = power_cable_desc(pump_flc)
    add("Q1", "KM-P1", "Power feeder pump", pump_cab, "")
    add("KM-P1", "OL-P1", "Pump motor branch", pump_cab, "")
    add("OL-P1", "M-P1", "Pump outgoing", pump_cab, "To pump motor")

    fan_flc = fan.flc_a_each if fan.flc_a_each > 0 else motor_flc(fan.motor_kw_each, fan.voltage_v, fan.phase)
    fan_cab = power_cable_desc(max(0.5, fan_flc), fan.phase)
    for i in range(min(fan.qty, 2)):
        add("Q1", f"KM-F{i+1}", f"Power feeder fan {i+1}", fan_cab, "")
        add(f"KM-F{i+1}", f"OL-F{i+1}", f"Fan motor branch {i+1}", fan_cab, "")
        add(f"OL-F{i+1}", f"M-F{i+1}", f"Fan outgoing {i+1}", fan_cab, "To condenser fan motor")

    # Control wires
    add("T1 secondary L", "F1", "Control supply", control_cable_desc(1.5, 1), "")
    add("F1", "Q2", "Control supply", control_cable_desc(1.5, 1), "")
    add("Q2", "E-STOP", "Master control rung", control_cable_desc(1.5, 1), "")
    add("E-STOP", "S0 STOP", "Master control rung", control_cable_desc(1.5, 1), "")
    add("S0 STOP", "S2 ON", "Master control rung", control_cable_desc(1.5, 1), "")
    add("S2 ON", "K0 coil", "Master control rung", control_cable_desc(1.5, 1), "")
    add("K0 NO", "LC bus", "Controlled live feed", control_cable_desc(1.5, 1), "")
    add("LC bus", "KM-P1 coil", "Pump control rung", control_cable_desc(1.5, 1), "Through SA1 and OL-P")
    add("LC bus", "YV1", "Cooling demand rung", control_cable_desc(1.5, 1), "Through TC1")
    add("LC bus", "KM-C1 coil", "Compressor control rung", control_cable_desc(1.5, 1), "Through FS1/HPS/LPS/FRZ/PR1/OL-C/AST")
    add("LC bus", "KM-F1 coil", "Fan 1 control rung", control_cable_desc(1.5, 1), "Through CPS1")
    if fan.qty >= 2:
        add("LC bus", "KM-F2 coil", "Fan 2 control rung", control_cable_desc(1.5, 1), "Through CPS2")
    add("LC bus", "HTR1", "Crankcase heater rung", control_cable_desc(1.5, 1), "Through KM-C NC contact")
    return pd.DataFrame(rows)


def terminal_schedule(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical, logic: Logic) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    term = 1

    def add(signal: str, internal: str, external: str, cable: str, remarks: str = ""):
        nonlocal term
        rows.append({
            "Terminal": f"TB1-{term:02d}",
            "Signal": signal,
            "Internal connection": internal,
            "External / field connection": external,
            "Field cable": cable,
            "Remarks": remarks,
        })
        term += 1

    add("Remote enable COM", "LC after K0", "Remote enable switch", control_cable_desc(1.5, 2), "Use only if remote start/stop required")
    add("Remote enable RET", "TC1 / enable chain", "Remote enable switch", control_cable_desc(1.5, 2), "")
    add("Flow switch COM", "Compressor permissive rung", "FS1 field switch", control_cable_desc(1.5, 2), water.flow_switch_type)
    add("Flow switch RET", "HPS input", "FS1 field switch", control_cable_desc(1.5, 2), "")
    add("HPS COM", "Safety chain", "HPS field switch", control_cable_desc(1.5, 2), "Manual reset preferred")
    add("HPS RET", "LPS input", "HPS field switch", control_cable_desc(1.5, 2), "")
    add("LPS COM", "Safety chain", "LPS field switch", control_cable_desc(1.5, 2), "Pump-down / low-pressure")
    add("LPS RET", "FRZ input", "LPS field switch", control_cable_desc(1.5, 2), "")
    add("CPS1 COM", "Fan 1 rung", "CPS1 field switch", control_cable_desc(1.5, 2), "Condenser pressure stage 1")
    add("CPS1 RET", "KM-F1 coil", "CPS1 field switch", control_cable_desc(1.5, 2), "")
    if fan.qty >= 2:
        add("CPS2 COM", "Fan 2 rung", "CPS2 field switch", control_cable_desc(1.5, 2), "Condenser pressure stage 2")
        add("CPS2 RET", "KM-F2 coil", "CPS2 field switch", control_cable_desc(1.5, 2), "")
    add("Freeze stat COM", "Safety chain", "FRZ1 thermostat", control_cable_desc(1.5, 2), f"Trip at {logic.freeze_stat_c:.1f} °C")
    add("Freeze stat RET", "PR1/OL-C input", "FRZ1 thermostat", control_cable_desc(1.5, 2), "")
    if elec.common_fault:
        add("Common fault NO", "Alarm relay", "BMS / dry contact", control_cable_desc(1.5, 2), "Volt-free contact")
        add("Common fault COM", "Alarm relay", "BMS / dry contact", control_cable_desc(1.5, 2), "")
    if elec.bms:
        add("Run status NO", "Run relay", "BMS / dry contact", control_cable_desc(1.5, 2), "Volt-free contact")
        add("Run status COM", "Run relay", "BMS / dry contact", control_cable_desc(1.5, 2), "")
    return pd.DataFrame(rows)


def contact_cross_reference(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical, logic: Logic) -> pd.DataFrame:
    rows = [
        {"Device": "K0", "Contact": "NO aux", "Used in": "Master seal-in path", "Purpose": "Self-hold after ON pushbutton", "Rung / reference": "Master control rung"},
        {"Device": "K0", "Contact": "NO aux", "Used in": "LC controlled live bus feed", "Purpose": "Makes lower control rungs live", "Rung / reference": "LC bus feed"},
        {"Device": "KM-P1", "Contact": "NO aux", "Used in": "Compressor permissive rung", "Purpose": "Pump contactor proven", "Rung / reference": "Compressor safety chain"},
        {"Device": "KM-C1", "Contact": "NO aux", "Used in": "Fan 1 rung", "Purpose": "Fan runs only when compressor is ON", "Rung / reference": "Fan 1 rung"},
        {"Device": "KM-C1", "Contact": "NO aux", "Used in": "Fan 2 rung", "Purpose": "Fan runs only when compressor is ON", "Rung / reference": "Fan 2 rung"},
        {"Device": "KM-C1", "Contact": "NC aux", "Used in": "Crankcase heater rung", "Purpose": "Heater ON when compressor is OFF", "Rung / reference": "Crankcase heater rung"},
        {"Device": "OL-C", "Contact": "NC aux", "Used in": "Compressor safety chain", "Purpose": "Stop compressor on overload trip", "Rung / reference": "Compressor safety chain"},
        {"Device": "OL-P", "Contact": "NC aux", "Used in": "Pump rung", "Purpose": "Stop / alarm on pump overload", "Rung / reference": "Pump rung"},
        {"Device": "PR1", "Contact": "NO healthy", "Used in": "Compressor safety chain", "Purpose": "Allow operation only with healthy 3-phase supply", "Rung / reference": "Compressor safety chain"},
    ]
    if fan.qty >= 1:
        rows.append({"Device": "CPS1", "Contact": "NO", "Used in": "Fan 1 rung", "Purpose": "Start fan stage 1 on condensing pressure rise", "Rung / reference": "Fan 1 rung"})
    if fan.qty >= 2:
        rows.append({"Device": "CPS2", "Contact": "NO", "Used in": "Fan 2 rung", "Purpose": "Start fan stage 2 on condensing pressure rise", "Rung / reference": "Fan 2 rung"})
    return pd.DataFrame(rows)


def electrical_standard_checks(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical, logic: Logic) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check: str, status: str, details: str, reference: str):
        rows.append({"Check": check, "Status": status, "Details": details, "Reference / standard": reference})

    # Main MCCB checks
    total_a = sum((c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)) for c in circuits)
    if project.configuration.startswith("Tandem") and circuits:
        c = circuits[0]
        total_a = (c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)) * project.number_of_compressors
    total_a += (water.pump_flc_a if water.pump_flc_a > 0 else flc_3ph(water.pump_kw, elec.main_voltage_v)) * water.pump_qty
    total_a += (fan.flc_a_each if fan.flc_a_each > 0 else motor_flc(fan.motor_kw_each, fan.voltage_v, fan.phase)) * fan.qty
    q1_rating = next_std(max(1.0, total_a * 1.25), STANDARD_BREAKERS_A)
    q1_ka = q1_breaking_capacity(elec.available_fault_ka)

    add("Main MCCB current rating", "PASS", f"Selected frame {q1_rating} A for estimated running current {total_a:.1f} A (125% sizing basis).", "IEC 60204-1 / IEC 60947-2")
    add("Main MCCB breaking capacity", "PASS" if q1_ka >= elec.available_fault_ka else "FAIL",
        f"Available fault level entered = {elec.available_fault_ka:.1f} kA; recommended MCCB Icu = {q1_ka} kA.",
        "IEC 60947-2")
    add("3-phase supply for compressor motors", "PASS" if elec.phase == "3-phase" else "FAIL",
        f"Main supply entered: {elec.phase}. Compressor and pump motor starters are assumed 3-phase.",
        "IEC 60204-1")
    add("Phase failure / phase sequence monitoring", "PASS" if elec.phase_relay else "WARN",
        "Phase relay is recommended for 3-phase chiller power circuits.", "Good engineering practice / IEC 60204-1")
    add("Emergency stop provision", "PASS" if elec.emergency_stop else "WARN",
        "Emergency stop should remove control permissive and stop hazardous motion / equipment.", "IEC 60204-1")
    add("Flow switch interlock", "PASS" if bool(water.flow_switch_type) else "FAIL",
        f"Water flow proving device: {water.flow_switch_type}.", "Good engineering practice")
    frz_ok = 0.5 < logic.freeze_stat_c < water.leaving_c
    add("Freeze thermostat setting", "PASS" if frz_ok else "WARN",
        f"Freeze stat {logic.freeze_stat_c:.1f} °C vs leaving water {water.leaving_c:.1f} °C.", "Chiller control best practice")
    ip_ok = (elec.panel_location == "Indoor") or \
            (elec.panel_location == "Outdoor under canopy" and elec.panel_ip in ["IP55", "IP65", "IP66"]) or \
            (elec.panel_location == "Outdoor exposed" and elec.panel_ip in ["IP65", "IP66"])
    add("Panel IP rating versus location", "PASS" if ip_ok else "WARN",
        f"Panel location: {elec.panel_location}; panel IP: {elec.panel_ip}.", "IEC 60529 / good practice")
    estimated_coil_va = (project.number_of_compressors + water.pump_qty + max(1, fan.qty if fan.contactor_per_fan else min(2, fan.qty)) + 2) * 20 + 8 * 5
    add("Control transformer VA adequacy", "PASS" if elec.control_transformer_va >= estimated_coil_va else "WARN",
        f"Entered transformer = {elec.control_transformer_va:.0f} VA; rough estimated coil/lamp load = {estimated_coil_va:.0f} VA.",
        "Control design best practice")
    add("Motor overload relay provision", "PASS",
        "Each motor branch includes overload relay selection and NC trip contact in the control logic.", "IEC 60947-4-1")
    add("Terminal and wire identification", "PASS",
        "App generates wire schedule and terminal schedule, but final panel drawings must still show wire markers and terminal numbering on the manufacturing drawings.",
        "IEC 60204-1")
    return pd.DataFrame(rows)


def bom_from(specs: pd.DataFrame, elec_sel: pd.DataFrame) -> pd.DataFrame:
    cols = ["System", "Tag", "Component", "Qty", "Specification", "Part Number", "Cable / Wire", "Remarks"]
    frames = []

    if specs is not None and not specs.empty:
        s = specs.copy()
        s["Part Number"] = ""
        s["Cable / Wire"] = ""
        for col in cols:
            if col not in s.columns:
                s[col] = ""
        frames.append(s[cols])

    if elec_sel is not None and not elec_sel.empty:
        e = elec_sel.copy()
        e["System"] = "Electrical"
        e = e.rename(columns={
            "Item": "Component",
            "Selection": "Specification",
            "Candidate part no.": "Part Number",
            "Recommended cable": "Cable / Wire",
            "Notes": "Remarks",
        })
        for col in cols:
            if col not in e.columns:
                e[col] = ""
        frames.append(e[cols])

    if not frames:
        return pd.DataFrame(columns=cols)

    return pd.concat(frames, ignore_index=True)


def refrigeration_part(kind: str, c: Circuit, device: str = "") -> str:
    """Preliminary candidate refrigeration control part numbers.

    These are catalogue-family suggestions only. Final exact part number depends on
    connection size, pressure range, coil voltage, refrigerant approval, reset type,
    ambient rating and local availability.
    """
    ref = c.refrigerant
    liq_mm = c.liquid_line_mm
    if kind == "hps":
        return "Danfoss KP 5 / KP 15 high pressure control - verify range and reset"
    if kind == "lps":
        return "Danfoss KP 1 / KP 15 low pressure control - verify range"
    if kind == "cps":
        return "Danfoss KP 5 / RT pressure control for condenser fan cycling"
    if kind == "yv1":
        if liq_mm <= 10:
            return f"Danfoss EVR 6 solenoid valve + coil, {ref}"
        if liq_mm <= 16:
            return f"Danfoss EVR 10/15 solenoid valve + coil, {ref}"
        return f"Danfoss EVR 20+ solenoid valve + coil, {ref} - size by capacity"
    if kind == "yv2":
        return "Danfoss EVR hot gas solenoid + KVC/CPCE style HGBP control - engineer size"
    if kind == "filter":
        return "Danfoss DML/DCL filter drier - size by line and refrigerant"
    if kind == "sight":
        return "Danfoss SGI/SGN sight glass moisture indicator"
    if kind == "txv":
        return "Danfoss TE/ETS valve family - select by capacity, refrigerant and pressure drop"
    return "Vendor select"


def refrigeration_controls_selection(circuits: List[Circuit], unit: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in circuits:
        df, warns, vals = pressure_settings(c, unit)
        if not vals:
            continue
        rows += [
            {
                "Circuit": c.name,
                "Tag": f"HPS-{c.name[-1] if c.name[-1].isdigit() else '1'}",
                "Component": "High pressure safety switch",
                "Pressure setting": ptxt(vals.get("hps_cutout", float("nan")), unit),
                "Pressure range required": "High side, manual reset preferred",
                "Candidate part no.": refrigeration_part("hps", c),
                "Notes": "Must be below compressor/system maximum high-side pressure."
            },
            {
                "Circuit": c.name,
                "Tag": f"LPS-{c.name[-1] if c.name[-1].isdigit() else '1'}",
                "Component": "Low pressure switch",
                "Pressure setting": f"Cut-in {ptxt(vals.get('lps_cutin', float('nan')), unit)} / cut-out {ptxt(vals.get('lps_cutout', float('nan')), unit)}",
                "Pressure range required": "Low side, auto reset for pump-down",
                "Candidate part no.": refrigeration_part("lps", c),
                "Notes": "Pump-down and low suction protection."
            },
            {
                "Circuit": c.name,
                "Tag": "CPS1",
                "Component": "Condenser fan stage 1 pressure switch",
                "Pressure setting": f"ON {ptxt(vals.get('cps1_on', float('nan')), unit)} / OFF {ptxt(vals.get('cps1_off', float('nan')), unit)}",
                "Pressure range required": "High side fan cycling",
                "Candidate part no.": refrigeration_part("cps", c),
                "Notes": "Starts first condenser fan stage."
            },
            {
                "Circuit": c.name,
                "Tag": "CPS2",
                "Component": "Condenser fan stage 2 pressure switch",
                "Pressure setting": f"ON {ptxt(vals.get('cps2_on', float('nan')), unit)} / OFF {ptxt(vals.get('cps2_off', float('nan')), unit)}",
                "Pressure range required": "High side fan cycling",
                "Candidate part no.": refrigeration_part("cps", c),
                "Notes": "Starts second condenser fan stage."
            },
        ]
        if c.liquid_solenoid_yv1:
            rows.append({
                "Circuit": c.name,
                "Tag": "YV1",
                "Component": "Liquid line solenoid valve",
                "Pressure setting": "Controlled by TC1 demand",
                "Pressure range required": f"Liquid line {c.liquid_line_mm:.1f} mm, {c.refrigerant}",
                "Candidate part no.": refrigeration_part("yv1", c),
                "Notes": "Used for pump-down control."
            })
        if c.hot_gas_bypass_yv2:
            rows.append({
                "Circuit": c.name,
                "Tag": "YV2/HGBP",
                "Component": "Hot gas bypass solenoid / regulator",
                "Pressure setting": f"Open {ptxt(vals.get('hgb_open', float('nan')), unit)} / close {ptxt(vals.get('hgb_close', float('nan')), unit)}",
                "Pressure range required": "Discharge to evaporator inlet/distributor",
                "Candidate part no.": refrigeration_part("yv2", c),
                "Notes": "Opens before LPS cut-out under low load."
            })
    return pd.DataFrame(rows)


def cable_schedule(project: Project, circuits: List[Circuit], water: Water, fan: Fan, elec: Electrical) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comp_qty = project.number_of_compressors if project.configuration.startswith("Tandem") else len(circuits)
    for i in range(comp_qty):
        c = circuits[0] if project.configuration.startswith("Tandem") else circuits[i]
        flc = c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)
        rows.append({
            "Cable tag": f"CAB-C{i+1}",
            "From": f"OL-C{i+1}",
            "To": f"M-C{i+1} compressor",
            "Load": f"{c.compressor_kw:.1f} kW compressor",
            "FLC A": round(flc, 2),
            "Cable recommendation": power_cable_desc(flc),
            "Basis": "125% current basis, Cu PVC/PVC - verify installation derating"
        })
    pump_flc = water.pump_flc_a if water.pump_flc_a > 0 else flc_3ph(water.pump_kw, elec.main_voltage_v)
    rows.append({
        "Cable tag": "CAB-P1",
        "From": "OL-P1",
        "To": "M-P1 chilled water pump",
        "Load": f"{water.pump_kw:.1f} kW pump",
        "FLC A": round(pump_flc, 2),
        "Cable recommendation": power_cable_desc(pump_flc),
        "Basis": "125% current basis, Cu PVC/PVC - verify installation derating"
    })
    fan_flc = fan.flc_a_each if fan.flc_a_each > 0 else motor_flc(fan.motor_kw_each, fan.voltage_v, fan.phase)
    for i in range(max(0, fan.qty)):
        rows.append({
            "Cable tag": f"CAB-F{i+1}",
            "From": f"OL-F{i+1 if i < 2 else 'X'}",
            "To": f"M-F{i+1} condenser fan",
            "Load": f"{fan.motor_kw_each:.2f} kW fan",
            "FLC A": round(fan_flc, 2),
            "Cable recommendation": power_cable_desc(max(0.5, fan_flc), fan.phase),
            "Basis": "125% current basis. Grouped fan circuits must be recalculated."
        })
    rows += [
        {"Cable tag":"CW-CTRL-01", "From":"Panel", "To":"Field safety switches", "Load":"FS1/HPS/LPS/FRZ/CPS", "FLC A":0.0, "Cable recommendation":control_cable_desc(1.5, 12), "Basis":"Control cable, numbered cores preferred"},
        {"Cable tag":"CW-BMS-01", "From":"Panel", "To":"BMS", "Load":"Remote start/common fault/run status", "FLC A":0.0, "Cable recommendation":control_cable_desc(1.5, 8), "Basis":"Volt-free contacts or as BMS specification"},
    ]
    return pd.DataFrame(rows)


def drawing_sheet_index() -> pd.DataFrame:
    rows = [
        {"Sheet": "E-001", "Title": "Electrical power and control schematic", "Output": "SVG + DXF + included in package", "Status": "Template generated"},
        {"Sheet": "R-001", "Title": "Freon / refrigerant circuit schematic", "Output": "SVG + DXF + included in package", "Status": "Template generated"},
        {"Sheet": "W-001", "Title": "Chilled water circuit schematic", "Output": "SVG + DXF + included in package", "Status": "Template generated"},
        {"Sheet": "S-001", "Title": "Wire schedule", "Output": "CSV + Excel", "Status": "Auto generated"},
        {"Sheet": "S-002", "Title": "Terminal schedule", "Output": "CSV + Excel", "Status": "Auto generated"},
        {"Sheet": "S-003", "Title": "Cable schedule", "Output": "CSV + Excel", "Status": "Auto generated"},
        {"Sheet": "S-004", "Title": "Contact cross-reference", "Output": "CSV + Excel", "Status": "Auto generated"},
        {"Sheet": "B-001", "Title": "BOM and component selections", "Output": "CSV + Excel", "Status": "Auto generated"},
        {"Sheet": "C-001", "Title": "Electrical standard checks", "Output": "CSV + Excel", "Status": "Preliminary checks"},
    ]
    return pd.DataFrame(rows)


def svg_to_basic_dxf(svg_text: str, title: str = "SCHEMATIC") -> str:
    """Very simple SVG-to-DXF converter for generated schematic line/rect/text content."""
    def dxf_line(x1, y1, x2, y2, layer="0"):
        return f"0\nLINE\n8\n{layer}\n10\n{x1}\n20\n{-y1}\n30\n0\n11\n{x2}\n21\n{-y2}\n31\n0\n"

    def dxf_text(x, y, text, height=7, layer="TEXT"):
        safe = re.sub(r"<.*?>", "", str(text)).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        safe = safe.replace("\n", " ")[:180]
        return f"0\nTEXT\n8\n{layer}\n10\n{x}\n20\n{-y}\n30\n0\n40\n{height}\n1\n{safe}\n"

    dxf = "0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n"
    dxf += dxf_text(20, 20, title, 14, "TITLE")

    for m in re.finditer(r'<line[^>]*x1="([0-9.]+)"[^>]*y1="([0-9.]+)"[^>]*x2="([0-9.]+)"[^>]*y2="([0-9.]+)"', svg_text):
        x1, y1, x2, y2 = map(float, m.groups())
        dxf += dxf_line(x1, y1, x2, y2, "LINES")

    for m in re.finditer(r'<rect[^>]*x="([0-9.]+)"[^>]*y="([0-9.]+)"[^>]*width="([0-9.]+)"[^>]*height="([0-9.]+)"', svg_text):
        x, y, w, h = map(float, m.groups())
        dxf += dxf_line(x, y, x+w, y, "BOX")
        dxf += dxf_line(x+w, y, x+w, y+h, "BOX")
        dxf += dxf_line(x+w, y+h, x, y+h, "BOX")
        dxf += dxf_line(x, y+h, x, y, "BOX")

    for m in re.finditer(r'<text[^>]*x="([0-9.]+)"[^>]*y="([0-9.]+)"[^>]*>(.*?)</text>', svg_text, flags=re.S):
        x, y, text = m.groups()
        dxf += dxf_text(float(x), float(y), text, 7, "TEXT")

    dxf += "0\nENDSEC\n0\nEOF\n"
    return dxf


def make_pdf_report(project, circuits, water, fan, elec, logic, specs, elec_sel, wire_df, terminal_df, cable_df, xref_df, checks_df, refrig_ctrl_df, bom) -> bytes:
    """Create a multi-page PDF report with schedules.

    Drawings are included in the package as SVG/DXF. The PDF includes schedules and checks.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    except Exception as exc:
        raise RuntimeError(f"ReportLab is not installed: {exc}")

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    def add_heading(txt):
        story.append(Paragraph(txt, styles["Heading1"]))
        story.append(Spacer(1, 8))

    def add_df(title, df, max_rows=28):
        add_heading(title)
        if df is None or df.empty:
            story.append(Paragraph("No data.", styles["Normal"]))
            story.append(PageBreak())
            return
        display = df.head(max_rows).astype(str)
        data = [list(display.columns)] + display.values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 6),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(tbl)
        story.append(PageBreak())

    add_heading(f"{project.project_name} - Preliminary Manufacturing Package")
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Configuration: {project.configuration}", styles["Normal"]))
    story.append(Paragraph("This PDF contains schedules and checks. SVG and DXF drawing sheets are included separately in the downloadable ZIP package.", styles["Normal"]))
    story.append(PageBreak())

    add_df("Drawing Sheet Index", drawing_sheet_index())
    add_df("Component Specifications", specs)
    add_df("Electrical Selections", elec_sel)
    add_df("Refrigeration Controls", refrig_ctrl_df)
    add_df("Wire Schedule", wire_df)
    add_df("Terminal Schedule", terminal_df)
    add_df("Cable Schedule", cable_df)
    add_df("Contact Cross-reference", xref_df)
    add_df("Electrical Standard Checks", checks_df)
    add_df("Bill of Material", bom)

    doc.build(story)
    return out.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_manufacturing_zip(esvg: str, rsvg: str, wsvg: str, xlsx: bytes, pdf_bytes: bytes, specs: pd.DataFrame, elec_sel: pd.DataFrame, wire_df: pd.DataFrame, terminal_df: pd.DataFrame, cable_df: pd.DataFrame, xref_df: pd.DataFrame, checks_df: pd.DataFrame, refrig_ctrl_df: pd.DataFrame, bom: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("01_drawings/electrical_circuit.svg", esvg)
        z.writestr("01_drawings/freon_circuit.svg", rsvg)
        z.writestr("01_drawings/chilled_water_circuit.svg", wsvg)
        z.writestr("02_dxf/electrical_circuit.dxf", svg_to_basic_dxf(esvg, "Electrical circuit"))
        z.writestr("02_dxf/freon_circuit.dxf", svg_to_basic_dxf(rsvg, "Freon circuit"))
        z.writestr("02_dxf/chilled_water_circuit.dxf", svg_to_basic_dxf(wsvg, "Chilled water circuit"))
        z.writestr("03_reports/chiller_design_report.xlsx", xlsx)
        z.writestr("03_reports/chiller_manufacturing_report.pdf", pdf_bytes)
        z.writestr("04_schedules/component_specs.csv", df_to_csv_bytes(specs))
        z.writestr("04_schedules/electrical_selection.csv", df_to_csv_bytes(elec_sel))
        z.writestr("04_schedules/refrigeration_controls.csv", df_to_csv_bytes(refrig_ctrl_df))
        z.writestr("04_schedules/wire_schedule.csv", df_to_csv_bytes(wire_df))
        z.writestr("04_schedules/terminal_schedule.csv", df_to_csv_bytes(terminal_df))
        z.writestr("04_schedules/cable_schedule.csv", df_to_csv_bytes(cable_df))
        z.writestr("04_schedules/contact_cross_reference.csv", df_to_csv_bytes(xref_df))
        z.writestr("04_schedules/electrical_checks.csv", df_to_csv_bytes(checks_df))
        z.writestr("04_schedules/bom.csv", df_to_csv_bytes(bom))
        z.writestr("README_MANUFACTURING_PACKAGE.txt",
                   "Preliminary auto-generated chiller manufacturing package. Verify all drawings, part numbers, cable sizes, protection settings and code compliance before manufacture.\n")
    return out.getvalue()


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
    """Clean reference-style electrical diagram.

    This version deliberately follows the uploaded reference:
    - left: power circuit
    - right: ladder control circuit
    - bottom: legend and notes
    Wire/terminal/cable details remain in generated schedules and are lightly referenced on the drawing.
    """
    w, h = 1700, 1100

    def mtext(x, y, content, cls="small", anchor="middle", color="#111", gap=13):
        lines = str(content).split("\n")
        out = ""
        for i, line in enumerate(lines):
            out += f'<text x="{x}" y="{y + i*gap}" text-anchor="{anchor}" class="{cls}" fill="{color}">{esc(line)}</text>'
        return out

    def tagtext(x, y, content, color="#003399", anchor="start"):
        return mtext(x, y, content, "small", anchor, color, 11)

    def contact_no(x, y, label="", wno=""):
        out = ln(x, y, x+16, y, "wire")
        out += ln(x+16, y-9, x+16, y+9, "wire")
        out += ln(x+34, y-9, x+34, y+9, "wire")
        out += ln(x+34, y, x+50, y, "wire")
        if label:
            out += mtext(x+25, y-25, label, "small")
        if wno:
            out += tagtext(x+18, y+22, wno)
        return out

    def contact_nc(x, y, label="", wno=""):
        out = ln(x, y, x+16, y, "wire")
        out += ln(x+16, y-9, x+16, y+9, "wire")
        out += ln(x+34, y-9, x+34, y+9, "wire")
        out += ln(x+16, y+9, x+34, y-9, "wire")
        out += ln(x+34, y, x+50, y, "wire")
        if label:
            out += mtext(x+25, y-25, label, "small")
        if wno:
            out += tagtext(x+18, y+22, wno)
        return out

    def coil(x, y, label, width=72):
        out = f'<rect x="{x}" y="{y-16}" width="{width}" height="32" fill="white" stroke="#111" stroke-width="1.5" rx="16" ry="16"/>'
        out += mtext(x+width/2, y-3, label, "small", "middle", "#111", 12)
        return out

    def lamp(x, y, label):
        out = f'<circle cx="{x}" cy="{y}" r="15" fill="white" stroke="#111" stroke-width="1.5"/>'
        out += ln(x-9, y-9, x+9, y+9, "wire")
        out += ln(x+9, y-9, x-9, y+9, "wire")
        out += mtext(x, y-23, label, "small")
        return out

    def small_box(x, y, ww, hh, label, cls="box"):
        return bx(x, y, ww, hh, label, cls)

    def three_pole_device(x, y, tag, label, rating=""):
        out = mtext(x+34, y-33, tag, "small")
        out += mtext(x+34, y-19, label, "small")
        if rating:
            out += mtext(x+34, y+68, rating, "small", "middle", "#003399")
        for i in range(3):
            xx = x + i*34
            out += ln(xx, y, xx, y+16, "wire")
            out += f'<circle cx="{xx}" cy="{y+18}" r="3" fill="white" stroke="#111" stroke-width="1.2"/>'
            out += ln(xx-8, y+26, xx+8, y+38, "wire")
            out += f'<circle cx="{xx}" cy="{y+44}" r="3" fill="white" stroke="#111" stroke-width="1.2"/>'
            out += ln(xx, y+46, xx, y+62, "wire")
        return out

    def overload_block(x, y, tag, label, setting=""):
        out = mtext(x+34, y-10, tag, "small")
        out += f'<rect x="{x-12}" y="{y}" width="92" height="46" fill="white" stroke="#111" stroke-width="1.4"/>'
        for i in range(3):
            xx = x + i*34
            out += ln(xx, y-18, xx, y, "wire")
            out += f'<rect x="{xx-8}" y="{y+10}" width="16" height="16" fill="white" stroke="#111" stroke-width="1"/>'
            out += ln(xx, y+46, xx, y+64, "wire")
        out += mtext(x+34, y+62, label, "small")
        if setting:
            out += mtext(x+34, y+88, setting, "small", "middle", "#003399")
        return out

    def motor_symbol(x, y, tag, label, rating, cable=""):
        out = f'<circle cx="{x}" cy="{y}" r="30" fill="white" stroke="#111" stroke-width="1.5"/>'
        out += mtext(x, y-4, "M", "head")
        out += mtext(x, y+12, "3~", "small")
        out += mtext(x, y+50, tag, "small")
        out += mtext(x, y+66, label, "small")
        out += mtext(x, y+94, rating, "small", "middle", "#111")
        if cable:
            out += mtext(x, y+110, cable, "small", "middle", "#003399")
        out += ln(x+30, y+17, x+54, y+17, "wire")
        out += f'<path d="M{x+54} {y+17} L{x+72} {y+17} M{x+57} {y+22} L{x+69} {y+22} M{x+60} {y+27} L{x+66} {y+27}" stroke="#111" stroke-width="1.2"/>'
        return out

    def notes_box(x, y, ww, hh, title, lines):
        out = f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}" fill="white" stroke="#111" stroke-width="1.3"/>'
        out += mtext(x+ww/2, y+17, title, "small")
        yy = y + 37
        for line in lines:
            out += mtext(x+10, yy, line, "small", "start", "#111", 12)
            yy += 15
        return out

    # Basic calculated values
    comp_qty = project.number_of_compressors if project.configuration.startswith("Tandem") else len(circuits)
    first_c = circuits[0] if circuits else None
    comp_flc = (first_c.compressor_flc_a if first_c and first_c.compressor_flc_a > 0 else flc_3ph(first_c.compressor_kw if first_c else 15, elec.main_voltage_v))
    pump_flc = water.pump_flc_a if water.pump_flc_a > 0 else flc_3ph(water.pump_kw, elec.main_voltage_v)
    fan_flc = fan.flc_a_each if fan.flc_a_each > 0 else motor_flc(fan.motor_kw_each, fan.voltage_v, fan.phase)
    total_a = comp_flc*comp_qty + pump_flc*water.pump_qty + fan_flc*fan.qty
    q1_rating = next_std(max(1.0, total_a*1.25), STANDARD_BREAKERS_A)
    q1_ka = q1_breaking_capacity(elec.available_fault_ka)

    # Start SVG
    s = svg_start(w, h, f"{project.project_name} - Electrical & Control Diagram")
    s += '<rect x="10" y="50" width="1680" height="1030" fill="white" stroke="#111" stroke-width="1.3"/>'
    s += '<line x1="905" y1="55" x2="905" y2="850" stroke="#111" stroke-width="1.4" stroke-dasharray="8 5"/>'
    s += mtext(455, 72, "1) POWER CIRCUIT", "head")
    s += mtext(1295, 72, f"2) CONTROL CIRCUIT ({elec.control_voltage})", "head")
    s += mtext(24, 100, f"{elec.phase} {elec.main_voltage_v:.0f} V\n{elec.frequency_hz:.0f} Hz", "small", "start")

    # Incoming power
    qx, qy = 105, 126
    for i, lab in enumerate(["L1","L2","L3"]):
        xx = qx + i*42
        s += mtext(xx, qy-28, lab, "small")
        s += ln(xx, qy-12, xx, 304, "wire")
    s += three_pole_device(qx, qy, "Q1", "MAIN ISOLATOR\n/ MCCB", f"{q1_rating}A {q1_ka}kA")
    s += tagtext(qx-20, 218, "P001/P002/P003", "#003399")

    # PR1 and T1
    s += small_box(258, 126, 125, 68, "PR1\nPHASE FAILURE /\nSEQUENCE RELAY")
    s += small_box(505, 120, 150, 78, f"T1\nCONTROL\nTRANSFORMER\n{elec.control_voltage}")
    s += ln(230, 158, 258, 158, "wire")
    s += ln(383, 158, 450, 158, "wire")
    # transformer stylized coil
    for i in range(5):
        s += f'<path d="M{455+i*8} 128 C{445+i*8} 136,{445+i*8} 181,{455+i*8} 189" stroke="#111" fill="none" stroke-width="1.2"/>'
        s += f'<path d="M{670+i*8} 128 C{680+i*8} 136,{680+i*8} 181,{670+i*8} 189" stroke="#111" fill="none" stroke-width="1.2"/>'
    s += ln(700, 158, 800, 158, "wire")
    s += f'<circle cx="803" cy="158" r="4" fill="white" stroke="#111" stroke-width="1.2"/>'
    s += mtext(745, 137, f"{elec.control_voltage}\nCONTROL SUPPLY", "small")
    s += tagtext(265, 210, "P004", "#003399")
    s += tagtext(702, 178, "C001/C002", "#003399")

    # 3 phase bus
    bus_y = 320
    s += ln(82, bus_y, 835, bus_y, "line")
    s += ln(82, bus_y+12, 835, bus_y+12, "line")
    s += ln(82, bus_y+24, 835, bus_y+24, "line")

    # Branches: compressor(s), pump, fan1, fan2.
    branches = []
    for i in range(comp_qty):
        c = circuits[0] if project.configuration.startswith("Tandem") else circuits[min(i, len(circuits)-1)]
        flc = c.compressor_flc_a if c.compressor_flc_a > 0 else flc_3ph(c.compressor_kw, elec.main_voltage_v)
        branches.append((f"KM-C{i+1}", f"OL-C{i+1}", f"M-C{i+1}", f"COMPRESSOR\n{i+1}", f"{c.compressor_kw:.1f} kW / {flc:.1f} A", f"CAB-C{i+1}", flc))
    branches.append(("KM-P1", "OL-P1", "M-P1", "CHILLED WATER\nPUMP", f"{water.pump_kw:.1f} kW / {pump_flc:.1f} A", "CAB-P1", pump_flc))
    for i in range(min(fan.qty, 2)):
        branches.append((f"KM-F{i+1}", f"OL-F{i+1}", f"M-F{i+1}", f"CONDENSER\nFAN {i+1}", f"{fan.motor_kw_each:.2f} kW / {fan_flc:.1f} A", f"CAB-F{i+1}", fan_flc))

    start_x = 100
    spacing = 145 if len(branches) <= 5 else 122
    for idx, (km, ol, mt, label, rating, cab, load_a) in enumerate(branches):
        x = start_x + idx*spacing
        for off in [0,12,24]:
            s += ln(x + off/2, bus_y+24, x + off/2, 365, "wire")
        s += three_pole_device(x-4, 365, km, "CONTACTOR")
        s += overload_block(x-4, 462, ol, "OVERLOAD\nRELAY", overload_range(load_a))
        s += motor_symbol(x+30, 602, mt, label, rating, cab)

    # Crankcase heater branch on right of power circuit
    s += ln(824, bus_y+24, 824, 462, "wire")
    s += small_box(804, 462, 40, 70, "FCH\n2A")
    s += ln(824, 532, 824, 570, "wire")
    s += small_box(796, 570, 56, 118, "HTR1\nCOMP.\nCRANKCASE\nHEATER")
    s += mtext(824, 710, "230 V AC", "small")

    # Control circuit rails
    xL, xN = 950, 1650
    rail_top, rail_bottom = 90, 825
    s += mtext(xL, 80, "L", "head")
    s += mtext(xN, 80, "N", "head")
    s += ln(xL, rail_top, xL, rail_bottom, "line")
    s += ln(xN, rail_top, xN, rail_bottom, "line")

    # Master control rung
    y = 112
    s += ln(xL, y, xL+38, y, "wire")
    s += small_box(xL+38, y-16, 55, 32, "F1\n2A")
    s += small_box(xL+122, y-16, 62, 32, "Q2\n6A")
    s += small_box(xL+220, y-16, 75, 32, "E-STOP\nNC")
    s += small_box(xL+330, y-16, 75, 32, "S0 STOP\nNC")
    s += ln(xL+93, y, xL+122, y, "wire")
    s += ln(xL+184, y, xL+220, y, "wire")
    s += ln(xL+295, y, xL+330, y, "wire")
    s += ln(xL+405, y, xN, y, "wire")
    s += tagtext(xL+4, y-14, "100")
    s += tagtext(xL+102, y+27, "101")
    s += tagtext(xL+410, y-14, "105")

    # K0 NO control bus
    y = 166
    s += contact_no(xL, y, "K0\nNO", "110")
    s += ln(xL+50, y, xL+235, y, "wire")
    s += mtext(xL+245, y+4, "CONTROL BUS AFTER F1, Q2, E-STOP, S0 & K0", "small", "start")
    s += mtext(xL+420, y+24, "LC BUS", "small", "start", "#990000")

    # Rung numbers and rungs
    rung_y = [244, 326, 410, 494, 578, 660, 742]
    for rn, yy in enumerate(rung_y, 1):
        s += f'<circle cx="{xL-28}" cy="{yy}" r="12" fill="white" stroke="#111" stroke-width="1.2"/>'
        s += mtext(xL-28, yy+4, str(rn), "small")

    # Rung 1 pump
    y = rung_y[0]
    s += contact_no(xL, y, "SA1\nPUMP\nAUTO/MAN", "120")
    s += small_box(xL+85, y-40, 155, 80, "AUTO\n\nMAN", "dash")
    s += contact_nc(xL+275, y, "OL-P\nNC", "121")
    s += contact_no(xL+365, y, "S1\nPUMP ON\nNO", "122")
    s += ln(xL+415, y, xN-95, y, "wire")
    s += coil(xN-95, y, "KM-P1")
    s += lamp(xN-38, y+35, "H2")

    # Rung 2 YV1
    y = rung_y[1]
    s += contact_no(xL, y, "TC1\nCOOL\nOUTPUT", "130")
    s += ln(xL+50, y, xN-110, y, "wire")
    s += mtext(xN-165, y-34, "SOLENOID VALVE\nFOR FREON", "small")
    s += coil(xN-110, y, "YV1")
    s += lamp(xN-38, y+35, "H5")

    # Rung 3 compressor
    y = rung_y[2]
    x = xL
    contacts = [
        ("KM-P\nNO", "300", False),
        (f"TD1\n{logic.pump_start_delay_s}s", "301", False),
        ("FS1\nNO", "302", False),
        ("HPS\nNC\nMAN RESET", "303", True),
        ("LPS\nNC\nPUMP-DOWN", "304", True),
        ("FRZ1\nNC", "305", True),
        ("PR1\nOK", "306", False),
        ("OL-C\nNC", "307", True),
        (f"AST\n{logic.anti_short_cycle_s}s", "308", False),
    ]
    step = 66
    for lab, wno, nc in contacts:
        s += contact_nc(x, y, lab, wno) if nc else contact_no(x, y, lab, wno)
        x += step
    s += ln(x, y, xN-95, y, "wire")
    s += coil(xN-95, y, "KM-C1")
    s += lamp(xN-38, y+35, "H3")

    # Rung 4 Fan1
    y = rung_y[3]
    x = xL
    for lab,wno,nc in [("SA2\nFANS\nAUTO/MAN","500",False),("KM-C\nNO","501",False),("OL-F1\nNC","502",True),("CPS1\nNO","503",False)]:
        s += contact_nc(x, y, lab, wno) if nc else contact_no(x, y, lab, wno)
        x += 102
    s += ln(x, y, xN-95, y, "wire")
    s += coil(xN-95, y, "KM-F1")
    s += lamp(xN-38, y+35, "H6")

    # Rung 5 Fan2
    y = rung_y[4]
    x = xL
    for lab,wno,nc in [("SA2\nFANS\nAUTO/MAN","510",False),("KM-C\nNO","511",False),("OL-F2\nNC","512",True),("CPS2\nNO","513",False)]:
        s += contact_nc(x, y, lab, wno) if nc else contact_no(x, y, lab, wno)
        x += 102
    s += ln(x, y, xN-95, y, "wire")
    s += coil(xN-95, y, "KM-F2")
    s += lamp(xN-38, y+35, "H7")

    # Rung 6 heater
    y = rung_y[5]
    s += contact_nc(xL, y, "KM-C\nNC", "600")
    s += ln(xL+50, y, xN-110, y, "wire")
    s += small_box(xN-110, y-19, 78, 38, "HTR1\nCCH")
    s += lamp(xN-38, y+35, "H4")

    # Rung 7 faults
    y = rung_y[6]
    x = xL
    for lab, lamp_tag, wno in [("HPS\nTRIP","H8","610"),("LPS\nTRIP","H9","611"),("OL-C\nTRIP","H11","612"),("OL-P\nTRIP","H12","613")]:
        s += contact_no(x, y, lab, wno)
        s += lamp(x+85, y, lamp_tag)
        x += 170

    # Terminal reference note
    s += mtext(1045, 814, "Field terminals: TB1-03/04 FS1, TB1-05/06 HPS, TB1-07/08 LPS, TB1-11/12 CPS1, TB1-13/14 CPS2", "small", "start", "#990000")

    # Bottom boxes
    s += notes_box(18, 865, 430, 165, "LEGEND (SYMBOLS & ABBREVIATIONS)", [
        "KM Contactor      OL Overload relay      M Motor",
        "FS Flow switch    HPS High pressure switch    LPS Low pressure switch",
        "CPS Condenser pressure switch    TC Temperature controller",
        "YV Solenoid valve    TD/AST Timers    K0 Master relay",
        "Blue text = wire/cable number. Terminal details are in schedules."
    ])
    s += notes_box(465, 865, 440, 165, "POWER CIRCUIT NOTES", [
        f"1. Q1 main MCCB approx. {q1_rating} A, Icu >= {q1_ka} kA.",
        "2. PR1 monitors phase failure and sequence.",
        "3. OL relays are selected from motor FLC/RLA.",
        "4. Cable tags CAB-C/P/F are listed in cable schedule.",
        "5. Final cable sizing requires derating and voltage-drop check."
    ])
    s += notes_box(925, 865, 745, 165, "CONTROL CIRCUIT NOTES (OPERATIONAL FEATURES)", [
        "1. K0 master relay creates protected LC control bus after F1, Q2, E-stop and stop.",
        "2. Pump-first: pump runs and FS1 must prove water flow before compressor starts.",
        "3. Pump-down: TC1 opens YV1; LPS stops compressor at low suction pressure.",
        "4. HPS is manual reset. CPS1/CPS2 stage condenser fans by head pressure.",
        "5. AST prevents rapid compressor restart. HTR1 energizes when compressor is OFF."
    ])

    s += "</svg>"
    return s


def show_svg(svg: str, height: int=650):
    html = f'<div style="width:100%; overflow:auto; border:1px solid #ddd; padding:8px">{svg}</div>'
    if hasattr(st, "html"):
        st.html(html)
    else:
        components.html(html, height=height)


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
        manufacturer=st.selectbox("Preferred electrical manufacturer", ["Schneider Electric", "Generic"], index=0)
    with c2:
        cv=st.selectbox("Control voltage", ["230 VAC", "110 VAC", "24 VAC", "24 VDC"])
        cs=st.selectbox("Compressor starter", ["DOL", "Star-delta", "Soft starter", "VFD"])
        ps=st.selectbox("Pump starter", ["DOL", "Star-delta", "Soft starter", "VFD"])
        fault_ka=nfloat("Available fault level kA",25.0,"faultka",1.0,1.0)
    with c3:
        ip=st.selectbox("Panel IP", ["IP54", "IP55", "IP65", "IP66"])
        loc=st.selectbox("Panel location", ["Indoor", "Outdoor under canopy", "Outdoor exposed"])
        method=st.selectbox("Control method", ["Hardwired relay logic", "PLC", "Dedicated chiller controller"])
    with c4:
        bms=st.checkbox("BMS",True)
        remote=st.checkbox("Remote start/stop",True)
        fault=st.checkbox("Common fault",True)
        pr=st.checkbox("Phase relay",True)
        estop=st.checkbox("Emergency stop",True)
        door=st.checkbox("Door interlock",False)
        va=nfloat("Control transformer VA",250.0,"ctva",50.0,0.0)
    return Electrical(v,ph,hz,cv,cs,ps,ip,loc,method,bms,remote,fault,pr,estop,door,va,manufacturer,fault_ka)


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




def excel_report(project, circuits, water, fan, elec, logic, ps_dfs, specs, elec_sel, wire_df, terminal_df, cable_df, xref_df, checks_df, refrig_ctrl_df, bom) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        rows=[]
        for section,obj in [("Project",project),("Water",water),("Fan",fan),("Electrical",elec),("Logic",logic)]:
            rows += [{"Section":section,"Input":k,"Value":v} for k,v in asdict(obj).items()]
        for c in circuits:
            rows += [{"Section":c.name,"Input":k,"Value":v} for k,v in asdict(c).items()]
        drawing_sheet_index().to_excel(writer, sheet_name="Sheet Index", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Inputs",index=False)
        if ps_dfs: pd.concat(ps_dfs,ignore_index=True).to_excel(writer, sheet_name="Pressure Settings",index=False)
        specs.to_excel(writer, sheet_name="Component Specs",index=False)
        elec_sel.to_excel(writer, sheet_name="Electrical Selection",index=False)
        refrig_ctrl_df.to_excel(writer, sheet_name="Refrig Controls", index=False)
        wire_df.to_excel(writer, sheet_name="Wire Schedule",index=False)
        terminal_df.to_excel(writer, sheet_name="Terminal Schedule",index=False)
        cable_df.to_excel(writer, sheet_name="Cable Schedule",index=False)
        xref_df.to_excel(writer, sheet_name="Contact Xref",index=False)
        checks_df.to_excel(writer, sheet_name="Electrical Checks",index=False)
        bom.to_excel(writer, sheet_name="BOM",index=False)
        pd.DataFrame([
            {"Note":"Preliminary app output only. Verify all component selections, wiring, pressure settings and code compliance before manufacturing."},
            {"Note":"Part numbers are candidate catalogue families. Confirm exact part number, coil voltage, auxiliary contacts, pressure range, reset type and local availability."},
            {"Note":"Cable sizes are preliminary. Final sizing must check installation method, ambient derating, grouping, voltage drop and short-circuit withstand."},
        ]).to_excel(writer, sheet_name="Notes",index=False)
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
    st.caption(f"Full preliminary Streamlit app: inputs → pressure switch settings → schematics → component specifications → BOM. App version: {APP_VERSION}")
    with st.sidebar:
        st.header("Settings")
        pressure_unit = st.radio("Pressure unit", ["bar(g)", "bar(abs)", "psig"], index=0)
        st.warning("Preliminary engineering tool only. Final design must be verified.")
    if PropsSI is None:
        st.error("CoolProp is not installed. Install requirements.txt before running pressure calculations.")

    tabs = st.tabs(["1 Project", "2 Compressor PDF", "3 Refrigerant", "4 Water + Fans", "5 Electrical", "6 Logic", "7 Outputs"])
    with tabs[0]:
        project = project_form()

    with tabs[1]:
        st.subheader("Compressor PDF upload and missing data check")
        parsed: Dict[str,Any] = {}
        uploaded = st.file_uploader("Upload compressor supplier PDF", type=["pdf"])
        if uploaded:
            text, warnings = extract_pdf_text(uploaded)
            for w in warnings:
                st.warning(w)
            if text:
                parsed = parse_compressor_pdf(text)
                st.success(f"Candidate fields found: {len(parsed)}. Please verify before using.")
                required = [
                    ("compressor_make","Make"),
                    ("compressor_model","Model"),
                    ("approved_refrigerants","Approved refrigerants"),
                    ("max_high_pressure_barg","Max high-side pressure"),
                    ("max_condensing_temp_c","Max condensing temp"),
                    ("min_evaporating_temp_c","Min evaporating temp"),
                    ("compressor_flc_a","FLC/RLA"),
                    ("compressor_lra_a","LRA"),
                ]
                st.dataframe(pd.DataFrame([{
                    "Data":lab,
                    "Status":"Found - verify" if k in parsed else "Missing",
                    "Value":parsed.get(k,"")
                } for k,lab in required]), width="stretch", hide_index=True)
                with st.expander("Extracted text"):
                    st.text_area("Text", text[:30000], height=250)

    with tabs[2]:
        circuits=[]
        if project.number_of_circuits == 1:
            circuits.append(circuit_form("c1","Circuit 1",parsed,project))
        else:
            t1,t2=st.tabs(["Circuit 1","Circuit 2"])
            with t1:
                circuits.append(circuit_form("c1","Circuit 1",parsed,project))
            with t2:
                circuits.append(circuit_form("c2","Circuit 2",parsed,project))

    total_cap = sum(c.cooling_capacity_kw for c in circuits)
    if project.configuration.startswith("Tandem") and circuits:
        total_cap = circuits[0].cooling_capacity_kw

    with tabs[3]:
        st.subheader("Chilled water inputs")
        water = water_form(total_cap)
        st.markdown("---")
        st.subheader("Condenser fan inputs")
        fan = fan_form()

    with tabs[4]:
        st.subheader("Electrical inputs")
        elec = electrical_form()

    with tabs[5]:
        st.subheader("Control logic inputs")
        logic = logic_form(project)

    with tabs[6]:
        st.header("Outputs")
        ps_dfs=[]
        for c in circuits:
            df,warns,vals = pressure_settings(c, pressure_unit)
            st.subheader(f"{c.name} pressure switches")
            for w in warns:
                st.warning(w)
            if vals:
                m1,m2,m3=st.columns(3)
                m1.metric("Normal suction",ptxt(vals["normal_suction"],pressure_unit))
                m2.metric("Normal condensing",ptxt(vals["normal_condensing"],pressure_unit))
                m3.metric("Subcooled liquid reference",ptxt(vals["subcooled_liquid"],pressure_unit))
            if not df.empty:
                ps_dfs.append(df)
                st.dataframe(df,width="stretch",hide_index=True)

        st.markdown("---")
        st.subheader("Diagrams")
        esvg = electrical_svg(project,circuits,water,fan,elec,logic)
        rsvg = refrigerant_svg(project,circuits,fan)
        wsvg = water_svg(project,water)
        dt1,dt2,dt3 = st.tabs(["Electrical", "Freon / refrigerant", "Chilled water"])
        with dt1:
            show_svg(esvg,720)
            st.markdown(svg_link(esvg,"electrical_circuit.svg","Download electrical SVG"), unsafe_allow_html=True)
        with dt2:
            show_svg(rsvg,620)
            st.markdown(svg_link(rsvg,"freon_circuit.svg","Download Freon SVG"), unsafe_allow_html=True)
        with dt3:
            show_svg(wsvg,600)
            st.markdown(svg_link(wsvg,"chilled_water_circuit.svg","Download chilled-water SVG"), unsafe_allow_html=True)

        st.markdown("---")
        specs = component_specs(project,circuits,water,fan,elec,logic)
        elec_sel = electrical_selection(project,circuits,water,fan,elec)
        refrig_ctrl_df = refrigeration_controls_selection(circuits, pressure_unit)
        wire_df = wire_schedule(project,circuits,water,fan,elec,logic)
        terminal_df = terminal_schedule(project,circuits,water,fan,elec,logic)
        cable_df = cable_schedule(project,circuits,water,fan,elec)
        xref_df = contact_cross_reference(project,circuits,water,fan,elec,logic)
        checks_df = electrical_standard_checks(project,circuits,water,fan,elec,logic)
        bom = bom_from(specs,elec_sel)

        st.subheader("Drawing sheet index")
        st.dataframe(drawing_sheet_index(),width="stretch",hide_index=True)

        st.subheader("Component specifications")
        st.dataframe(specs,width="stretch",hide_index=True)

        st.subheader("Electrical selections with candidate part numbers and cable recommendations")
        st.dataframe(elec_sel,width="stretch",hide_index=True)

        st.subheader("Refrigeration controls with candidate part numbers")
        st.dataframe(refrig_ctrl_df,width="stretch",hide_index=True)

        st.subheader("Wire schedule")
        st.dataframe(wire_df,width="stretch",hide_index=True)

        st.subheader("Terminal schedule")
        st.dataframe(terminal_df,width="stretch",hide_index=True)

        st.subheader("Cable schedule")
        st.dataframe(cable_df,width="stretch",hide_index=True)

        st.subheader("Contact cross-reference")
        st.dataframe(xref_df,width="stretch",hide_index=True)

        st.subheader("Electrical standard checks")
        st.dataframe(checks_df,width="stretch",hide_index=True)

        st.subheader("Bill of material")
        st.dataframe(bom,width="stretch",hide_index=True)

        st.subheader("Sequence logic")
        if project.configuration.startswith("Tandem"):
            st.code(
                f"""TANDEM COMMON CIRCUIT
Control ON → pump → flow proven → TC1 stage 1 → common YV1 opens → LPS closes → lead compressor starts.
If load remains high by {logic.stage2_on_offset_k:.1f} K after {logic.lag_start_delay_s}s, lag compressor starts.
On unloading, lag compressor stops first. Lead stops by pump-down when TC1 is satisfied.
Common HPS/LPS/FS/FRZ/PR trips stop both compressors. Individual overload trips stop the affected compressor.""",
                language="text"
            )
        else:
            st.code(
                f"""START
Control ON → K0 ON → pump starts → wait {logic.pump_start_delay_s}s → flow proves → TC1 calls cooling → YV1 opens → suction rises → LPS closes → compressor starts.

NORMAL STOP
TC1 satisfied → YV1 closes → compressor pumps down → LPS opens → compressor stops → anti-short-cycle timer {logic.anti_short_cycle_s}s starts → pump off delay {logic.pump_off_delay_s}s.

SAFETY STOP
HPS / flow fail / freeze / overload / phase fault opens → compressor stops immediately.""",
                language="text"
            )

        xlsx = excel_report(project,circuits,water,fan,elec,logic,ps_dfs,specs,elec_sel,wire_df,terminal_df,cable_df,xref_df,checks_df,refrig_ctrl_df,bom)
        try:
            pdf_report = make_pdf_report(project,circuits,water,fan,elec,logic,specs,elec_sel,wire_df,terminal_df,cable_df,xref_df,checks_df,refrig_ctrl_df,bom)
        except Exception as exc:
            pdf_report = b""
            st.warning(f"PDF report could not be generated. Install reportlab if needed. Details: {exc}")

        st.download_button("Download Excel report", xlsx, "chiller_design_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if pdf_report:
            st.download_button("Download multi-sheet PDF report", pdf_report, "chiller_manufacturing_report.pdf", "application/pdf")
        st.download_button("Download electrical SVG", esvg, "electrical_circuit.svg", "image/svg+xml")
        st.download_button("Download Freon SVG", rsvg, "freon_circuit.svg", "image/svg+xml")
        st.download_button("Download chilled-water SVG", wsvg, "chilled_water_circuit.svg", "image/svg+xml")
        st.download_button("Download electrical DXF", svg_to_basic_dxf(esvg, "Electrical circuit"), "electrical_circuit.dxf", "application/dxf")
        st.download_button("Download Freon DXF", svg_to_basic_dxf(rsvg, "Freon circuit"), "freon_circuit.dxf", "application/dxf")
        st.download_button("Download chilled-water DXF", svg_to_basic_dxf(wsvg, "Chilled water circuit"), "chilled_water_circuit.dxf", "application/dxf")
        if pdf_report:
            package_zip = make_manufacturing_zip(esvg,rsvg,wsvg,xlsx,pdf_report,specs,elec_sel,wire_df,terminal_df,cable_df,xref_df,checks_df,refrig_ctrl_df,bom)
            st.download_button("Download complete manufacturing package ZIP", package_zip, "chiller_manufacturing_package.zip", "application/zip")

        st.warning("Part numbers and cable sizes are preliminary candidate selections only. Final manufacturing drawings must be checked and approved by a qualified electrical engineer.")

if __name__ == "__main__":
    main()
