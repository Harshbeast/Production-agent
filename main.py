# ============================================
# main.py - Industrial APS Scheduler (Dashboard UI)
# Zen70 = zen70_pv + zen70_py + zen70_pz
# ============================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

# Load .env before anything else so GEMINI_API_KEY is available.
load_dotenv()

# Resolve Gemini key once — used by the chat popup throughout the file.
gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

from creater import create_schedule
from agent import SchedulerAssistant, execute_write_csv, execute_set_constant

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Industrial APS Scheduler",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Industrial APS Scheduler")


# ============================================
# FILE PATHS
# ============================================
material_file = "input/material_constraints.csv"
targets_file = "input/targets.csv"
actual_file = "input/actual_production.csv"
config_file_input = "input/configurations.csv"
mad_file = "input/special_orders.csv"
tracking_file = "Zenition Tracking(Sheet1).csv"

previous_plan_file = "output/or_tools_schedule.csv"
daily_consumption_file = "output/daily_material_consumption.csv"
config_schedule_file = "output/config_schedule.csv"
mad_report_file = "output/mad_compliance_report.csv"
remaining_material_file = "output/remaining_material.csv"
complete_targets_file = "output/complete_targets.csv"
material_avl_on_date_file = "output/material_availability_on_date.csv"

ZEN70_TYPES = ["zen70_pv", "zen70_py", "zen70_pz"]

# ============================================
# HELPERS
# ============================================
def file_empty(path):
    if not os.path.exists(path):
        return True
    try:
        return pd.read_csv(path).empty
    except Exception:
        return True


def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    date_str = str(date_str).strip()
    for fmt in ["%d-%m-%y", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def get_business_period(start_date_obj):
    """Keep your current simple window; adjust if you restore 28→27 logic."""
    d = start_date_obj
    period_start = start_date_obj - timedelta(days=1)
    period_end = d.replace(day=27)
    return period_start, period_end


def normalize_product(name, products):
    """Map System_Type to targets product key (zen70 family stays zen70)."""
    system_type = str(name).strip().lower().replace(" ", "")
    if system_type in products:
        return system_type
    # subtypes → family for target reduction
    if system_type in ("zen70pv", "zen70_pv", "zen70py", "zen70_py", "zen70pz", "zen70_pz"):
        if "zen70" in products:
            return "zen70"
        return system_type.replace("zen70", "zen70_") if "_" not in system_type else system_type
    if system_type == "zen70" and "zen70" in products:
        return "zen70"
    for p in products:
        if p.replace("zen", "") == system_type.replace("zen", "").replace("_", ""):
            return p
    return None


def build_special_orders_from_tracking(tracking_path, planning_start_date):
    if not os.path.exists(tracking_path):
        return None, None, None, None

    df = pd.read_csv(tracking_path)
    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in ["production_order", "productionorder"]:
            col_map[col] = "Production_order"
        elif key == "country":
            col_map[col] = "Country"
        elif key in ["system_type", "systemtype"]:
            col_map[col] = "System_Type"
        elif key in ["expected_mad", "expectedmad"]:
            col_map[col] = "Expected_MAD"
        elif key == "assembly":
            col_map[col] = "Assembly"
        elif key in ["assembly_date", "assemblydate"]:
            col_map[col] = "Assembly_date"
    df = df.rename(columns=col_map)

    needed = ["Production_order", "Country", "System_Type", "Expected_MAD", "Assembly", "Assembly_date"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Tracking file missing columns: {missing}")
        return None, None, None, None

    data = df[needed].copy()
    data["Expected_MAD"] = pd.to_datetime(
        data["Expected_MAD"].astype(str).str.replace("-", "/", regex=False),
        errors="coerce"
    )
    data["Assembly_date"] = pd.to_datetime(
        data["Assembly_date"].astype(str).str.replace("-", "/", regex=False),
        errors="coerce"
    )

    period_start, period_end = get_business_period(planning_start_date)
    cutoff_date = pd.Timestamp(period_start)

    df_filtered = data[
        data["Assembly_date"].isna() |
        (data["Assembly_date"] >= cutoff_date)
    ].copy()

    df_filtered["Expected_MAD"] = df_filtered["Expected_MAD"].dt.strftime("%d-%m-%Y")
    df_filtered["Assembly_date"] = df_filtered["Assembly_date"].dt.strftime("%d-%m-%Y")

    os.makedirs("input", exist_ok=True)
    df_filtered.to_csv(mad_file, index=False)
    return df_filtered, cutoff_date, period_start, period_end


ZEN70_TYPES = ["zen70_pv", "zen70_py", "zen70_pz"]

def zen70_subtype_from_order(order_id: str):
    s = str(order_id).upper()
    if "PV" in s:
        return "zen70_pv"
    if "PY" in s:
        return "zen70_py"
    if "PZ" in s:
        return "zen70_pz"
    return None


def prepare_planning_inputs(targets_df, material_df, config_df, mad_df, start_date_str):
    """
    Completed → reduce targets / config targets / stock.
    Zen70 completed reduces family target zen70.
    BOM columns may be zen70_pv / zen70_py / zen70_pz (not zen70).
    """
    products = [str(p).lower() for p in targets_df["Product"].tolist()]
    start_dt = parse_date(start_date_str)
    if start_dt is None:
        start_dt = datetime.strptime(start_date_str, "%d-%m-%y")

    configs = defaultdict(dict)
    if config_df is not None and not config_df.empty:
        for _, row in config_df.iterrows():
            p = str(row["Product"]).strip().lower()
            cfg = str(row["Config_Name"]).strip()
            mat = str(row["Material"]).strip() if pd.notna(row.get("Material")) else None
            qty = int(row["Qty_Per_System"]) if pd.notna(row.get("Qty_Per_System")) else 0
            target = int(row["Target_Qty"]) if pd.notna(row.get("Target_Qty")) else None
            if cfg not in configs[p]:
                configs[p][cfg] = {"materials": {}, "target": 0}
            if target is not None:
                configs[p][cfg]["target"] = target
            if mat and mat.lower() not in ["", "nan", "none"]:
                configs[p][cfg]["materials"][mat] = qty

    # Material usage: prefer subtype columns; for zen70 family use max of subtypes if no zen70 col
    material_usage = {}
    for _, row in material_df.iterrows():
        mat = row["Material"]
        material_usage[mat] = {}
        for p in products:
            if p == "zen70":
                usages = []
                for tp in ZEN70_TYPES:
                    if tp in material_df.columns:
                        usages.append(int(row[tp]))
                material_usage[mat][p] = max(usages) if usages else 0
            elif p in material_df.columns:
                material_usage[mat][p] = int(row[p])
            else:
                material_usage[mat][p] = 0

    completed_by_product = defaultdict(int)
    completed_by_config = defaultdict(int)
    open_rows = []

    if mad_df is not None and not mad_df.empty:
        for _, row in mad_df.iterrows():
            assembly = str(row.get("Assembly", "")).strip().lower()
            country = str(row.get("Country", "")).strip()
            order_id = str(row.get("Production_order", row.get("Production_Order", ""))).strip()

            system_type = normalize_product(row.get("System_Type", ""), products)
            if system_type is None:
                continue

            # PV / PY / PZ from Production_order
            subtype = zen70_subtype_from_order(order_id)

            # For materials / config: prefer subtype when Zen70
            mat_product = system_type
            if system_type == "zen70" and subtype is not None:
                mat_product = subtype          # zen70_pv / zen70_py / zen70_pz
            elif system_type in ("zen70_pv", "zen70_py", "zen70_pz"):
                mat_product = system_type
                subtype = system_type

            config_name = None
            if mat_product in configs and country in configs[mat_product]:
                config_name = country
            elif system_type in configs and country in configs[system_type]:
                config_name = country

            if assembly == "completed":
                # Target: always count against family zen70 when it is a Zen70
                if system_type == "zen70" or system_type in ("zen70_pv", "zen70_py", "zen70_pz"):
                    completed_by_product["zen70"] += 1
                else:
                    completed_by_product[system_type] += 1

                # Materials / config tracking: by subtype when known
                completed_by_config[(mat_product, config_name)] += 1
            else:
                open_rows.append(row)

    mad_df_open = pd.DataFrame(open_rows) if open_rows else pd.DataFrame(
        columns=list(mad_df.columns) if mad_df is not None and not mad_df.empty else []
    )

    # if mad_df is not None and not mad_df.empty:
    #     for _, row in mad_df.iterrows():
    #         assembly = str(row.get("Assembly", "")).strip().lower()
    #         country = str(row.get("Country", "")).strip()
    #         system_type = normalize_product(row.get("System_Type", ""), products)
    #         order_id = str(row.get("Production_order", row.get("Production_Order", ""))).strip()
    #         subtype = zen70_subtype_from_order(order_id)  # need order_id from row
    #         if system_type is None:
    #             continue

    #         config_name = None
    #         if system_type in configs and country in configs[system_type]:
    #             config_name = country

    #         if assembly == "completed":
    #             completed_by_product[system_type] += 1
    #             completed_by_config[(system_type, config_name)] += 1
    #         else:
    #             open_rows.append(row)

    # mad_df_open = pd.DataFrame(open_rows) if open_rows else pd.DataFrame(
    #     columns=list(mad_df.columns) if mad_df is not None and not mad_df.empty else []
    # )

    new_targets = []
    for _, row in targets_df.iterrows():
        p = str(row["Product"]).lower()
        original = int(row["Target_Qty"])
        done = completed_by_product.get(p, 0)
        new_targets.append({
            "Product": p,
            "Target_Qty": max(0, original - done),
            "Priority": int(row["Priority"]) if "Priority" in row else 10,
            "Max_Per_Day": int(row["Max_Per_Day"]) if "Max_Per_Day" in row and pd.notna(row.get("Max_Per_Day")) else None,
        })
    targets_rem = pd.DataFrame(new_targets)

    config_rem = config_df.copy() if config_df is not None else pd.DataFrame()
    if not config_rem.empty:
        rows = []
        seen = set()
        target_map = {}
        for _, row in config_rem.iterrows():
            p = str(row["Product"]).strip().lower()
            cfg = str(row["Config_Name"]).strip()
            mat = row.get("Material", "")
            qty = row.get("Qty_Per_System", 0)
            original_target = int(row["Target_Qty"]) if pd.notna(row.get("Target_Qty")) else 0
            key = (p, cfg)
            if key not in seen:
                done = completed_by_config.get((p, cfg), 0)
                target_map[key] = max(0, original_target - done)
                seen.add(key)
            rows.append({
                "Product": p,
                "Config_Name": cfg,
                "Material": mat if pd.notna(mat) else "",
                "Qty_Per_System": qty if pd.notna(qty) else 0,
                "Target_Qty": target_map.get(key, 0)
            })
        config_rem = pd.DataFrame(rows)

    stock = {row["Material"]: int(row["Available_Qty"]) for _, row in material_df.iterrows()}

    for _, row in material_df.iterrows():
        mat = str(row["Material"]).strip()
        qty_str = str(row.get("Incoming_Qty", "")).strip()
        date_str = str(row.get("Date", "")).strip()
        if qty_str in ["", "0", "nan", "NaN", "0.0"] or date_str in ["", "nan", "NaN"]:
            continue
        date_list = [x.strip() for x in date_str.split(",") if x.strip()]
        qty_list = [x.strip() for x in qty_str.split(",") if x.strip()]
        for d_text, q_text in zip(date_list, qty_list):
            dt = parse_date(d_text)
            try:
                qty = int(float(q_text))
            except Exception:
                continue
            if dt is None or qty <= 0:
                continue
            if dt <= start_dt:
                stock[mat] = stock.get(mat, 0) + qty

    for (p, cfg), cnt in completed_by_config.items():
        if cnt <= 0:
            continue
        if cfg is not None and p in configs and cfg in configs[p]:
            for mat, qty_per in configs[p][cfg]["materials"].items():
                stock[mat] = max(0, stock.get(mat, 0) - cnt * qty_per)
        else:
            for mat, usage_dict in material_usage.items():
                usage = usage_dict.get(p, 0)
                if usage > 0:
                    stock[mat] = max(0, stock.get(mat, 0) - cnt * usage)

    future_incoming = defaultdict(list)
    for _, row in material_df.iterrows():
        mat = str(row["Material"]).strip()
        qty_str = str(row.get("Incoming_Qty", "")).strip()
        date_str = str(row.get("Date", "")).strip()
        if qty_str in ["", "0", "nan", "NaN", "0.0"] or date_str in ["", "nan", "NaN"]:
            continue
        date_list = [x.strip() for x in date_str.split(",") if x.strip()]
        qty_list = [x.strip() for x in qty_str.split(",") if x.strip()]
        for d_text, q_text in zip(date_list, qty_list):
            dt = parse_date(d_text)
            try:
                qty = int(float(q_text))
            except Exception:
                continue
            if dt is None or qty <= 0:
                continue
            if dt > start_dt:
                future_incoming[mat].append((dt.strftime("%d-%m-%y"), qty))

    mat_rows = []
    for _, row in material_df.iterrows():
        mat = row["Material"]
        new_row = row.copy()
        new_row["Available_Qty"] = int(stock.get(mat, 0))
        fut = future_incoming.get(mat, [])
        if fut:
            new_row["Incoming_Qty"] = ",".join(str(q) for _, q in fut)
            new_row["Date"] = ",".join(d for d, _ in fut)
        else:
            new_row["Incoming_Qty"] = ""
            new_row["Date"] = ""
        mat_rows.append(new_row)
    material_df_rem = pd.DataFrame(mat_rows)

    summary = {
        "completed_by_product": dict(completed_by_product),
        "completed_by_config": {
            f"{p}|{c if c else 'BASE'}": n for (p, c), n in completed_by_config.items()
        },
        "open_orders": len(mad_df_open),
    }
    return targets_rem, material_df_rem, config_rem, mad_df_open, summary


# ============================================
# DATE + TRACKING
# ============================================
today_dt = datetime.today()
today_str = today_dt.strftime("%d-%m-%y")
today_date = today_dt.date()

def _biz_period_bounds(d):
    """28→27 business month bounds as date objects."""
    if d.day >= 28:
        p_start = d.replace(day=28)
        next_month = d.month % 12 + 1
        next_year = d.year + (1 if d.month == 12 else 0)
        p_end = d.replace(year=next_year, month=next_month, day=27)
    else:
        p_end = d.replace(day=27)
        prev_month = d.month - 1 if d.month > 1 else 12
        prev_year = d.year - 1 if d.month == 1 else d.year
        p_start = d.replace(year=prev_year, month=prev_month, day=28)
    return p_start, p_end

biz_start, biz_end = _biz_period_bounds(today_date)

c_from, c_to = st.columns(2)
with c_from:
    plan_start = st.date_input(
        "Plan from",
        value=max(today_date, biz_start),
        key="plan_start",
    )
with c_to:
    plan_end = st.date_input(
        "Plan to",
        value=max(biz_end, plan_start),
        min_value=plan_start,
        key="plan_end",
    )

if plan_start > plan_end:
    st.error("'Plan from' must not be after 'Plan to'.")
    st.stop()

start_date_str = plan_start.strftime("%d-%m-%y")
end_date_str = plan_end.strftime("%d-%m-%y")
planning_start_dt = datetime.combine(plan_start, datetime.min.time())
period_start, period_end = get_business_period(planning_start_dt)

# ---- Calendar working-day selector ----
st.markdown("#### Working days")
st.markdown(
    """<style>
    /* Selected calendar day — light green.
       Direct selector; works regardless of Streamlit version or DOM nesting. */
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background-color: #86efac !important;
        border-color:     #86efac !important;
        color:            #14532d !important;
    }
    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background-color: #4ade80 !important;
        border-color:     #4ade80 !important;
        color:            #14532d !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

_DAY_ABBR  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_cal_key   = f"cal_{plan_start}_{plan_end}"
_cal_start = plan_start - timedelta(days=plan_start.weekday())
_cal_end   = plan_end   + timedelta(days=6 - plan_end.weekday())

# Initialise session state when the date range is new or first load.
# Key encodes the range so changing plan_start/plan_end resets to weekday defaults.
if _cal_key not in st.session_state:
    _init = set()
    _d = plan_start
    while _d <= plan_end:
        if _d.weekday() < 5:
            _init.add(_d)
        _d += timedelta(days=1)
    st.session_state[_cal_key] = _init

_selected = st.session_state[_cal_key]

# Day-of-week header row
_hcols = st.columns(7)
for _i, _name in enumerate(_DAY_ABBR):
    _color = "#999" if _i >= 5 else "#444"
    _hcols[_i].markdown(
        f"<div style='text-align:center;font-weight:700;color:{_color};"
        f"font-size:13px;padding:4px 0'>{_name}</div>",
        unsafe_allow_html=True,
    )

# Week rows — every cell is a full-width button so all 7 columns stay aligned.
# Primary (coloured) = selected working day  |  Secondary = not selected  |  Disabled = outside range
_cur = _cal_start
_last_shown_month = None
while _cur <= _cal_end:
    _week = [_cur + timedelta(days=_i) for _i in range(7)]

    # Month label when month changes within the plan range
    _first_in = next((d for d in _week if plan_start <= d <= plan_end), None)
    if _first_in and _first_in.month != _last_shown_month:
        _last_shown_month = _first_in.month
        st.markdown(
            f"<div style='font-size:11px;color:#aaa;margin:6px 0 2px 0'>"
            f"{_first_in.strftime('%B %Y')}</div>",
            unsafe_allow_html=True,
        )

    _wcols = st.columns(7)
    for _i, _day in enumerate(_week):
        _in_range = plan_start <= _day <= plan_end
        _btn_key  = f"{_cal_key}_{_day.isoformat()}"
        _is_sel   = _day in _selected
        if _in_range:
            if _wcols[_i].button(
                str(_day.day),
                key=_btn_key,
                use_container_width=True,
                type="primary" if _is_sel else "secondary",
            ):
                if _is_sel:
                    _selected.discard(_day)
                else:
                    _selected.add(_day)
                st.rerun()
        else:
            _wcols[_i].button(
                str(_day.day),
                key=_btn_key,
                use_container_width=True,
                disabled=True,
            )

    _cur += timedelta(days=7)

working_dates = sorted(
    [d.strftime("%d-%m-%y") for d in _selected if plan_start <= d <= plan_end],
    key=lambda s: datetime.strptime(s, "%d-%m-%y"),
)

if not working_dates:
    st.error("No working days selected — pick at least one day.")
    st.stop()

st.caption(
    f"Ref. period **{biz_start.strftime('%d-%m-%Y')} → {biz_end.strftime('%d-%m-%Y')}** · "
    f"Planning window **{plan_start.strftime('%d-%m-%Y')} → {plan_end.strftime('%d-%m-%Y')}** · "
    f"**{len(working_dates)}** working days · Today **{today_str}**"
)

if os.path.exists(tracking_file):
    built = build_special_orders_from_tracking(tracking_file, planning_start_dt)
    if built[0] is not None:
        mad_df, cutoff_date, _, _ = built
        st.caption(f"Assembly cutoff: **{cutoff_date.strftime('%d-%m-%Y')}** (from tracking sheet)")
    else:
        mad_df = pd.read_csv(mad_file) if os.path.exists(mad_file) else pd.DataFrame()
else:
    mad_df = pd.read_csv(mad_file) if os.path.exists(mad_file) else pd.DataFrame(
        columns=["Production_order", "Country", "System_Type", "Expected_MAD", "Assembly", "Assembly_date"]
    )

# ============================================
# LOAD MASTER DATA
# ============================================
original_material_df = pd.read_csv(material_file)
if "Included" not in original_material_df.columns:
    original_material_df["Included"] = True

material_df = original_material_df.copy()
material_df["Included"] = material_df["Included"].map({
    True: True, "Yes": True, 1: True, "true": True,
    False: False, "No": False, 0: False, "false": False
}).fillna(True)
material_df = material_df[material_df["Included"] == True].copy()
material_df = material_df.drop(columns=["Included"], errors="ignore")

targets_df = pd.read_csv(targets_file)

if os.path.exists(config_file_input):
    config_df = pd.read_csv(config_file_input)
else:
    config_df = pd.DataFrame(
        columns=["Product", "Config_Name", "Material", "Qty_Per_System", "Target_Qty"]
    )

if not mad_df.empty:
    if "Assembly" not in mad_df.columns:
        mad_df["Assembly"] = ""
    assembly_norm = mad_df["Assembly"].astype(str).str.strip().str.lower()
    completed_df = mad_df[assembly_norm == "completed"].copy()
    open_df = mad_df[assembly_norm != "completed"].copy()
else:
    completed_df = pd.DataFrame()
    open_df = pd.DataFrame()

# ============================================
# LOAD PLAN + KPI
# ============================================
plan_df = pd.read_csv(previous_plan_file) if not file_empty(previous_plan_file) else pd.DataFrame()

total_planned = 0
if not plan_df.empty:
    if "Total" in plan_df.columns:
        total_planned = int(plan_df["Total"].sum())
    else:
        skip = {"Date", "zen70_total"}
        prod_cols = [c for c in plan_df.columns if c not in skip]
        total_planned = int(plan_df[prod_cols].sum().sum())

on_time, missed = 0, 0
if not file_empty(mad_report_file):
    _mr = pd.read_csv(mad_report_file)
    if not _mr.empty and "Status" in _mr.columns:
        on_time = int((_mr["Status"] == "On Time").sum())
        missed = int((_mr["Status"] == "Missed").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Planned systems", total_planned)
k2.metric("Open orders", len(open_df))
k3.metric("MAD On Time", on_time)
k4.metric("MAD Missed", missed)

# ============================================
# CONTROLS
# ============================================
with st.expander("Advanced solver settings", expanded=False):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        solve_time = st.slider("Solver time limit (s)", 30, 300, 90, step=15, key="solve_time")
    with adv_col2:
        if "solver_status" in st.session_state:
            s = st.session_state["solver_status"]
            if s == "OPTIMAL":
                st.success(f"Last solve: {s}")
            else:
                st.warning(f"Last solve: {s}")
        else:
            st.info("No plan run yet.")

st.markdown("### Planning")
c1, c2 = st.columns(2)
with c1:
    if st.button("Create Initial Plan", type="secondary", use_container_width=True):
        with st.spinner("Preparing inputs and solving..."):
            targets_rem, material_rem, config_rem, mad_open, summary = prepare_planning_inputs(
                targets_df, material_df, config_df, mad_df, start_date_str
            )
            st.session_state["prep_summary"] = summary

            result = create_schedule(
                material_df=material_rem,
                targets_df=targets_rem,
                start_date=start_date_str,
                working_dates=working_dates,
                end_date=end_date_str,
                config_df=config_rem if not config_rem.empty else None,
                mad_df=mad_open if not mad_open.empty else None,
                solve_time=solve_time,
            )

            if result[0] is None:
                st.error("Plan failed. Check constraints / materials / MAD / BOM columns (zen70_pv/py/pz).")
            else:
                (
                    schedule_df,
                    mat_status,
                    target_status,
                    daily_cons,
                    config_sched,
                    mad_report,
                    material_availability_df,
                    solver_status_str,
                ) = result

                os.makedirs("output", exist_ok=True)
                schedule_df.to_csv(previous_plan_file, index=False)
                mat_status.to_csv(remaining_material_file, index=False)
                target_status.to_csv(complete_targets_file, index=False)
                daily_cons.to_csv(daily_consumption_file, index=False)
                config_sched.to_csv(config_schedule_file, index=False)
                mad_report.to_csv(mad_report_file, index=False)
                material_availability_df.to_csv(material_avl_on_date_file, index=False)
                st.session_state["solver_status"] = solver_status_str
                st.success(f"Plan created ({solver_status_str}).")
                st.rerun()

with c2:
    if st.button("Clear plan", use_container_width=True):
        for f in [
            previous_plan_file, daily_consumption_file, config_schedule_file,
            mad_report_file, remaining_material_file, complete_targets_file,
            material_avl_on_date_file,
        ]:
            if os.path.exists(f):
                pd.DataFrame().to_csv(f, index=False)
        st.success("Cleared.")
        st.rerun()

if "prep_summary" in st.session_state:
    with st.expander("Preprocessing summary"):
        st.json(st.session_state["prep_summary"])

if plan_df.empty:
    st.warning("No plan yet. Set date and click **Create Initial Plan**.")

# ============================================
# COMBINED VIEW
# ============================================
st.markdown("### Production · Material · Configuration")

if not plan_df.empty and os.path.exists(daily_consumption_file):
    prod_view = plan_df.set_index("Date").T
    prod_view = prod_view.drop(index=["Total"], errors="ignore")
    # optional: keep zen70_total as info row
    numeric_idx = [i for i in prod_view.index if i != "zen70_total"]
    prod_view.loc[numeric_idx, "TOTAL"] = prod_view.loc[numeric_idx].sum(axis=1)

    daily_df = pd.read_csv(daily_consumption_file)
    mat_view = daily_df.set_index("Date").T
    mat_view["TOTAL"] = mat_view.sum(axis=1)

    if not file_empty(config_schedule_file):
        cfg_df = pd.read_csv(config_schedule_file)
        config_view = cfg_df.set_index("Config") if "Config" in cfg_df.columns else pd.DataFrame()
    else:
        config_view = pd.DataFrame()

    all_cols = list(dict.fromkeys(
        list(prod_view.columns) + list(mat_view.columns) + list(config_view.columns)
    ))
    prod_view = prod_view.reindex(columns=all_cols, fill_value=0)
    mat_view = mat_view.reindex(columns=all_cols, fill_value=0)
    parts = [prod_view, mat_view]
    if not config_view.empty:
        config_view = config_view.reindex(columns=all_cols, fill_value=0)
        parts.append(config_view)

    st.dataframe(pd.concat(parts), use_container_width=True, height=520)
else:
    st.info("Combined view appears after planning.")

# ============================================
# MATERIAL AVAILABILITY CHART
# ============================================
st.markdown("### Material availability over time")
st.caption("Net remaining stock per material after cumulative production consumption.")

if not file_empty(material_avl_on_date_file):
    _avl_df = pd.read_csv(material_avl_on_date_file)
    _mat_cols = [c for c in _avl_df.columns if c != "Date"]

    _zero_mats = [m for m in _mat_cols if (_avl_df[m] <= 0).any()]
    if _zero_mats:
        st.warning(f"Materials hitting zero during plan: {', '.join(_zero_mats)}")

    _sel_mats = st.multiselect(
        "Materials to display",
        _mat_cols,
        default=_mat_cols[:min(6, len(_mat_cols))],
        key="avl_chart_main",
    )
    if _sel_mats:
        st.line_chart(_avl_df.set_index("Date")[_sel_mats], use_container_width=True)
else:
    st.info("Material availability chart appears after planning.")

# ============================================
# MAD COMPLIANCE
# ============================================
st.markdown("### MAD Compliance")

if not file_empty(mad_report_file):
    mad_report = pd.read_csv(mad_report_file)

    if mad_report.empty:
        st.info("No open MAD rows (all completed or none matched).")
    elif "Status" not in mad_report.columns:
        st.dataframe(mad_report, use_container_width=True)
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            status_filter = st.multiselect(
                "Status",
                ["On Time", "Missed"],
                default=["On Time", "Missed"],
                key="mad_status"
            )
        with f2:
            products = sorted(mad_report["System_Type"].dropna().astype(str).unique()) if "System_Type" in mad_report.columns else []
            product_filter = st.multiselect("Product", products, default=products, key="mad_prod")
        with f3:
            missed_only = st.toggle("Missed only", value=False, key="mad_missed_only")

        view = mad_report.copy()
        if status_filter:
            view = view[view["Status"].isin(status_filter)]
        if product_filter and "System_Type" in view.columns:
            view = view[view["System_Type"].isin(product_filter)]
        if missed_only:
            view = view[view["Status"] == "Missed"]

        def style_mad(row):
            if row.get("Status") == "Missed":
                return ["background-color: #ffe0e0"] * len(row)
            if row.get("Status") == "On Time":
                return ["background-color: #e8f8e8"] * len(row)
            return [""] * len(row)

        st.dataframe(view.style.apply(style_mad, axis=1), use_container_width=True, height=380)

        a, b = st.columns(2)
        a.success(f"On Time: {on_time}")
        if missed > 0:
            b.error(f"Missed: {missed}")
        else:
            b.success("Missed: 0")
else:
    st.info("MAD report appears after planning.")

with st.expander("Special orders (Completed / Open)"):
    t1, t2 = st.tabs(["Completed", "Open / On Hold"])
    with t1:
        if completed_df.empty:
            st.info("None")
        else:
            st.dataframe(completed_df, use_container_width=True, height=220)
    with t2:
        if open_df.empty:
            st.info("None")
        else:
            st.dataframe(open_df, use_container_width=True, height=220)

# ============================================
# TARGETS & MATERIALS
# ============================================
st.markdown("### Targets & materials")
c_a, c_b = st.columns(2)
with c_a:
    st.caption("Remaining targets (zen70 = family; subtypes shown if planned)")
    if not file_empty(complete_targets_file):
        st.dataframe(pd.read_csv(complete_targets_file), use_container_width=True)
    else:
        st.info("—")
with c_b:
    st.caption("Remaining material")
    if not file_empty(remaining_material_file):
        st.dataframe(pd.read_csv(remaining_material_file), use_container_width=True)
    else:
        st.info("—")

with st.expander("Net material availability by date"):
    if not file_empty(material_avl_on_date_file):
        st.dataframe(pd.read_csv(material_avl_on_date_file), use_container_width=True, height=280)
    else:
        st.info("Generated after planning.")

# ============================================
# MATERIAL CONSTRAINTS
# ============================================
st.markdown("### Material constraints")
st.caption("Include / exclude materials used by the planner")

display_df = original_material_df[["Material", "Available_Qty"]].copy()
display_df["Include"] = original_material_df["Included"].astype(bool)

edited_df = st.data_editor(
    display_df,
    column_config={
        "Include": st.column_config.CheckboxColumn("Include", default=True),
        "Material": st.column_config.TextColumn("Material", disabled=True),
        "Available_Qty": st.column_config.NumberColumn("Available", disabled=True),
    },
    use_container_width=True,
    height=280,
    key="material_editor"
)

if st.button("Save material inclusion", type="secondary"):
    original_material_df["Included"] = edited_df["Include"].values
    original_material_df.to_csv(material_file, index=False)
    st.success("Saved")
    st.rerun()

# ============================================
# MOVE CONFIG
# ============================================
with st.expander("Move configuration"):
    if not file_empty(config_schedule_file):
        config_df_view = pd.read_csv(config_schedule_file)
        if not config_df_view.empty and "Config" in config_df_view.columns:
            config_options = config_df_view["Config"].tolist()
            date_options = [c for c in config_df_view.columns if c not in ["Config", "TOTAL"]]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                selected_config = st.selectbox("Configuration", config_options, key="move_cfg")
            with m2:
                from_date = st.selectbox("From", date_options, key="move_from")
            with m3:
                to_date = st.selectbox("To", date_options, key="move_to")
            with m4:
                qty = st.number_input("Qty", min_value=1, max_value=5, value=1, key="move_qty")

            if st.button("Apply move", type="secondary"):
                current = config_df_view.set_index("Config")
                current_from = int(current.loc[selected_config, from_date])
                current_to = int(current.loc[selected_config, to_date])
                if current_from < qty:
                    st.error(f"Only {current_from} on {from_date}")
                else:
                    product, cfg_name = selected_config.split(" | ")
                    forced = {
                        (product, cfg_name, from_date): current_from - qty,
                        (product, cfg_name, to_date): current_to + qty,
                    }
                    targets_rem, material_rem, config_rem, mad_open, _ = prepare_planning_inputs(
                        targets_df, material_df, config_df, mad_df, start_date_str
                    )
                    result = create_schedule(
                        material_df=material_rem,
                        targets_df=targets_rem,
                        start_date=start_date_str,
                        working_dates=working_dates,
                        end_date=end_date_str,
                        config_df=config_rem if not config_rem.empty else None,
                        mad_df=mad_open if not mad_open.empty else None,
                        forced_configs=forced,
                        solve_time=solve_time,
                    )
                    if result[0] is None:
                        st.error("Move not possible.")
                    else:
                        (
                            schedule_df,
                            mat_status,
                            target_status,
                            daily_cons,
                            config_sched,
                            mad_report,
                            material_availability_df,
                            solver_status_str,
                        ) = result
                        schedule_df.to_csv(previous_plan_file, index=False)
                        mat_status.to_csv(remaining_material_file, index=False)
                        target_status.to_csv(complete_targets_file, index=False)
                        daily_cons.to_csv(daily_consumption_file, index=False)
                        config_sched.to_csv(config_schedule_file, index=False)
                        mad_report.to_csv(mad_report_file, index=False)
                        material_availability_df.to_csv(material_avl_on_date_file, index=False)
                        st.session_state["solver_status"] = solver_status_str
                        st.success(f"Moved {qty} × {selected_config}")
                        st.rerun()
        else:
            st.info("No configuration rows.")
    else:
        st.info("No configuration schedule yet.")

# ============================================
# DAY COMPLETION
# ============================================
with st.expander("Day completion / actuals"):
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Mark today completed", use_container_width=True) and not plan_df.empty:
            today_row = plan_df[plan_df["Date"] == today_str]
            if not today_row.empty:
                if file_empty(actual_file):
                    updated = today_row
                else:
                    updated = pd.concat([pd.read_csv(actual_file), today_row], ignore_index=True)
                updated.to_csv(actual_file, index=False)
                st.success("Actual updated")
                st.rerun()
            else:
                st.warning("No plan row for today.")

    with btn_col2:
        if st.button("Replan from today", use_container_width=True, type="secondary"):
            with st.spinner("Replanning from today..."):
                targets_rem, material_rem, config_rem, mad_open, summary = prepare_planning_inputs(
                    targets_df, material_df, config_df, mad_df, today_str
                )
                replan_dates = [
                    d for d in working_dates
                    if datetime.strptime(d, "%d-%m-%y").date() >= today_date
                ]
                result = create_schedule(
                    material_df=material_rem,
                    targets_df=targets_rem,
                    start_date=today_str,
                    working_dates=replan_dates,
                    end_date=end_date_str,
                    config_df=config_rem if not config_rem.empty else None,
                    mad_df=mad_open if not mad_open.empty else None,
                    solve_time=st.session_state.get("solve_time", 90),
                )
                if result[0] is None:
                    st.error("Replan failed. Check constraints / materials.")
                else:
                    (
                        schedule_df,
                        mat_status,
                        target_status,
                        daily_cons,
                        config_sched,
                        mad_report,
                        material_availability_df,
                        solver_status_str,
                    ) = result
                    os.makedirs("output", exist_ok=True)
                    schedule_df.to_csv(previous_plan_file, index=False)
                    mat_status.to_csv(remaining_material_file, index=False)
                    target_status.to_csv(complete_targets_file, index=False)
                    daily_cons.to_csv(daily_consumption_file, index=False)
                    config_sched.to_csv(config_schedule_file, index=False)
                    mad_report.to_csv(mad_report_file, index=False)
                    material_availability_df.to_csv(material_avl_on_date_file, index=False)
                    st.session_state["solver_status"] = solver_status_str
                    st.success(f"Replanned from today ({solver_status_str}).")
                    st.rerun()

    if not file_empty(actual_file):
        st.dataframe(pd.read_csv(actual_file), use_container_width=True)

# ============================================
# AI CHAT POPUP
# (sidebar styled as a floating popup via CSS;
#  no JS bridge — Streamlit chat_input handles
#  submission natively)
# ============================================


# ---- FAB + popup-CSS injection (runs on every rerun) ----
components.html("""
<script>
(function () {
  var p = window.parent.document;

  if (!p.getElementById('aip-style')) {
    var s = p.createElement('style');
    s.id = 'aip-style';
    s.textContent = [
      '#aip-fab{position:fixed;bottom:1.6rem;right:1.6rem;z-index:99999;',
      'width:52px;height:52px;border-radius:50%;background:#86efac;border:none;',
      'cursor:pointer;font-size:22px;display:flex;align-items:center;',
      'justify-content:center;box-shadow:0 4px 14px rgba(0,0,0,.3);',
      'transition:transform .15s,background .15s}',
      '#aip-fab:hover{transform:scale(1.12);background:#4ade80}',
      /* sidebar → floating popup */
      'section[data-testid="stSidebar"]{',
      'position:fixed!important;bottom:5.4rem!important;right:1.6rem!important;',
      'left:auto!important;top:auto!important;',
      'width:370px!important;height:560px!important;',
      'border-radius:16px!important;overflow:hidden!important;',
      'box-shadow:0 20px 60px rgba(0,0,0,.65)!important;',
      'z-index:99998!important;background:#0f172a!important;',
      'opacity:0!important;visibility:hidden!important;',
      'transform:translateY(12px)!important;',
      'transition:opacity .2s,visibility .2s,transform .2s!important}',
      'section[data-testid="stSidebar"].aip-open{',
      'opacity:1!important;visibility:visible!important;',
      'transform:translateY(0)!important}',
      '[data-testid="stSidebarResizeHandle"],[data-testid="stSidebarCollapsedControl"],',
      '[data-testid="collapsedControl"]{display:none!important}',
      'section[data-testid="stSidebar"]>div:first-child{padding-top:0!important}',

      /* ── Force dark theme inside popup regardless of Streamlit theme ── */

      /* All text */
      'section[data-testid="stSidebar"] p,',
      'section[data-testid="stSidebar"] span,',
      'section[data-testid="stSidebar"] li,',
      'section[data-testid="stSidebar"] label,',
      'section[data-testid="stSidebar"] a,',
      'section[data-testid="stSidebar"] code,',
      'section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]',
      '{color:#cbd5e1!important}',

      /* Chat message avatars & bubbles */
      'section[data-testid="stSidebar"] [data-testid="stChatMessage"]',
      '{background:transparent!important}',
      'section[data-testid="stSidebar"] [data-testid="stChatMessage"]>div>div',
      '{background:#1e293b!important;color:#cbd5e1!important;border-radius:10px!important}',

      /* Chat input box */
      'section[data-testid="stSidebar"] [data-testid="stChatInput"]',
      '{background:#1e293b!important;border-top:1px solid #334155!important}',
      'section[data-testid="stSidebar"] [data-testid="stChatInputTextArea"] textarea,',
      'section[data-testid="stSidebar"] [data-testid="stChatInput"] textarea',
      '{background:#1e293b!important;color:#e2e8f0!important;',
      'border:1px solid #334155!important;border-radius:10px!important}',
      'section[data-testid="stSidebar"] [data-testid="stChatInput"] textarea::placeholder',
      '{color:#64748b!important}',
      'section[data-testid="stSidebar"] [data-testid="stChatInputSubmitButton"] button',
      '{background:#1d4ed8!important;color:#fff!important}',

      /* Info / warning / error boxes */
      'section[data-testid="stSidebar"] [data-testid="stNotification"],',
      'section[data-testid="stSidebar"] .stAlert,',
      'section[data-testid="stSidebar"] [data-baseweb="notification"]',
      '{background:#1e293b!important;color:#cbd5e1!important;border-color:#334155!important}',

      /* Buttons (Confirm / Cancel) */
      'section[data-testid="stSidebar"] button[kind="secondary"],',
      'section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]',
      '{background:#1e293b!important;color:#e2e8f0!important;',
      'border:1px solid #334155!important}',
      'section[data-testid="stSidebar"] button[kind="secondary"]:hover',
      '{background:#334155!important}',

      /* Scrollbar */
      'section[data-testid="stSidebar"] *::-webkit-scrollbar{width:4px}',
      'section[data-testid="stSidebar"] *::-webkit-scrollbar-thumb',
      '{background:#334155;border-radius:2px}',
      'section[data-testid="stSidebar"] *::-webkit-scrollbar-track{background:transparent}',
    ].join('');
    p.head.appendChild(s);
  }

  if (!p.getElementById('aip-fab')) {
    var fab = p.createElement('button');
    fab.id = 'aip-fab';
    fab.title = 'AI Planning Assistant';
    fab.innerHTML = '&#x1F916;';
    p.body.appendChild(fab);
  }

  /* Re-attach every rerun so closure always has current scope */
  p.getElementById('aip-fab').onclick = function () {
    var sb = p.querySelector('section[data-testid="stSidebar"]');
    if (!sb) return;
    var open = sb.classList.toggle('aip-open');
    sessionStorage.setItem('aip_open', open ? '1' : '0');
  };

  /* Restore open state */
  if (sessionStorage.getItem('aip_open') === '1') {
    var sb = p.querySelector('section[data-testid="stSidebar"]');
    if (sb) sb.classList.add('aip-open');
  }
})();
</script>
""", height=0, scrolling=False)

# ---- Sidebar = native Streamlit chat (no JS bridge needed) ----
with st.sidebar:
    # Gradient header
    st.markdown(
        """<div style="background:linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 60%,#0891b2 100%);
                    padding:.9rem 1.1rem .8rem;margin:-1rem -1rem .5rem -1rem">
          <p style="font-size:10px;letter-spacing:2px;color:rgba(255,255,255,.7);
                    text-transform:uppercase;margin:0 0 3px">AI Planning Assistant</p>
          <h3 style="font-size:17px;font-weight:700;color:#fff;margin:0">
            Production Scheduler</h3>
        </div>""",
        unsafe_allow_html=True,
    )

    # State init
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending_operation" not in st.session_state:
        st.session_state.pending_operation = None

    # Assistant init
    if gemini_key:
        if (
            "chat_assistant" not in st.session_state
            or st.session_state.get("_gemini_key_used") != gemini_key
        ):
            try:
                st.session_state.chat_assistant   = SchedulerAssistant(gemini_key)
                st.session_state._gemini_key_used = gemini_key
            except Exception as _exc:
                st.error(f"AI assistant error: {_exc}")
    else:
        st.info("Set GEMINI_API_KEY in .env to enable chat.")

    # Pending confirmation
    _pending = st.session_state.get("pending_operation")
    if _pending:
        st.warning(
            f"**Proposed change**\n\n{_pending['description']}\n\n_Nothing written yet._"
        )
        _c1, _c2 = st.columns(2)
        if _c1.button("✓ Confirm", use_container_width=True, key="aip_confirm"):
            if _pending["type"] == "write_csv":
                _ok, _m = execute_write_csv(_pending["path"], _pending["updates"])
            else:
                _ok, _m = execute_set_constant(_pending["name"], _pending["value"])
            _r = (
                f"✅ Done — {_pending['description']}.\n\nClick **Create Initial Plan** to regenerate."
                if _ok else f"❌ Write failed: {_m}"
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": _r})
            st.session_state.pending_operation = None
            st.rerun()
        if _c2.button("✕ Cancel", use_container_width=True, key="aip_cancel"):
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": "Change cancelled — nothing written."}
            )
            st.session_state.pending_operation = None
            st.rerun()

    # Chat history
    for _m in st.session_state.get("chat_messages", [])[-40:]:
        with st.chat_message(_m["role"]):
            st.markdown(_m["content"])

    # Chat input (native — no JS bridge)
    _user_input = st.chat_input(
        "Ask about the schedule, materials, targets…",
        disabled=bool(st.session_state.get("pending_operation")),
        key="aip_chat_input",
    )
    if _user_input:
        st.session_state.chat_messages.append({"role": "user", "content": _user_input})
        if "chat_assistant" in st.session_state:
            with st.spinner("Thinking…"):
                try:
                    _reply, _new_pending = st.session_state.chat_assistant.send_message(
                        _user_input
                    )
                except Exception as _exc:
                    _reply, _new_pending = f"Error: {_exc}", None
            if _new_pending:
                st.session_state.pending_operation = _new_pending
            st.session_state.chat_messages.append({"role": "assistant", "content": _reply})
        else:
            st.session_state.chat_messages.append(
                {"role": "assistant",
                 "content": "⚠️ AI not ready. Check GEMINI_API_KEY in .env."}
            )
        st.rerun()

