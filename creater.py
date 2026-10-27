import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from collections import defaultdict

# =========================================================
# CONFIGURATION
# =========================================================
max_systems_per_day = 6
diversity_weight = 2
smoothness_weight = 1
max_products_per_day = 8
earliness_weight = 1000
mad_penalty_weight = 500

ZEN70_TYPES = ["zen70_pv", "zen70_py", "zen70_pz"]
ZEN70_FAMILY_DAILY_MAX = 3   # total Zen70 of any type per day

product_max = {
    "zen10": 1,
    "zen30": 1,
    "zen50": 2,
    "zen70_pv": 3,
    "zen70_py": 3,
    "zen70_pz": 3,
    "zen90": 3,
}


def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    date_str = str(date_str).strip()
    for fmt in ["%d-%m-%y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def normalize_system_type(system_type, products):
    system_type = str(system_type).strip().lower().replace(" ", "")
    if system_type in products:
        return system_type
    # Zen 70 family → logical key "zen70" for MAD/orders
    if system_type in ("zen70", "zen70pv", "zen70py", "zen70pz"):
        if system_type == "zen70pv":
            return "zen70_pv"
        if system_type == "zen70py":
            return "zen70_py"
        if system_type == "zen70pz":
            return "zen70_pz"
        return "zen70"
    for p in products:
        if p.replace("zen", "") == system_type.replace("zen", ""):
            return p
    return None


def zen70_subtype_from_order(order_id: str):
    s = str(order_id).upper()
    if "PV" in s:
        return "zen70_pv"
    if "PY" in s:
        return "zen70_py"
    if "PZ" in s:
        return "zen70_pz"
    return None


def create_schedule(material_df, targets_df, start_date, config_df=None, mad_df=None, forced_configs=None, end_date=None, solve_time=90):
    """
    Returns:
        schedule_df,
        material_status_df,
        target_status_df,
        daily_consumption_df,
        config_schedule_df,
        mad_report_df,
        material_availability_df
    """
    # Expand zen70 target product into three planning products
    raw_products = [str(p).lower() for p in targets_df["Product"].tolist()]
    products = []
    for p in raw_products:
        if p == "zen70":
            products.extend(ZEN70_TYPES)
        else:
            products.append(p)
    products = list(dict.fromkeys(products))

    # ---------- Date Handling (Business month: 28 → 27) ----------
    try:
        start_date_obj = datetime.strptime(start_date, "%d-%m-%Y")
    except ValueError:
        start_date_obj = datetime.strptime(start_date, "%d-%m-%y")

    d = start_date_obj
    if d.day >= 28:
        period_start = d.replace(day=28)
        if d.month == 12:
            period_end = d.replace(year=d.year + 1, month=1, day=27)
        else:
            period_end = d.replace(month=d.month + 1, day=27)
    else:
        period_end = d.replace(day=27)
        if d.month == 1:
            period_start = d.replace(year=d.year - 1, month=12, day=28)
        else:
            period_start = d.replace(month=d.month - 1, day=28)

    if end_date is not None:
        for fmt in ["%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d"]:
            try:
                custom_end = datetime.strptime(str(end_date).strip(), fmt)
                period_end = custom_end
                break
            except ValueError:
                continue

    effective_start = max(d, period_start)

    working_dates = []
    current = effective_start
    while current <= period_end:
        if current.weekday() < 5:
            working_dates.append(current.strftime("%d-%m-%y"))
        current += timedelta(days=1)

    if not working_dates:
        print("Warning: No working days in the selected range")
        return None, None, None, None, None, None, None, None

    planning_start = period_start
    planning_end = period_end
    print(f"Business period: {period_start.strftime('%d-%m-%y')} → {period_end.strftime('%d-%m-%y')}")
    print(f"Planning days:   {working_dates[0]} → {working_dates[-1]} ({len(working_dates)} days)")

    # ---------- Basic Data ----------
    targets = {str(row["Product"]).lower(): int(row["Target_Qty"]) for _, row in targets_df.iterrows()}
    priorities = {str(row["Product"]).lower(): int(row["Priority"]) for _, row in targets_df.iterrows()}

    # Priority for subtypes inherits zen70 priority
    zen70_priority = priorities.get("zen70", 10)
    for p in ZEN70_TYPES:
        priorities[p] = zen70_priority

    # Per-product daily cap — read from targets.csv Max_Per_Day, fall back to hardcoded dict
    _product_max = dict(product_max)
    if "Max_Per_Day" in targets_df.columns:
        for _, row in targets_df.iterrows():
            p = str(row["Product"]).lower()
            if pd.notna(row.get("Max_Per_Day")):
                mpd = int(row["Max_Per_Day"])
                if p == "zen70":
                    for tp in ZEN70_TYPES:
                        _product_max[tp] = mpd
                else:
                    _product_max[p] = mpd
    for p in products:
        if p not in _product_max:
            print(f"WARNING: {p} has no Max_Per_Day in targets.csv and no hardcoded default; using 1")
            _product_max[p] = 1

    materials = material_df["Material"].tolist()
    material_stock = {row["Material"]: int(row["Available_Qty"]) for _, row in material_df.iterrows()}

    # BOM columns = planning products (zen70_pv / py / pz must exist in CSV)
    bom_cols = [c for c in material_df.columns if str(c).lower() in products or str(c).lower() in ZEN70_TYPES]
    material_usage = {}
    for _, row in material_df.iterrows():
        mat = row["Material"]
        material_usage[mat] = {}
        for p in products:
            col = p
            if col in material_df.columns:
                material_usage[mat][p] = int(row[col])
            else:
                material_usage[mat][p] = 0

    # ---------- Configuration Processing ----------
    configs = defaultdict(dict)
    if config_df is not None and not config_df.empty:
        for _, row in config_df.iterrows():
            p = str(row["Product"]).strip().lower()
            # allow config on zen70 family → apply to all subtypes if Product=zen70
            cfg = str(row["Config_Name"]).strip()
            mat = str(row["Material"]).strip() if pd.notna(row["Material"]) else None
            qty = int(row["Qty_Per_System"]) if pd.notna(row.get("Qty_Per_System")) else 0
            target = int(row["Target_Qty"]) if pd.notna(row.get("Target_Qty")) else None

            target_products = ZEN70_TYPES if p == "zen70" else [p]
            for tp in target_products:
                if cfg not in configs[tp]:
                    configs[tp][cfg] = {"target": 0, "materials": {}}
                if target is not None:
                    configs[tp][cfg]["target"] = target
                if mat and mat.lower() not in ["", "nan", "none"]:
                    configs[tp][cfg]["materials"][mat] = qty

    # ---------- MAD / special orders ----------
    # Family key "zen70" for generic Zen70 orders; subtype keys if order specifies PV/PY/PZ
    mad_orders = []
    orders_by_pc = defaultdict(list)

    if mad_df is not None and not mad_df.empty:
        for _, row in mad_df.iterrows():
            order_id = str(row.get("Production_order", row.get("Production_Order", ""))).strip()
            country = str(row.get("Country", "")).strip()
            raw_type = str(row.get("System_Type", "")).strip().lower().replace(" ", "")
            mad_raw = row.get("Expected_MAD", None)
            mad_dt = parse_date(mad_raw)
            if mad_dt is None:
                continue
            mad_str = mad_dt.strftime("%d-%m-%y")

            system_type = normalize_system_type(raw_type, products + ["zen70"])
            if system_type is None:
                continue

            # NEW: pin subtype from Production_order
            if system_type == "zen70":
                subtype = zen70_subtype_from_order(order_id)
                if subtype is not None:
                    system_type = subtype

            # Config only if exact product has that country config
            config_name = None
            if system_type in configs and country in configs[system_type]:
                config_name = country
            elif system_type == "zen70":
                # family-level config name if any subtype has it
                for tp in ZEN70_TYPES:
                    if tp in configs and country in configs[tp]:
                        config_name = country
                        break

            order = {
                "Production_Order": order_id,
                "Country": country,
                "System_Type": system_type,  # zen70 or zen70_pv etc.
                "Config_Name": config_name,
                "Expected_MAD": mad_str,
                "MAD_datetime": mad_dt,
            }
            mad_orders.append(order)
            orders_by_pc[(system_type, config_name)].append(order)

    mad_requirements = {}
    for (prod, cfg), orders in orders_by_pc.items():
        orders_sorted = sorted(orders, key=lambda o: o["MAD_datetime"])
        cumulative = 0
        for order in orders_sorted:
            cumulative += 1
            key = (prod, cfg, order["Expected_MAD"])
            mad_requirements[key] = max(mad_requirements.get(key, 0), cumulative)

    # ---------- Arrivals ----------
    arrivals = defaultdict(lambda: defaultdict(int))
    for _, row in material_df.iterrows():
        mat = str(row["Material"]).strip()
        if not mat:
            continue
        qty_str = str(row.get("Incoming_Qty", "")).strip()
        date_str = str(row.get("Date", "")).strip()
        if qty_str in ["", "0", "nan", "NaN", "0.0"] or date_str in ["", "nan", "NaN"]:
            continue
        date_list = [x.strip() for x in date_str.split(",") if x.strip()]
        qty_list = [x.strip() for x in qty_str.split(",") if x.strip()]
        for d_text, q_text in zip(date_list, qty_list):
            try:
                qty = int(float(q_text))
                if qty <= 0:
                    continue
                for fmt in ["%d-%m-%y", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y", "%m/%d/%Y"]:
                    try:
                        dt = datetime.strptime(d_text, fmt)
                        arrivals[dt.strftime("%d-%m-%y")][mat] += qty
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

    # ---------- Material Availability ----------
    material_available = {}
    full_calendar = []
    current = planning_start
    while current <= planning_end:
        full_calendar.append(current.strftime("%d-%m-%y"))
        current += timedelta(days=1)

    for mat in materials:
        material_available[mat] = {}
        cum = material_stock.get(mat, 0)
        for day_str in full_calendar:
            if day_str in arrivals and mat in arrivals[day_str]:
                cum += arrivals[day_str][mat]
            dt = datetime.strptime(day_str, "%d-%m-%y")
            if dt.weekday() < 5:
                material_available[mat][day_str] = cum
        for wd in working_dates:
            if wd not in material_available[mat]:
                material_available[mat][wd] = cum

    # ---------- CP Model ----------
    model = cp_model.CpModel()

    x = {(p, d): model.NewIntVar(0, _product_max.get(p, 1), f"x_{p}_{d}")
         for p in products for d in working_dates}
    y = {(p, d): model.NewBoolVar(f"y_{p}_{d}")
         for p in products for d in working_dates}

    for p in products:
        for d in working_dates:
            model.Add(x[p, d] <= _product_max.get(p, 1) * y[p, d])
            model.Add(x[p, d] >= y[p, d])

    # Targets: normal products individual; zen70 family sum
    for p in products:
        if p in ZEN70_TYPES:
            continue
        model.Add(sum(x[p, d] for d in working_dates) <= targets.get(p, 0))

    if "zen70" in targets:
        model.Add(
            sum(x[p, d] for p in ZEN70_TYPES for d in working_dates)
            <= targets["zen70"]
        )

    for d in working_dates:
        model.Add(sum(x[p, d] for p in products) <= max_systems_per_day)
        model.Add(sum(y[p, d] for p in products) <= max_products_per_day)
        # family daily cap for zen70
        model.Add(sum(x[p, d] for p in ZEN70_TYPES if p in products) <= ZEN70_FAMILY_DAILY_MAX)

    for p in products:
        for i in range(1, len(working_dates)):
            d1, d2 = working_dates[i - 1], working_dates[i]
            model.Add(x[p, d2] - x[p, d1] <= 1)
            model.Add(x[p, d1] - x[p, d2] <= 1)

    # Configuration variables
    z = {}
    for p in configs:
        if p not in products:
            continue
        for cfg, info in configs[p].items():
            for d in working_dates:
                z[p, cfg, d] = model.NewIntVar(0, _product_max.get(p, 1), f"z_{p}_{cfg}_{d}")
                model.Add(z[p, cfg, d] <= x[p, d])
            model.Add(sum(z[p, cfg, d] for d in working_dates) <= info["target"])
        for d in working_dates:
            model.Add(sum(z[p, cfg, d] for cfg in configs[p]) <= x[p, d])

    if forced_configs:
        for (p, cfg, d), qty in forced_configs.items():
            if (p, cfg, d) in z:
                model.Add(z[p, cfg, d] == int(qty))

    # Config smoothness — prevent day-to-day jumps > 1 on z variables
    for p in configs:
        if p not in products:
            continue
        for cfg in configs[p]:
            for i in range(1, len(working_dates)):
                d1, d2 = working_dates[i - 1], working_dates[i]
                if (p, cfg, d1) in z and (p, cfg, d2) in z:
                    model.Add(z[p, cfg, d2] - z[p, cfg, d1] <= 1)
                    model.Add(z[p, cfg, d1] - z[p, cfg, d2] <= 1)

    # MAD constraints
    mad_shortage_vars = {}
    for (prod, config_name, mad_date), required_qty in mad_requirements.items():
        days_up_to_mad = [
            day for day in working_dates
            if datetime.strptime(day, "%d-%m-%y") <= datetime.strptime(mad_date, "%d-%m-%y")
        ]
        if not days_up_to_mad:
            continue

        if config_name is not None:
            # configured: sum z over relevant products
            terms = []
            if prod == "zen70":
                for tp in ZEN70_TYPES:
                    if (tp, config_name, days_up_to_mad[0]) in z or (tp, config_name) in [(a, b) for a, b, _ in z.keys()]:
                        for day in days_up_to_mad:
                            if (tp, config_name, day) in z:
                                terms.append(z[tp, config_name, day])
            elif prod in products:
                for day in days_up_to_mad:
                    if (prod, config_name, day) in z:
                        terms.append(z[prod, config_name, day])
            if not terms:
                continue
            produced_by_mad = model.NewIntVar(0, required_qty + 20, f"prod_{prod}_{config_name}_{mad_date}")
            model.Add(produced_by_mad == sum(terms))
        else:
            # BASE family or single product — subtract configured units so base-only is tracked
            if prod == "zen70":
                terms = []
                for tp in ZEN70_TYPES:
                    for day in days_up_to_mad:
                        cfg_vars = [z[tp, cfg, day] for cfg in configs.get(tp, {})
                                    if (tp, cfg, day) in z]
                        terms.append(x[tp, day] - sum(cfg_vars) if cfg_vars else x[tp, day])
            elif prod in products:
                terms = []
                for day in days_up_to_mad:
                    cfg_vars = [z[prod, cfg, day] for cfg in configs.get(prod, {})
                                if (prod, cfg, day) in z]
                    terms.append(x[prod, day] - sum(cfg_vars) if cfg_vars else x[prod, day])
            else:
                continue
            produced_by_mad = model.NewIntVar(0, required_qty + 50, f"prod_{prod}_base_{mad_date}")
            model.Add(produced_by_mad == sum(terms))

        shortage = model.NewIntVar(0, required_qty, f"shortage_{prod}_{config_name}_{mad_date}")
        model.Add(shortage >= required_qty - produced_by_mad)
        model.Add(shortage >= 0)
        mad_shortage_vars[(prod, config_name, mad_date)] = shortage

    # Material constraints (BOM + config override)
    for mat in materials:
        for i, day in enumerate(working_dates):
            consumption_terms = []
            for prod in products:
                bom_usage = material_usage[mat].get(prod, 0)
                for d in working_dates[: i + 1]:
                    configured_on_d = []
                    if prod in configs:
                        for cfg in configs[prod]:
                            if (prod, cfg, d) in z:
                                configured_on_d.append(z[prod, cfg, d])
                    if configured_on_d:
                        total_configured = sum(configured_on_d)
                        if bom_usage > 0:
                            consumption_terms.append((x[prod, d] - total_configured) * bom_usage)
                        for cfg, info in configs[prod].items():
                            if (prod, cfg, d) not in z:
                                continue
                            if mat in info["materials"]:
                                consumption_terms.append(z[prod, cfg, d] * info["materials"][mat])
                            elif bom_usage > 0:
                                consumption_terms.append(z[prod, cfg, d] * bom_usage)
                    else:
                        if bom_usage > 0:
                            consumption_terms.append(x[prod, d] * bom_usage)
            if consumption_terms:
                model.Add(sum(consumption_terms) <= material_available[mat][day])

    # Objective
    daily_load = {day: model.NewIntVar(0, max_systems_per_day, f"load_{day}") for day in working_dates}
    for day in working_dates:
        model.Add(daily_load[day] == sum(x[p, day] for p in products))

    diffs = []
    for i in range(1, len(working_dates)):
        d1, d2 = working_dates[i - 1], working_dates[i]
        diff = model.NewIntVar(0, max_systems_per_day, f"diff_{i}")
        model.AddAbsEquality(diff, daily_load[d2] - daily_load[d1])
        diffs.append(diff)

    day_weights = {day: len(working_dates) - i for i, day in enumerate(working_dates)}
    earliness_bonus = sum(x[p, day] * day_weights[day] for p in products for day in working_dates)
    total_production = sum(x[p, day] for p in products for day in working_dates)
    total_priority = sum(x[p, day] * priorities.get(p, 1) for p in products for day in working_dates)
    diversity = sum(y[p, day] for p in products for day in working_dates)

    config_reward = 0
    for p in configs:
        for cfg in configs[p]:
            for day in working_dates:
                if (p, cfg, day) in z:
                    config_reward += z[p, cfg, day] * 80

    mad_penalty = sum(var * mad_penalty_weight for var in mad_shortage_vars.values())

    # Soft under-production penalty — incentivise hitting targets even under material pressure
    deficit_vars = {}
    for p_raw in raw_products:
        t = targets.get(p_raw, 0)
        if t <= 0:
            continue
        dv = model.NewIntVar(0, t, f"deficit_{p_raw}")
        if p_raw == "zen70":
            model.Add(dv >= t - sum(x[tp, d] for tp in ZEN70_TYPES for d in working_dates))
        elif p_raw in products:
            model.Add(dv >= t - sum(x[p_raw, d] for d in working_dates))
        model.Add(dv >= 0)
        deficit_vars[p_raw] = dv

    deficit_penalty = sum(
        deficit_vars[p] * priorities.get(p, 1) * 30
        for p in deficit_vars
    )

    model.Maximize(
        total_priority * 10
        + total_production
        + diversity_weight * diversity
        + earliness_weight * earliness_bonus
        + config_reward * 10
        - smoothness_weight * sum(diffs)
        - mad_penalty * 100
        - deficit_penalty
    )

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = 4
    solver.parameters.max_time_in_seconds = solve_time
    status = solver.Solve(model)
    solver_status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE (not proven optimal)"

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None, None, None, None, None, None, None, None

    # ---------- Outputs ----------
    production_plan = []
    for day in working_dates:
        row = {"Date": day}
        total = 0
        for p in products:
            qty = solver.Value(x[p, day])
            row[p] = qty
            total += qty
        row["zen70_total"] = sum(solver.Value(x[p, day]) for p in ZEN70_TYPES if p in products)
        row["Total"] = total
        production_plan.append(row)

    schedule_df = pd.DataFrame(production_plan)
    schedule_df["Date_dt"] = pd.to_datetime(schedule_df["Date"], format="%d-%m-%y")
    schedule_df = schedule_df.sort_values("Date_dt").drop(columns=["Date_dt"]).reset_index(drop=True)

    daily_consumption = {day: {mat: 0 for mat in materials} for day in working_dates}
    for day in working_dates:
        for mat in materials:
            for prod in products:
                bom_usage = material_usage[mat].get(prod, 0)
                x_val = solver.Value(x[prod, day])
                total_z = 0
                if prod in configs:
                    for cfg, info in configs[prod].items():
                        if (prod, cfg, day) not in z:
                            continue
                        z_val = solver.Value(z[prod, cfg, day])
                        total_z += z_val
                        if mat in info["materials"]:
                            daily_consumption[day][mat] += z_val * info["materials"][mat]
                        else:
                            daily_consumption[day][mat] += z_val * bom_usage
                daily_consumption[day][mat] += (x_val - total_z) * bom_usage

    daily_consumption_df = pd.DataFrame([{"Date": day, **daily_consumption[day]} for day in working_dates])

    # Cumulative material consumption up to each working day (used for miss-diagnosis)
    _cum_used = {mat: 0 for mat in materials}
    cumulative_consumption_by_day = {}
    for day in working_dates:
        for mat in materials:
            _cum_used[mat] += daily_consumption[day].get(mat, 0)
        cumulative_consumption_by_day[day] = dict(_cum_used)

    config_rows = []
    for p in configs:
        for cfg in configs[p]:
            row = {"Config": f"{p} | {cfg}"}
            total = 0
            for day in working_dates:
                qty = solver.Value(z[p, cfg, day]) if (p, cfg, day) in z else 0
                row[day] = qty
                total += qty
            row["TOTAL"] = total
            config_rows.append(row)
    config_schedule_df = pd.DataFrame(config_rows)

    material_status = []
    for material in materials:
        total_used = sum(daily_consumption[day][material] for day in working_dates)
        final_available = material_available[material][working_dates[-1]] if working_dates else 0
        material_status.append({
            "Material": material,
            "Available": final_available,
            "Used": total_used,
            "Remaining": final_available - total_used,
        })
    material_status_df = pd.DataFrame(material_status)

    # Target status: family zen70 + other products
    target_status = []
    for p in raw_products:
        if p == "zen70":
            planned = sum(schedule_df[tp].sum() for tp in ZEN70_TYPES if tp in schedule_df.columns)
            target_status.append({
                "Product": "zen70",
                "Target": targets.get("zen70", 0),
                "Planned": planned,
                "Remaining": targets.get("zen70", 0) - planned,
            })
            for tp in ZEN70_TYPES:
                if tp in schedule_df.columns:
                    target_status.append({
                        "Product": tp,
                        "Target": "(part of zen70)",
                        "Planned": int(schedule_df[tp].sum()),
                        "Remaining": "",
                    })
        else:
            planned = schedule_df[p].sum() if p in schedule_df.columns else 0
            target_status.append({
                "Product": p,
                "Target": targets.get(p, 0),
                "Planned": planned,
                "Remaining": targets.get(p, 0) - planned,
            })
    target_status_df = pd.DataFrame(target_status)

    # MAD report — zen70 family uses merged timeline of all subtypes
    mad_report = []
    for (prod, cfg), orders in orders_by_pc.items():
        orders_sorted = sorted(orders, key=lambda o: o["MAD_datetime"])

        production_timeline = []
        if cfg is not None:
            if prod == "zen70":
                day_qty = defaultdict(int)
                for tp in ZEN70_TYPES:
                    for day in working_dates:
                        if (tp, cfg, day) in z:
                            day_qty[day] += solver.Value(z[tp, cfg, day])
                for day in working_dates:
                    for _ in range(day_qty[day]):
                        production_timeline.append(day)
            elif prod in products:
                for day in working_dates:
                    if (prod, cfg, day) in z:
                        for _ in range(solver.Value(z[prod, cfg, day])):
                            production_timeline.append(day)
        else:
            if prod == "zen70":
                for day in working_dates:
                    qty = sum(int(schedule_df.loc[schedule_df["Date"] == day, tp].values[0])
                              for tp in ZEN70_TYPES if tp in schedule_df.columns)
                    # subtract configured on those types
                    for tp in ZEN70_TYPES:
                        if tp in configs:
                            for cname in configs[tp]:
                                if (tp, cname, day) in z:
                                    qty -= solver.Value(z[tp, cname, day])
                    for _ in range(max(0, qty)):
                        production_timeline.append(day)
            elif prod in products:
                for day in working_dates:
                    qty_today = int(schedule_df.loc[schedule_df["Date"] == day, prod].values[0])
                    configured_qty = 0
                    if prod in configs:
                        for cfg_name in configs[prod]:
                            if (prod, cfg_name, day) in z:
                                configured_qty += solver.Value(z[prod, cfg_name, day])
                    for _ in range(max(0, qty_today - configured_qty)):
                        production_timeline.append(day)

        for idx, order in enumerate(orders_sorted):
            mad = order["Expected_MAD"]
            order_id = order["Production_Order"]
            country = order["Country"]

            if idx < len(production_timeline):
                produced_date = production_timeline[idx]
                if datetime.strptime(produced_date, "%d-%m-%y") <= datetime.strptime(mad, "%d-%m-%y"):
                    status = "On Time"
                    reason = produced_date
                else:
                    status = "Missed"
                    reason = f"Produced on {produced_date}, after MAD {mad}"
            else:
                status = "Missed"
                reason = "Not produced at all / Not enough systems planned"

            if status == "Missed":
                days_up_to_mad = [
                    day for day in working_dates
                    if datetime.strptime(day, "%d-%m-%y") <= datetime.strptime(mad, "%d-%m-%y")
                ]
                if not days_up_to_mad:
                    reason += " | MAD is before planning horizon"
                else:
                    max_possible = len(days_up_to_mad) * (
                        ZEN70_FAMILY_DAILY_MAX if prod == "zen70" else _product_max.get(prod, 1)
                    )
                    total_demand = sum(
                        1 for (p, c), olist in orders_by_pc.items()
                        for o in olist
                        if p == prod and o["MAD_datetime"] <= order["MAD_datetime"]
                    )
                    if total_demand > max_possible:
                        reason += " | Insufficient capacity before MAD"
                    else:
                        # Check net material availability on last day before MAD
                        last_day = days_up_to_mad[-1]
                        short_mats = []
                        for mat in materials:
                            usage = material_usage[mat].get(prod, 0)
                            if usage > 0:
                                net = (
                                    material_available.get(mat, {}).get(last_day, 0)
                                    - cumulative_consumption_by_day.get(last_day, {}).get(mat, 0)
                                )
                                if net < usage:
                                    short_mats.append(mat)
                        if short_mats:
                            reason += f" | Material shortage: {', '.join(short_mats)}"
                        else:
                            reason += " | Capacity, material, or sequencing limit before MAD"

            mad_report.append({
                "Production_Order": order_id,
                "System_Type": prod,
                "Config_Name": cfg if cfg else "BASE",
                "Country": country,
                "Expected_MAD": mad,
                "Status": status,
                "Reason": reason,
            })

    mad_report_df = pd.DataFrame(mad_report)

    # Net material availability after plan
    availability_rows = []
    cumulative_used = {mat: 0 for mat in materials}
    for day in working_dates:
        row = {"Date": day}
        for mat in materials:
            gross = material_available.get(mat, {}).get(day, 0)
            cumulative_used[mat] += daily_consumption[day].get(mat, 0)
            row[mat] = gross - cumulative_used[mat]
        availability_rows.append(row)
    material_availability_df = pd.DataFrame(availability_rows)

    return (
        schedule_df,
        material_status_df,
        target_status_df,
        daily_consumption_df,
        config_schedule_df,
        mad_report_df,
        material_availability_df,
        solver_status_str,
    )