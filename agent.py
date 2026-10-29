"""
agent.py — Gemini-powered planning assistant for the Production Scheduler.

Public API
----------
SchedulerAssistant(api_key)          — create/re-create when the key changes
    .send_message(text) -> (reply, pending_op | None)

execute_write_csv(path, updates)     -> (ok: bool, msg: str)
execute_set_constant(name, value)    -> (ok: bool, msg: str)
"""

import os
import re
import json
import pathlib

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env so GEMINI_API_KEY is available via os.environ when the app starts.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

_WRITE_WHITELIST = {
    "input/targets.csv",
    "input/material_constraints.csv",
    "input/configurations.csv",
    "input/special_orders.csv",
    "input/actual_production.csv",
}

_EDITABLE_CONSTANTS = [
    "max_systems_per_day",
    "diversity_weight",
    "smoothness_weight",
    "max_products_per_day",
    "earliness_weight",
    "mad_penalty_weight",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an AI assistant embedded in an Industrial APS (Advanced Planning \
and Scheduling) Streamlit app that schedules production of Philips medical imaging systems.

## Products
zen10, zen30, zen50, zen70 (family → subtypes zen70_pv / zen70_py / zen70_pz), zen90.

## Input files you may READ and EDIT (after user confirmation)
- input/targets.csv              — Product, Target_Qty, Priority, Max_Per_Day
  (Max_Per_Day for zen70 controls the daily family cap for all three subtypes)
- input/material_constraints.csv — Material, zen10, zen30, zen50, zen70_pv, zen70_py,
                                   zen70_pz, zen90, Available_Qty, required_Qty,
                                   Incoming_Qty, Date, Included
- input/configurations.csv       — Product, Config_Name, Material, Qty_Per_System, Target_Qty
- input/special_orders.csv       — Production_order, Country, System_Type, Expected_MAD,
                                   Assembly, Assembly_date
- input/actual_production.csv    — Date, product columns, Total

## Output files (READ ONLY — never call write_csv on these)
- output/or_tools_schedule.csv, output/daily_material_consumption.csv,
  output/config_schedule.csv, output/mad_compliance_report.csv,
  output/remaining_material.csv, output/complete_targets.csv,
  output/material_availability_on_date.csv

## Solver constants in creater.py you may propose to change
- max_systems_per_day   (default 6)    — global hard cap: total systems per working day
- diversity_weight      (default 2)    — reward for scheduling multiple product types per day
- smoothness_weight     (default 1)    — penalty for day-to-day production load changes
- max_products_per_day  (default 8)    — max distinct product types allowed per day
- earliness_weight      (default 1000) — reward for front-loading production
- mad_penalty_weight    (default 500)  — penalty per unit short of a customer MAD deadline

## Rules
1. Always call read_csv or get_constant FIRST before proposing any change — never guess values.
2. For edits, call write_csv or set_constant. These register a PROPOSED change only.
   Nothing is written until the user clicks Confirm in the UI.
3. State clearly: file · column · current value → new value.
4. After a confirmed change, remind the user to click Create Initial Plan to regenerate.
5. Output files are read-only. Redirect "edit the schedule" requests to changing inputs.
6. Be concise. Use bullet points for data summaries.
"""

# ---------------------------------------------------------------------------
# Tool declarations (Gemini function-calling schema)
# ---------------------------------------------------------------------------
_TOOL_DECLARATIONS = [
    {
        "name": "list_files",
        "description": (
            "Return the full inventory of input and output CSV files in this project."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_csv",
        "description": (
            "Read a CSV file and return its column names, row count, and up to 50 rows as JSON. "
            "Use this before proposing any change so you know the current values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from project root, e.g. 'input/targets.csv'",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_csv",
        "description": (
            "Propose a cell-level edit to an input CSV file. "
            "The change is QUEUED for user confirmation — it does NOT write until Confirm is clicked. "
            "Use only on input/ files. Do NOT use on output/ files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the input CSV, e.g. 'input/targets.csv'",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English summary shown to the user, e.g. "
                        "'Change zen10 Max_Per_Day from 1 to 2'"
                    ),
                },
                "updates": {
                    "type": "array",
                    "description": "One entry per cell to change.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "match_column": {
                                "type": "string",
                                "description": "Column used to identify the target row",
                            },
                            "match_value": {
                                "type": "string",
                                "description": "Value that identifies the target row",
                            },
                            "set_column": {
                                "type": "string",
                                "description": "Column whose value will be changed",
                            },
                            "set_value": {
                                "type": "string",
                                "description": "New value (always a string; cast on write)",
                            },
                        },
                        "required": [
                            "match_column",
                            "match_value",
                            "set_column",
                            "set_value",
                        ],
                    },
                },
            },
            "required": ["path", "description", "updates"],
        },
    },
    {
        "name": "get_constant",
        "description": "Read the current value of a solver constant from creater.py.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Constant name",
                    "enum": _EDITABLE_CONSTANTS,
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_constant",
        "description": (
            "Propose changing a solver constant in creater.py. "
            "The change is QUEUED for user confirmation — nothing is written until Confirm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": _EDITABLE_CONSTANTS,
                },
                "value": {
                    "type": "number",
                    "description": "New numeric value",
                },
            },
            "required": ["name", "value"],
        },
    },
]

# ---------------------------------------------------------------------------
# Read-only tool implementations
# ---------------------------------------------------------------------------

def _tool_list_files() -> dict:
    inp = PROJECT_ROOT / "input"
    out = PROJECT_ROOT / "output"
    return {
        "input_files":  sorted(f"input/{p.name}"  for p in inp.glob("*.csv")),
        "output_files": sorted(f"output/{p.name}" for p in out.glob("*.csv")),
    }


def _tool_read_csv(path: str) -> dict:
    # Validate path stays inside project
    full = (PROJECT_ROOT / path).resolve()
    if not str(full).startswith(str(PROJECT_ROOT)):
        return {"error": "Path outside project root is not allowed."}
    if not full.exists():
        return {"error": f"File not found: {path}"}
    try:
        df = pd.read_csv(full)
        rows = df.head(50).to_dict(orient="records")
        result = {
            "columns": list(df.columns),
            "row_count": len(df),
            "rows": rows,
        }
        if len(df) > 50:
            result["truncated"] = True
        return result
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_constant(name: str) -> dict:
    src = (PROJECT_ROOT / "creater.py").read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*([\d.]+)", src, re.M)
    if m:
        raw = m.group(1)
        return {"name": name, "value": float(raw) if "." in raw else int(raw)}
    return {"error": f"Constant '{name}' not found in creater.py"}

# ---------------------------------------------------------------------------
# Confirmation-side executors (called from main.py after user clicks Confirm)
# ---------------------------------------------------------------------------

def execute_write_csv(path: str, updates: list) -> tuple:
    """Apply cell-level updates to a whitelisted input CSV.

    Returns (True, "Saved") on success, (False, error_message) on failure.
    """
    if path not in _WRITE_WHITELIST:
        return False, f"'{path}' is not in the write whitelist."
    full = PROJECT_ROOT / path
    try:
        df = pd.read_csv(full)
        for upd in updates:
            mc, mv, sc, sv = (
                upd["match_column"],
                str(upd["match_value"]),
                upd["set_column"],
                upd["set_value"],
            )
            if mc not in df.columns:
                return False, f"Column '{mc}' not found in {path}."
            if sc not in df.columns:
                return False, f"Column '{sc}' not found in {path}."
            mask = df[mc].astype(str) == mv
            if not mask.any():
                return False, f"No row where {mc} == '{mv}' in {path}."
            # Cast the incoming string to the column's existing dtype so pandas
            # doesn't reject e.g. '4' into an int64 column (pandas ≥ 2.0).
            try:
                sv_typed = pd.Series([sv]).astype(df[sc].dtype).iloc[0]
            except (ValueError, TypeError):
                sv_typed = sv  # fall back to raw string
            df.loc[mask, sc] = sv_typed
        # Re-infer numeric types where possible
        df = df.infer_objects(copy=False)
        df.to_csv(full, index=False)
        return True, "Saved"
    except Exception as exc:
        return False, str(exc)


def execute_set_constant(name: str, value) -> tuple:
    """Patch a top-level numeric constant in creater.py.

    Returns (True, message) on success, (False, error_message) on failure.
    """
    creater = PROJECT_ROOT / "creater.py"
    try:
        src = creater.read_text(encoding="utf-8")
        new_src = re.sub(
            rf"^({re.escape(name)}\s*=\s*)[\d.]+",
            rf"\g<1>{value}",
            src,
            flags=re.M,
        )
        if new_src == src:
            return False, f"Constant '{name}' not found or value unchanged in creater.py."
        creater.write_text(new_src, encoding="utf-8")
        return True, f"{name} set to {value}"
    except Exception as exc:
        return False, str(exc)

# ---------------------------------------------------------------------------
# SchedulerAssistant
# ---------------------------------------------------------------------------

class SchedulerAssistant:
    """Wraps a Gemini chat session with function-calling tools."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name="gemini-3.5-flash-lite",
            tools=[{"function_declarations": _TOOL_DECLARATIONS}],
            system_instruction=_SYSTEM_PROMPT,
        )
        self._session = self._model.start_chat(history=[])

    def send_message(self, user_msg: str) -> tuple:
        """Send a user message and drive the full tool-calling loop.

        Returns:
            reply_text   — assistant's final text response
            pending_op   — dict describing a proposed write, or None
        """
        try:
            response = self._session.send_message(user_msg)
        except Exception as exc:
            return f"Error calling Gemini API: {exc}", None

        return self._process(response)

    # ------------------------------------------------------------------
    def _process(self, response) -> tuple:
        pending_op = None

        for _ in range(8):   # max tool rounds per user turn
            fn_calls = [
                p.function_call
                for p in response.parts
                if hasattr(p, "function_call") and p.function_call.name
            ]
            if not fn_calls:
                break

            tool_parts = []
            for fc in fn_calls:
                result, op = self._dispatch(fc.name, dict(fc.args))
                if op:
                    pending_op = op
                tool_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response={"result": json.dumps(result, default=str)},
                        )
                    )
                )

            try:
                response = self._session.send_message(tool_parts)
            except Exception as exc:
                return f"Error returning tool results to Gemini: {exc}", pending_op

        text = "".join(
            p.text for p in response.parts if hasattr(p, "text") and p.text
        ).strip()
        return text or "(No response from assistant)", pending_op

    # ------------------------------------------------------------------
    def _dispatch(self, name: str, args: dict) -> tuple:
        """Execute a tool call. Returns (result_dict, pending_op | None)."""

        if name == "list_files":
            return _tool_list_files(), None

        if name == "read_csv":
            return _tool_read_csv(args.get("path", "")), None

        if name == "get_constant":
            return _tool_get_constant(args.get("name", "")), None

        if name == "write_csv":
            path = args.get("path", "")
            if not path.startswith("input/"):
                return {"error": "Output files are read-only."}, None
            op = {
                "type":        "write_csv",
                "path":        path,
                "description": args.get("description", "Edit CSV"),
                "updates":     args.get("updates", []),
            }
            return {
                "status":  "pending_confirmation",
                "message": "Change queued for user confirmation in the UI.",
            }, op

        if name == "set_constant":
            n = args.get("name", "")
            v = args.get("value")
            current = _tool_get_constant(n)
            cur_val = current.get("value", "?")
            op = {
                "type":        "set_constant",
                "name":        n,
                "value":       v,
                "description": f"Set `{n}` from **{cur_val}** → **{v}** in creater.py",
            }
            return {
                "status":  "pending_confirmation",
                "message": "Change queued for user confirmation in the UI.",
            }, op

        return {"error": f"Unknown tool: {name}"}, None
