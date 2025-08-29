import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# robust SQLDatabase import across versions
try:
    from langchain_community.utilities.sql_database import SQLDatabase
except Exception:
    try:
        from langchain_community.utilities import SQLDatabase
    except Exception:
        from langchain.sql_database import SQLDatabase

# dotenv loader
try:
    from dotenv import load_dotenv, find_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

class SQLPlan(BaseModel):
    intent: str = Field(description="task type, e.g. list/lookup/analytics")
    dialect: str = Field(description="sql dialect, e.g. mysql")
    tables: List[str] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    filters: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    sql: str = Field(description="final SELECT sql")
    confidence: float = Field(description="0..1")

SYSTEM_PROMPT = (
    "You are a precise SQL planner for MySQL/MariaDB.\n"
    "Return ONLY valid JSON conforming to the SQLPlan schema.\n"
    "Rules:\n"
    "- Use ONLY tables/columns from the provided schema.\n"
    "- Produce ONE final SQL SELECT query (no DDL/DML/UPDATE/INSERT/DELETE).\n"
    "- Prefer explicit JOINs with ON; never use NATURAL JOIN.\n"
    "- Quote identifiers with backticks only if needed.\n"
    "- Some date-like fields may be VARCHAR; when filtering/ordering by date, wrap with STR_TO_DATE using:\n"
    "  COALESCE(STR_TO_DATE(col,'%Y-%m-%d'),STR_TO_DATE(col,'%d-%m-%Y'),STR_TO_DATE(col,'%d/%m/%Y'),STR_TO_DATE(col,'%Y/%m/%d'))\n"
    "- If ambiguous, make minimal safe assumptions and list them.\n"
    "- Default LIMIT 200 if user doesn’t specify.\n"
    "- Dialect: mysql."
)

FEWSHOT_TEMPLATE = "Examples (NL → SQL):\n{fewshots}"
USER_PROMPT = (
    "User question:\n{user_query}\n\n"
    "Schema (truncated summary):\n{schema_note}\n\n"
    "Return ONLY the JSON object for SQLPlan."
)

SELECT_ONLY_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE | re.DOTALL)

def load_env(env_file: Optional[str]) -> None:
    if env_file:
        if not DOTENV_AVAILABLE:
            raise RuntimeError("python-dotenv not installed. Install: pip install python-dotenv")
        if not os.path.isfile(env_file):
            raise FileNotFoundError(f".env file not found: {env_file}")
        load_dotenv(env_file, override=False)
    else:
        if DOTENV_AVAILABLE:
            # tries to locate nearest .env upward from CWD; safe no-op if none
            path = find_dotenv(usecwd=True)
            if path:
                load_dotenv(path, override=False)

def require_env(var_name: str) -> str:
    val = os.getenv(var_name, "").strip()
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return val

def load_domain_pack(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_schema_catalog(domain: Dict[str, Any]) -> Dict[str, Any]:
    tables = {}
    fks = []
    for t in domain.get("tables", []):
        tname = t["name"]
        tables[tname] = {
            "columns": [c["name"] for c in t.get("columns", [])],
            "column_meta": {c["name"]: c for c in t.get("columns", [])}
        }
        for c in t.get("columns", []):
            if "references_table" in c and c["references_table"]:
                fks.append({
                    "from_table": tname,
                    "from_col": c["name"],
                    "to_table": c["references_table"],
                    "to_col": c.get("references_column", "")
                })
    return {"tables": tables, "fks": fks}

def make_schema_note(domain: Dict[str, Any], max_cols_per_table: int = 14) -> str:
    lines = []
    for t in domain.get("tables", []):
        tname = t["name"]
        cols = [c["name"] for c in t.get("columns", [])]
        shown = ", ".join(cols[:max_cols_per_table]) + (", ..." if len(cols) > max_cols_per_table else "")
        lines.append(f"- {tname}({shown})")
    fk_lines = []
    for t in domain.get("tables", []):
        for c in t.get("columns", []):
            if c.get("references_table"):
                fk_lines.append(f"{t['name']}.{c['name']} -> {c['references_table']}.{c.get('references_column','')}")
    if fk_lines:
        lines.append("\nForeign keys:\n" + "\n".join([f"- {l}" for l in fk_lines]))
    return "\n".join(lines)

def collect_fewshot_examples(domain: Dict[str, Any]) -> List[Dict[str, str]]:
    return domain.get("natural_language_to_sql_examples", []) or []

def fewshot_block(examples: List[Dict[str, str]]) -> str:
    if not examples:
        return "None."
    b = []
    for ex in examples[:5]:
        nl = ex.get("question") or ex.get("nl") or ""
        sql = ex.get("sql_query") or ex.get("sql") or ""
        b.append(f"- NL: {nl}\n  SQL: {sql}")
    return "\n".join(b)

def validate_plan(plan: SQLPlan, catalog: Dict[str, Any]) -> List[str]:
    errs = []
    if not SELECT_ONLY_RE.search(plan.sql):
        errs.append("SQL must start with SELECT.")
    known_tables = set(catalog["tables"].keys())
    for t in plan.tables:
        if t not in known_tables:
            errs.append(f"Unknown table: {t}")
    known_cols = {f"{t}.{c}" for t, meta in catalog["tables"].items() for c in meta["columns"]}
    for col in plan.columns:
        if "." in col and col not in known_cols:
            errs.append(f"Unknown column: {col}")
    for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]:
        if kw in plan.sql.upper():
            errs.append("Forbidden keyword (write operation) detected.")
            break
    return errs

def try_parse_json_block(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

def plan_sql(user_query: str, domain_path: str, model_name: str = "gemini-1.5-pro") -> SQLPlan:
    domain = load_domain_pack(domain_path)
    catalog = build_schema_catalog(domain)
    schema_note = make_schema_note(domain)
    examples = collect_fewshot_examples(domain)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("system", FEWSHOT_TEMPLATE.format(fewshots=fewshot_block(examples))),
            ("human", USER_PROMPT),
        ]
    )

    # ensure key exists
    require_env("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, convert_system_message_to_human=True)

    # primary: structured output
    structured_llm = llm.with_structured_output(SQLPlan)
    try:
        result = (prompt | structured_llm).invoke({"user_query": user_query, "schema_note": schema_note})
        plan = result if isinstance(result, SQLPlan) else SQLPlan(**result)
    except Exception:
        # fallback: parse JSON from raw text
        raw = (prompt | llm).invoke({"user_query": user_query, "schema_note": schema_note})
        text = getattr(raw, "content", str(raw))
        parsed = try_parse_json_block(text)
        if not parsed:
            raise ValueError("Model did not return valid JSON for SQLPlan.")
        plan = SQLPlan(**parsed)

    errs = validate_plan(plan, catalog)
    if errs:
        raise ValueError("Plan validation failed: " + "; ".join(errs))
    return plan

@dataclass
class ToolDecision:
    tool: str
    params: Dict[str, Any]

def decide_tool_and_params(plan: SQLPlan) -> ToolDecision:
    return ToolDecision(tool="sql.query", params={"query": plan.sql})

def maybe_execute_sql(db_url: Optional[str], sql: str):
    if not db_url:
        return None
    db = SQLDatabase.from_uri(db_url)
    return db.run(sql)

def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--domain-pack", default="db_nlp.json")
    ap.add_argument("--model", default="gemini-1.5-pro")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--env-file", default=None, help="Path to a .env file (optional)")
    args = ap.parse_args()

    # load env
    load_env(args.env_file)

    try:
        plan = plan_sql(args.query, args.domain_pack, args.model)
    except Exception as e:
        print("ERROR planning SQL:", e)
        sys.exit(1)

    decision = decide_tool_and_params(plan)
    print("\n=== SQL PLAN ===")
    print(plan.model_dump_json(indent=2))
    print("\n=== TOOL ===")
    print(decision.tool, decision.params)

    if args.execute:
        if not args.db_url:
            print("Provide --db-url to execute.")
            sys.exit(2)
        print("\n=== EXECUTION RESULT (truncated) ===")
        out = maybe_execute_sql(args.db_url, plan.sql)
        print(out if out is not None else "(no output)")

if __name__ == "__main__":
    main()
