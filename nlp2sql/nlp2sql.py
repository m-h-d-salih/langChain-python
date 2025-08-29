#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini-only NL→SQL router for MySQL using your db.json schema.

- EXACT identifiers only (no renaming)
- READ-ONLY guard: SELECT / WITH / SHOW / DESCRIBE only
- Stage 2.1: LLM plans a tool call (mysql.query | no_action)
- Stage 2.2: Validate (tables & columns), single statement, read-only
- Relationship-aware (uses context.relationships first; shared-column hints as fallback)
- Auto-repair loop on invalid SQL (qualified/unqualified checks)
- .env auto-load (GOOGLE_API_KEY, GEMINI_ROUTER_MODEL)
- REST transport (stable on Windows), no system→human warning
- Schema SLICER + domain forcing:
    * calibration → tools_register
    * “sales order” → any table(s) with Sales_Order_No (joins preferred)
    * seaman book → table(s) with Seaman_Doc (+ join to professional_details for names)
- Ingests schema-provided examples (context.examples / context.example_queries) and prioritizes them
- Robust JSON parsing (balanced-brace scan) + REPROMPT-to-JSON stabilizer
- Deterministic SALES ORDER fallback when LLM still won’t return JSON

Usage:
  python nlp2sql.py --query "List all checklist entries that reference Sales Order No for redundancy validation." --schema_path ./db.json --print_json
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional, Set

# --- load .env early ---
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# ---------------- LLM (Gemini only) ----------------
def init_llm(model_name: Optional[str] = None):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        raise RuntimeError(
            "Missing langchain-google-genai. Install it:\n"
            "  pip install -U langchain-google-genai google-generativeai"
        ) from e

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set. Put it in your .env or set the env var.")

    model = (model_name or os.getenv("GEMINI_ROUTER_MODEL") or "gemini-1.5-pro").strip()
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        transport="rest",
        convert_system_message_to_human=False,   # avoids the deprecation warning
        max_output_tokens=1200,
    )

# ------------- Read-only / SQL guards --------------
READ_ONLY_PATTERN = re.compile(r"^\s*(SELECT|WITH|SHOW|DESCRIBE|DESC)\b", re.IGNORECASE | re.DOTALL)
FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|SET\s+|USE\s+|CALL\b|COMMIT\b|ROLLBACK\b|LOCK\b|UNLOCK\b|LOAD\b|OUTFILE\b|INFILE\b|EXPLAIN\b)\b",
    re.IGNORECASE,
)

def is_read_only(sql: str) -> bool:
    s = (sql or "").strip()
    parts = [p for p in s.split(";") if p.strip()]
    if len(parts) > 1:
        return False
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"--[^\n]*", " ", s)
    s = s.strip()
    if not READ_ONLY_PATTERN.match(s):
        return False
    if FORBIDDEN_PATTERN.search(s):
        return False
    return True

# ------------- Schema + relationships --------------
def _get_ctx(obj: dict) -> dict:
    return (obj.get("bundle", {}).get("context")
            or obj.get("context")
            or obj)

def parse_tables_and_relationships(schema_text: str) -> Tuple[List[dict], List[dict]]:
    obj = json.loads(schema_text)
    ctx = _get_ctx(obj)
    return (ctx.get("tables") or []), (ctx.get("relationships") or [])

def build_schema_catalog(schema_text: str) -> Dict[str, List[str]]:
    tables, _ = parse_tables_and_relationships(schema_text)
    if not tables:
        raise ValueError("No tables found in schema JSON at context.tables.")
    out: Dict[str, List[str]] = {}
    for t in tables:
        tname = t.get("name")
        cols = [c.get("name") for c in (t.get("columns") or []) if c.get("name")]
        if tname:
            out[tname] = cols
    return out

def build_type_map(schema_text: str) -> Dict[str, Dict[str, str]]:
    tables, _ = parse_tables_and_relationships(schema_text)
    tmap: Dict[str, Dict[str, str]] = {}
    for t in tables:
        tname = t.get("name")
        cols = t.get("columns") or []
        if not tname:
            continue
        tmap[tname] = {}
        for c in cols:
            cname = c.get("name")
            ctype = (c.get("type") or "").lower()
            if cname:
                tmap[tname][cname] = ctype
    return tmap

def build_explicit_relationships(schema_text: str) -> List[Dict[str, str]]:
    rels: List[Dict[str, str]] = []
    _, raw = parse_tables_and_relationships(schema_text)
    for r in raw:
        st = r.get("source_table"); sc = r.get("source_column")
        tt = r.get("target_table"); tc = r.get("target_column")
        if st and sc and tt and tc:
            rels.append({
                "source_table": st, "source_column": sc,
                "target_table": tt, "target_column": tc,
                "type": r.get("type",""), "join_example": r.get("join_example","")
            })
    return rels

def shared_columns_map(catalog: Dict[str, List[str]]) -> Dict[Tuple[str,str], List[str]]:
    """Case-insensitive 'shared columns' hints (for JOIN guidance only)."""
    tables = list(catalog.keys())
    rel: Dict[Tuple[str,str], List[str]] = {}
    for i in range(len(tables)):
        for j in range(i+1, len(tables)):
            a, b = tables[i], tables[j]
            sa = {c.lower(): c for c in catalog[a]}
            sb = {c.lower(): c for c in catalog[b]}
            inter = sorted(set(sa.keys()) & set(sb.keys()))
            if inter:
                rel[(a,b)] = inter
    return rel

def relationships_text(explicit: List[Dict[str,str]], shared: Dict[Tuple[str,str], List[str]],
                       max_pairs: int = 80, max_cols: int = 10) -> str:
    lines: List[str] = []
    if explicit:
        lines.append("EXPLICIT RELATIONSHIPS (preferred):")
        for r in explicit[:max_pairs]:
            lines.append(
                f"- {r['source_table']}.{r['source_column']} ↔ {r['target_table']}.{r['target_column']}"
                + (f" | type: {r.get('type','')}" if r.get('type') else "")
                + (f" | example: {r.get('join_example','')}" if r.get('join_example') else "")
            )
    else:
        lines.append("EXPLICIT RELATIONSHIPS (preferred): (none provided)")

    if shared:
        lines.append("\nSHARED-COLUMN HINTS (case-insensitive; fallback only):")
        count = 0
        for (a,b), cols in shared.items():
            count += 1
            if count > max_pairs:
                break
            cols_show = ", ".join(cols[:max_cols]) + ("" if len(cols) <= max_cols else ", …")
            lines.append(f"- {a} ↔ {b}: {cols_show}")
    else:
        lines.append("\nSHARED-COLUMN HINTS (fallback only): (none detected)")
    return "\n".join(lines)

# ------------- tokenization/synonyms --------------
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9_]+")

def tokenize_identifier(name: str) -> Set[str]:
    if not name:
        return set()
    s = str(name)
    base = s.lower()
    parts = {base}
    for tok in base.split("_"):
        if tok:
            parts.add(tok)
    for tok in _CAMEL.sub(" ", s).lower().split():
        if tok:
            parts.add(tok)
    return parts

def normalize_simple(s: str) -> str:
    return re.sub(r"[_\s]+", "", (s or "").lower())

def expand_synonyms(words: Set[str]) -> Set[str]:
    expanded = set(words)
    if {"seaman","book","seamanbook","cdc"} & words:
        expanded |= {"seaman","book","seaman_doc","cdc","certificate","doc","document"}
    if {"calibration","calibrate","calibrated","caliberation","due","expiry","validity","certificate","cert","date"} & words:
        expanded |= {"calibration","calib","cal","due","expiry","validity","certificate","cert","next","date"}
    if {"tool","tools","gauge","instrument","equipment","device","meter"} & words:
        expanded |= {"tool","tools","gauge","instrument","equipment","device","meter"}
    if {"qty","quantity","pending","balance"} & words:
        expanded |= {"qty","quantity","pending","balance"}
    if "place" in words:
        expanded |= {"place","location","city","country"}
    if {"professional","professionals"} & words:
        expanded |= {"professional","professionals","employee","personnel"}
    if {"sales","order","salesorder"} & words:
        expanded |= {"sales","order","sales_order","sales_order_no","so","so_no"}
    return expanded

def query_tokens(user_query: str) -> Set[str]:
    raw = {w.lower() for w in _WORD.findall(user_query or "")}
    uq = user_query.lower()
    if "sales order" in uq:
        raw |= {"sales","order","sales_order","sales_order_no"}
    return expand_synonyms(raw)

# ------------- ranking helpers (for EXAMPLES ONLY) -------------
def seaman_rank(colname: str) -> int:
    n = colname.lower()
    score = 0
    if "seaman_doc" in n: score += 100
    if "seaman" in n and "doc" in n: score += 90
    if "cdc" in n: score += 70
    if "seaman_book_no" in n: score += 60
    if "seaman_book" in n: score += 50
    if "expiry" in n: score += 10
    return score

def calibration_rank(colname: str, ctype: str = "") -> int:
    n = colname.lower(); t = (ctype or "").lower()
    score = 0
    if "calibration" in n or "calibrated" in n or "calibration_" in n: score += 60
    if "cal_date" in n or "cal date" in n or (n.startswith("cal_") and "date" in n): score += 55
    if "due_date" in n or "due date" in n: score += 45
    if "calib" in n: score += 30
    if "certificate" in n or "cert" in n: score += 15
    if "verification" in n: score += 12
    if "validity" in n or "expiry" in n or "expire" in n: score += 12
    if "next" in n: score += 8
    if "date" in n: score += 8
    if "date" in n and ("date" in t or "datetime" in t or "timestamp" in t): score += 10
    return score

def toolish_rank(colname: str) -> int:
    n = colname.lower()
    score = 0
    if "tool" in n: score += 30
    if "gauge" in n: score += 25
    if "instrument" in n: score += 22
    if "equipment" in n or "device" in n or "meter" in n: score += 18
    if "description" in n or "name" in n or "instrument_type" in n: score += 10
    return score

# ------------- month-year parsing -----------------
MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
def parse_month_year(user_query: str) -> Optional[Tuple[int,int,str]]:
    uq = user_query.lower()
    y = re.search(r"\b(20\d{2}|19\d{2})\b", uq)
    m = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", uq)
    if m and y:
        month = MONTHS[m.group(1)]
        year = int(y.group(1))
        disp = m.group(1).capitalize() + f" {year}"
        return month, year, disp
    return None

# ------------- domain forcing ---------------------
def force_tables_for_query(user_query: str, all_tables: List[dict], catalog: Dict[str, List[str]], types: Dict[str, Dict[str,str]]) -> List[str]:
    tokens = query_tokens(user_query)
    names = {t.get("name"): t for t in all_tables if t.get("name")}
    forced: List[str] = []

    # Calibration
    if {"calibration","caliberation","calib","cal"} & tokens or {"tool","tools","gauge","instrument"} & tokens:
        if "tools_register" in names:
            cols = catalog.get("tools_register", [])
            ttypes = types.get("tools_register", {})
            cal_cols = [c for c in cols if calibration_rank(c, ttypes.get(c,"")) >= 30]
            if cal_cols:
                forced = ["tools_register"]

    # Seaman doc
    if {"seaman","book","seamanbook","cdc"} & tokens:
        seaman_tables = [t for t in names if any(c.lower() == "seaman_doc" for c in catalog.get(t, []))]
        if seaman_tables:
            if "professional_details" in names and "professional_details" not in seaman_tables:
                forced = seaman_tables + ["professional_details"]
            else:
                forced = seaman_tables

    # Sales order
    uq = user_query.lower()
    if "sales order" in uq or "sales_order" in uq or "sales_order_no" in uq:
        so_tables = [t for t, cols in catalog.items() if any(c.lower() == "sales_order_no" for c in cols)]
        if so_tables:
            forced = list(dict.fromkeys(so_tables))
    return forced

# ------------- schema slicing ---------------------
def slice_schema(schema_text: str, user_query: str, max_tables: int = 14) -> Tuple[str, List[str], List[dict]]:
    obj = json.loads(schema_text)
    ctx = _get_ctx(obj)
    all_tables = ctx.get("tables") or []
    explicit = ctx.get("relationships") or []

    # prebuild maps
    catalog_full = {}
    types_full = {}
    for t in all_tables:
        tn = t.get("name")
        if not tn:
            continue
        catalog_full[tn] = [c.get("name") for c in (t.get("columns") or []) if c.get("name")]
        types_full[tn] = {c.get("name"): (c.get("type") or "").lower() for c in (t.get("columns") or []) if c.get("name")}

    # 1) Domain forcing
    forced = force_tables_for_query(user_query, all_tables, catalog_full, types_full)
    if forced:
        kept_names = set(forced)
        for r in explicit:
            st, tt = r.get("source_table"), r.get("target_table")
            if st in kept_names or tt in kept_names:
                kept_names.add(st); kept_names.add(tt)
        kept_tables = [t for t in all_tables if t.get("name") in kept_names]
        kept_rels = [r for r in explicit if r.get("source_table") in kept_names and r.get("target_table") in kept_names]
        sliced = {"context": {"tables": kept_tables}}
        if kept_rels:
            sliced["context"]["relationships"] = kept_rels
        return json.dumps(sliced, ensure_ascii=False), [t["name"] for t in kept_tables if t.get("name")], kept_rels

    # 2) Score-based slicer
    qwords = query_tokens(user_query)
    qnorms = {normalize_simple(w) for w in qwords}

    def score_table(t: dict) -> int:
        tname = (t.get("name") or "")
        cols = [c.get("name") or "" for c in (t.get("columns") or [])]
        score = 0
        t_tokens = tokenize_identifier(tname)
        if t_tokens & qwords: score += 6
        if normalize_simple(tname) in qnorms: score += 2
        bump_sea = 0; bump_cal = 0; bump_tool = 0
        for c in cols:
            c_tokens = tokenize_identifier(c)
            if c_tokens & qwords: score += 2
            if normalize_simple(c) in qnorms: score += 1
            for qw in qnorms:
                if qw and normalize_simple(c).find(qw) >= 0:
                    score += 1
            ctype = types_full.get(tname, {}).get(c, "")
            bump_sea = max(bump_sea, seaman_rank(c))
            bump_cal = max(bump_cal, calibration_rank(c, ctype))
            bump_tool = max(bump_tool, toolish_rank(c))
        score += min(bump_sea//5, 12) + min(bump_cal//4, 15) + min(bump_tool//10, 8)
        if ("tool" in qnorms or "calib" in qnorms or "calibration" in qnorms or "date" in qnorms) and tname.lower() == "tools_register":
            score += 6
        return score

    scored = [(score_table(t), t) for t in all_tables]
    scored.sort(key=lambda x: x[0], reverse=True)
    base_tables = [t for s,t in scored if s > 0][:max_tables]
    if not base_tables:
        all_tables_sorted = sorted(all_tables, key=lambda t: len(t.get("columns") or []))
        base_tables = all_tables_sorted[:min(6, len(all_tables_sorted))]

    kept_names = {t["name"] for t in base_tables if t.get("name")}
    for r in explicit:
        st, tt = r.get("source_table"), r.get("target_table")
        if st in kept_names or tt in kept_names:
            kept_names.add(st); kept_names.add(tt)

    kept_tables = [t for t in all_tables if t.get("name") in kept_names]
    kept_rels = [r for r in explicit if r.get("source_table") in kept_names and r.get("target_table") in kept_names]

    sliced = {"context": {"tables": kept_tables}}
    if kept_rels:
        sliced["context"]["relationships"] = kept_rels

    return json.dumps(sliced, ensure_ascii=False), [t["name"] for t in kept_tables if t.get("name")], kept_rels

# ------------- SQL parsing/validation -------------
_alias_pat = re.compile(
    r"\b(from|join)\s+([`\"a-zA-Z0-9_.]+)(?:\s+(?:as\s+)?([`\"a-zA-Z0-9_]+))?",
    re.IGNORECASE
)

def parse_aliases(sql: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for m in _alias_pat.finditer(sql):
        table_raw = m.group(2).strip('`"')
        alias_raw = m.group(3)
        table_name = table_raw.split(".")[-1]
        if alias_raw:
            alias_name = alias_raw.strip('`"')
            aliases[alias_name] = table_name
        aliases.setdefault(table_name, table_name)
    return aliases

def extract_qualified_cols(sql: str) -> List[Tuple[str,str]]:
    refs = []
    for m in re.finditer(r"([`\"a-zA-Z0-9_]+)\s*\.\s*([`\"a-zA-Z0-9_]+)", sql):
        a = m.group(1).strip('`"')
        c = m.group(2).strip('`"')
        refs.append((a,c))
    return refs

def extract_tables_from_sql(sql: str) -> List[str]:
    found = []
    for kw in ("from", "join"):
        for m in re.finditer(rf"\b{kw}\b\s+([`\"a-zA-Z0-9_.]+)", sql, flags=re.I):
            raw = m.group(1)
            tbl = raw.strip('`"').split(".")[-1]
            tbl = re.split(r"\s+as\s+|\s+", tbl, flags=re.I)[0]
            if tbl not in found:
                found.append(tbl)
    return found

SQL_KEYWORDS = {
    "select","from","where","join","inner","left","right","full","outer","on","and","or","not",
    "group","by","order","having","limit","offset","as","asc","desc","distinct","union","all",
    "case","when","then","else","end","with","exists","in","between","like","is","null","coalesce",
    "count","sum","avg","min","max","cast","convert","date","datetime","timestamp","year","month","day",
}

def _strip_strings_and_numbers(s: str) -> str:
    s = re.sub(r"'([^'\\]|\\.)*'", " ", s)
    s = re.sub(r'"([^"\\]|\\.)*"', " ", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", " ", s)
    return s

def _unqualified_tokens(s: str) -> List[str]:
    s = _strip_strings_and_numbers(s)
    toks = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", s):
        name = m.group(1)
        if name.lower() in SQL_KEYWORDS:
            continue
        after = s[m.end():]
        if after.lstrip().startswith("("):
            continue
        if after.startswith("."):
            continue
        before = s[:m.start()]
        if before.endswith("."):
            continue
        toks.append(name)
    return toks

def _segment(sql: str, start_kw: str, end_kws: List[str]) -> Optional[str]:
    m = re.search(rf"\b{start_kw}\b", sql, flags=re.I)
    if not m:
        return None
    start = m.end()
    end = len(sql)
    for kw in end_kws:
        m2 = re.search(rf"\b{kw}\b", sql[start:], flags=re.I)
        if m2:
            cand_end = start + m2.start()
            if cand_end < end:
                end = cand_end
    return sql[start:end]

def validate_tables_and_columns(sql: str, catalog: Dict[str, List[str]]) -> Tuple[bool, List[str], str]:
    errors: List[str] = []
    aliases = parse_aliases(sql)
    tables_used: List[str] = sorted(set(aliases.values()))

    unknown_tables = [t for t in tables_used if t not in catalog]
    if unknown_tables:
        errors.append(f"Unknown tables: {unknown_tables}")

    qrefs = extract_qualified_cols(sql)
    for a, c in qrefs:
        if a not in aliases:
            errors.append(f"Unknown alias/table reference '{a}' in '{a}.{c}'")
            continue
        table = aliases[a]
        cols = catalog.get(table, [])
        if c not in cols:
            sample = ", ".join(cols[:12]) + ("" if len(cols) <= 12 else ", …")
            errors.append(f"Unknown column '{c}' on table '{table}'. Allowed columns: {sample}")

    if len(tables_used) > 1:
        for clause, starters, stoppers in [
            ("SELECT", ["select"], ["from"]),
            ("WHERE", ["where"], ["group","order","having","limit","offset"]),
            ("GROUP BY", ["group"], ["order","having","limit","offset"]),
            ("HAVING", ["having"], ["order","limit","offset"]),
            ("ORDER BY", ["order"], ["limit","offset"]),
        ]:
            seg = None
            for st in starters:
                seg = _segment(sql, st, stoppers)
                if seg: break
            if not seg:
                continue
            tokens = _unqualified_tokens(seg)
            if tokens:
                errors.append(f"In {clause}, qualify columns with table/alias (found unqualified: {sorted(set(tokens))}).")
    else:
        table = tables_used[0] if tables_used else None
        if table and table in catalog:
            table_cols = set(catalog[table])
            for clause, starters, stoppers in [
                ("SELECT", ["select"], ["from"]),
                ("WHERE", ["where"], ["group","order","having","limit","offset"]),
                ("GROUP BY", ["group"], ["order","having","limit","offset"]),
                ("HAVING", ["having"], ["order","limit","offset"]),
                ("ORDER BY", ["order"], ["limit","offset"]),
            ]:
                seg = None
                for st in starters:
                    seg = _segment(sql, st, stoppers)
                    if seg: break
                if not seg:
                    continue
                tokens = _unqualified_tokens(seg)
                unknown = [t for t in tokens if t not in table_cols]
                if unknown:
                    sample = ", ".join(sorted(list(table_cols))[:16]) + ("" if len(table_cols) <= 16 else ", …")
                    errors.append(f"In {clause}, unknown column(s) for table '{table}': {sorted(set(unknown))}. Allowed: {sample}")

    if errors:
        lines = ["--- VALIDATION FEEDBACK ---",
                 "Fix the SQL using only identifiers present in the schema.",
                 "Aliases → tables mapping your SQL created:"]
        for a,t in aliases.items():
            lines.append(f"  {a} → {t}")
        lines.append("\nAllowed columns per table (subset):")
        for t, cols in catalog.items():
            sample = ", ".join(cols[:16]) + ("" if len(cols) <= 16 else ", …")
            lines.append(f"  {t}: {sample}")
        lines.append("\nErrors:")
        for e in errors:
            lines.append(f"  - {e}")
        feedback = "\n".join(lines)
        return False, errors, feedback

    return True, [], ""

# ------------- helpers for examples ---------------
def guess_name_column(catalog: Dict[str, List[str]], table: str) -> str:
    prefs = ["Customer_Name","Name","Full_Name","Employee_Name","Professional_Name","Title","Description","Instrument_Type"]
    cols = catalog.get(table, [])
    for p in prefs:
        if p in cols:
            return p
    for c in cols:
        if "name" in c.lower() or "title" in c.lower() or "description" in c.lower():
            return c
    return cols[0] if cols else "S_No"

def find_explicit_join(a: str, b: str, rels: List[Dict[str,str]]) -> Optional[Tuple[str,str]]:
    for r in rels:
        st, sc, tt, tc = r.get("source_table"), r.get("source_column"), r.get("target_table"), r.get("target_column")
        if st == a and tt == b:
            return f"p.{sc} = x.{tc}", r.get("type","")
        if st == b and tt == a:
            return f"p.{tc} = x.{sc}", r.get("type","")
    return None

def month_year_predicate(table: str, col: str, coltype: str, month_year: Optional[Tuple[int,int,str]]) -> str:
    if not month_year:
        return "1=1"
    m, y, disp = month_year
    if any(k in coltype for k in ["date","datetime","timestamp"]):
        return f"YEAR({table}.{col}) = {y} AND MONTH({table}.{col}) = {m}"
    return f"{table}.{col} LIKE '%{disp}%'"

# ---------- use schema-provided examples (priority) ----------
def extract_schema_examples(schema_text: str) -> List[Dict[str, Any]]:
    """Supports context.examples or context.example_queries with {question, sql_query}."""
    try:
        obj = json.loads(schema_text)
        ctx = _get_ctx(obj)
    except Exception:
        return []
    raw = ctx.get("examples") or ctx.get("example_queries") or []
    out: List[Dict[str, Any]] = []
    for it in raw:
        q = it.get("question") or it.get("query") or it.get("user") or ""
        sql = it.get("sql_query") or it.get("sql") or ""
        if not q or not sql:
            continue
        tabs = extract_tables_from_sql(sql)
        out.append({
            "user": q,
            "json": {"tool_name":"mysql.query","params":{"sql": sql if sql.strip().endswith(";") else sql.strip()+";","tables": tabs},"confidence":0.95}
        })
    return out

# ------------- dynamic few-shots ------------------
def build_examples_json(user_query: str,
                        kept_tables: List[str],
                        catalog: Dict[str, List[str]],
                        types: Dict[str, Dict[str, str]],
                        explicit_rels: List[Dict[str,str]],
                        schema_examples: List[Dict[str, Any]]) -> str:
    q_month = parse_month_year(user_query)
    ex: List[Dict[str, Any]] = []

    # 0) schema-provided examples first (highest priority)
    ex.extend(schema_examples[:20])

    # 1) per-table basics
    for t in kept_tables:
        cols = catalog.get(t, [])
        ttypes = types.get(t, {})

        ex.append({
            "user": f"how many rows are in {t}",
            "json": {"tool_name":"mysql.query","params":{"sql":f"SELECT COUNT(*) AS total_count FROM {t};","tables":[t]},"confidence":0.85}
        })

        name_col = guess_name_column(catalog, t)
        ex.append({
            "user": f"show sample names in {t}",
            "json": {"tool_name":"mysql.query","params":{"sql":f"SELECT {name_col} FROM {t} LIMIT 20;","tables":[t]},"confidence":0.8}
        })

        # non-null doc/cert/file
        for c in cols:
            cn = c.lower()
            if any(k in cn for k in ["doc","file","path","certificate","cert","upload"]):
                ex.append({
                    "user": f"list rows in {t} where {c} exists",
                    "json": {"tool_name":"mysql.query","params":{"sql":f"SELECT * FROM {t} WHERE {t}.{c} IS NOT NULL AND {t}.{c} <> '' LIMIT 50;","tables":[t]},"confidence":0.85}
                })

        # qty-like numeric
        for c in cols:
            ctype = ttypes.get(c,"")
            if any(k in c.lower() for k in ["qty","quantity","balance","pending"]) or any(k in ctype for k in ["int","decimal","float","double","numeric"]):
                ex.append({
                    "user": f"show positive values for {t}.{c}",
                    "json": {"tool_name":"mysql.query","params":{"sql":f"SELECT {c} FROM {t} WHERE CAST(NULLIF({c},'') AS DECIMAL(18,4)) > 0 LIMIT 50;","tables":[t]},"confidence":0.8}
                })

        # month-year filters for date-like columns
        for c in cols:
            ctype = ttypes.get(c,"")
            if "date" in c.lower() or any(k in ctype for k in ["date","datetime","timestamp"]):
                pred = month_year_predicate(t, c, ctype, q_month or (4,2025,"April 2025"))
                ex.append({
                    "user": f"filter {t} by {c} in April 2025",
                    "json": {"tool_name":"mysql.query","params":{"sql":f"SELECT {c} FROM {t} WHERE {pred} LIMIT 50;","tables":[t]},"confidence":0.86}
                })

    # 2) explicit relationship join samples
    for r in explicit_rels[:12]:
        a, ac, b, bc = r["source_table"], r["source_column"], r["target_table"], r["target_column"]
        if a in kept_tables and b in kept_tables:
            a_name = guess_name_column(catalog, a)
            b_name = guess_name_column(catalog, b)
            ex.append({
                "user": f"join {a} and {b}",
                "json": {"tool_name":"mysql.query","params":{"sql":f"SELECT p.{a_name}, x.{b_name} FROM {a} AS p JOIN {b} AS x ON p.{ac} = x.{bc} LIMIT 20;","tables":[a,b],"notes":f"Explicit: {a}.{ac} = {b}.{bc}"},"confidence":0.86}
            })

    # 3) seaman presence example
    sea_hits = []
    for t in kept_tables:
        for c in catalog.get(t, []):
            rank = seaman_rank(c)
            if rank > 0:
                sea_hits.append((rank, t, c))
    sea_hits.sort(reverse=True)
    if sea_hits:
        _, t, c = sea_hits[0]
        join = find_explicit_join("professional_details", t, explicit_rels)
        if join and "professional_details" in kept_tables:
            name_col = guess_name_column(catalog, "professional_details")
            ex.append({
                "user": "list all professionals who have seaman book",
                "json": {"tool_name":"mysql.query","params":{
                    "sql":f"SELECT p.{name_col}, x.{c} FROM professional_details AS p JOIN {t} AS x ON {join[0]} WHERE x.{c} IS NOT NULL AND x.{c} <> '';",
                    "tables":["professional_details", t],
                    "notes":"seaman doc presence via explicit relationship"},"confidence":0.9}
            })
        else:
            ex.append({
                "user": "list all professionals who have seaman book",
                "json": {"tool_name":"mysql.query","params":{
                    "sql":f"SELECT * FROM {t} WHERE {t}.{c} IS NOT NULL AND {t}.{c} <> '' LIMIT 50;",
                    "tables":[t]},"confidence":0.85}
            })

    # 4) calibration April 2025 → prefer tools_register
    if "tools_register" in kept_tables:
        cols = catalog["tools_register"]; ttypes = types.get("tools_register", {})
        best_col = None; best_score = -1
        for c in cols:
            s = calibration_rank(c, ttypes.get(c,""))
            if s > best_score and ("date" in c.lower() or "cal" in c.lower() or "due" in c.lower()):
                best_col = c; best_score = s
        tool_col = "Description" if "Description" in cols else ("Instrument_Type" if "Instrument_Type" in cols else guess_name_column(catalog,"tools_register"))
        pred = month_year_predicate("tools_register", best_col or "Cal_Date", ttypes.get(best_col or "Cal_Date",""), parse_month_year(user_query) or (4,2025,"April 2025"))
        ex.append({
            "user": "list down the tools which calibration date is this april 2025",
            "json": {"tool_name":"mysql.query","params":{
                "sql":f"SELECT tools_register.{tool_col}, tools_register.{best_col or 'Cal_Date'} FROM tools_register WHERE {pred};",
                "tables":["tools_register"],
                "notes":"prefer tools_register for calibration"}, "confidence":0.92}
        })

    if len(ex) > 80:
        ex = ex[:80]
    return json.dumps(ex, ensure_ascii=False)

# ---------------- Prompts -------------------------
PLANNER_SYSTEM_TEXT = (
    "You are Ergontec’s MySQL SQL writer.\n\n"
    "Return STRICT JSON only:\n"
    "{\"tool_name\":\"...\", \"params\":{...}, \"confidence\": <0..1>, \"rationale\": \"...\"}\n\n"
    "Rules:\n"
    "- tool_name ∈ {\"mysql.query\",\"no_action\"}.\n"
    "- Use ONLY identifiers (table/column names) present in the provided schema JSON, EXACT spelling/casing.\n"
    "- Prefer EXPLICIT RELATIONSHIPS from the schema JSON for JOINs (e.g., A.x ↔ B.y). If none apply, you may use shared columns present in both tables (case-insensitive hints are provided).\n"
    "- When joining two or more tables, FULLY QUALIFY all column references as table_or_alias.column.\n"
    "- Produce exactly ONE read-only SQL statement (SELECT / WITH ... SELECT / SHOW / DESCRIBE). Never write operations.\n"
    "- If ambiguous or not answerable from schema, return:\n"
    "  {\"tool_name\":\"no_action\",\"params\":{\"reason\":\"<short reason>\"},\"confidence\":0,\"rationale\":\"<why>\"}\n"
    "- When returning mysql.query:\n"
    "  - params.sql: the SQL string (end with semicolon)\n"
    "  - params.tables: array of tables used (exact names from schema)\n"
    "  - params.notes: optional short note\n"
    "Be concise and deterministic.\n"
    "\n"
    "Hard requirements for Sales Order queries:\n"
    "- If more than one table has Sales_Order_No AND a relationship exists (e.g., project_details.Sales_Order_No ↔ project_checklist.Sales_Order_No), you MUST join them.\n"
)

PLANNER_USER_TEMPLATE = """User query:
{user_query}

Tools:
- mysql.query(sql, tables[], notes?)
- no_action(reason)

Sliced Schema (JSON):
{schema_json}

Relationship guide (EXPLICIT first, then shared-column hints):
{relationship_guide}

Examples (schema-provided first, then dynamic; JSON only):
{examples_json}

Your JSON response only (no markdown, no fences):
"""

REPAIR_USER_TEMPLATE = """Your previous JSON tool call used invalid columns/tables or unqualified columns.

Fix it now. Keep the same constraints:
- EXACT identifiers only from the schema JSON.
- Use JOINs based on the EXPLICIT relationships first (preferred), then shared-column hints if necessary.
- Fully qualify columns when multiple tables are used.
- One read-only statement only.

Validation feedback to address:
{validator_feedback}

Return corrected STRICT JSON now:
"""

# ---------------- JSON helpers (robust + stabilizer) --------------------
def _extract_first_json_object(text: str) -> Dict[str, Any]:
    # try straight parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # try fenced ```json block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # scan for first balanced {...}
    s = text
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = s[start:i+1]
                        try:
                            return json.loads(cand)
                        except Exception:
                            break
        start = s.find("{", start + 1)

    raise ValueError("LLM did not return valid JSON.")

def force_json(text: str) -> Dict[str, Any]:
    return _extract_first_json_object(text)

# ---------------- LLM call helper (+ reprompt) -----------------
def _invoke_llm(llm, system_text: str, human_text: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_text}"),
        ("human", "{human_text}"),
    ])
    msg = prompt | llm
    raw = msg.invoke({"system_text": system_text, "human_text": human_text})
    return getattr(raw, "content", str(raw))

def llm_plan(llm, system_text: str, human_text: str) -> Dict[str, Any]:
    """
    Try normal call → parse JSON.
    If it fails, REPROMPT with a tiny instruction to output ONLY valid JSON.
    """
    text = _invoke_llm(llm, system_text, human_text)
    try:
        return force_json(text)
    except Exception:
        # Reprompt to force JSON only (short coercer)
        coercer = (
            "Return ONLY valid JSON with keys: tool_name, params, confidence, rationale.\n"
            "No markdown. No extra text. If unsure, respond with:\n"
            "{\"tool_name\":\"no_action\",\"params\":{\"reason\":\"unsure\"},\"confidence\":0,\"rationale\":\"coercer\"}"
        )
        text2 = _invoke_llm(llm, "You must output strict JSON.", coercer + "\n\nPrevious content (for reference only):\n" + text)
        return force_json(text2)

# ---------------- Deterministic SALES ORDER fallback ------------
def deterministic_sales_order(user_query: str, catalog: Dict[str, List[str]], rels: List[Dict[str,str]]) -> Optional[Dict[str, Any]]:
    uq = user_query.lower()
    if not ("sales order" in uq or "sales_order" in uq or "sales_order_no" in uq):
        return None

    # find tables with Sales_Order_No
    tables_with_so = [t for t, cols in catalog.items() if "Sales_Order_No" in cols]
    if not tables_with_so:
        return None

    # prefer project_details + project_checklist if relationship present
    has_pd = "project_details" in tables_with_so
    has_pc = "project_checklist" in tables_with_so
    joined = False
    sql = None
    tabs: List[str] = []

    if has_pd and has_pc:
        # verify explicit relationship exists
        rel_ok = any((r.get("source_table")=="project_details" and r.get("target_table")=="project_checklist" and
                      r.get("source_column")=="Sales_Order_No" and r.get("target_column")=="Sales_Order_No")
                     or
                     (r.get("source_table")=="project_checklist" and r.get("target_table")=="project_details" and
                      r.get("source_column")=="Sales_Order_No" and r.get("target_column")=="Sales_Order_No")
                     for r in rels)
        if rel_ok:
            # pick a couple of visible columns in checklist if present
            checklist_cols = catalog.get("project_checklist", [])
            pick = [c for c in checklist_cols if c.startswith("Item")] or checklist_cols[:2] or ["Sales_Order_No"]
            cols_sql = ", ".join(["pd.Sales_Order_No"] + [f"pc.{c}" for c in pick])
            sql = f"SELECT {cols_sql} FROM project_details AS pd JOIN project_checklist AS pc ON pd.Sales_Order_No = pc.Sales_Order_No;"
            tabs = ["project_details", "project_checklist"]
            joined = True

    if not sql:
        # if we cannot join, at least list rows that have Sales_Order_No somewhere
        t = tables_with_so[0]
        sql = f"SELECT * FROM {t} WHERE {t}.Sales_Order_No IS NOT NULL;"
        tabs = [t]

    return {
        "tool_name": "mysql.query",
        "params": {"sql": sql, "tables": tabs, "notes": "deterministic Sales_Order_No route" if joined else "single-table Sales_Order_No presence"},
        "confidence": 0.92 if joined else 0.75,
        "rationale": "deterministic fallback"
    }

# ---------------- Router --------------------------
def route(user_query: str, schema_text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    # slice schema (with domain forcing)
    sliced_schema_json, kept_table_names, kept_rels = slice_schema(schema_text, user_query, max_tables=14)

    catalog = build_schema_catalog(sliced_schema_json)
    types = build_type_map(sliced_schema_json)
    explicit_rels = kept_rels
    shared_rels = shared_columns_map(catalog)
    rel_text = relationships_text(explicit_rels, shared_rels)

    # schema-provided examples (from your db.json) + dynamic ones
    schema_examples = extract_schema_examples(schema_text)
    examples_json = build_examples_json(user_query, kept_table_names, catalog, types, explicit_rels, schema_examples)

    llm = init_llm(model_name or os.getenv("GEMINI_ROUTER_MODEL"))

    # 1) plan
    human_text = PLANNER_USER_TEMPLATE.format(
        user_query=user_query,
        schema_json=sliced_schema_json,
        relationship_guide=rel_text,
        examples_json=examples_json,
    )

    try:
        data = llm_plan(llm, PLANNER_SYSTEM_TEXT, human_text)
    except Exception:
        # LAST-RESORT deterministic SALES ORDER fallback if applicable
        fallback = deterministic_sales_order(user_query, catalog, explicit_rels)
        if fallback:
            return fallback
        # else surface the error
        return {"tool_name": "no_action", "params": {"reason": "LLM did not return valid JSON."}, "confidence": 0.0, "rationale": "exception"}

    # 2) validate & optional repair loop
    for attempt in range(3):
        tool = (data.get("tool_name") or "").strip()
        params = data.get("params") or {}

        if tool not in {"mysql.query", "no_action"}:
            # deterministic fallback for sales order
            fallback = deterministic_sales_order(user_query, catalog, explicit_rels)
            if fallback:
                return fallback
            return {"tool_name": "no_action", "params": {"reason": f"invalid tool_name '{tool}'"}, "confidence": 0.0, "rationale": "validation"}

        if tool == "no_action":
            # deterministic fallback for sales order
            fallback = deterministic_sales_order(user_query, catalog, explicit_rels)
            if fallback:
                return fallback
            return {
                "tool_name": "no_action",
                "params": {"reason": params.get("reason") or "unspecified"},
                "confidence": float(data.get("confidence", 0)) if isinstance(data.get("confidence", 0), (int, float)) else 0.0,
                "rationale": data.get("rationale", "LLM declined")
            }

        # mysql.query
        sql = (params.get("sql") or "").strip()
        if not sql:
            return {"tool_name": "no_action", "params": {"reason": "missing sql"}, "confidence": 0.0, "rationale": "validation"}
        if not sql.endswith(";"):
            sql += ";"
            params["sql"] = sql
        if not is_read_only(sql):
            return {"tool_name": "no_action", "params": {"reason": "non read-only or multi-statement SQL"}, "confidence": 0.0, "rationale": "safety"}

        ok, errs, feedback = validate_tables_and_columns(sql, catalog)
        if ok:
            aliases = parse_aliases(sql)
            final_tables = sorted(set(aliases.values())) or (params.get("tables") or [])
            if not final_tables:
                final_tables = sorted(set(extract_tables_from_sql(sql)))
            params["tables"] = final_tables
            return {
                "tool_name": "mysql.query",
                "params": {"sql": sql, "tables": final_tables, "notes": params.get("notes", "")},
                "confidence": float(data.get("confidence", 0.9)) if isinstance(data.get("confidence", 0.9), (int, float)) else 0.9,
                "rationale": data.get("rationale", "validated")
            }

        if attempt < 2:
            # Ask the LLM to repair with the validator feedback
            repair_human = REPAIR_USER_TEMPLATE.format(validator_feedback=feedback)
            try:
                data = llm_plan(llm, PLANNER_SYSTEM_TEXT, repair_human)
                continue
            except Exception:
                # if the LLM bails here, try deterministic fallback for sales order
                fallback = deterministic_sales_order(user_query, catalog, explicit_rels)
                if fallback:
                    return fallback
                return {"tool_name": "no_action", "params": {"reason": "LLM did not return valid JSON."}, "confidence": 0.0, "rationale": "exception"}

        return {"tool_name": "no_action", "params": {"reason": "validation failed: " + "; ".join(errs)}, "confidence": 0.0, "rationale": "invalid SQL per schema"}

# ---------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Natural language question")
    ap.add_argument("--schema_path", required=True, help="Path to db.json")
    ap.add_argument("--sql_only", action="store_true", help="Print only SQL when mysql.query is chosen")
    ap.add_argument("--print_json", action="store_true", help="Print final JSON tool call")
    ap.add_argument("--model", default=None, help="Gemini model name (or set GEMINI_ROUTER_MODEL in .env)")
    args = ap.parse_args()

    try:
        with open(args.schema_path, "r", encoding="utf-8") as f:
            schema_text = f.read()
    except Exception as e:
        print(f"Failed to read schema at {args.schema_path}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        out = route(args.query, schema_text, model_name=args.model)
    except Exception as e:
        out = {"tool_name": "no_action", "params": {"reason": f"router error: {e}"},
               "confidence": 0.0, "rationale": "exception"}

    if args.sql_only and out.get("tool_name") == "mysql.query":
        print(out["params"]["sql"].strip()); return

    if args.print_json:
        print(json.dumps(out, ensure_ascii=False))
        if out.get("tool_name") == "mysql.query":
            print("\n[mysql.query] SQL:\n" + out["params"]["sql"])
            print("Tables:", out["params"]["tables"])
            note = out["params"].get("notes")
            if note:
                print("Notes:", note)
    else:
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Set GOOGLE_API_KEY in your environment or .env file.", file=sys.stderr)
        sys.exit(2)
    main()
