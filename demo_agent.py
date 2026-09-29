"""demo_agent.py - Machine Memory Agent core (Hindsight recall/retain + Groq)."""
import os, re, json, datetime, threading, time
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_ID = os.environ.get("BANK_ID", "factory-floor-v2")
LLM_MODEL = "openai/gpt-oss-120b"
MCH_RE = re.compile(r"MCH-\d+")
ID_RE = re.compile(r"(?:MAINT-\d{4}-\d{3}|LIVE-\d{3})")

with open(os.path.join(HERE, "machine_maintenance_history.json")) as f:
    HISTORY = json.load(f)
BY_ID = {r["id"]: r for r in HISTORY}
KNOWN = {r["machine_id"] for r in HISTORY}
TYPE_BY_MACHINE = {r["machine_id"]: r["machine_type"] for r in HISTORY}
LIVE_FILE = os.path.join(HERE, "live_incidents.json")
_live_lock = threading.Lock()

# ASSUMPTION: downtime cost varies a lot by asset class - a stalled CNC line idles paid
# operators and a full production cell, a chiller running warm rarely stops anything outright.
# Matched by substring against machine_type so new machine types fall back to a plant-wide default
# instead of silently using whatever the first machine on the fleet happens to cost.
COST_PER_HOUR_DEFAULT = 3500
COST_PER_HOUR_BY_TYPE = {
    "CNC Milling Machine": 6000, "Industrial Robot Arm": 7000, "Hydraulic Press": 5000,
    "Industrial Air Compressor": 4200, "Centrifugal Pump": 3500, "Industrial Chiller": 3000,
    "Conveyor Belt Motor": 2200,
}

def cost_per_hour(machine):
    t = TYPE_BY_MACHINE.get(machine, "")
    for key, rate in COST_PER_HOUR_BY_TYPE.items():
        if key in t: return rate
    return COST_PER_HOUR_DEFAULT

def friendly_error(e):
    """Translate a raw client exception into something a technician can act on,
    while the real message still reaches the server log for debugging."""
    print(f"[memory service error] {e}")
    msg = str(e).lower()
    if "timeout" in msg: return "Memory service timed out. Try again in a moment."
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
        return "Memory service rejected the API key. Check HINDSIGHT_API_KEY/GROQ_API_KEY."
    if "429" in msg or "rate" in msg or "capacity" in msg:
        return "Memory service is at capacity. Wait a few seconds and retry."
    if "connection" in msg or "resolve" in msg or "network" in msg:
        return "Could not reach the memory service. Check your network connection."
    return f"Memory service error: {str(e)[:200]}"

TRANSIENT_MARKERS = ("timeout", "429", "503", "capacity", "connection", "temporarily", "task")

def _is_transient(e):
    """'task' catches the asyncio 'Timeout context manager should be used inside a
    task' class of error - a real one we hit once under load and never fully pinned
    down; retrying once with a brand-new client is cheap insurance against it."""
    m = str(e).lower()
    return any(k in m for k in TRANSIENT_MARKERS)

def call_with_retry(fn, attempts=2, delay=1.2):
    """Call fn() (a zero-arg thunk making one client call) up to `attempts` times,
    retrying only errors that look transient. Never retries auth/permission errors -
    those won't fix themselves on a second try."""
    last = None
    for i in range(attempts):
        try:
            return fn(), None
        except Exception as e:
            last = e
            if i < attempts - 1 and _is_transient(e):
                print(f"[memory service retry {i+1}/{attempts-1}] {e}")
                time.sleep(delay)
                continue
            return None, e
    return None, last

_g = None
def hs():
    # A fresh client per call, not a cached global: the sync client wraps each call in
    # its own asyncio event loop, and a cached aiohttp session from an earlier loop
    # breaks ("Timeout context manager should be used inside a task") once reused
    # under Flask's threaded dev server. Construction itself is cheap (no I/O).
    from hindsight_client import Hindsight
    key = os.environ.get("HINDSIGHT_API_KEY")
    if not key:
        raise RuntimeError("HINDSIGHT_API_KEY is not set. Export it and restart.")
    return Hindsight(base_url=os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io"), api_key=key)

def groq():
    global _g
    if _g is None:
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Export it and restart.")
        _g = Groq(api_key=key)
    return _g

def llm(system, user, json_mode=False):
    kw = dict(model=LLM_MODEL, temperature=0.2,
              messages=[{"role": "system", "content": system + ("\nRespond only with valid raw JSON." if json_mode else "")},
                        {"role": "user", "content": user}])
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    out, e = call_with_retry(lambda: groq().chat.completions.create(**kw).choices[0].message.content)
    if e:
        return {"error": friendly_error(e)} if json_mode else f"({friendly_error(e)})"
    try:
        return json.loads(out) if json_mode else out
    except Exception as e:
        return {"error": f"Model returned invalid JSON: {e}"} if json_mode else out

# ---------- live incidents (persisted, flagged unreviewed) ----------
def load_live():
    try:
        with open(LIVE_FILE) as f: return json.load(f)
    except Exception: return []

def save_live(items):
    with open(LIVE_FILE, "w") as f: json.dump(items, f, indent=1)

def reset_live():
    with _live_lock:
        save_live([])

# ---------- CMMS outbox (demo) ----------
# Stands in for a real CMMS integration (Fiix, UpKeep, Maximo, SAP PM): a real deployment
# posts cmms_prevention_draft to that system's work-order API behind a webhook instead of
# this local file, and a reliability engineer approves it there rather than here. This
# makes the integration's *shape* demoable now - draft -> queued work order, visible in a
# queue - without pretending a real CMMS is on the other end of the wire.
CMMS_FILE = os.path.join(HERE, "cmms_outbox.json")
_cmms_lock = threading.Lock()

def load_outbox():
    try:
        with open(CMMS_FILE) as f: return json.load(f)
    except Exception: return []

def save_outbox(items):
    with open(CMMS_FILE, "w") as f: json.dump(items, f, indent=1)

def reset_outbox():
    with _cmms_lock:
        save_outbox([])

def send_to_cmms(machine, draft, source="pre-repair check"):
    if machine not in KNOWN:
        return {"error": f"Unknown machine {machine}."}
    if not (draft or "").strip():
        return {"error": "No prevention draft to send."}
    now = datetime.datetime.now(datetime.timezone.utc)
    with _cmms_lock:
        items = load_outbox()
        rec = {"id": f"WO-{now.year}-{len(items)+1:04d}", "machine_id": machine, "draft": draft.strip(),
               "source": source, "status": "pending_approval", "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "system": "demo outbox (would be Fiix/UpKeep/Maximo/SAP PM in production)"}
        items.append(rec); save_outbox(items)
    return rec

# ---------- memory ----------
def recall(query, machine=None, n=8):
    q = f"{machine} {query}" if machine and machine not in query.upper() else query
    r, e = call_with_retry(lambda: hs().recall(bank_id=BANK_ID, query=q, max_tokens=1500, budget="low"))
    if e: return [], friendly_error(e)
    return [x.text for x in (getattr(r, "results", None) or [])[:n]], None

def retain(content, context, timestamp=None, tags=None):
    kw = dict(bank_id=BANK_ID, content=content, context=context)
    if timestamp: kw["timestamp"] = timestamp
    if tags: kw["tags"] = tags
    _, e = call_with_retry(lambda: hs().retain(**kw))
    if e: return False, friendly_error(e)
    return True, None

def reflect(query, context=None, tags=None, response_schema=None, budget="mid"):
    r, e = call_with_retry(lambda: hs().reflect(bank_id=BANK_ID, query=query, context=context, tags=tags,
                                                 tags_match="any", budget=budget, max_tokens=1600,
                                                 response_schema=response_schema, include_facts=True))
    if e: return None, friendly_error(e)
    return r, None

def confidence(query, texts):
    ids = set(MCH_RE.findall(query.upper()))
    if ids and not (ids & KNOWN):
        return {"score": 0, "label": "0% match", "type": "Unknown machine: no fleet history"}
    if not texts:
        return {"score": 0, "label": "0% match", "type": "No matching memory"}
    blob = " ".join(texts)
    rec_m = set(MCH_RE.findall(blob))
    recs = {i for i in ID_RE.findall(blob) if i in BY_ID}
    score = 25
    if ids and ids & rec_m: score += 40; kind = "Same-machine recurrence"
    elif ids: score += 5; kind = "Weak: machine not in recalled memory"
    else: score += 20; kind = "Symptom / cross-fleet correlation"
    score += min(20, len(recs) * 7)
    dates = sorted(r["date"] for r in HISTORY)
    if any(BY_ID[i]["date"] >= dates[-4] for i in recs): score += 10
    score = min(95, score)
    return {"score": score, "label": f"{score}% match", "type": kind}

def sources_from(texts):
    seen = []
    for i in ID_RE.findall(" ".join(texts)):
        if i in BY_ID and i not in seen: seen.append(i)
    return [dict(id=i, machine=BY_ID[i]["machine_id"], date=BY_ID[i]["date"], tech=BY_ID[i]["technician"],
                 downtime=BY_ID[i]["downtime_hours"], root_cause=BY_ID[i]["root_cause"]) for i in seen]

def cross_fleet_from(texts, machine, n=3):
    """Cross-machine learning: other machines Hindsight itself surfaced as relevant."""
    out, seen = [], set()
    for i in ID_RE.findall(" ".join(texts)):
        if i not in BY_ID: continue
        r = BY_ID[i]
        if machine and r["machine_id"] == machine: continue
        if r["machine_id"] in seen: continue
        seen.add(r["machine_id"]); out.append(r)
        if len(out) >= n: break
    return out

def ledger(machine):
    return [x for x in load_live() if x["machine_id"] == machine]

# ---------- diagnosis ----------
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "triage_level": {"type": "string", "enum": ["critical", "warning", "nominal", "novel"]},
        "badge_label": {"type": "string", "description": "short label, e.g. REPEAT FAILURE"},
        "primary_machine": {"type": "string", "description": "MCH-XXX or None"},
        "verdict_summary": {"type": "string", "description": "2 sentences"},
        "dead_end_warning": {"type": "string", "description": "failed past fixes and who tried them, or empty string"},
        "root_cause": {"type": "string", "description": "systemic reason it recurs"},
        "action_checklist": {"type": "array", "items": {"type": "string"}, "description": "steps with exact tolerances/part numbers"},
        "citations": {"type": "array", "items": {"type": "string"}, "description": "record IDs used, e.g. MAINT-2025-001"},
        "cmms_prevention_draft": {"type": "string", "description": "schedule/checklist change that prevents recurrence"},
    },
    "required": ["triage_level", "badge_label", "primary_machine", "verdict_summary", "root_cause", "action_checklist", "citations", "cmms_prevention_draft"],
}
INSTRUCTIONS = ("You are the Chief Reliability Engineer for a plant, reasoning over this bank's maintenance memory. "
                "Decide whether the symptoms match a known failure pattern for this machine, or a pattern seen elsewhere "
                "in the fleet. State only what is grounded in retrieved memory; if memory is thin, say so plainly. "
                "Never invent record IDs, technician names, dates, or part numbers.")

def check_before_repair(symptoms, lang="English"):
    ids = MCH_RE.findall(symptoms.upper())
    machine = ids[0] if ids else None
    if machine and machine not in KNOWN:
        return {"machine": machine, "confidence": confidence(symptoms, []), "sources": [], "cross_fleet": [], "downtime_hours": 0,
                "financial_risk_usd": 0, "cost_per_hour_used": COST_PER_HOUR_DEFAULT,
                "agent": {"triage_level": "novel", "badge_label": "NOVEL SIGNATURE", "primary_machine": "None",
                          "verdict_summary": f"{machine} has no history in memory. No grounded diagnosis is possible; follow the OEM manual and log the outcome so the fleet learns from it.",
                          "dead_end_warning": "", "root_cause": "Unknown", "action_checklist": ["Isolate the machine safely", "Follow OEM troubleshooting", "Log the result using Fix worked / didn't work"],
                          "citations": [], "cmms_prevention_draft": ""}}
    led = ledger(machine) if machine else []
    led_txt = "\n".join(f"- {x['id']} ({'unreviewed' if not x['reviewed'] else 'reviewed'}) fix '{x['fix']}' {'WORKED' if x['worked'] else 'FAILED - do not repeat'}" for x in led) or "none"
    query = (f"{INSTRUCTIONS}\n\nSYMPTOMS: {symptoms}\n\nWrite all text values in {lang}; keep IDs and part numbers unchanged.")
    ctx = f"TECHNICIAN OUTCOME LEDGER for this machine (recent feedback, may not be indexed into recall yet):\n{led_txt}"
    r, err = reflect(query, context=ctx, response_schema=RESPONSE_SCHEMA)
    if err:
        a, facts, mental_models, directives = {"error": err}, [], [], []
    else:
        a = dict(r.structured_output or {})
        if r.structured_output_error: a["error"] = r.structured_output_error
        facts = (r.based_on.memories if r.based_on else None) or []
        mental_models = (r.based_on.mental_models if r.based_on else None) or []
        directives = (r.based_on.directives if r.based_on else None) or []
    texts = [f.text for f in facts]
    # fact.text is Hindsight's extracted/paraphrased summary and often drops the literal
    # [MAINT-2025-XXX] tag; fact.context is the raw context string we retained it with,
    # which reliably carries the record ID. Use both when matching IDs, text alone for display.
    id_blob = texts + [f.context for f in facts if f.context]
    # reflect() sometimes answers straight from this machine's digital twin (mental model)
    # instead of the underlying fact snippets - it's synthesized from exactly that machine's
    # retained records, so when it's used, those records are legitimately "sources" too.
    if machine and any(m.id == twin_id(machine) for m in mental_models):
        id_blob += [f"[{rid}] machine {machine}" for rid in BY_ID if BY_ID[rid]["machine_id"] == machine]
    conf = confidence(symptoms, id_blob)
    srcs = sources_from(id_blob)
    cross = cross_fleet_from(id_blob, machine)
    allowed = {s["id"] for s in srcs} | {c["id"] for c in cross} | {x["id"] for x in led}
    cites = [c for c in (a.get("citations") or []) if isinstance(c, str) and any(c.startswith(i) for i in allowed)]
    a["citations"] = [next(i for i in allowed if c.startswith(i)) for c in cites]
    # the model can only cite an ID it literally saw; when it answered from the digital twin's
    # prose alone (no bracketed IDs in it) citations comes back empty even though srcs is populated -
    # fall back to srcs so downtime/financial risk isn't understated for a well-grounded answer.
    hrs = max([BY_ID[c]["downtime_hours"] for c in a["citations"] if c in BY_ID] or [s["downtime"] for s in srcs] or [0])
    rate = cost_per_hour(machine)
    return {"machine": machine, "agent": a, "confidence": conf, "sources": srcs, "raw_recall": texts, "recall_error": err,
            "cross_fleet": [dict(id=r["id"], machine=r["machine_id"], type=r["machine_type"], cause=r["root_cause"]) for r in cross],
            "downtime_hours": hrs, "financial_risk_usd": hrs * rate, "cost_per_hour_used": rate,
            "mental_models_used": [dict(id=m.id, text=m.text) for m in mental_models],
            "directives_applied": [dict(id=d.id, name=d.name, content=d.content) for d in directives]}

# ---------- digital twin (per-machine mental model) ----------
def twin_id(machine):
    return f"digital-twin-{machine.lower()}"

def digital_twin(machine):
    if machine not in KNOWN:
        return {"error": f"Unknown machine {machine}."}
    try:
        mm = hs().get_mental_model(bank_id=BANK_ID, mental_model_id=twin_id(machine), detail="content")
        content = mm.content or ""
        generating = "Generating content" in content
        return {"machine": machine, "content": None if generating else content,
                "generating": generating, "last_refreshed": mm.last_refreshed_at, "is_stale": mm.is_stale}
    except Exception as e:
        return {"error": str(e)}

def compare(question):
    texts, _ = recall(question, (MCH_RE.findall(question.upper()) or [None])[0])
    sysm = "You are an industrial maintenance assistant. Answer concisely (max 120 words) with a diagnosis and next steps."
    return {"question": question, "confidence": confidence(question, texts),
            "without_memory": llm(sysm, question),
            "with_memory": llm(sysm + " Use the plant history provided; cite record IDs and mention failed past fixes.",
                               f"{question}\n\nPLANT HISTORY:\n" + ("\n".join(f"- {t}" for t in texts) or "none"))}

# ---------- closed loop + improvement ----------
def log_feedback(machine, fix, worked, tech="Technician"):
    if machine not in KNOWN:
        return {"retained": False, "error": f"Unknown machine {machine}."}
    now = datetime.datetime.now(datetime.timezone.utc)
    # Locked so two concurrent feedback submissions can't both read the same
    # live[] length and mint the same LIVE-NNN id, silently dropping one.
    with _live_lock:
        live = load_live()
        rec = {"id": f"LIVE-{len(live)+1:03d}", "machine_id": machine, "fix": fix, "worked": worked, "technician": tech,
               "date": now.strftime("%Y-%m-%d"), "reviewed": False}
        live.append(rec); save_live(live)
    ok, err = retain(f"[{rec['id']}] Machine {machine} repair outcome on {rec['date']}: {fix}. Result: "
                     f"{'WORKED' if worked else 'FAILED - do not repeat'}. Logged by {tech} (unreviewed).",
                     f"closed-loop feedback | {rec['id']} | machine {machine}", now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     tags=[machine, "closed-loop-feedback"])
    return {"retained": ok, "error": err, "id": rec["id"]}

def improvement(machine, question, fix, worked):
    before = check_before_repair(question)
    fb = log_feedback(machine, fix, worked, "Demo technician")
    after = check_before_repair(question)
    b, a = before["agent"], after["agent"]
    return {"before": b, "after": a, "feedback": fb,
            "changed": (b.get("dead_end_warning") or "") != (a.get("dead_end_warning") or "") or b.get("action_checklist") != a.get("action_checklist")}

# ---------- fleet / stats / sensors ----------
STREAMS = {
    "MCH-017": dict(signal="Vibration", unit="mm/s", ref="MAINT-2025-001", start=2.1, fail=9.8, step="day", data=[2.1, 2.2, 2.6, 3.9, 5.6, 7.4, 8.9]),
    "MCH-042": dict(signal="Discharge temp", unit="C", ref="MAINT-2025-003", start=85, fail=118, step="hour", data=[85, 86, 88, 92, 99, 106, 112]),
    "MCH-028": dict(signal="Motor temp", unit="C", ref="MAINT-2025-007", start=65, fail=95, step="hour", data=[65, 65, 66, 66, 67, 66, 67]),
}
MAX_TICK = 6

def sensors(tick):
    tick = max(0, min(MAX_TICK, int(tick)))
    out, alerts = {}, []
    for m, s in STREAMS.items():
        d = s["data"][:tick + 1]
        latest = d[-1]
        slope = (d[-1] - d[max(0, len(d) - 3)]) / max(1, min(2, len(d) - 1))
        prog = (latest - s["start"]) / (s["fail"] - s["start"])
        alert = prog >= 0.4 and slope > 0
        out[m] = {"signal": s["signal"], "unit": s["unit"], "series": d, "lo": s["start"], "hi": s["fail"], "latest": latest, "alert": alert}
        if alert:
            ref = BY_ID[s["ref"]]
            eta = (s["fail"] - latest) / slope
            alerts.append({"machine": m, "ref": s["ref"], "tech": ref["technician"], "date": ref["date"],
                           "message": f"{m} {s['signal'].lower()} rose {s['start']}\u2192{latest} {s['unit']}, matching the pattern of {s['ref']} ({ref['failure']}). About {eta:.1f} {s['step']}s to the failure level."})
    return {"tick": tick, "max_tick": MAX_TICK, "machines": out, "alerts": alerts}

def overview():
    live = load_live()
    alerts = {a["machine"] for a in sensors(MAX_TICK)["alerts"]}
    per, first, avoided, avoided_dollars, repeats = {}, {}, 0, 0, 0
    for r in sorted(HISTORY, key=lambda r: r["date"]):
        m = per.setdefault(r["machine_id"], dict(machine_id=r["machine_id"], type=r["machine_type"], incidents=0, repeats=0, last=r["date"]))
        m["incidents"] += 1; m["last"] = max(m["last"], r["date"])
        f = first.setdefault(r["machine_id"], r)
        if "repeat-failure" in r["tags"] and f is not r:
            saved_hours = max(0, f["downtime_hours"] - r["downtime_hours"])
            m["repeats"] += 1; repeats += 1; avoided += saved_hours
            avoided_dollars += saved_hours * cost_per_hour(r["machine_id"])
    today = datetime.date.today()
    for m in per.values():
        m["failed_fixes"] = sum(1 for x in live if x["machine_id"] == m["machine_id"] and not x["worked"])
        score = m["repeats"] + (2 if m["machine_id"] in alerts else 0) + m["failed_fixes"]
        m["risk"] = "critical" if score >= 3 else "warning" if score >= 1 else "nominal"
        m["days_since"] = (today - datetime.date.fromisoformat(m["last"])).days
    order = {"critical": 0, "warning": 1, "nominal": 2}
    # blended rate is a display convenience only; per-machine diagnoses always use cost_per_hour(machine)
    blended_rate = round(avoided_dollars / avoided) if avoided else COST_PER_HOUR_DEFAULT
    return {"machines": sorted(per.values(), key=lambda m: (order[m["risk"]], m["machine_id"])),
            "stats": {"memories": len(HISTORY) + len(live), "live": len(live), "repeat_patterns": repeats, "hours_saved": avoided,
                      "dollars_saved": avoided_dollars, "cost_per_hour": blended_rate}}

def handoff(lang="English"):
    ov, al = overview(), sensors(MAX_TICK)["alerts"]
    ctx = json.dumps({"fleet": ov["machines"], "sensor_alerts": [a["message"] for a in al], "live_outcomes": load_live()[-5:]})
    return llm("You write shift handoff notes for a plant. Use only the data given. Return JSON: "
               '{"headline":"one sentence","watch_list":["..."],"do_first":["..."]}. ' + f"Write in {lang}.", ctx, json_mode=True)

# ---------- eval (retrieval-level, no LLM) ----------
EVAL_CASES = [
    ("MCH-017 vibration climbing, bearing housing warm", "MCH-017"), ("MCH-009 spindle current up, chatter marks", "MCH-009"),
    ("MCH-042 discharge temperature high, short cycling", "MCH-042"), ("MCH-028 motor casing hot, belt tracking drifting", "MCH-028"),
    ("MCH-055 hydraulic pressure fluctuating, fluid stains at seals", "MCH-055"), ("MCH-063 refrigerant pressure falling, compressor running longer", "MCH-063"),
    ("MCH-071 joint 3 clicking, servo current spikes", "MCH-071"), ("high pitched whine and rising vibration on the line 3 pump", "MCH-017"),
    ("MCH-999 hydraulic leakage on new robot", None), ("MCH-500 strange vibration", None)]

def run_eval():
    rows, passed = [], 0
    for q, exp in EVAL_CASES:
        texts, _ = recall(q, exp)
        c = confidence(q, texts)
        blob = " ".join(texts)
        valid = all(i in BY_ID or i.startswith("LIVE-") for i in ID_RE.findall(blob))
        ok = (c["score"] == 0) if exp is None else (exp in blob and c["score"] >= 50)
        ok = ok and valid
        passed += ok
        rows.append({"query": q, "expected": exp or "no history", "confidence": c["label"], "ids_valid": valid, "passed": ok})
    return {"passed": passed, "total": len(EVAL_CASES), "accuracy": round(100 * passed / len(EVAL_CASES)), "rows": rows}
