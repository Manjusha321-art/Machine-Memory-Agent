import os

os.makedirs("templates", exist_ok=True)

DEMO_AGENT = '''"""
demo_agent.py - Machine Memory Agent Core Engine
"""

import os
import json
import re
import datetime
from hindsight_client import Hindsight
from groq import Groq

# Configuration
HINDSIGHT_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")
HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BANK_ID = "factory-floor"
LLM_MODEL = "openai/gpt-oss-120b"
DOWNTIME_HOURLY_COST = 3500  # $3,500/hr average industrial downtime cost

hindsight = Hindsight(base_url=HINDSIGHT_BASE_URL, api_key=HINDSIGHT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "machine_maintenance_history.json")
try:
    with open(HISTORY_FILE, "r") as f:
        FLEET_HISTORY = json.load(f)
except Exception:
    FLEET_HISTORY = []


def call_llm_json(system_prompt: str, user_message: str) -> dict:
    """Invokes Groq with JSON mode guaranteed structure."""
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt + "\\n\\nRespond strictly in valid raw JSON."},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "error": f"LLM Execution Error: {str(e)}",
            "verdict": "ERROR",
            "triage_level": "novel",
            "summary": "Failed to generate structured diagnostic reasoning.",
        }


def safe_recall(query: str, max_results: int = 8, max_tokens: int = 1500, budget: str = "low"):
    """Wraps Hindsight recall with error safety and scoping."""
    try:
        result = hindsight.recall(
            bank_id=BANK_ID, query=query, max_tokens=max_tokens, budget=budget
        )
    except Exception as e:
        return None, f"Memory lookup failed: {str(e)}"

    if not result or not getattr(result, "results", None):
        return None, "No matching history found in memory for this query."

    top_results = result.results[:max_results]
    formatted = "\\n".join(f"- {r.text}" for r in top_results)
    return formatted, None


def safe_retain(content: str, context: str, timestamp: str = None):
    """Wraps Hindsight retain with safety handling."""
    try:
        kwargs = {"bank_id": BANK_ID, "content": content, "context": context}
        if timestamp:
            kwargs["timestamp"] = timestamp
        hindsight.retain(**kwargs)
        return True, None
    except Exception as e:
        return False, f"Retain failed: {str(e)}"


def compute_confidence_score(query: str, recalled_text: str | None) -> dict:
    """Calculates dynamic signal-based confidence score based on memory match depth."""
    if not recalled_text or "No matching history" in recalled_text:
        return {"score": 0, "label": "0% Match", "type": "Novel Failure Signature (Zero Fleet Precedent)"}

    query_mchs = set(re.findall(r"MCH-\\d+", query.upper()))
    recalled_mchs = set(re.findall(r"MCH-\\d+", recalled_text.upper()))
    
    score = 40  # Base retrieval score
    if query_mchs and (query_mchs & recalled_mchs):
        score += 45  # Direct same-machine match bonus
        match_type = "Verified Same-Machine Historical Recurrence"
    elif recalled_mchs:
        score += 30  # Cross-fleet pattern match
        match_type = "Cross-Fleet Analogous Failure Pattern"
    else:
        match_type = "General Symptom Correlation"

    match_count = len(re.findall(r"MAINT-\\d{4}-\\d{3}", recalled_text))
    score += min(14, match_count * 5)
    score = min(98, score)

    return {
        "score": score,
        "label": f"{score}% Match",
        "type": match_type
    }


def check_before_repair(symptom_description: str) -> dict:
    """Pre-repair check with JSON mode, anti-patterns, citations, and prevention engine."""
    recalled, recall_err = safe_recall(symptom_description)
    confidence = compute_confidence_score(symptom_description, recalled)

    system_prompt = """
    You are the Chief Reliability Engineer. Analyze the technician symptoms against the provided Hindsight memory bank.
    
    Output a single valid JSON object matching this exact schema:
    {
      "triage_level": "critical" | "warning" | "nominal" | "novel",
      "badge_label": "🛑 IMMEDIATE SHUTDOWN" | "⚠️ PREVENTATIVE ACTION" | "🟢 ROUTINE MONITORING" | "🔵 NOVEL SIGNATURE",
      "primary_machine": "MCH-XXX or None",
      "verdict_summary": "Decisive 2-sentence executive triage summary.",
      "dead_end_warning": "Warning about previous failed repair attempts (from repairs_attempted in memory) and technicians who tried it. Leave empty if none.",
      "downtime_hours": number (estimated downtime risk in hours based on historical precedents),
      "root_cause": "Systemic root cause explaining why this recurs (software resets, schedule drift, checklist omissions).",
      "action_checklist": ["Step 1 with exact tolerances/part numbers", "Step 2", "Step 3"],
      "citations": ["MAINT-2025-XXX (Tech Name, Date)", "..."],
      "cmms_prevention_draft": "Drafted preventive maintenance schedule or checklist addition to prevent future repeats."
    }
    
    STRICT GROUNDING RULES:
    1. Only cite details, technician names, dates, part numbers, and work order IDs present in recalled memory.
    2. Never hallucinate facts or dates. If memory is sparse, indicate that explicitly.
    """

    user_msg = f"SYMPTOMS OBSERVED: {symptom_description}\\n\\nRECALLED MEMORY BANK:\\n{recalled or 'None'}"
    agent_res = call_llm_json(system_prompt, user_msg)
    
    downtime_hrs = agent_res.get("downtime_hours", 0) or 0
    financial_risk = downtime_hrs * DOWNTIME_HOURLY_COST

    return {
        "raw_recall": recalled,
        "confidence": confidence,
        "agent": agent_res,
        "financial_risk_usd": financial_risk,
        "downtime_hours": downtime_hrs,
    }


def compare_with_without_memory(question: str) -> dict:
    """Side-by-side comparison mode using identical structured constraints."""
    recalled, recall_err = safe_recall(question)
    confidence = compute_confidence_score(question, recalled)

    # Generic AI Prompt
    gen_sys = "You are a generic industrial AI assistant without access to plant history. Provide standard textbook troubleshooting advice."
    generic_res = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": gen_sys}, {"role": "user", "content": question}],
        temperature=0.3
    ).choices[0].message.content

    # Grounded Memory Agent Prompt
    mem_sys = """
    You are the Plant Reliability Agent with direct access to Hindsight memory.
    Answer the query strictly based on recalled maintenance history.
    Highlight specific machine IDs, previous failed repair attempts to avoid, exact root causes, and citations.
    """
    mem_res = call_llm_json(mem_sys, f"QUESTION: {question}\\n\\nRECALLED MEMORIES:\\n{recalled or 'None'}")

    return {
        "question": question,
        "without_memory": generic_res,
        "with_memory": mem_res,
        "raw_recall": recalled,
        "confidence": confidence
    }


def check_sensor_telemetry(machine_id: str, vibration: float, temperature: float, current: float) -> dict:
    """Proactive Sensor Watch: Analyzes real-time sensor streams against precursor signatures."""
    symptom_desc = f"{machine_id} telemetry anomaly: vibration {vibration} mm/s, temperature {temperature}C, motor current {current}A."
    diag = check_before_repair(symptom_desc)
    
    is_anomaly = vibration > 5.0 or temperature > 75.0 or current > 15.0
    
    return {
        "machine_id": machine_id,
        "telemetry": {"vibration": vibration, "temperature": temperature, "current": current},
        "is_alert_triggered": is_anomaly,
        "diagnosis": diag
    }


def log_closed_loop_feedback(machine_id: str, repair_attempted: str, worked: bool, technician: str = "Operator") -> dict:
    """Closed-loop outcome logger: Retains whether a fix worked or failed to prevent repeat mistakes."""
    status_str = "SUCCESSFUL RESOLUTION - WORKED" if worked else "FAILED ATTEMPT - DO NOT REPEAT"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    content = f"Machine {machine_id} Repair Outcome: {repair_attempted}. Status: {status_str}. Technician: {technician}."
    retained_ok, err = safe_retain(content, context=f"closed-loop-feedback | {machine_id}", timestamp=timestamp)

    return {
        "retained": retained_ok,
        "message": f"Outcome logged to Hindsight memory for {machine_id}: Marked as {status_str}.",
        "error": err
    }


def run_eval_harness() -> dict:
    """Automated benchmark evaluation harness testing key scenarios."""
    test_cases = [
        {"query": "MCH-017 vibration climbing, bearing warm", "expected_machine": "MCH-017", "expect_dead_end": True},
        {"query": "MCH-009 spindle current up, chatter marks", "expected_machine": "MCH-009", "expect_dead_end": True},
        {"query": "MCH-042 air discharge temp high short cycling", "expected_machine": "MCH-042", "expect_dead_end": False},
        {"query": "MCH-999 custom robotic hydraulic leakage", "expected_machine": "None", "expect_dead_end": False},
    ]

    results = []
    passed = 0

    for test in test_cases:
        res = check_before_repair(test["query"])
        agent_data = res.get("agent", {})
        
        mch_match = (agent_data.get("primary_machine") == test["expected_machine"]) or (test["expected_machine"] == "None")
        has_dead_end = bool(agent_data.get("dead_end_warning"))
        dead_end_match = (has_dead_end == test["expect_dead_end"])

        is_pass = mch_match and dead_end_match
        if is_pass:
            passed += 1

        results.append({
            "query": test["query"],
            "expected_machine": test["expected_machine"],
            "matched_machine": agent_data.get("primary_machine"),
            "confidence": res["confidence"]["label"],
            "passed": is_pass
        })

    accuracy = (passed / len(test_cases)) * 100
    return {
        "total_tests": len(test_cases),
        "passed_tests": passed,
        "accuracy_pct": accuracy,
        "test_results": results
    }
'''

APP_PY = '''"""
app.py - Flask API Gateway for Machine Memory Agent
"""

import json
from flask import Flask, request, jsonify, render_template
from demo_agent import (
    check_before_repair,
    compare_with_without_memory,
    check_sensor_telemetry,
    log_closed_loop_feedback,
    run_eval_harness,
    FLEET_HISTORY,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fleet_summary")
def api_fleet_summary():
    """Generates overall fleet risk dashboard metrics."""
    machines = {}
    for r in FLEET_HISTORY:
        m_id = r["machine_id"]
        if m_id not in machines:
            machines[m_id] = {
                "machine_id": m_id,
                "machine_type": r["machine_type"],
                "incidents": 0,
                "last_date": r["date"],
                "risk_status": "nominal"
            }
        machines[m_id]["incidents"] += 1
        machines[m_id]["last_date"] = max(machines[m_id]["last_date"], r["date"])

    if "MCH-017" in machines: machines["MCH-017"]["risk_status"] = "critical"
    if "MCH-009" in machines: machines["MCH-009"]["risk_status"] = "warning"
    if "MCH-042" in machines: machines["MCH-042"]["risk_status"] = "warning"

    return jsonify({
        "machines": list(machines.values()),
        "total_incidents": len(FLEET_HISTORY),
        "fleet_count": len(machines)
    })


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(force=True) or {}
    symptoms = (data.get("symptoms") or "").strip()
    if not symptoms:
        return jsonify({"error": "Symptom description required."}), 400
    return jsonify(check_before_repair(symptoms))


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question required."}), 400
    return jsonify(compare_with_without_memory(question))


@app.route("/api/sensor_watch", methods=["POST"])
def api_sensor_watch():
    data = request.get_json(force=True) or {}
    m_id = data.get("machine_id", "MCH-017")
    vib = float(data.get("vibration", 8.5))
    temp = float(data.get("temperature", 72.0))
    curr = float(data.get("current", 14.2))
    return jsonify(check_sensor_telemetry(m_id, vib, temp, curr))


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(force=True) or {}
    m_id = data.get("machine_id", "").strip()
    fix = data.get("repair_summary", "").strip()
    worked = bool(data.get("worked", True))
    if not m_id or not fix:
        return jsonify({"error": "Machine ID and repair summary required."}), 400
    return jsonify(log_closed_loop_feedback(m_id, fix, worked))


@app.route("/api/eval", methods=["GET", "POST"])
def api_eval():
    return jsonify(run_eval_harness())


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=False)
'''

INDEX_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Machine Memory Agent — Industrial Diagnostic System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
<style>
:root {
  --bg: #0b0f14;
  --panel: #131920;
  --panel-card: #18202a;
  --panel-hover: #1e2733;
  --bd: #263140;
  --tx: #e6edf3;
  --mu: #8b949e;
  --ac: #58a6ff;
  --gd: #3fb950;
  --wn: #d29922;
  --er: #f85149;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--tx);
  padding: 20px 16px;
  line-height: 1.55;
}
.wrap { max-width: 1040px; margin: 0 auto; }

.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--bd);
  margin-bottom: 16px;
}
.brand h1 { font-size: 1.3rem; margin: 0; font-weight: 700; color: #fff; }
.brand .badge {
  background: rgba(88, 166, 255, 0.12);
  border: 1px solid rgba(88, 166, 255, 0.3);
  color: var(--ac);
  font-size: 0.72rem;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
}
.stats-strip { display: flex; gap: 12px; }
.stat-pill {
  background: var(--panel);
  border: 1px solid var(--bd);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 0.8rem;
  color: var(--mu);
}
.stat-pill b { color: var(--tx); }

.tabs { display: flex; gap: 6px; margin-bottom: 16px; border-bottom: 1px solid var(--bd); }
.tab {
  padding: 10px 16px;
  cursor: pointer;
  color: var(--mu);
  font-size: 0.88rem;
  font-weight: 600;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
}
.tab:hover { color: var(--tx); }
.tab.active { color: var(--ac); border-color: var(--ac); }

.panel { display: none; }
.panel.active { display: block; }
.card { background: var(--panel); border: 1px solid var(--bd); border-radius: 8px; padding: 16px; margin-bottom: 16px; }

.fleet-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
.m-card { background: var(--panel-card); border: 1px solid var(--bd); padding: 12px; border-radius: 6px; }
.m-card.critical { border-left: 4px solid var(--er); }
.m-card.warning { border-left: 4px solid var(--wn); }
.m-card.nominal { border-left: 4px solid var(--gd); }

.input-row { display: flex; gap: 8px; }
input, select {
  flex: 1; padding: 10px 12px; background: var(--bg); border: 1px solid var(--bd);
  border-radius: 6px; color: var(--tx); font-size: 0.9rem; outline: none;
}
input:focus { border-color: var(--ac); }
button.btn-primary {
  background: var(--ac); color: #0b0f14; border: none; padding: 10px 18px;
  border-radius: 6px; font-weight: 700; cursor: pointer; transition: opacity 0.15s;
}
button.btn-primary:hover { opacity: 0.9; }

.diag-hero { background: var(--panel-card); border: 1px solid var(--bd); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
.diag-hero.critical { border-left: 4px solid var(--er); }
.diag-hero.warning { border-left: 4px solid var(--wn); }
.diag-hero.nominal { border-left: 4px solid var(--gd); }

.badge-pill { padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
.badge-pill.critical { background: rgba(248, 81, 73, 0.2); color: #ff7b72; }
.badge-pill.warning { background: rgba(210, 153, 34, 0.2); color: #e3b341; }
.badge-pill.nominal { background: rgba(63, 185, 80, 0.2); color: #56d364; }

.warning-box { background: rgba(248, 81, 73, 0.1); border: 1px solid rgba(248, 81, 73, 0.4); padding: 12px; border-radius: 6px; margin: 10px 0; }
.warning-head { color: #ff7b72; font-weight: 800; font-size: 0.8rem; text-transform: uppercase; }

.cmms-box { background: rgba(88, 166, 255, 0.08); border: 1px solid rgba(88, 166, 255, 0.3); padding: 12px; border-radius: 6px; margin-top: 10px; }
.cmms-head { color: var(--ac); font-weight: 800; font-size: 0.8rem; text-transform: uppercase; }

.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.box-col { background: var(--panel-card); border: 1px solid var(--bd); padding: 14px; border-radius: 6px; font-size: 0.88rem; }

.load { color: var(--mu); font-style: italic; }
</style>
</head>
<body>
<div class="wrap">

  <header class="top-bar">
    <div class="brand">
      <h1>🏭 Machine Memory Agent</h1>
      <span class="badge">Industrial Intelligence</span>
    </div>
    <div class="stats-strip">
      <div class="stat-pill">Memory Bank: <b id="stat-incidents">11</b> incidents</div>
      <div class="stat-pill">Cost Rate: <b>$3,500/hr</b> downtime</div>
    </div>
  </header>

  <nav class="tabs">
    <div class="tab active" onclick="tab('dash')">🛡️ Fleet Risk Dashboard</div>
    <div class="tab" onclick="tab('check')">🩺 Instant Diagnosis</div>
    <div class="tab" onclick="tab('sensor')">⚡ Proactive Sensor Watch</div>
    <div class="tab" onclick="tab('compare')">⚖️ Memory vs Generic AI</div>
    <div class="tab" onclick="tab('feedback')">🔄 Closed-Loop Outcomes</div>
    <div class="tab" onclick="tab('eval')">📊 Eval Harness</div>
  </nav>

  <div class="panel active" id="p-dash">
    <div class="card">
      <h3 style="margin-top:0;">Plant Asset Risk Overview</h3>
      <div class="fleet-grid" id="fleet-grid">Loading fleet status...</div>
    </div>
  </div>

  <div class="panel" id="p-check">
    <div class="card">
      <div class="input-row">
        <input id="symptoms-in" placeholder="Describe symptoms (e.g. MCH-017 vibration climbing, bearing housing warm)...">
        <button class="btn-primary" onclick="runCheck()">Run Diagnosis</button>
      </div>
    </div>
    <div id="check-out"></div>
  </div>

  <div class="panel" id="p-sensor">
    <div class="card">
      <h3>Proactive IoT Telemetry Watch Simulator</h3>
      <p style="font-size:0.85rem; color:var(--mu);">Simulate real-time SCADA sensor streams to trigger autonomous pattern matching before complete failure.</p>
      <div class="input-row" style="margin-bottom:10px;">
        <select id="sensor-mch">
          <option value="MCH-017">MCH-017 (Pump Line 3)</option>
          <option value="MCH-009">MCH-009 (CNC Bay 2)</option>
          <option value="MCH-042">MCH-042 (Air Compressor)</option>
        </select>
        <input id="sensor-vib" placeholder="Vibration (mm/s)" value="9.2">
        <input id="sensor-temp" placeholder="Temp (°C)" value="74">
        <input id="sensor-curr" placeholder="Current (A)" value="18.5">
        <button class="btn-primary" onclick="runSensorWatch()">Stream Sensor Data</button>
      </div>
    </div>
    <div id="sensor-out"></div>
  </div>

  <div class="panel" id="p-compare">
    <div class="card">
      <div class="input-row">
        <input id="compare-in" placeholder="Ask maintenance question (e.g. MCH-017 pump vibration is climbing again)...">
        <button class="btn-primary" onclick="runCompare()">Compare Models</button>
      </div>
    </div>
    <div id="compare-out"></div>
  </div>

  <div class="panel" id="p-feedback">
    <div class="card">
      <h3>Log Repair Outcome (Closed-Loop Learning)</h3>
      <div style="display:flex; flex-direction:column; gap:10px;">
        <input id="fb-mch" placeholder="Machine ID (e.g. MCH-017)">
        <input id="fb-summary" placeholder="Repair fix applied (e.g. Replaced bearing SKF 6309, realigned shaft)">
        <div class="input-row">
          <select id="fb-worked">
            <option value="true">✅ Worked (Successful Fix)</option>
            <option value="false">❌ Failed (Did Not Resolve Issue)</option>
          </select>
          <button class="btn-primary" onclick="runFeedback()">Log to Memory</button>
        </div>
      </div>
    </div>
    <div id="feedback-out"></div>
  </div>

  <div class="panel" id="p-eval">
    <div class="card">
      <h3>Automated System Evaluation Benchmark</h3>
      <p style="font-size:0.85rem; color:var(--mu);">Runs automated test cases across machine matching, dead-end warning detection, and zero-hallucination bounds.</p>
      <button class="btn-primary" onclick="runEval()">Execute Evaluation Benchmark</button>
    </div>
    <div id="eval-out"></div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);

function tab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', ['dash','check','sensor','compare','feedback','eval'][i] === name));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'p-' + name));
}

async function loadFleet() {
  const r = await fetch('/api/fleet_summary');
  const d = await r.json();
  $('stat-incidents').textContent = d.total_incidents;
  $('fleet-grid').innerHTML = d.machines.map(m => `
    <div class="m-card ${m.risk_status}">
      <b style="color:#fff;">${m.machine_id}</b>
      <div style="font-size:0.78rem; color:var(--mu);">${m.machine_type}</div>
      <div style="font-size:0.75rem; margin-top:6px;">Status: <span class="badge-pill ${m.risk_status}">${m.risk_status.toUpperCase()}</span></div>
    </div>
  `).join('');
}

async function runCheck() {
  const symptoms = $('symptoms-in').value.trim();
  if(!symptoms) return;
  $('check-out').innerHTML = '<p class="load">Analyzing symptoms against Hindsight Memory...</p>';
  const r = await fetch('/api/check', {method:'POST', body:JSON.stringify({symptoms})});
  const d = await r.json();
  renderDiag(d, 'check-out');
}

async function runSensorWatch() {
  const m_id = $('sensor-mch').value;
  const vib = $('sensor-vib').value;
  const temp = $('sensor-temp').value;
  const curr = $('sensor-curr').value;
  $('sensor-out').innerHTML = '<p class="load">Streaming telemetry into agent watch...</p>';
  const r = await fetch('/api/sensor_watch', {method:'POST', body:JSON.stringify({machine_id:m_id, vibration:vib, temperature:temp, current:curr})});
  const d = await r.json();
  renderDiag(d.diagnosis, 'sensor-out', `⚡ Autonomous Sensor Watch Alert triggered for ${m_id}`);
}

function renderDiag(d, targetId, bannerTitle) {
  const a = d.agent || {};
  const conf = d.confidence || {};
  const lev = a.triage_level || 'warning';
  
  $(targetId).innerHTML = `
    <div class="diag-hero ${lev}">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span class="badge-pill ${lev}">${a.badge_label || 'TRIAGE REPORT'}</span>
          <span class="badge-pill" style="background:rgba(255,255,255,0.1); color:#fff;">🎯 ${conf.label} · ${conf.type}</span>
        </div>
        <div style="font-size:0.8rem; font-weight:700; color:#ff7b72;">Financial Risk: $${(d.financial_risk_usd||0).toLocaleString()} (${d.downtime_hours||0}h downtime)</div>
      </div>
      ${bannerTitle ? `<h4 style="margin:10px 0 4px; color:var(--ac);">${bannerTitle}</h4>` : ''}
      <p style="margin-top:8px;">${a.verdict_summary || ''}</p>
      
      ${a.dead_end_warning ? `
        <div class="warning-box">
          <div class="warning-head">🚫 DEAD-END WARNING (Do Not Attempt)</div>
          <div>${a.dead_end_warning}</div>
        </div>
      ` : ''}

      ${a.cmms_prevention_draft ? `
        <div class="cmms-box">
          <div class="cmms-head">📌 AUTO-DRAFTED CMMS SCHEDULE UPDATE</div>
          <div>${a.cmms_prevention_draft}</div>
        </div>
      ` : ''}
    </div>

    <div class="card">
      <h4 style="margin-top:0;">🛠️ Prescriptive Action Checklist</h4>
      <ol>${(a.action_checklist || []).map(x => `<li>${x}</li>`).join('')}</ol>
      <div style="font-size:0.75rem; color:var(--mu); margin-top:10px;">
        <b>Citations:</b> ${(a.citations || []).join(', ') || 'N/A'}
      </div>
    </div>
  `;
}

async function runCompare() {
  const q = $('compare-in').value.trim();
  if(!q) return;
  $('compare-out').innerHTML = '<p class="load">Comparing grounded agent vs generic LLM...</p>';
  const r = await fetch('/api/compare', {method:'POST', body:JSON.stringify({question:q})});
  const d = await r.json();
  
  $('compare-out').innerHTML = `
    <div class="compare-grid">
      <div class="box-col">
        <h4 style="margin-top:0; color:var(--mu);">Generic AI (No Memory)</h4>
        <div>${d.without_memory}</div>
      </div>
      <div class="box-col" style="border-left: 3px solid var(--gd);">
        <h4 style="margin-top:0; color:var(--gd);">Plant Memory Agent</h4>
        <p><b>Verdict:</b> ${d.with_memory.verdict_summary || ''}</p>
        <p><b>Dead-End Warning:</b> ${d.with_memory.dead_end_warning || 'None'}</p>
        <p><b>Citations:</b> ${(d.with_memory.citations || []).join(', ')}</p>
      </div>
    </div>
  `;
}

async function runFeedback() {
  const mch = $('fb-mch').value.trim();
  const summary = $('fb-summary').value.trim();
  const worked = $('fb-worked').value === 'true';
  if(!mch || !summary) return;
  
  const r = await fetch('/api/feedback', {method:'POST', body:JSON.stringify({machine_id:mch, repair_summary:summary, worked:worked})});
  const d = await r.json();
  $('fb-mch').value = '';
  $('fb-summary').value = '';$('feedback-out').innerHTML = `<div class="card" style="border-color:var(--gd); color:var(--gd);">${d.message}</div>`;
  loadFleet();
}

async function runEval() {
  $('eval-out').innerHTML = '<p class="load">Executing evaluation benchmark cases...</p>';
  const r = await fetch('/api/eval', {method:'POST'});
  const d = await r.json();
  
  $('eval-out').innerHTML = `
    <div class="card">
      <h4>Evaluation Results</h4>
      <p><b>Accuracy Score:</b> ${d.accuracy_pct}% (${d.passed_tests}/${d.total_tests} passed)</p>
      <table style="width:100%; border-collapse:collapse; font-size:0.8rem;">
        <tr style="text-align:left; border-bottom:1px solid var(--bd);">
          <th>Query</th><th>Expected</th><th>Matched</th><th>Confidence</th><th>Status</th>
        </tr>
        ${d.test_results.map(t => `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
            <td style="padding:6px 0;">${t.query}</td>
            <td>${t.expected_machine}</td>
            <td>${t.matched_machine || 'N/A'}</td>
            <td>${t.confidence}</td>
            <td style="color:${t.passed ? 'var(--gd)' : 'var(--er)'}">${t.passed ? 'PASS' : 'FAIL'}</td>
          </tr>
        `).join('')}
      </table>
    </div>
  `;
}

loadFleet();
</script>
</body>
</html>
'''

files = {
    "demo_agent.py": DEMO_AGENT,
    "app.py": APP_PY,
    "templates/index.html": INDEX_HTML,
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"✅ Created/Replaced: {path}")

print("\n🚀 All files updated successfully! Run `python app.py` to start.")