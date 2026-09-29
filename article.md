# How Hindsight Stopped Our Maintenance Agent From Repeating Dead-End Fixes

At 2:00 AM on a packaging line, a centrifugal pump (`MCH-017`) starts whining and its bearing housing climbs past 70°C. The night-shift technician checks the symptoms, pumps fresh grease into the bearing housing, and closes the ticket—unaware that six months ago on the day shift, another technician tried that exact same fix, watched the vibration keep climbing to 9.8 mm/s, and lost 14 hours of production when the drive-end bearing seized.

Industrial plants do not suffer from a lack of data; they suffer from amnesia between shifts. Service reports sit in siloed tickets, dead-end repair attempts rarely make it into standard operating procedures, and generic LLMs offer textbook advice ("check lubrication and inspect alignment") that ignores what actually happened on that specific serial number last quarter.

To solve this, I built the **Machine Memory Agent** ([GitHub repository](https://github.com/Manjusha321-art/Machine-Memory-Agent)): a reliability system that gives every industrial asset on a factory floor its own persistent, self-updating memory using the [Hindsight open-source repository on GitHub](https://github.com/vectorize-io/hindsight) and Groq (`openai/gpt-oss-120b`). Instead of treating maintenance logs as static documents for keyword search, the system retains technician field notes, synthesizes living per-machine digital twins, enforces plant safety policies as standing memory directives, and learns from failed floor repairs in real time.

![Machine Memory Agent Overview & Fleet Telemetry](https://raw.githubusercontent.com/Manjusha321-art/Machine-Memory-Agent/main/screenshots/1_dashboard_overview.png)

---

## How the System Hangs Together

When a floor technician describes a symptom (by typing or using hands-free voice input) or when a live telemetry stream crosses a precursor slope threshold, the request flows through a Flask gateway (`app.py`) into the core reasoning engine (`demo_agent.py`).

```text
Technician Input (Voice / Text) or Live Sensor Telemetry Anomaly
                              │
                              ▼
                        demo_agent.py
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
     RETAIN                RECALL                REFLECT
 (field notes &       (fast retrieval      (schema-constrained
  live outcomes)       for comparison       triage over memories,
        │              & eval suite)        mental models & directives)
        └─────────────────────┼─────────────────────┘
                              ▼
                      Hindsight Memory
                  (bank: "factory-floor-v2")
```

Rather than bolting a basic vector store onto a chat prompt, the architecture leans on [Vectorize's guide to what agent memory is](https://vectorize.io/what-is-agent-memory) and uses all five primitives documented in the [Hindsight developer documentation](https://hindsight.vectorize.io/):

1. **`retain()`** — Ingests natural-language maintenance field notes (`MAINT-2025-001` through `MAINT-2025-011`) and live technician outcomes (`LIVE-001`, etc.) with structured `context`, ISO timestamps, and machine/failure tags.
2. **`recall()`** — Powers low-latency retrieval (`budget="low"`, `max_tokens=1500`) for side-by-side baseline comparisons and our 10-case automated retrieval benchmark.
3. **`reflect()`** — Executes multi-memory judgment queries with a strict JSON `response_schema` and `include_facts=True`, returning both the structured diagnosis and a `based_on` provenance object listing the exact memories, mental models, and directives used.
4. **Mental Models (`create_mental_model`)** — Maintains a self-updating Digital Twin (`digital-twin-mch-017`, etc.) for every machine on the floor, configured with `trigger={"refresh_after_consolidation": True}` so the profile updates automatically as new repairs are retained.
5. **Directives (`create_directive`)** — Encodes plant-wide safety rules (such as mandatory Lockout-Tagout before touching rotating equipment) and asset-specific guardrails directly inside the memory bank rather than burying them in fragile application prompts.

![Fleet Risk Grid and Incident Scenario Picker](https://raw.githubusercontent.com/Manjusha321-art/Machine-Memory-Agent/main/screenshots/2_scenarios_fleet.png)

---

## The Core Technical Story: Why `recall()` Wasn't Enough

When I wrote the first version of the diagnostic endpoint, I used the pattern almost every developer reaches for first: call `recall()`, stuff the top text chunks into a system prompt, and ask an LLM to output JSON.

It worked for simple single-record lookups, but it broke down on the exact scenarios a Chief Reliability Engineer actually cares about:
* **Repeat-failure synthesis:** Understanding *why* `MCH-017` kept eating SKF 6309 bearings required connecting a February incident (`MAINT-2025-001`, where a cost-cutting review stretched lubrication from monthly to quarterly) with an August incident (`MAINT-2025-002`, where a CMMS software update quietly reset the lubrication template back to quarterly).
* **Policy enforcement:** Certain machines have hard operational rules—for example, `MCH-042` (an industrial air compressor) must never have its thermal overload trip reset a second time without first verifying intake filter replacement.
* **Verifiable provenance:** On a factory floor, if an agent tells a technician *not* to try a fix, it has to prove which historical work order and technician tried it before.

Switching the diagnostic path in `demo_agent.py` from `recall()` + custom prompting to Hindsight's `reflect()` with Mental Models and Directives solved all three problems at the memory layer.

### 1. Self-Updating Digital Twins via Mental Models

In `setup_advanced_memory.py`, each machine gets a dedicated Hindsight Mental Model tagged with its asset ID and configured to refresh automatically after memory consolidation:

```python
for m in machines:
    mid = f"digital-twin-{m.lower()}"
    client.create_mental_model(
        bank_id=BANK_ID,
        id=mid,
        name=f"{m} digital twin",
        source_query=(
            f"Build a living profile of machine {m} ({by_machine[m]}): its known failure modes and "
            f"precursor signals, root causes, fixes that worked, fixes that failed and should not be "
            f"repeated, and its current risk level based on the most recent incidents and technician "
            f"feedback. Be specific: part numbers, tolerances, technician names, dates."
        ),
        tags=[m],
        max_tokens=900,
        trigger={"refresh_after_consolidation": True},
    )
```

Right below that, we register standing plant policies as Hindsight Directives:

```python
DIRECTIVES = [
    dict(
        name="grounding-policy",
        content=(
            "State only facts, technician names, dates, and part numbers present in retrieved memory. "
            "If retrieved memory is thin or absent for a machine, say so explicitly rather than guessing."
        ),
        priority=10,
        tags=None,
    ),
    dict(
        name="loto-before-rotating-equipment",
        content=(
            "Before any bearing, motor, coupling, or seal replacement on rotating equipment, the action checklist "
            "must include lockout-tagout (LOTO) verification as the first step."
        ),
        priority=5,
        tags=None,
    ),
    dict(
        name="mch-042-filter-before-reset",
        content=(
            "MCH-042 (Industrial Air Compressor - Unit B): do not recommend clearing a thermal overload trip a second "
            "time without first confirming intake filter replacement."
        ),
        priority=1,
        tags=["MCH-042"],
    ),
]
```

### 2. Schema-Constrained `reflect()` with Anti-Hallucination Citation Filtering

When `check_before_repair()` runs in `demo_agent.py`, it calls `reflect()` with `RESPONSE_SCHEMA` and `include_facts=True`. Even more importantly, we inspect `r.based_on` to verify every citation the model claims against the actual record IDs present in the recalled facts and context:

```python
def reflect(query, context=None, tags=None, response_schema=None, budget="mid"):
    r, e = call_with_retry(
        lambda: hs().reflect(
            bank_id=BANK_ID,
            query=query,
            context=context,
            tags=tags,
            tags_match="any",
            budget=budget,
            max_tokens=1600,
            response_schema=response_schema,
            include_facts=True,
        )
    )
    if e:
        return None, friendly_error(e)
    return r, None
```

Inside `check_before_repair()`, we combine `f.text` (Hindsight's extracted fact) with `f.context` (which preserves the exact `MAINT-2025-XXX` or `LIVE-XXX` identifier) and strip any citation that wasn't genuinely retrieved in `r.based_on`:

```python
facts = (r.based_on.memories if r.based_on else None) or []
mental_models = (r.based_on.mental_models if r.based_on else None) or []
directives = (r.based_on.directives if r.based_on else None) or []

texts = [f.text for f in facts]
id_blob = texts + [f.context for f in facts if f.context]
if machine and any(m.id == twin_id(machine) for m in mental_models):
    id_blob += [f"[{rid}] machine {machine}" for rid in BY_ID if BY_ID[rid]["machine_id"] == machine]

srcs = sources_from(id_blob)
cross = cross_fleet_from(id_blob, machine)
allowed = {s["id"] for s in srcs} | {c["id"] for c in cross} | {x["id"] for x in led}
cites = [c for c in (a.get("citations") or []) if isinstance(c, str) and any(c.startswith(i) for i in allowed)]
a["citations"] = [next(i for i in allowed if c.startswith(i)) for c in cites]
```

If someone queries an unknown machine like `MCH-999 hydraulic leakage on a newly installed robot`, the agent refuses to guess: it returns a `0% match` confidence score, flags the triage level as `novel`, and instructs the technician to follow the OEM manual and log the outcome so the fleet learns from it.

![MCH-017 Diagnosis with Dead-End Warning, Standing Directives, Citations & Digital Twin](https://raw.githubusercontent.com/Manjusha321-art/Machine-Memory-Agent/main/screenshots/3_diagnosis_digital_twin.png)

---

## Concrete Behavior: Before vs. After Memory Grows

A static memory bank only proves that seeding works. To prove that the agent improves over time, the system includes a closed-loop outcome logger (`log_feedback`) and a 3-step live improvement workflow (`improvement`).

Take `MCH-055` (Hydraulic Press — Forming Station). In its initial history (`MAINT-2025-008`), cylinder seals failed due to contaminated hydraulic fluid, and no prior failed fixes were recorded (`repairs_attempted: []`).

1. **Step 1 (Baseline Diagnosis):** We query `MCH-055 hydraulic press pressure fluctuating, fluid stains near cylinder seals`. The agent identifies the cylinder seal wear and contaminated fluid from `MAINT-2025-008`, with no dead-end warning on record.
2. **Step 2 (Closed-Loop Retain):** A technician tries a shortcut—*"Topped up hydraulic reservoir and bled air line without replacing cylinder seals"*—and clicks **Didn't work**. `log_feedback()` immediately retains `LIVE-001` into Hindsight:
   ```python
   ok, err = retain(
       f"[{rec['id']}] Machine {machine} repair outcome on {rec['date']}: {fix}. Result: "
       f"{'WORKED' if worked else 'FAILED - do not repeat'}. Logged by {tech} (unreviewed).",
       f"closed-loop feedback | {rec['id']} | machine {machine}",
       now.strftime("%Y-%m-%dT%H:%M:%SZ"),
       tags=[machine, "closed-loop-feedback"],
   )
   ```
3. **Step 3 (Post-Feedback Diagnosis):** The exact same symptom query is run again. This time, the diagnosis surfaces a prominent **Dead-End Warning** citing `LIVE-001` (*"Do NOT repeat failed attempt by Demo technician: 'Topped up hydraulic reservoir and bled air line without replacing cylinder seals'"*) and prepends an explicit avoidance step to the action checklist.

The same contrast appears in the side-by-side **With vs. Without Memory** comparison for `MCH-009` (CNC Milling Machine):
* **Without Memory (Generic LLM):** Suggests generic inspection steps with zero awareness of prior plant tickets.
* **With Hindsight Memory (`72% match`):** Warns that technician A. Kulkarni (`MAINT-2025-005`) and D. Iyer (`MAINT-2025-006`) already faced this exact failure pattern, highlights the drifted coolant nozzle root cause, and instructs the technician to add daily coolant-nozzle alignment verification to the preventive checklist.

![Side-by-Side Comparison: Without Memory vs. With Hindsight Memory](https://raw.githubusercontent.com/Manjusha321-art/Machine-Memory-Agent/main/screenshots/4_with_vs_without_memory.png)

In addition to per-incident triage, the system synthesizes fleet-wide risk, sensor precursor alerts, and recent live outcomes into an actionable **Shift Handoff** brief for incoming reliability teams:

![Automated Shift Handoff Brief](https://raw.githubusercontent.com/Manjusha321-art/Machine-Memory-Agent/main/screenshots/5_shift_handoff.png)

---

## Lessons Learned Building With Agent Memory

1. **Keep raw identifiers in `context` when retaining natural-language notes.** Hindsight's extraction pass paraphrases `content` into clean semantic facts, which occasionally drops bracketed ticket IDs like `[MAINT-2025-001]`. Passing `"maintenance record MAINT-2025-001 | machine MCH-017"` in the `context` parameter guarantees that `fact.context` always carries the exact ticket ID for downstream citation verification.
2. **Always bound `recall()` and `reflect()` token budgets.** My earliest implementation called `recall()` without `budget` or `max_tokens` limits, pulling back ~65 memory fragments across all 11 machines on every query. Setting `budget="low"` and `max_tokens=1500` on `recall()` (and `budget="mid"`, `max_tokens=1600` on `reflect()`) cut latency and eliminated irrelevant cross-machine noise.
3. **Instantiate sync Hindsight clients per request in multi-threaded servers.** When browser page loads fired `/api/overview`, `/api/sensors`, and `/api/check` concurrently under Flask's threaded server, sharing a single cached `Hindsight` instance across threads occasionally triggered an `aiohttp` event-loop error (`Timeout context manager should be used inside a task`). Constructing a fresh `Hindsight` client per call—wrapped in a 2-attempt retry helper (`call_with_retry`) that retries transient timeouts/503s but fails fast on `401/403` auth errors—made concurrent requests completely stable.
4. **Separate advisory memory from schedule mutation.** Having `reflect()` generate a `cmms_prevention_draft` is only useful if it reaches the plant's scheduling system without blindly overwriting production intervals. Queuing drafts as `pending_approval` work orders (`WO-2026-0001`) gives reliability managers a human-in-the-loop gate before changes hit Fiix, UpKeep, Maximo, or SAP PM.
