# Machine Memory Agent

An AI agent that gives every industrial machine its own persistent memory —
so a technician facing a failure today can instantly draw on what happened
the last time, even if that repair was done by someone else, months ago.

## The Problem

A technician fixes a machine and writes a service report. Months later, a
different technician sees similar symptoms on the same machine — but has no
idea it's happened before. The knowledge is trapped in old reports nobody
reads, or in one person's memory. The same failures get repeated, and the
same diagnostic time gets spent from scratch every time.

## The Idea

Every repair becomes a memory: what the machine looked/sounded like before
it failed, what actually failed, what was tried, what fixed it, and why it
happened. When a technician describes what they're currently seeing, the
agent checks its memory for a match — and tells them plainly if this has
happened before.

## Architecture

```
Technician input (symptoms / current sensor readings)
              │
              ▼
        demo_agent.py
              │
      ┌───────┴───────┐
      │               │
   RECALL          REFLECT
 (fast lookup)   (reasoning over
      │           multiple memories)
      │               │
      └───────┬───────┘
              ▼
        Hindsight Memory
      (bank: "factory-floor-v2")
              │
              ▼
         Groq LLM
   (openai/gpt-oss-120b)
              │
              ▼
   Grounded, machine-specific
        response
```

## How Hindsight Memory Is Used

This project uses all three of Hindsight's core operations, plus two of
its higher-level constructs (mental models and directives):

**Retain** (`seed_memory.py`) — Each of the 11 historical maintenance
records is retained as a natural-language field note (not raw structured
data), so Hindsight's extraction pass can pull out entities, facts, and
temporal information the way it would from a real technician's report.
Each record includes: machine ID, precursor signals observed before
failure, the failure itself, repairs attempted, the fix that actually
worked, and the root cause.

**Recall** — Used in the side-by-side demo mode (`compare`) and the
automated 10-case benchmark evaluation (`run_eval`). A technician's
question is used as a recall query against the bank; the retrieved
memories are then passed to the LLM as grounding context before it
answers. This is intentionally lightweight — recall's job here is fast
retrieval, and the LLM does the reasoning on top of it.

**Reflect** — Used in the pre-repair check mode (`check_before_repair`).
Reflect is suited to judgment questions rather than plain lookup — "does
this match a known failure pattern, and what should be done about it" is
a reasoning task over multiple memories, not a single retrieval, so this
mode calls `hindsight.reflect()` directly with a JSON `response_schema`
and `include_facts=True`, instead of doing recall + a separate LLM call.
The structured output *is* the diagnosis; the `based_on` field it returns
names the exact memories, mental models, and directives that produced it,
which is what the "Why does the agent think this?" panel renders — the
grounding isn't a hand-rolled prompt convention, it's what the memory
system itself reports using.

Why this distinction matters: recall is fast and cheap, appropriate when
you already know you'll do further reasoning downstream (the comparison
demo). Reflect is heavier but appropriate when the question itself
requires synthesizing across multiple memories before an answer makes
sense (the pre-repair check).

**Mental models** (`setup_advanced_memory.py`) — Each machine gets a
"digital twin": a mental model (`digital-twin-mch-017`, etc.) whose
`source_query` asks Hindsight to synthesize that machine's full failure
history, root causes, and which fixes worked or failed into a living
profile. It's created with `trigger={"refresh_after_consolidation": True}`,
so it keeps itself current as new incidents and technician feedback get
retained — it isn't a cached summary we generate once. Cross-fleet
correlation (`cross_fleet_from`) is also powered this way: it's whatever
other machines Hindsight's own reflection surfaced as relevant, not a
local keyword/tag match against the seed file.

**Directives** — Standing plant policy encoded as memory rather than
prompt text: a grounding rule ("never invent IDs, dates, or part
numbers"), a safety rule (require lockout-tagout before touching rotating
equipment), and a machine-specific rule for MCH-042. `reflect()` applies
these automatically, and they show up in the response's
`directives_applied` — visible proof the agent is enforcing standing
policy, not just retrieving incident history.

## What Makes This Different From a Chatbot With Search

A chatbot with search waits for someone to ask the right question and dig
through old reports themselves. This agent proactively recognizes "I've
seen this shape before" — the same way a technician who's worked at the
plant for years would say "wait, we tried that, it didn't work because
X." The memory isn't just a data lookup; it's what turns a generic
answer into a specific, accountable one.

## Setup

```bash
pip install -r requirements.txt

export HINDSIGHT_API_KEY="your-hindsight-cloud-api-key"
export HINDSIGHT_BASE_URL="https://api.hindsight.vectorize.io"
export GROQ_API_KEY="your-groq-api-key"

python seed_memory.py --reset      # loads the 11 synthetic maintenance records
python setup_advanced_memory.py    # creates per-machine digital twins + standing directives
python app.py                      # launch the interactive web application
```

See `SETUP.md` for detailed, step-by-step instructions.

## Interactive Web Application (`app.py`)

```bash
python app.py
```
Open `http://localhost:5000`. Features:
- **Live Fleet & Sensor Watch** — Real-time sparkline telemetry across monitored machines with proactive precursor anomaly detection and 1-click alert diagnosis.
- **Hands-Free Voice & Text Input** — Browser speech recognition for hands-free floor diagnostics.
- **Tab 1: Diagnose & Digital Twin** — 20 realistic scenarios (including cross-fleet and novel/unknown machine edge cases), Hindsight `reflect()` triage, dead-end warnings, financial downtime risk (`$/hr` by asset class), standing Directives enforcement, citation provenance, self-updating Hindsight Mental Model Digital Twin, 1-click CMMS work-order drafting, and closed-loop technician outcome logging.
- **Tab 2: Closed-Loop Improvement** — Automated 3-step live learning proof: diagnoses a machine, retains a failed repair (`LIVE-XXX`) into Hindsight, and re-runs the diagnosis to show the new dead-end warning and updated checklist side-by-side.
- **Tab 3: With vs. Without Memory** — Side-by-side comparison between a generic LLM (0 plant records) and the Hindsight-grounded agent.
- **Tab 4: Shift Handoff, CMMS Outbox & 10-Case Evaluation** — Generates shift handoff briefs, displays queued CMMS preventive work orders (`WO-YYYY-XXXX`), and runs a 10-case retrieval & grounding benchmark.

Try:
```
MCH-017 pump vibration is climbing again, bearing housing feels warm
```
```
MCH-009 spindle motor current is up over baseline, seeing chatter marks
```

Both are designed to surface repeat-failure patterns already present in
the seeded data (`machine_maintenance_history.json`).

## Live Learning: Proving Memory Grows, Not Just Recalls

A static, seeded memory bank can look impressive without actually
demonstrating that the system learns. To make that concrete rather than
asserted, `retain_new_incident_and_recall()` retains a brand-new incident
in real time — as if a technician just logged a fresh repair — and
immediately queries for it. The result shows the new incident already
present and ranked in the recalled memories seconds after being retained,
which is the clearest evidence that this is a system whose knowledge
grows with use, not a fixed lookup table dressed up as an agent.

## An Honest Limitation

Early versions of this agent had real problems, all caught during testing
rather than assumed away:

1. **Hallucination.** The agent occasionally invented plausible-sounding
   details that weren't actually in retrieved memory — a fabricated
   repair date, a technician name that didn't exist in any record. Fixed
   with an explicit prompt-level constraint: state only what's present in
   retrieved memory, and say plainly when something isn't on record
   rather than guessing.

2. **Unbounded recall.** The initial recall implementation didn't set a
   token/result budget, so it returned every memory in the entire bank
   (~65 fragments) for any query, regardless of relevance. Fixed by using
   Hindsight's `budget` and `max_tokens` recall parameters plus a manual
   result cap, after confirming the exact installed client's method
   signature rather than assuming one from documentation.

3. **Occasional transient timeouts under load.** Once real concurrent
   traffic was tested (a page load fires overview, sensors, and a
   diagnosis request together), the sync client occasionally raised a
   timeout - sometimes a genuine hiccup from the memory service, at least
   once traced to an async event-loop/session mismatch under Flask's
   threaded dev server. Fixed two ways: `hs()` now builds a fresh client
   per call instead of caching one across requests, and every memory/LLM
   call retries once automatically on anything that looks transient
   (timeout, 429/503, connection reset) - but never on an auth failure,
   which won't fix itself on a second try. A failure that survives the
   retry still renders as a plain-language message in the UI instead of a
   broken page or a raw stack trace.

All three fixes mattered for the same underlying reason: a system meant to be
trusted on a factory floor needs to be traceable and focused, not just
articulate.

## Path to Production

This demo runs on one synthetic plant and one shared API key. Getting it
onto a real factory floor is mostly integration work, not a redesign:

- **Multi-tenancy is already the shape of the code, not a rewrite.**
  `BANK_ID` is the only thing scoping a deployment to one plant; one bank
  per site (or per production line, for a plant large enough to want
  that) is a config change, not new logic. Auth is the real gap: this
  demo uses one shared key for everyone, production needs per-technician
  SSO and a role split (who can log a "fix worked/failed" outcome vs. who
  can only read the diagnosis).

- **`cmms_prevention_draft` is deliberately shaped to leave, not just be
  read — and that's now demoed, not just described.** The "Send to CMMS"
  button (`send_to_cmms()`, `/api/cmms/send`, `/api/cmms/outbox`) queues
  the draft as a work order with `status: pending_approval`, visible in
  an outbox in the Handoff tab. It's a mock queue, not a real Fiix/UpKeep/
  Maximo/SAP PM call — swapping the local outbox file for that system's
  webhook is the actual integration work — but the *shape* of "diagnosis
  produces a draft, a human approves it before it touches the real
  schedule" is working end to end today, not just asserted in a README.
  That approval step is deliberate: the difference between "an agent that
  talks" and "an agent that changes the maintenance schedule" is exactly
  why the draft stays advisory.

- **The sensor feed is synthetic on purpose, but the alert logic isn't
  throwaway.** `sensors()`'s "climbing toward a known failure signature"
  check is the same shape whether the data comes from a canned array or a
  live OPC-UA/MQTT tag from the plant's SCADA system — swapping the
  source is the integration; the pattern-matching against precursor
  signals in memory doesn't change.

- **Directives don't have to be hand-written.** The three in this demo
  were written for the pitch; in practice they'd be seeded from the
  plant's existing safety manual and SOPs, which turns "policy compliance
  built into every diagnosis" from a demo trick into the actual selling
  point for a reliability manager evaluating this.

- **Buyer and rollout:** this is sold to a plant reliability or
  maintenance manager, not to a single technician — the value compounds
  across a fleet and across staff turnover, which one technician's local
  knowledge never does. The honest rollout path is a pilot on one asset
  class with a repeat-failure history (exactly what this demo's synthetic
  data models), not a fleet-wide launch on day one.

## Tech Stack

- **Memory:** [Hindsight](https://github.com/vectorize-io/hindsight) —
  [docs](https://hindsight.vectorize.io/)
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Data:** synthetic maintenance records modeled on realistic industrial
  failure patterns (bearing wear, thermal overload, lubrication scheduling
  failures, coolant misalignment, etc.)

Learn more about agent memory in general at
[Vectorize's overview](https://vectorize.io/what-is-agent-memory).

## Project Structure

```
machine-memory-agent/
├── machine_maintenance_history.json   # 11 synthetic maintenance records
├── seed_memory.py                     # retain pipeline
├── setup_advanced_memory.py           # creates digital twins (mental models) + directives
├── demo_agent.py                      # recall/reflect/retain logic, retry/error handling, CMMS outbox
├── app.py                             # Flask web UI wrapping demo_agent.py
├── templates/
│   └── index.html                     # web UI page
├── test_demo_agent.py                 # unit tests: retry logic, confidence, sensors, concurrency
├── test_app.py                        # integration tests: Flask routes and edge cases
├── live_incidents.json                # closed-loop technician feedback (demo state)
├── cmms_outbox.json                   # mock CMMS work-order queue (demo state)
├── SETUP.md                           # step-by-step setup instructions
└── README.md
```

Run the tests with `pytest` — 35 tests, no API keys required.
