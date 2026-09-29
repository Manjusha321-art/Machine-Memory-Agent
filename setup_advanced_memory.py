"""setup_advanced_memory.py - create per-machine mental models (digital twins) and
standing directives (safety/SOP policy) in the bank. Run once after seed_memory.py.

Mental models: a self-updating natural-language profile per machine, synthesized by
Hindsight from all retained facts about it, and refreshed automatically as new
incidents/feedback are retained (trigger=refresh_after_consolidation).

Directives: standing rules that reflect() applies automatically. Untagged directives
always apply (e.g. the grounding policy); tagged ones apply only to matching machines.
"""
import json, os, sys, time
from hindsight_client import Hindsight

BANK_ID = os.environ.get("BANK_ID", "factory-floor-v2")
key = os.environ.get("HINDSIGHT_API_KEY")
if not key:
    sys.exit("HINDSIGHT_API_KEY is not set. Export it, then run again.")

client = Hindsight(base_url=os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io"), api_key=key)
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "machine_maintenance_history.json")) as f:
    records = json.load(f)
machines = sorted({r["machine_id"] for r in records})
by_machine = {m: [r["machine_type"] for r in records if r["machine_id"] == m][0] for m in machines}


def twin_id(machine):
    return f"digital-twin-{machine.lower()}"


print(f"Creating {len(machines)} digital twins (mental models)...")
for m in machines:
    mid = twin_id(m)
    try:
        client.create_mental_model(
            bank_id=BANK_ID, id=mid, name=f"{m} digital twin",
            source_query=(f"Build a living profile of machine {m} ({by_machine[m]}): its known failure modes and "
                          f"precursor signals, root causes, fixes that worked, fixes that failed and should not be "
                          f"repeated, and its current risk level based on the most recent incidents and technician "
                          f"feedback. Be specific: part numbers, tolerances, technician names, dates."),
            tags=[m], max_tokens=900, trigger={"refresh_after_consolidation": True})
        print(" created", mid)
    except Exception as e:
        print(" (skip, may already exist)", mid, "-", e)

print("Refreshing twins so content is ready immediately (normally happens on its own after new retains)...")
for m in machines:
    try:
        client.refresh_mental_model(bank_id=BANK_ID, mental_model_id=twin_id(m))
    except Exception as e:
        print(" refresh failed for", twin_id(m), "-", e)

print("Waiting for generation (this takes a couple of minutes for all machines)...")
pending = set(machines)
deadline = time.time() + 180
while pending and time.time() < deadline:
    time.sleep(8)
    for m in list(pending):
        try:
            mm = client.get_mental_model(bank_id=BANK_ID, mental_model_id=twin_id(m), detail="content")
            content = getattr(mm, "content", None) or ""
            if content and "Generating content" not in content:
                pending.discard(m)
                print(f" ready: {m} ({len(content)} chars)")
        except Exception:
            pass
if pending:
    print(f"Still generating for {sorted(pending)} - they'll be ready within a minute or two; the app degrades gracefully until then.")

print("\nCreating standing directives...")
DIRECTIVES = [
    dict(name="grounding-policy", content=(
        "State only facts, technician names, dates, and part numbers present in retrieved memory. "
        "If retrieved memory is thin or absent for a machine, say so explicitly rather than guessing."),
        priority=10, tags=None),
    dict(name="loto-before-rotating-equipment", content=(
        "Before any bearing, motor, coupling, or seal replacement on rotating equipment, the action checklist "
        "must include lockout-tagout (LOTO) verification as the first step."),
        priority=5, tags=None),
    dict(name="mch-042-filter-before-reset", content=(
        "MCH-042 (Industrial Air Compressor - Unit B): do not recommend clearing a thermal overload trip a second "
        "time without first confirming intake filter replacement - repeated trips without filter service caused "
        "two downtime incidents in 2025."),
        priority=1, tags=["MCH-042"]),
]
for d in DIRECTIVES:
    try:
        client.create_directive(bank_id=BANK_ID, name=d["name"], content=d["content"], priority=d["priority"],
                                 is_active=True, tags=d["tags"])
        print(" created directive:", d["name"])
    except Exception as e:
        print(" (skip, may already exist)", d["name"], "-", e)

print("\nDone.")
