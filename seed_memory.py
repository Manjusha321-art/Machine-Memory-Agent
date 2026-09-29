"""seed_memory.py - retain maintenance records into Hindsight (run once)."""
import json, os, sys, time
from hindsight_client import Hindsight

BANK_ID = os.environ.get("BANK_ID", "factory-floor-v2")
key = os.environ.get("HINDSIGHT_API_KEY")
if not key:
    sys.exit("HINDSIGHT_API_KEY is not set. Export it, then run again.")

def note(r):
    tried = "; ".join(r["repairs_attempted"]) or "none before the final fix"
    return (f"[{r['id']}] Machine {r['machine_id']} ({r['machine_type']}) failure on {r['date']}. "
            f"Precursor signals: {'; '.join(r['precursor_signals'])}. Failure: {r['failure']}. Severity: {r['severity']}. "
            f"Downtime: {r['downtime_hours']} hours. Repairs attempted that did not work: {tried}. "
            f"Final fix: {r['final_fix']}. Root cause: {r['root_cause']}. Technician: {r['technician']}.")

client = Hindsight(base_url=os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io"), api_key=key)

if "--reset" in sys.argv:
    try:
        client.delete_bank(BANK_ID)
        print(f"Deleted existing bank '{BANK_ID}' to reseed cleanly.")
    except Exception as e:
        print(f"(no existing bank to delete, or delete failed: {e})")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "machine_maintenance_history.json")) as f:
    records = json.load(f)
for r in records:
    client.retain(bank_id=BANK_ID, content=note(r),
                  context=f"maintenance record {r['id']} | machine {r['machine_id']} | tags: {', '.join(r['tags'])}",
                  timestamp=f"{r['date']}T09:00:00Z", tags=[r["machine_id"], *r["tags"]])
    print("retained", r["id"], r["machine_id"]); time.sleep(0.3)
print(f"Done: {len(records)} records in bank '{BANK_ID}'. Wait ~1 minute for indexing before demoing.")
