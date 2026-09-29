"""app.py - Flask gateway for Machine Memory Agent."""
from flask import Flask, request, jsonify, render_template
import demo_agent as A

app = Flask(__name__)

def body():
    return request.get_json(force=True, silent=True) or {}

@app.errorhandler(Exception)
def on_error(e):
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/overview")
def overview():
    return jsonify(A.overview())

@app.route("/api/sensors")
def sensors():
    return jsonify(A.sensors(request.args.get("tick", 0, type=int)))

@app.route("/api/check", methods=["POST"])
def check():
    d = body(); s = (d.get("symptoms") or "").strip()
    if not s: return jsonify({"error": "Describe the symptoms first."}), 400
    return jsonify(A.check_before_repair(s, d.get("lang") or "English"))

@app.route("/api/compare", methods=["POST"])
def compare():
    q = (body().get("question") or "").strip()
    if not q: return jsonify({"error": "Enter a question."}), 400
    return jsonify(A.compare(q))

@app.route("/api/feedback", methods=["POST"])
def feedback():
    d = body(); m = (d.get("machine_id") or "").strip().upper(); fix = (d.get("fix") or "").strip()
    if not m or not fix: return jsonify({"error": "Machine ID and the fix that was tried are required."}), 400
    return jsonify(A.log_feedback(m, fix, bool(d.get("worked", True))))

@app.route("/api/improvement", methods=["POST"])
def improvement():
    d = body()
    return jsonify(A.improvement(d.get("machine_id", "MCH-017"), d.get("question", "MCH-017 vibration climbing, bearing housing warm"),
                                 d.get("fix", "Re-greased bearing housing"), bool(d.get("worked", False))))

@app.route("/api/reset_live", methods=["POST"])

def reset_live():
    A.reset_live(); A.reset_outbox(); return jsonify({"ok": True})

@app.route("/api/twin/<machine>")
def twin(machine):
    return jsonify(A.digital_twin(machine.upper()))

@app.route("/api/cmms/send", methods=["POST"])
def cmms_send():
    d = body(); m = (d.get("machine_id") or "").strip().upper(); draft = d.get("draft") or ""
    return jsonify(A.send_to_cmms(m, draft))

@app.route("/api/cmms/outbox")
def cmms_outbox():
    return jsonify({"items": A.load_outbox()})

@app.route("/api/handoff", methods=["POST"])
def handoff():
    return jsonify(A.handoff(body().get("lang") or "English"))

@app.route("/api/eval", methods=["POST"])
def run_eval():
    return jsonify(A.run_eval())

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
