"""Integration tests for the Flask routes in app.py: request validation, edge cases
(unknown machine, missing fields), and that known-machine paths wire correctly into
demo_agent without actually reaching Hindsight/Groq (those calls are monkeypatched).

Run with: pytest
No HINDSIGHT_API_KEY / GROQ_API_KEY needed - nothing here touches the network.
"""
import types
import pytest
import demo_agent as A
import app as flaskapp


@pytest.fixture
def client():
    flaskapp.app.testing = True
    return flaskapp.app.test_client()


@pytest.fixture(autouse=True)
def isolate_live_file(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "LIVE_FILE", str(tmp_path / "live_incidents.json"))
    monkeypatch.setattr(A, "CMMS_FILE", str(tmp_path / "cmms_outbox.json"))
    A.reset_live()
    A.reset_outbox()


def fake_reflect_response(structured_output, memories=None, mental_models=None, directives=None):
    based_on = types.SimpleNamespace(memories=memories or [], mental_models=mental_models or [], directives=directives or [])
    return types.SimpleNamespace(structured_output=structured_output, structured_output_error=None, based_on=based_on)


def test_index_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Machine Memory Agent" in r.data


def test_overview_shape(client):
    d = client.get("/api/overview").get_json()
    assert "machines" in d and "stats" in d
    assert {"memories", "live", "repeat_patterns", "hours_saved", "dollars_saved", "cost_per_hour"} <= set(d["stats"])
    assert all({"machine_id", "type", "risk", "incidents"} <= set(m) for m in d["machines"])


def test_sensors_shape(client):
    d = client.get("/api/sensors?tick=3").get_json()
    assert d["tick"] == 3
    assert "machines" in d and "alerts" in d


def test_check_requires_symptoms(client):
    r = client.post("/api/check", json={"symptoms": "   "})
    assert r.status_code == 400
    assert "Describe the symptoms" in r.get_json()["error"]


def test_check_unknown_machine_is_novel_without_touching_network(client, monkeypatch):
    def boom(*a, **k): raise AssertionError("should not call reflect for an unknown machine")
    monkeypatch.setattr(A, "reflect", boom)
    d = client.post("/api/check", json={"symptoms": "MCH-999 strange noise"}).get_json()
    assert d["agent"]["triage_level"] == "novel"


def test_check_known_machine_uses_reflect(client, monkeypatch):
    fake = fake_reflect_response({
        "triage_level": "critical", "badge_label": "TEST", "primary_machine": "MCH-017",
        "verdict_summary": "test", "dead_end_warning": "", "root_cause": "test",
        "action_checklist": ["step"], "citations": [], "cmms_prevention_draft": "",
    })
    monkeypatch.setattr(A, "reflect", lambda *a, **k: (fake, None))
    d = client.post("/api/check", json={"symptoms": "MCH-017 pump vibration climbing"}).get_json()
    assert d["agent"]["triage_level"] == "critical"
    assert d["machine"] == "MCH-017"
    assert d["cost_per_hour_used"] == A.cost_per_hour("MCH-017")


def test_check_surfaces_reflect_error_gracefully_as_200(client, monkeypatch):
    """A memory-service failure should render as a message in the payload, not a 500 -
    the frontend renders agent.error inline rather than showing a broken page."""
    monkeypatch.setattr(A, "reflect", lambda *a, **k: (None, "Memory service timed out. Try again in a moment."))
    r = client.post("/api/check", json={"symptoms": "MCH-017 pump vibration climbing"})
    assert r.status_code == 200
    assert "timed out" in r.get_json()["agent"]["error"]


def test_compare_requires_question(client):
    r = client.post("/api/compare", json={"question": ""})
    assert r.status_code == 400


def test_feedback_requires_fields(client):
    r = client.post("/api/feedback", json={"machine_id": "MCH-017"})
    assert r.status_code == 400


def test_feedback_unknown_machine_rejected_without_network(client, monkeypatch):
    def boom(*a, **k): raise AssertionError("should not call retain for an unknown machine")
    monkeypatch.setattr(A, "retain", boom)
    d = client.post("/api/feedback", json={"machine_id": "MCH-999", "fix": "x", "worked": True}).get_json()
    assert d["retained"] is False


def test_feedback_known_machine_logged(client, monkeypatch):
    monkeypatch.setattr(A, "retain", lambda *a, **k: (True, None))
    d = client.post("/api/feedback", json={"machine_id": "mch-017", "fix": "replaced bearing", "worked": True}).get_json()
    assert d["retained"] is True
    assert d["id"].startswith("LIVE-")


def test_reset_live_clears_ledger(client, monkeypatch):
    monkeypatch.setattr(A, "retain", lambda *a, **k: (True, None))
    client.post("/api/feedback", json={"machine_id": "MCH-017", "fix": "x", "worked": False})
    assert len(A.load_live()) == 1
    r = client.post("/api/reset_live")
    assert r.get_json() == {"ok": True}
    assert A.load_live() == []


def test_twin_unknown_machine_without_touching_network(client, monkeypatch):
    def boom(): raise AssertionError("should not call hs() for an unknown machine")
    monkeypatch.setattr(A, "hs", boom)
    d = client.get("/api/twin/MCH-999").get_json()
    assert "error" in d


def test_twin_known_machine(client, monkeypatch):
    fake_mm = types.SimpleNamespace(content="## Twin content", last_refreshed_at="2026-01-01", is_stale=False)
    fake_client = types.SimpleNamespace(get_mental_model=lambda **k: fake_mm)
    monkeypatch.setattr(A, "hs", lambda: fake_client)
    d = client.get("/api/twin/MCH-017").get_json()
    assert d["content"] == "## Twin content"


def test_cmms_send_and_outbox_round_trip(client):
    d = client.post("/api/cmms/send", json={"machine_id": "MCH-017", "draft": "Lock the schedule to monthly."}).get_json()
    assert d["id"].startswith("WO-")
    assert d["status"] == "pending_approval"
    outbox = client.get("/api/cmms/outbox").get_json()["items"]
    assert len(outbox) == 1 and outbox[0]["machine_id"] == "MCH-017"


def test_cmms_send_unknown_machine_rejected(client):
    d = client.post("/api/cmms/send", json={"machine_id": "MCH-999", "draft": "x"}).get_json()
    assert "error" in d
    assert client.get("/api/cmms/outbox").get_json()["items"] == []


def test_reset_live_also_clears_cmms_outbox(client):
    client.post("/api/cmms/send", json={"machine_id": "MCH-017", "draft": "x"})
    assert len(A.load_outbox()) == 1
    client.post("/api/reset_live")
    assert A.load_outbox() == []
