"""Unit tests for the pure, network-free logic in demo_agent.py: the transient-error
retry wrapper, confidence scoring, source/cross-fleet extraction, sensor alert
thresholds, and the live-incident file lock that prevents concurrent feedback
submissions from colliding on the same LIVE-NNN id.

Run with: pytest
No HINDSIGHT_API_KEY / GROQ_API_KEY needed - nothing here touches the network.
"""
import concurrent.futures
import demo_agent as A


def test_retry_succeeds_after_one_transient_failure(monkeypatch):
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Timeout context manager should be used inside a task")
        return "ok"
    result, err = A.call_with_retry(flaky)
    assert result == "ok" and err is None and calls["n"] == 2


def test_retry_gives_up_after_max_attempts_on_persistent_transient_error(monkeypatch):
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    calls = {"n": 0}
    def always_times_out():
        calls["n"] += 1
        raise TimeoutError("request timeout")
    result, err = A.call_with_retry(always_times_out, attempts=2)
    assert result is None and err is not None and calls["n"] == 2


def test_retry_does_not_retry_auth_errors():
    calls = {"n": 0}
    def unauthorized():
        calls["n"] += 1
        raise RuntimeError("401 Unauthorized")
    result, err = A.call_with_retry(unauthorized, attempts=3)
    assert result is None and err is not None and calls["n"] == 1


def test_confidence_unknown_machine_is_zero():
    c = A.confidence("MCH-999 strange vibration", [])
    assert c["score"] == 0
    assert "Unknown machine" in c["type"]


def test_confidence_known_machine_no_memory_is_zero():
    c = A.confidence("MCH-017 pump vibration climbing", [])
    assert c["score"] == 0
    assert c["type"] == "No matching memory"


def test_confidence_same_machine_recurrence_scores_high():
    texts = ["[MAINT-2025-001] machine MCH-017 bearing failure", "[MAINT-2025-002] machine MCH-017 repeat bearing failure"]
    c = A.confidence("MCH-017 pump vibration climbing again, bearing housing warm", texts)
    assert c["type"] == "Same-machine recurrence"
    assert c["score"] >= 65


def test_confidence_cross_fleet_when_no_machine_named():
    texts = ["[MAINT-2025-007] machine MCH-028 belt tension"]
    c = A.confidence("conveyor motor running hot, belt drifting", texts)
    assert c["type"] == "Symptom / cross-fleet correlation"


def test_sources_from_dedupes_and_ignores_unknown_ids():
    blob = ["[MAINT-2025-001] machine MCH-017 ...", "[MAINT-2025-001] machine MCH-017 duplicate mention",
            "[MAINT-2025-999] this id does not exist"]
    srcs = A.sources_from(blob)
    assert [s["id"] for s in srcs] == ["MAINT-2025-001"]
    assert srcs[0]["machine"] == "MCH-017"


def test_cross_fleet_from_excludes_current_machine():
    blob = ["[MAINT-2025-001] machine MCH-017 ...", "[MAINT-2025-007] machine MCH-028 ..."]
    cross = A.cross_fleet_from(blob, "MCH-017")
    assert [r["machine_id"] for r in cross] == ["MCH-028"]
    cross_no_machine = A.cross_fleet_from(blob, None)
    assert {r["machine_id"] for r in cross_no_machine} == {"MCH-017", "MCH-028"}


def test_twin_id_format():
    assert A.twin_id("MCH-017") == "digital-twin-mch-017"


def test_sensors_no_alert_at_tick_zero():
    s = A.sensors(0)
    assert s["alerts"] == []


def test_sensors_alerts_by_final_tick():
    s = A.sensors(A.MAX_TICK)
    assert len(s["alerts"]) >= 1
    assert all(a["machine"] in s["machines"] for a in s["alerts"])


def test_log_feedback_unknown_machine_rejected():
    r = A.log_feedback("MCH-999", "replaced part", True)
    assert r["retained"] is False
    assert "Unknown machine" in r["error"]


def test_log_feedback_is_thread_safe(tmp_path, monkeypatch):
    """Regression test for the race this session fixed: concurrent feedback submissions
    reading live[] at the same length used to mint the same LIVE-NNN id and one write
    would silently clobber the other."""
    monkeypatch.setattr(A, "LIVE_FILE", str(tmp_path / "live_incidents.json"))
    monkeypatch.setattr(A, "retain", lambda *a, **k: (True, None))
    A.reset_live()

    def submit(i):
        return A.log_feedback("MCH-017", f"fix attempt {i}", worked=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(20)))

    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), f"duplicate LIVE ids assigned under concurrency: {ids}"
    assert len(A.load_live()) == 20


def test_send_to_cmms_unknown_machine_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CMMS_FILE", str(tmp_path / "cmms_outbox.json"))
    r = A.send_to_cmms("MCH-999", "some draft")
    assert "error" in r
    assert A.load_outbox() == []


def test_send_to_cmms_rejects_empty_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CMMS_FILE", str(tmp_path / "cmms_outbox.json"))
    r = A.send_to_cmms("MCH-017", "   ")
    assert "error" in r


def test_send_to_cmms_queues_a_work_order(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CMMS_FILE", str(tmp_path / "cmms_outbox.json"))
    A.reset_outbox()
    r = A.send_to_cmms("MCH-017", "Lock the lubrication schedule to monthly.")
    assert r["id"].startswith("WO-")
    assert r["machine_id"] == "MCH-017"
    assert r["status"] == "pending_approval"
    assert len(A.load_outbox()) == 1


def test_send_to_cmms_is_thread_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CMMS_FILE", str(tmp_path / "cmms_outbox.json"))
    A.reset_outbox()

    def submit(i):
        return A.send_to_cmms("MCH-017", f"draft {i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(20)))

    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids)), f"duplicate WO ids assigned under concurrency: {ids}"
    assert len(A.load_outbox()) == 20
