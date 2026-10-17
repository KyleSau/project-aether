from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app import app

c = TestClient(app)


def test_health_and_data():
    assert c.get("/api/health").json()["ok"]
    assert c.get("/api/vehicle").json()["schema"] == "aether-vehicle/1"
    assert c.get("/api/thermo").json()["schema"] == "aether-thermo/1"


def test_simulate_reaches_orbit():
    r = c.post("/api/simulate", json={}).json()
    assert r["ok"] and r["orbit_achieved"]


def test_openfoam_not_connected():
    s = c.get("/api/openfoam/status").json()
    assert s["status"] in ("NOT CONNECTED", "CONNECTED")
    assert s["requirements"]
