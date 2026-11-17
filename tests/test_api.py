import os

import pytest
from fastapi.testclient import TestClient

from app.api import app, MODEL_PATH, MAX_BATCH

# The model file is not in git, so skip these tests if it hasn't been trained
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Train the model first: python3 train_final.py",
)

# Two real Criteo customers with known scores from earlier checks
HIGH = {"f0": 21.96814727783203, "f1": 10.059654235839844, "f2": 8.297974586486816,
        "f3": 4.679881572723389, "f4": 10.280525207519531, "f5": 4.115453243255615,
        "f6": -4.595459461212158, "f7": 4.83381462097168, "f8": 3.874502658843994,
        "f9": 18.380111694335938, "f10": 5.300374984741211, "f11": -0.16867922246456146}

LOW = {"f0": 22.331823348999023, "f1": 10.059654235839844, "f2": 8.214383125305176,
       "f3": 4.679881572723389, "f4": 10.280525207519531, "f5": 4.115453243255615,
       "f6": -2.4111146926879883, "f7": 4.83381462097168, "f8": 3.971858024597168,
       "f9": 13.190055847167969, "f10": 5.300374984741211, "f11": -0.16867922246456146}


@pytest.fixture(scope="module")
def client():
    # The "with" block runs startup, which loads the model
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["target"] == "visit"
    assert len(body["features"]) == 12
    assert body["score_p90"] > body["score_median"]


def test_score_known_values(client):
    r = client.post("/score", json={"customers": [HIGH, LOW]})
    assert r.status_code == 200
    first, second = r.json()["results"]
    assert first["uplift"] == pytest.approx(0.04923, abs=1e-4)
    assert second["uplift"] == pytest.approx(0.00006, abs=1e-4)


def test_score_ranking_is_consistent(client):
    first, second = client.post("/score", json={"customers": [HIGH, LOW]}).json()["results"]
    # Higher uplift must mean a smaller "top X%" number
    assert first["top_percent"] < second["top_percent"]
    assert 0 <= first["top_percent"] <= 100
    assert 0 <= second["top_percent"] <= 100


def test_recommend_known_decision(client):
    r = client.post("/recommend", json={"customers": [HIGH, LOW], "value": 1.0, "cost": 0.01})
    assert r.status_code == 200
    body = r.json()
    assert body["n_customers"] == 2
    assert body["n_contact"] == 1
    assert [x["contact"] for x in body["results"]] == [True, False]
    assert body["expected_total_profit"] == pytest.approx(0.03923, abs=1e-4)


def test_recommend_high_cost_contacts_nobody(client):
    r = client.post("/recommend", json={"customers": [HIGH, LOW], "value": 1.0, "cost": 5.0})
    body = r.json()
    assert body["n_contact"] == 0
    assert body["expected_total_profit"] == 0


def test_recommend_respects_max_contacts(client):
    # With zero cost both customers have positive profit, so the cap decides
    r = client.post("/recommend", json={"customers": [HIGH, LOW], "value": 1.0,
                                        "cost": 0.0, "max_contacts": 1})
    body = r.json()
    assert body["n_contact"] == 1
    assert body["results"][0]["contact"] is True   # the higher-profit customer is kept


def test_missing_feature_is_rejected(client):
    bad = {k: v for k, v in HIGH.items() if k != "f5"}
    assert client.post("/score", json={"customers": [bad]}).status_code == 422


def test_empty_customer_list_is_rejected(client):
    assert client.post("/score", json={"customers": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    too_many = [HIGH] * (MAX_BATCH + 1)
    assert client.post("/score", json={"customers": too_many}).status_code == 422


def test_negative_cost_is_rejected(client):
    r = client.post("/recommend", json={"customers": [HIGH], "value": 1.0, "cost": -1.0})
    assert r.status_code == 422


def test_non_numeric_feature_is_rejected(client):
    bad = dict(HIGH, f0="not a number")
    assert client.post("/score", json={"customers": [bad]}).status_code == 422