from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_identical_phrases():

    phrases = ["What profits were clocked by Cholamandalam in 2020?",
               "Which Lloyd's syndicates owns the maximum number of assets?"]

    for phrase in phrases:

        response = client.post("/closest_phrase", json={"text": phrase})
    
        assert response.status_code == 200
        assert response.json()["closest_phrase"] == phrase
