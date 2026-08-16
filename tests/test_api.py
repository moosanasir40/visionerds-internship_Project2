import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def test_normal_chat():
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "test-session-1", "message": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["path_taken"] == "normal_reply"
        assert "reply" in data

def test_tool_action_chat():
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "test-session-1", "message": "load review"})
        assert response.status_code == 200
        data = response.json()
        assert data["path_taken"] == "tool_action:load_saved_review"