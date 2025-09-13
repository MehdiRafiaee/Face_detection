import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Face Recognition System" in response.text

def test_register_user_no_file():
    response = client.post("/api/v1/register/", data={"name": "test"})
    assert response.status_code == 422  # Missing file

# More tests can be added here
