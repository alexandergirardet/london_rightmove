from fastapi.testclient import TestClient

from app.backend.app.main import app

client = TestClient(app)
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'Hello': 'World'}

def test_get_property():
    response = client.get("/get_property/140642000")
    print(response.json())
    assert response.status_code == 200
    # assert response.json() == {"detail": "Item not found"}