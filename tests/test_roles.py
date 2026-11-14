from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from auth import create_doctor
from conftest import auth_headers, doctor_headers

SECRET = "test-secret-not-for-production"


async def test_signup_creates_patient(client):
    resp = await client.post(
        "/api/signup", json={"username": "alice", "password": "password123"}
    )
    assert resp.json()["role"] == "patient"
    me = (await client.get(
        "/api/me", headers={"Authorization": f"Bearer {resp.json()['token']}"}
    )).json()
    assert me == {"username": "alice", "role": "patient", "display_name": None}


async def test_doctor_login_returns_role_and_display_name(client):
    await doctor_headers(client)
    resp = await client.post(
        "/api/login", json={"username": "dr.ting", "password": "doctor-password-1"}
    )
    assert resp.json()["role"] == "doctor"
    me = (await client.get(
        "/api/me", headers={"Authorization": f"Bearer {resp.json()['token']}"}
    )).json()
    assert me["role"] == "doctor"
    assert me["display_name"] == "Dr. Ting Kai"


async def test_create_doctor_rejects_taken_username(client):
    await doctor_headers(client)
    async with client.app.state.session_factory() as session:
        with pytest.raises(ValueError):
            await create_doctor(session, "dr.ting", "doctor-password-1", "Other")


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/documents/active"),
        ("GET", "/api/checklist"),
        ("GET", "/api/care-team"),
        ("POST", "/api/ask"),
        ("GET", "/api/conversation"),
        ("POST", "/api/documents"),
    ],
)
async def test_doctor_gets_403_on_patient_routes(client, method, path):
    headers = await doctor_headers(client)
    resp = await client.request(method, path, headers=headers)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "This page is for patients."


def _token(user_id, username, **extra):
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        **extra,
    }
    return {"Authorization": "Bearer " + jwt.encode(payload, SECRET, algorithm="HS256")}


async def test_old_token_without_role_still_works(client):
    await auth_headers(client, "alice")
    resp = await client.get("/api/care-team", headers=_token(1, "alice"))
    assert resp.status_code == 200


async def test_role_comes_from_database(client):
    await auth_headers(client, "alice")
    resp = await client.get("/api/doctor/ping", headers=_token(1, "alice", role="doctor"))
    assert resp.status_code == 403
    assert resp.json()["detail"] == "This page is for doctors."


async def test_doctor_ping_allows_doctor(client):
    headers = await doctor_headers(client)
    resp = await client.get("/api/doctor/ping", headers=headers)
    assert resp.json() == {"ok": True}
