"""ASGI tests avoid the blocking thread portal used by Starlette TestClient."""

import asyncio

import httpx

from cad_env.api import CADAPI


def request(app, method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


class TestCADAPI:
    def test_initialization(self):
        api = CADAPI()
        assert api.app is not None and api.env is not None

    def test_root_endpoint(self):
        response = request(CADAPI().app, "GET", "/")
        assert response.status_code == 200
        assert "version" in response.json()

    def test_health_check(self):
        response = request(CADAPI().app, "GET", "/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_create_document(self):
        response = request(
            CADAPI().app, "POST", "/documents/create", json={"name": "TestDoc"}
        )
        assert response.status_code == 200
        assert response.json()["document_id"]

    def test_execute_command_reports_backend_status(self):
        api = CADAPI()
        request(api.app, "POST", "/documents/create", json={"name": "TestDoc"})
        response = request(
            api.app,
            "POST",
            "/commands/execute",
            json={
                "command": "create_box",
                "parameters": {"length": 10, "width": 5, "height": 3},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == data["result"]["success"]
        if data["result"].get("simulation"):
            assert not data["success"]
