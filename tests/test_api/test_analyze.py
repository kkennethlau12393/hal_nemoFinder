"""Tests for API endpoints (analyze, health, jobs).

Uses httpx.AsyncClient with ASGITransport against the FastAPI app.
Database sessions and Celery tasks are mocked to keep these as unit tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session():
    """Return an AsyncMock that behaves like an AsyncSession."""
    session = AsyncMock()
    # Make commit / refresh no-ops
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    return session


def _override_get_session(mock_session):
    """Return an async generator dependency override for get_session."""
    async def _override():
        yield mock_session
    return _override


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------


class TestAnalyzeEndpointAcceptsText:
    """A valid text submission should return 201 with a job_id."""

    async def test_analyze_endpoint_accepts_text(self) -> None:
        mock_session = _mock_session()
        # When refresh is called the job object gets its id populated.
        # We simulate that by setting the id attribute on whatever was added.
        fake_job_id = uuid4()

        original_add = mock_session.add

        def _side_effect_add(obj):
            obj.id = fake_job_id
            obj.status = "pending"

        mock_session.add = MagicMock(side_effect=_side_effect_add)

        app.dependency_overrides = {}
        from src.db.session import get_session
        app.dependency_overrides[get_session] = _override_get_session(mock_session)

        with patch("src.api.endpoints.analyze.celery_app") as mock_celery:
            mock_celery.send_task = MagicMock()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/analyze",
                    json={"text": "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has MW of 180.16 g/mol."},
                )

            assert response.status_code == 201
            body = response.json()
            assert "job_id" in body
            assert body["status"] == "pending"
            mock_celery.send_task.assert_called_once()

        app.dependency_overrides = {}


class TestAnalyzeEndpointRejectsEmpty:
    """An empty text submission should return 422."""

    async def test_analyze_endpoint_rejects_empty(self) -> None:
        mock_session = _mock_session()

        from src.db.session import get_session
        app.dependency_overrides[get_session] = _override_get_session(mock_session)

        with patch("src.api.endpoints.analyze.celery_app"):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/analyze",
                    json={"text": ""},
                )

            assert response.status_code == 422

        app.dependency_overrides = {}


class TestAnalyzeEndpointRejectsWhitespaceOnly:
    """Whitespace-only text should also be rejected."""

    async def test_analyze_endpoint_rejects_whitespace(self) -> None:
        mock_session = _mock_session()

        from src.db.session import get_session
        app.dependency_overrides[get_session] = _override_get_session(mock_session)

        with patch("src.api.endpoints.analyze.celery_app"):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/analyze",
                    json={"text": "   \n\t  "},
                )

            assert response.status_code == 422

        app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Health endpoint should return 200."""

    async def test_health_endpoint(self) -> None:
        mock_session = _mock_session()
        # Simulate SELECT 1 succeeding
        mock_session.execute = AsyncMock(return_value=MagicMock())

        from src.db.session import get_session
        app.dependency_overrides[get_session] = _override_get_session(mock_session)

        with (
            patch("src.api.endpoints.health._check_redis", new_callable=AsyncMock, return_value=True),
            patch("src.api.endpoints.health._check_verifiers", new_callable=AsyncMock, return_value={"chemical": True}),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/v1/health")

            assert response.status_code == 200
            body = response.json()
            assert "status" in body
            assert "database" in body

        app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# GET /api/v1/jobs
# ---------------------------------------------------------------------------


class TestJobsList:
    """Jobs list endpoint should return 200 with a list."""

    async def test_jobs_list(self) -> None:
        mock_session = _mock_session()
        # Simulate an empty result set
        mock_result = MagicMock()
        mock_result.all = MagicMock(return_value=[])
        mock_session.execute = AsyncMock(return_value=mock_result)

        from src.db.session import get_session
        app.dependency_overrides[get_session] = _override_get_session(mock_session)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/jobs")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)

        app.dependency_overrides = {}
