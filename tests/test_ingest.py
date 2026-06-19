"""Tests for the CDSE Sentinel-1 ingest module.

Every test mocks the HTTP layer by patching the :class:`requests.Session`
methods used by :mod:`oilspill.pipeline.ingest`. No test performs a real network
call; a session whose ``get``/``post`` raise is injected to prove it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from oilspill.pipeline import ingest
from oilspill.pipeline.ingest import (
    ODATA_PRODUCTS_URL,
    TOKEN_URL,
    Product,
    download_product,
    get_access_token,
    search_products,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cdse"

# A small AOI polygon over the Adriatic, as a GeoJSON Feature.
_AOI: dict[str, Any] = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [
                [12.0, 41.5],
                [14.5, 41.5],
                [14.5, 43.5],
                [12.0, 43.5],
                [12.0, 41.5],
            ]
        ],
    },
}


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


class _FakeResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self._chunks = chunks or []
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict[str, Any]:
        assert self._json is not None
        return self._json

    def iter_content(self, chunk_size: int = 1) -> list[bytes]:
        del chunk_size
        return self._chunks

    def close(self) -> None:
        self.closed = True


class _RecordingSession:
    """A session that records calls and returns canned responses (never networks)."""

    def __init__(
        self,
        *,
        get_response: _FakeResponse | None = None,
        post_response: _FakeResponse | None = None,
    ) -> None:
        self.get_response = get_response
        self.post_response = post_response
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []
        self.closed = False

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "params": params, "headers": headers, "kwargs": kwargs})
        assert self.get_response is not None
        return self.get_response

    def post(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _FakeResponse:
        self.post_calls.append({"url": url, "data": data, "kwargs": kwargs})
        assert self.post_response is not None
        return self.post_response

    def close(self) -> None:
        self.closed = True


class _ExplodingSession:
    """A session whose every HTTP method raises, proving no real call happens."""

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted via get()")

    def post(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted via post()")

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the module's session factory return an exploding session by default.

    Tests that need canned responses pass an explicit ``session=`` argument; the
    factory is only hit if a test forgets to, which then fails loudly instead of
    reaching the network.
    """
    monkeypatch.setattr(ingest, "_make_session", lambda **_: _ExplodingSession())


# --------------------------------------------------------------------------- #
# get_access_token
# --------------------------------------------------------------------------- #
def test_get_access_token_parses_response() -> None:
    fixture = _load_fixture("token_response.json")
    session = _RecordingSession(post_response=_FakeResponse(json_data=fixture))

    token = get_access_token("user@example.com", "secret", session=session)

    assert token == fixture["access_token"]
    assert len(session.post_calls) == 1
    call = session.post_calls[0]
    assert call["url"] == TOKEN_URL
    assert call["data"]["grant_type"] == "password"
    assert call["data"]["client_id"] == "cdse-public"
    assert call["data"]["username"] == "user@example.com"
    assert call["data"]["password"] == "secret"


def test_get_access_token_missing_credentials() -> None:
    with pytest.raises(ValueError, match="credentials missing"):
        get_access_token("", "", session=_RecordingSession())


def test_get_access_token_no_token_in_response() -> None:
    session = _RecordingSession(post_response=_FakeResponse(json_data={"expires_in": 600}))
    with pytest.raises(ValueError, match="did not contain an access_token"):
        get_access_token("u", "p", session=session)


# --------------------------------------------------------------------------- #
# search_products
# --------------------------------------------------------------------------- #
def test_search_products_builds_odata_query_and_parses() -> None:
    fixture = _load_fixture("products_search.json")
    session = _RecordingSession(get_response=_FakeResponse(json_data=fixture))

    products = search_products(
        _AOI,
        datetime(2022, 5, 18),
        datetime(2022, 5, 21),
        product_type="IW_GRDH",
        polarisation="VV",
        max_results=10,
        session=session,
    )

    # Request was built against the right endpoint with the right params.
    assert len(session.get_calls) == 1
    call = session.get_calls[0]
    assert call["url"] == ODATA_PRODUCTS_URL
    params = call["params"]
    odata_filter = params["$filter"]
    assert "Collection/Name eq 'SENTINEL-1'" in odata_filter
    assert "OData.CSC.Intersects" in odata_filter
    assert "ContentDate/Start ge 2022-05-18T00:00:00.000Z" in odata_filter
    assert "ContentDate/Start le 2022-05-21T00:00:00.000Z" in odata_filter
    assert "IW_GRDH" in odata_filter
    assert "POLYGON((12.0 41.5" in odata_filter
    assert params["$top"] == "10"

    # The whole thing round-trips through a real URL encoder cleanly.
    encoded = requests.models.RequestEncodingMixin._encode_params(params)  # type: ignore[attr-defined]
    qs = parse_qs(urlparse("https://x?" + encoded).query)
    assert "OData.CSC.Intersects" in qs["$filter"][0]

    # Two products in the fixture, both VV-capable (1SDV), so both are returned.
    assert len(products) == 2
    first = products[0]
    assert isinstance(first, Product)
    assert first.id == "b1e2c3d4-1111-2222-3333-444455556666"
    assert first.name.startswith("S1A_IW_GRDH_1SDV_20220520")
    assert first.size == 1073741824
    assert first.online is True
    assert first.content_date_start == "2022-05-20T17:33:05.123Z"
    assert first.download_url.endswith(f"Products({first.id})/$value")


def test_search_products_filters_by_polarisation() -> None:
    fixture = _load_fixture("products_search.json")
    # Rename one product to a VV-only single-pol (1SSH = HH only) -> dropped for VV.
    fixture["value"][1]["Name"] = "S1A_IW_GRDH_1SSH_20220519T173240_x_x_x_x.SAFE"
    session = _RecordingSession(get_response=_FakeResponse(json_data=fixture))

    products = search_products(
        _AOI,
        datetime(2022, 5, 18),
        datetime(2022, 5, 21),
        polarisation="VV",
        session=session,
    )

    assert len(products) == 1
    assert products[0].name.startswith("S1A_IW_GRDH_1SDV")


def test_search_products_empty_value() -> None:
    session = _RecordingSession(get_response=_FakeResponse(json_data={"value": []}))
    products = search_products(_AOI, datetime(2022, 5, 18), datetime(2022, 5, 21), session=session)
    assert products == []


# --------------------------------------------------------------------------- #
# download_product
# --------------------------------------------------------------------------- #
def _product(size: int) -> Product:
    return Product(
        id="b1e2c3d4-1111-2222-3333-444455556666",
        name="S1A_IW_GRDH_1SDV_20220520T173305_TEST",
        size=size,
    )


def test_download_product_writes_file(tmp_path: Path) -> None:
    body = [b"abcd", b"efgh", b"ij"]  # 10 bytes total
    product = _product(size=10)
    session = _RecordingSession(get_response=_FakeResponse(status_code=200, chunks=body))

    out = download_product(product, tmp_path, token="TOK", session=session)

    assert out == tmp_path / f"{product.name}.zip"
    assert out.read_bytes() == b"abcdefghij"
    # Bearer auth sent, no Range header on a fresh download.
    call = session.get_calls[0]
    assert call["headers"]["Authorization"] == "Bearer TOK"
    assert "Range" not in call["headers"]
    assert call["kwargs"]["stream"] is True
    assert call["url"] == product.download_url


def test_download_product_resumes_partial(tmp_path: Path) -> None:
    product = _product(size=10)
    dest = tmp_path / f"{product.name}.zip"
    # Simulate a previous interrupted download: first 4 bytes already on disk.
    dest.write_bytes(b"abcd")

    # Server honours the Range request and returns the remaining 6 bytes (206).
    remainder = [b"efgh", b"ij"]
    session = _RecordingSession(get_response=_FakeResponse(status_code=206, chunks=remainder))

    out = download_product(product, tmp_path, token="TOK", session=session)

    # A Range header from the existing offset was sent.
    call = session.get_calls[0]
    assert call["headers"]["Range"] == "bytes=4-"
    assert call["headers"]["Authorization"] == "Bearer TOK"
    # File was appended to and is now complete.
    assert out.read_bytes() == b"abcdefghij"
    assert out.stat().st_size == 10


def test_download_product_range_ignored_restarts(tmp_path: Path) -> None:
    """If the server ignores Range (returns 200), the file is rewritten in full."""
    product = _product(size=10)
    dest = tmp_path / f"{product.name}.zip"
    dest.write_bytes(b"XXXX")  # stale partial

    full_body = [b"abcde", b"fghij"]
    session = _RecordingSession(get_response=_FakeResponse(status_code=200, chunks=full_body))

    out = download_product(product, tmp_path, token="TOK", session=session)

    # Range was still requested (partial existed) but 200 means full restart.
    assert session.get_calls[0]["headers"]["Range"] == "bytes=4-"
    assert out.read_bytes() == b"abcdefghij"


def test_download_product_size_mismatch_raises(tmp_path: Path) -> None:
    product = _product(size=999)  # expected size that won't match
    session = _RecordingSession(get_response=_FakeResponse(status_code=200, chunks=[b"abc"]))

    with pytest.raises(OSError, match="does not match expected"):
        download_product(product, tmp_path, token="TOK", session=session)


# --------------------------------------------------------------------------- #
# no-network guard
# --------------------------------------------------------------------------- #
def test_factory_session_is_exploding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: without an injected session, the autouse guard blocks network."""
    monkeypatch.setenv("CDSE_USER", "u")
    monkeypatch.setenv("CDSE_PASS", "p")
    with pytest.raises(AssertionError, match="network access attempted"):
        get_access_token("u", "p")
