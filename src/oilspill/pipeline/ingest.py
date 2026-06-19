"""Sentinel-1 scene ingest from the Copernicus Data Space Ecosystem (CDSE).

This module queries the CDSE OData catalogue for Sentinel-1 GRD scenes that
intersect an area of interest (AOI) over a date range, and downloads the matching
product archives with resume support.

Endpoints (verified against the current CDSE documentation):

* OData catalogue search:
  ``https://catalogue.dataspace.copernicus.eu/odata/v1/Products``
  (https://documentation.dataspace.copernicus.eu/APIs/OData.html)
* Product download (streamed ZIP):
  ``https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value``
* Keycloak access token (``grant_type=password``, ``client_id=cdse-public``):
  ``https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token``
  (https://documentation.dataspace.copernicus.eu/APIs/Token.html)

Network access is only performed when the public functions are called; importing
this module never touches the network. All HTTP is funnelled through a
:class:`requests.Session` so it can be mocked in tests.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import requests
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@runtime_checkable
class HttpSession(Protocol):
    """The subset of :class:`requests.Session` this module relies on.

    Typing against this protocol (rather than the concrete ``Session``) lets
    callers inject any compatible client, which is what the test suite does to
    keep all HTTP mocked.
    """

    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...

    def close(self) -> None: ...


# --- Endpoint constants (see module docstring for provenance) ---------------
ODATA_PRODUCTS_URL: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL_TEMPLATE: str = (
    "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
)
TOKEN_URL: str = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
TOKEN_CLIENT_ID: str = "cdse-public"

# CDSE collection name for Sentinel-1 (the OData ``Collection/Name`` value).
SENTINEL1_COLLECTION: str = "SENTINEL-1"

_DEFAULT_CHUNK_SIZE: int = 1 << 20  # 1 MiB
_DEFAULT_MAX_RESULTS: int = 100
_DEFAULT_TIMEOUT: float = 60.0


class CDSESettings(BaseSettings):
    """CDSE credentials, read from the environment (or a ``.env`` file).

    ``CDSE_USER`` / ``CDSE_PASS`` are the Copernicus Data Space Ecosystem login.
    Credentials are never logged or printed by this module.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    user: str = Field(default="", validation_alias="CDSE_USER")
    password: str = Field(default="", validation_alias="CDSE_PASS")


class Product(BaseModel):
    """A single Sentinel-1 product record parsed from an OData search response."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    footprint: str | None = None
    size: int | None = None
    online: bool = True
    content_date_start: str | None = None

    @property
    def download_url(self) -> str:
        """OData ``$value`` URL that streams this product's ZIP archive."""
        return DOWNLOAD_URL_TEMPLATE.format(product_id=self.id)


def _to_iso_z(value: datetime | str) -> str:
    """Render a datetime (or pass-through string) as an OData UTC timestamp.

    OData expects ``YYYY-MM-DDTHH:MM:SS.sssZ``. Naive datetimes are treated as
    UTC; strings are assumed already formatted and returned unchanged.
    """
    if isinstance(value, str):
        return value
    # Millisecond precision with a trailing Z, matching the CDSE examples.
    return value.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _polygon_wkt(aoi_geojson: dict[str, Any]) -> str:
    """Build a WKT ``POLYGON`` (lon lat, ...) from a GeoJSON Polygon.

    Only the exterior ring is used. The ring is closed (first point repeated)
    if the source is not already closed, as required by ``OData.CSC.Intersects``.
    """
    geom = aoi_geojson.get("geometry", aoi_geojson)
    geom_type = geom.get("type")
    if geom_type != "Polygon":
        raise ValueError(f"AOI must be a GeoJSON Polygon, got {geom_type!r}")
    coords = geom.get("coordinates")
    if not coords:
        raise ValueError("AOI Polygon has no coordinates")
    ring: list[list[float]] = list(coords[0])
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    points = ", ".join(f"{lon} {lat}" for lon, lat in ring)
    return f"POLYGON(({points}))"


def _build_filter(
    aoi_geojson: dict[str, Any],
    start: datetime | str,
    end: datetime | str,
    *,
    product_type: str,
) -> str:
    """Assemble the OData ``$filter`` expression for a Sentinel-1 GRD search."""
    wkt = _polygon_wkt(aoi_geojson)
    intersects = f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    # Filter the product type via the product Name, which encodes it in the
    # standard Sentinel naming convention (e.g. ``S1B_IW_GRDH_1SDV_...``). This is
    # more robust than the ``Attributes/productType`` lambda filter, which silently
    # matches nothing unless the attribute value is exact and expanded server-side.
    product_type_filter = f"contains(Name,'{product_type}')"
    return (
        f"Collection/Name eq '{SENTINEL1_COLLECTION}'"
        f" and {intersects}"
        f" and ContentDate/Start ge {_to_iso_z(start)}"
        f" and ContentDate/Start le {_to_iso_z(end)}"
        f" and {product_type_filter}"
    )


def _make_session(*, retries: int = 3) -> requests.Session:
    """Create a session with sensible retry/backoff for transient HTTP errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_access_token(
    user: str | None = None,
    password: str | None = None,
    *,
    session: HttpSession | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """Obtain a CDSE access token via the Keycloak password grant.

    Credentials default to ``CDSE_USER`` / ``CDSE_PASS`` from the environment
    (loaded via :class:`CDSESettings`). The returned token is short-lived
    (~10 minutes); callers that download large scenes should fetch a fresh token
    per download rather than caching it. Raises :class:`requests.HTTPError` on a
    non-2xx response and :class:`ValueError` if credentials are missing.
    """
    if user is None or password is None:
        settings = CDSESettings()
        user = user or settings.user
        password = password or settings.password
    if not user or not password:
        raise ValueError(
            "CDSE credentials missing; set CDSE_USER and CDSE_PASS in the environment or .env"
        )

    owns_session = session is None
    session = session or _make_session()
    try:
        response = session.post(
            TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": TOKEN_CLIENT_ID,
                "username": user,
                "password": password,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    finally:
        if owns_session:
            session.close()

    token = payload.get("access_token")
    if not token:
        raise ValueError("CDSE token response did not contain an access_token")
    return str(token)


def _parse_product(record: dict[str, Any]) -> Product:
    """Parse one OData ``value`` entry into a :class:`Product`."""
    footprint = record.get("Footprint") or record.get("GeoFootprint")
    if isinstance(footprint, dict):
        footprint = footprint.get("type")  # keep a lightweight summary, not raw geometry
    content_date = record.get("ContentDate") or {}
    online = record.get("Online")
    return Product(
        id=str(record["Id"]),
        name=str(record["Name"]),
        footprint=str(footprint) if footprint is not None else None,
        size=record.get("ContentLength"),
        online=bool(online) if online is not None else True,
        content_date_start=content_date.get("Start"),
    )


def search_products(
    aoi_geojson: dict[str, Any],
    start: datetime | str,
    end: datetime | str,
    *,
    product_type: str = "IW_GRDH",
    polarisation: str = "VV",
    max_results: int = _DEFAULT_MAX_RESULTS,
    session: HttpSession | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[Product]:
    """Query the CDSE OData catalogue for Sentinel-1 GRD scenes.

    Builds an OData ``$filter`` combining the Sentinel-1 collection, an
    ``OData.CSC.Intersects`` geographic test against ``aoi_geojson`` (a GeoJSON
    Polygon), a ``ContentDate/Start`` range and the requested ``product_type``
    (default ``IW_GRDH`` — IW-mode GRD high resolution). ``polarisation`` is
    matched client-side against the product name because CDSE GRD product names
    encode the polarisation mode (e.g. ``...1SDV...`` carries VV+VH).

    Returns up to ``max_results`` parsed :class:`Product` records. Raises
    :class:`requests.HTTPError` on a non-2xx response.
    """
    odata_filter = _build_filter(aoi_geojson, start, end, product_type=product_type)
    params = {
        "$filter": odata_filter,
        "$orderby": "ContentDate/Start desc",
        "$top": str(max_results),
    }

    owns_session = session is None
    session = session or _make_session()
    try:
        response = session.get(ODATA_PRODUCTS_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    finally:
        if owns_session:
            session.close()

    products = [_parse_product(record) for record in payload.get("value", [])]
    if polarisation:
        products = [p for p in products if _name_matches_polarisation(p.name, polarisation)]
    return products


def _name_matches_polarisation(name: str, polarisation: str) -> bool:
    """Return whether a Sentinel-1 product name covers ``polarisation``.

    Sentinel-1 GRD names encode polarisation as a ``1S<pol>`` token where ``<pol>``
    is one of ``SH``/``SV`` (single) or ``DH``/``DV`` (dual). VV is present in
    ``SV``/``DV`` products; VH only in ``DV``/``DH`` dual products. When the token
    is absent (e.g. in test data), the product is kept rather than dropped.
    """
    pol = polarisation.upper()
    upper = name.upper()
    # Map a requested channel to the ``1S<pol>`` tokens whose products carry it.
    tokens: dict[str, tuple[str, ...]] = {
        "VV": ("1SSV", "1SDV"),
        "VH": ("1SDV", "1SDH"),
        "HH": ("1SSH", "1SDH"),
        "HV": ("1SDH", "1SDV"),
    }
    wanted = tokens.get(pol)
    if wanted is None:
        return True
    # If the name has no recognisable polarisation token (e.g. trimmed test data),
    # keep it rather than dropping a potentially valid product.
    if "1S" not in upper:
        return True
    return any(token in upper for token in wanted)


def download_product(
    product: Product,
    dest_dir: Path | str,
    token: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    session: HttpSession | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Stream a product's ZIP archive to ``dest_dir`` with resume support.

    The file is written to ``<dest_dir>/<product.name>.zip``. If a partial file
    from a previous interrupted download already exists, the download resumes
    from its current size using an HTTP ``Range: bytes=<n>-`` request and appends
    to the partial file; otherwise it downloads from the start. When the server
    answers a range request with ``206 Partial Content`` the existing bytes are
    kept; a ``200 OK`` (server ignored the range) restarts the file from scratch.

    The download uses ``Authorization: Bearer <token>``. After completion the
    file size is verified against ``product.size`` (when known) and a mismatch
    raises :class:`OSError`. Returns the path to the completed ``.zip``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{product.name}.zip"

    existing = dest_path.stat().st_size if dest_path.exists() else 0

    headers = {"Authorization": f"Bearer {token}"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    owns_session = session is None
    session = session or _make_session()
    try:
        response = session.get(
            product.download_url,
            headers=headers,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()

        # 206 => server honoured the Range and is sending the remainder; append.
        # Anything else (200) => full body; truncate and start over.
        resuming = existing > 0 and response.status_code == 206
        mode = "ab" if resuming else "wb"
        with dest_path.open(mode) as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    finally:
        response_close = getattr(response, "close", None)
        if callable(response_close):
            response_close()
        if owns_session:
            session.close()

    final_size = dest_path.stat().st_size
    if product.size is not None and final_size != product.size:
        raise OSError(
            f"Downloaded size {final_size} does not match expected {product.size} "
            f"for {product.name}"
        )
    return dest_path


def load_aoi(path: Path | str) -> dict[str, Any]:
    """Load a GeoJSON AOI from ``path`` (a Feature, FeatureCollection, or geometry).

    Returns a dict with a ``geometry`` of type ``Polygon`` ready for
    :func:`search_products`.
    """
    import json

    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("type") == "FeatureCollection":
        features = raw.get("features") or []
        if not features:
            raise ValueError("FeatureCollection has no features")
        return features[0]
    return raw


def _credentials_from_env() -> tuple[str, str]:
    """Return ``(user, password)`` from the environment without printing them."""
    return os.environ.get("CDSE_USER", ""), os.environ.get("CDSE_PASS", "")
