import base64
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
import uuid
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union
from io import StringIO
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from shapely.geometry import Point
from shapely.ops import unary_union

try:
    import pycountry
except ImportError:
    pycountry = None

ISO_SOURCEAUTHORITY_NAME_DEFAULT = "ISO 3166 Country Codes"
ISO_SOURCEAUTHORITY_URL_DEFAULT = "https://www.iso.org/iso-3166-country-codes.html"
ISO_ALPHA2_SEARCH_URL_DEFAULT = "https://www.iso.org/obp/ui/#search"

# ---------- GitHub persistence config ----------
# Put these in Streamlit secrets (Streamlit Cloud -> App -> Settings -> Secrets)
# GITHUB_TOKEN="ghp_..."
# GITHUB_REPO="username/repo"
# HISTORY_PATH="history/upload_history.json"   # fallback only; each user gets their own file
# ACCOUNTS_PATH="accounts/users.json"
# BRANCH="main"


def _github_persistence_available() -> tuple[bool, str]:
    """
    True/False whether GITHUB_TOKEN + GITHUB_REPO are configured, plus a reason
    string when they aren't. Catches broadly on purpose: any failure here just
    means "persistence is off for this run", never "the app won't start".
    """
    try:
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("GITHUB_REPO")
        return bool(token) and bool(repo), ""
    except Exception as e:
        return False, str(e)


def _gh_headers():
    return {
        "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }


def gh_read_json(path: str, branch: str = "main"):
    """Return (obj, sha). If file does not exist, return ([], None)."""
    repo = st.secrets["GITHUB_REPO"]
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(), params={"ref": branch}, timeout=30)
    if r.status_code == 404:
        return [], None
    r.raise_for_status()
    payload = r.json()
    content = base64.b64decode(payload["content"]).decode("utf-8")
    return json.loads(content), payload["sha"]


def gh_write_json(path: str, obj, message: str, sha: str | None, branch: str = "main"):
    repo = st.secrets["GITHUB_REPO"]
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    content_str = json.dumps(obj, indent=2, default=str)
    b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
    body = {"message": message, "content": b64, "branch": branch}
    if sha is not None:
        body["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Per-user accounts (login / sign up)
# ============================================================
# Accounts are stored as a single JSON object on GitHub (ACCOUNTS_PATH),
# keyed by a sanitized, lowercase username, e.g.:
#   {"frank": {"password_hash": "...", "salt": "...", "created_at": "..."}}
# Passwords are never stored in plain text -- PBKDF2-HMAC-SHA256 with a
# random per-user salt (stdlib hashlib, no extra dependency needed).
#
# If GitHub persistence isn't configured (no secrets.toml), accounts and
# history simply live in memory for that session only -- login/sign-up
# still works, it just won't survive an app restart. This mirrors how the
# rest of this app degrades gracefully when a source authority is
# unavailable, rather than crashing.

ACCOUNTS_PATH_DEFAULT = "accounts/users.json"
_PBKDF2_ITERATIONS = 200_000


def _sanitize_username(raw: str) -> str:
    """Lowercase, trim, keep only [a-z0-9_-] -- used both as the account key
    and as the per-user history filename, so it must be filesystem/URL safe."""
    raw = (raw or "").strip().lower()
    return re.sub(r"[^a-z0-9_-]", "", raw)


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


def _load_accounts() -> dict:
    if "_accounts_cache" in st.session_state:
        return st.session_state["_accounts_cache"]

    accounts: dict = {}
    available, _reason = _github_persistence_available()
    if available:
        try:
            path = st.secrets.get("ACCOUNTS_PATH", ACCOUNTS_PATH_DEFAULT)
            branch = st.secrets.get("BRANCH", "main")
            loaded, sha = gh_read_json(path, branch=branch)
            if isinstance(loaded, dict):
                accounts = loaded
            st.session_state["_accounts_sha"] = sha
        except Exception:
            st.session_state["_accounts_sha"] = None
    st.session_state["_accounts_cache"] = accounts
    st.session_state["_accounts_persistence_available"] = available
    return accounts


def _save_accounts(accounts: dict) -> bool:
    st.session_state["_accounts_cache"] = accounts
    available, _reason = _github_persistence_available()
    if not available:
        return False
    path = st.secrets.get("ACCOUNTS_PATH", ACCOUNTS_PATH_DEFAULT)
    branch = st.secrets.get("BRANCH", "main")
    for attempt in range(2):
        try:
            _current, sha = gh_read_json(path, branch=branch)
            gh_write_json(path=path, obj=accounts, message="Update accounts", sha=sha, branch=branch)
            return True
        except Exception:
            if attempt == 0:
                continue
            return False
    return False


def _create_account(username: str, password: str) -> tuple[bool, str]:
    accounts = _load_accounts()
    if username in accounts:
        return False, "That username is already taken."
    salt_hex = os.urandom(16).hex()
    accounts[username] = {
        "password_hash": _hash_password(password, salt_hex),
        "salt": salt_hex,
        "created_at": utc_now_iso(),
    }
    _save_accounts(accounts)
    return True, "Account created."


def _check_login(username: str, password: str) -> bool:
    accounts = _load_accounts()
    record = accounts.get(username)
    if not record:
        return False
    return _hash_password(password, record.get("salt", "")) == record.get("password_hash")


def _user_history_path(username: str) -> str:
    return f"history/user_{username}.json"


def _switch_to_user(username: str):
    """Sets the logged-in user and (re)loads *their own* history only."""
    st.session_state["auth_username"] = username
    st.session_state["_user_history_path"] = _user_history_path(username)
    st.session_state["_history_loaded"] = False
    st.session_state["upload_history"] = []
    st.session_state["_history_paused_versions"] = set()
    for k in ("current_version_id", "current_uploaded_at"):
        st.session_state.pop(k, None)
    load_persisted_history()


def _log_out():
    for k in (
        "auth_username", "_user_history_path", "_history_loaded", "upload_history",
        "current_version_id", "current_uploaded_at", "_new_load_event",
        "_history_paused_versions", "_active_upload_version_id",
    ):
        st.session_state.pop(k, None)


def render_auth_gate() -> bool:
    """
    Draws the login / sign-up UI. Returns True (and draws nothing else) once
    someone is logged in for this session; otherwise draws the form and
    returns False so the caller can st.stop().
    """
    if st.session_state.get("auth_username"):
        return True

    st.subheader("Log in or create an account")
    available, _reason = _github_persistence_available()
    if not available:
        st.info(
            "GitHub persistence isn't configured, so accounts and history will only "
            "last for this browser session (they won't survive an app restart). "
            "That's fine for trying the app out."
        )

    tab_login, tab_signup = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_username")
            login_pass = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            username = _sanitize_username(login_user)
            if not username or not login_pass:
                st.error("Enter a username and password.")
            elif _check_login(username, login_pass):
                _switch_to_user(username)
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_user = st.text_input("Choose a username", key="signup_username")
            new_pass = st.text_input("Choose a password", type="password", key="signup_password")
            new_pass2 = st.text_input("Confirm password", type="password", key="signup_password2")
            submitted2 = st.form_submit_button("Create account")
        if submitted2:
            username = _sanitize_username(new_user)
            if not username:
                st.error("Username must contain at least one letter, number, - or _.")
            elif len(new_pass) < 4:
                st.error("Password must be at least 4 characters.")
            elif new_pass != new_pass2:
                st.error("Passwords do not match.")
            else:
                ok, msg = _create_account(username, new_pass)
                if ok:
                    _switch_to_user(username)
                    st.success("Account created -- you're logged in.")
                    st.rerun()
                else:
                    st.error(msg)

    return False


def load_persisted_history():
    """Loads the CURRENTLY LOGGED-IN user's persisted history from GitHub
    into session_state (once per login)."""
    if st.session_state.get("_history_loaded", False):
        return

    st.session_state.setdefault("upload_history", [])
    available, reason = _github_persistence_available()
    if not available:
        st.session_state["_history_persistence_available"] = False
        st.session_state["_history_persistence_reason"] = (
            reason or "GITHUB_TOKEN/GITHUB_REPO not set -- history will only last this session."
        )
        st.session_state["_history_loaded"] = True
        return

    try:
        path = st.session_state.get("_user_history_path") or st.secrets.get(
            "HISTORY_PATH", "history/upload_history.json"
        )
        branch = st.secrets.get("BRANCH", "main")
        history, _sha = gh_read_json(path, branch=branch)
        if not isinstance(history, list):
            history = []
        st.session_state["upload_history"] = [h for h in history if isinstance(h, dict)]
        st.session_state["_history_persistence_available"] = True
    except Exception as e:
        st.session_state["_history_persistence_available"] = False
        st.session_state["_history_persistence_reason"] = str(e)
    st.session_state["_history_loaded"] = True


def append_persisted_history(new_record: dict):
    """
    Appends a record to the CURRENT user's GitHub JSON file with a simple
    retry to handle concurrent edits. Also appends to st.session_state.
    """
    if "upload_history" not in st.session_state or not isinstance(
        st.session_state["upload_history"], list
    ):
        st.session_state["upload_history"] = []
    st.session_state["upload_history"].append(new_record)

    available, _reason = _github_persistence_available()
    if not available:
        return

    path = st.session_state.get("_user_history_path") or st.secrets.get(
        "HISTORY_PATH", "history/upload_history.json"
    )
    branch = st.secrets.get("BRANCH", "main")

    # Retry once in case of SHA conflict
    for attempt in range(2):
        try:
            history, sha = gh_read_json(path, branch=branch)
            if not isinstance(history, list):
                history = []
            history.append(new_record)
            gh_write_json(
                path=path,
                obj=history,
                message=f"Add upload history {new_record.get('version_id','')}",
                sha=sha,
                branch=branch,
            )
            return
        except requests.HTTPError as e:
            # 409 / 422 can occur if SHA changed between read and write
            if attempt == 0:
                continue
            st.warning(f"Could not save upload history to GitHub (kept in this session only): {e}")
            return
        except Exception as e:
            st.warning(f"Could not save upload history to GitHub (kept in this session only): {e}")
            return


def _is_history_paused_for(*version_ids) -> bool:
    """True if any of the given ids belongs to the dataset that was active at
    the moment history was last cleared -- used to stop that SAME upload from
    instantly reappearing in history on the next rerun. Tracking resumes on
    its own the moment a genuinely different dataset is uploaded, since a new
    file's id will never be in this paused set."""
    paused = st.session_state.get("_history_paused_versions") or set()
    return any(v in paused for v in version_ids if v)


def clear_user_history():
    """
    Wipes the CURRENT user's history (in-session and on GitHub) in the
    Panels/dashboard view only. The dataset that's still sitting in the file
    uploader right now is remembered so it is NOT immediately re-added back
    into history on the next rerun -- tracking only starts again once the
    person actually uploads a new (different) dataset.
    """
    st.session_state["upload_history"] = []
    st.session_state["_history_paused_versions"] = {
        v for v in (
            st.session_state.get("current_version_id"),
            st.session_state.get("_active_upload_version_id"),
        ) if v
    }
    available, _reason = _github_persistence_available()
    if not available:
        return
    path = st.session_state.get("_user_history_path") or st.secrets.get(
        "HISTORY_PATH", "history/upload_history.json"
    )
    branch = st.secrets.get("BRANCH", "main")
    try:
        _current, sha = gh_read_json(path, branch=branch)
        gh_write_json(path=path, obj=[], message="Clear upload history", sha=sha, branch=branch)
    except Exception as e:
        st.warning(f"Could not clear history on GitHub (cleared locally only): {e}")


def start_new_history_session():
    """Starts a fresh tracked upload 'version' without touching previously
    saved history -- the next upload is logged as a brand-new entry even if
    it's the same file as before."""
    st.session_state["current_version_id"] = (
        f"v{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    st.session_state["current_uploaded_at"] = utc_now_iso()
    st.session_state["_new_load_event"] = False
    st.session_state["_history_paused_versions"] = set()

# ============================================================
# Helper functions
# ============================================================


# ============================================================
# Real bdqval:sourceAuthority data for VALIDATION_COORDINATESCOUNTRYCODE_CONSISTENT:
#   "10m-admin-1 boundaries UNION with Exclusive Economic Zones"
#   = Natural Earth 10m Admin 1 -- States, Provinces
#     UNION Marine Regions Exclusive Economic Zones (EEZ)
# ============================================================

# Confirmed live (via naturalearthdata.com + independent mirrors -- see
# gist.github.com/DanielJWood/b71237cc200831acf8e637c05ce2c375 and
# go-spatial/tegola-osm's natural_earth.sh, both pointing at the same file):
# Natural Earth's public S3 mirror, no auth required. ~14MB zipped shapefile.
NATURAL_EARTH_ADMIN1_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"

# Confirmed live via a real GetCapabilities/GetFeature request (marineregions.org/webservices.php):
# Marine Regions' public GeoServer WFS, no auth/registration required.
# typeNames="eez" is correct as-is (no "MarineRegions:" prefix needed --
# the endpoint path already scopes the request to that workspace).
MARINE_REGIONS_EEZ_WFS_URL = "https://geo.vliz.be/geoserver/MarineRegions/wfs"

# Geometry simplification tolerance in degrees (~111m at the equator) applied
# once at load time. Both source datasets ship at very high coastline detail;
# simplifying keeps memory/distance-calculation cost sane without materially
# changing the outcome of a 3000m-default buffer check.
_BOUNDARY_SIMPLIFY_TOLERANCE_DEG = 0.001


def _alpha3_to_alpha2(alpha3) -> str:
    """
    ISO 3166-1 alpha-3 -> alpha-2, via pycountry (a maintained ISO 3166
    database: `pip install pycountry`). Needed for two reasons:

      1) Natural Earth's own iso_a2 field has a known, longstanding gap for
         at least France and Norway (shows "-99" instead of "FR"/"NO") --
         see nvkelso/natural-earth-vector issues #284 and #947; the
         "_eh"-suffixed fallback field has the same bug per issue #252. So
         admin-1 boundaries fall back through the more reliable adm0_a3
         (alpha-3) field via this table whenever iso_a2 looks invalid.
      2) Marine Regions' EEZ layer has NO alpha-2 field at all -- only
         alpha-3 (iso_ter1 / iso_sov1; confirmed live via its WFS, e.g.
         "ASM"/"USA" for American Samoa, not "AS"/"US"), so this table is
         the only way to get an iso_a2 to union EEZ polygons by.

    Accepts anything (str, None, float NaN, pandas.NA, etc.) since this is
    called via `.apply()` over a real-world data column that can contain
    missing values -- `pd.isna()` catches those before any string method
    is used on them. (A bare `alpha3 or ""` is NOT safe here: `float('nan')`
    is truthy in Python, so `nan or ""` evaluates to `nan` itself, not `""`,
    and calling `.strip()` on that float raises AttributeError.)

    Returns "" (never raises) if the input is missing/NaN, pycountry isn't
    installed, or the code isn't recognized, so a missing/bad code just
    drops that one polygon from the union rather than crashing the whole
    load.
    """
    if pd.isna(alpha3):
        return ""
    alpha3 = str(alpha3).strip().upper()
    if not alpha3 or pycountry is None:
        return ""
    try:
        country = pycountry.countries.get(alpha_3=alpha3)
        return country.alpha_2 if country else ""
    except (LookupError, AttributeError):
        return ""


@st.cache_data(persist="disk", show_spinner="Downloading Natural Earth admin-1 boundaries (~14MB, first run only)...")
def load_source_authority_gdf():
    """
    Real bdqval:sourceAuthority data: Natural Earth 10m Admin 1 -- States,
    Provinces (https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/).
    Downloaded once from NATURAL_EARTH_ADMIN1_URL and cached to local disk
    (persist="disk") so subsequent app runs/restarts don't re-download it.
    Returns None (-> EXTERNAL_PREREQUISITES_NOT_MET for every test that
    requires this authority) if the download or the shapefile is unusable.
    """
    if pycountry is None:
        st.warning(
            "pycountry is not installed -- run `pip install pycountry`. Without it, "
            "countries whose Natural Earth iso_a2 field is broken (e.g. France, Norway) "
            "cannot be resolved, and no EEZ polygons can be resolved at all."
        )

    try:
        r = requests.get(NATURAL_EARTH_ADMIN1_URL, timeout=90)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not download Natural Earth admin-1 boundaries from {NATURAL_EARTH_ADMIN1_URL}: {e}")
        return None

    try:
        extract_dir = Path(tempfile.gettempdir()) / "dq_validator_natural_earth_admin1"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(extract_dir)
        shp_path = next(extract_dir.glob("*.shp"), None)
        if shp_path is None:
            st.error("Downloaded Natural Earth archive did not contain a .shp file.")
            return None
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        st.error(f"Could not parse the Natural Earth admin-1 shapefile: {e}")
        return None

    if gdf is None or gdf.empty:
        return None

    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")

    def _resolve_iso_a2(row):
        code = str(row.get("iso_a2", "")).strip().upper()
        if re.fullmatch(r"[A-Z]{2}", code):
            return code
        # Known Natural Earth gap (e.g. France, Norway show "-99") -- fall
        # back through the country-level alpha-3 code instead.
        return _alpha3_to_alpha2(str(row.get("adm0_a3", "")))

    gdf["iso_a2"] = gdf.apply(_resolve_iso_a2, axis=1)
    gdf = gdf[gdf["iso_a2"] != ""].copy()
    if gdf.empty:
        return None

    gdf["geometry"] = gdf.geometry.simplify(_BOUNDARY_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    name_col = "name" if "name" in gdf.columns else gdf.columns[0]
    gdf = gdf.rename(columns={name_col: "name"})[["iso_a2", "name", "geometry"]].reset_index(drop=True)
    return gdf


@st.cache_data(persist="disk", show_spinner="Downloading Marine Regions EEZ boundaries (first run only)...")
def load_eez_gdf():
    """
    Real bdqval:sourceAuthority data: Marine Regions Exclusive Economic
    Zones, fetched live from Marine Regions' public GeoServer WFS
    (https://www.marineregions.org/webservices.php) -- confirmed reachable
    without auth/registration via a live GetFeature request. UNIONed with
    the admin-1 land boundaries above for VALIDATION_COORDINATESCOUNTRYCODE_CONSISTENT.

    Cached to local disk (persist="disk") so this only downloads once.
    Returns None if the service is unreachable -- per
    _country_union_geometries()'s docstring, a missing EEZ layer is treated
    as "this country has no EEZ data available" (matching the spec's "plus
    its Exclusive Economic Zone... if any"), NOT as the whole source
    authority being unavailable -- that gate is on load_source_authority_gdf()
    (the admin-1 layer) alone.
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": "eez",
        "outputFormat": "application/json",
        "propertyName": "iso_ter1,geoname",
    }
    try:
        r = requests.get(MARINE_REGIONS_EEZ_WFS_URL, params=params, timeout=120)
        r.raise_for_status()
        gdf = gpd.read_file(io.BytesIO(r.content))
    except Exception as e:
        st.warning(f"Could not download Marine Regions EEZ boundaries (continuing without EEZ data): {e}")
        return None

    if gdf is None or gdf.empty:
        return None

    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")

    if "iso_ter1" not in gdf.columns:
        st.warning("Marine Regions EEZ response did not include the expected 'iso_ter1' field.")
        return None

    gdf["iso_a2"] = gdf["iso_ter1"].astype(str).apply(_alpha3_to_alpha2)
    gdf = gdf[gdf["iso_a2"] != ""].copy()
    if gdf.empty:
        return None

    gdf["geometry"] = gdf.geometry.simplify(_BOUNDARY_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    name_col = "geoname" if "geoname" in gdf.columns else gdf.columns[0]
    gdf = gdf.rename(columns={name_col: "name"})[["iso_a2", "name", "geometry"]].reset_index(drop=True)
    return gdf


def _normalize_state_name(x: str) -> str:
    return " ".join(str(x).strip().split())

# ============================================================
# BDQ Tests (subset)
# ============================================================
def test_country_not_empty(country_series, country_code_series):
    results = []
    for country, country_code in zip(country_series, country_code_series):
        country_empty = pd.isna(country) or str(country).strip() == ""
        country_code_val = str(country_code).strip().upper() if pd.notna(country_code) else ""
        country_val = str(country).strip().lower() if pd.notna(country) else ""

        if not country_empty:
            results.append("COMPLIANT")
        elif country_code_val == "XZ" and (country_empty or country_val == "high seas"):
            results.append("COMPLIANT")
        else:
            results.append("NOT_COMPLIANT")
    return pd.Series(results, index=country_series.index)


# ============================================================
# VALIDATION_COORDINATESCOUNTRYCODE_CONSISTENT
# https://rs.tdwg.org/bdqtest/terms/...
#
# Expected Response:
#   EXTERNAL_PREREQUISITES_NOT_MET if bdqval:sourceAuthority is not available;
#   INTERNAL_PREREQUISITES_NOT_MET if one or more of dwc:decimalLatitude,
#     dwc:decimalLongitude, or dwc:countryCode are bdqval:Empty or invalid;
#   COMPLIANT if the coordinates fall on or within the boundary defined by the
#     UNION of the country boundary (from dwc:countryCode) plus its Exclusive
#     Economic Zone as found in bdqval:sourceAuthority, if any, plus an
#     exterior buffer given by bdqval:spatialBufferInMeters;
#   otherwise NOT_COMPLIANT.
#
# Default parameters:
#   sourceAuthority = "10m-admin-1 boundaries UNION with Exclusive Economic Zones"
#       ({https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/}
#        spatial UNION {https://www.marineregions.org/downloads.php#marbound})
#   spatialBufferInMeters = 3000
# ============================================================

DEFAULT_SPATIAL_BUFFER_M = 3000


def _prepare_gdf_for_buffering(gdf: gpd.GeoDataFrame, metric_crs: str = "EPSG:3857"):
    """Reproject a lon/lat GeoDataFrame into a metric CRS so buffering/distance is in real meters."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(metric_crs)


def _country_union_geometries(gdf: gpd.GeoDataFrame, eez_gdf: Optional[gpd.GeoDataFrame], metric_crs: str = "EPSG:3857"):
    """
    Build, per iso_a2 code, the metric-CRS geometry that is the spatial UNION of:
      - all polygons in `gdf` (admin-1 boundaries) sharing that iso_a2, and
      - all polygons in `eez_gdf` (EEZ) sharing that iso_a2, if `eez_gdf` is provided.
    Returns {iso_a2_upper: shapely geometry in metric_crs}.
    """
    gdf_metric = _prepare_gdf_for_buffering(gdf, metric_crs)
    eez_metric = _prepare_gdf_for_buffering(eez_gdf, metric_crs) if (eez_gdf is not None and not eez_gdf.empty) else None

    codes = set(gdf_metric["iso_a2"].astype(str).str.strip().str.upper())
    if eez_metric is not None:
        codes |= set(eez_metric["iso_a2"].astype(str).str.strip().str.upper())

    union_geoms = {}
    for code in codes:
        geoms = list(gdf_metric.loc[gdf_metric["iso_a2"].astype(str).str.strip().str.upper() == code, "geometry"])
        if eez_metric is not None:
            geoms += list(eez_metric.loc[eez_metric["iso_a2"].astype(str).str.strip().str.upper() == code, "geometry"])
        if geoms:
            union_geoms[code] = unary_union(geoms)
    return union_geoms


def test_coordinates_countrycode_consistent(
        lat_series: pd.Series,
        lon_series: pd.Series,
        country_code_series: pd.Series,
        gdf: gpd.GeoDataFrame = None,
        eez_gdf: gpd.GeoDataFrame = None,
        spatial_buffer_m: float = DEFAULT_SPATIAL_BUFFER_M,
) -> pd.Series:
    n = len(lat_series)
    idx = lat_series.index

    # ---- EXTERNAL_PREREQUISITES_NOT_MET: source authority unavailable ----
    if gdf is None or gdf.empty:
        return pd.Series(["EXTERNAL_PREREQUISITES_NOT_MET"] * n, index=idx)

    # Country boundary UNION EEZ, pre-computed per iso_a2 in a metric CRS
    union_geoms = _country_union_geometries(gdf, eez_gdf)
    metric_crs = "EPSG:3857"

    results = []
    for lat, lon, cc in zip(lat_series, lon_series, country_code_series):
        # ---- INTERNAL_PREREQUISITES_NOT_MET: lat/lon empty or invalid ----
        try:
            if pd.isna(lat) or pd.isna(lon):
                raise ValueError
            lat_f = float(lat)
            lon_f = float(lon)
            if not (-90 <= lat_f <= 90) or not (-180 <= lon_f <= 180):
                raise ValueError
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        # ---- INTERNAL_PREREQUISITES_NOT_MET: countryCode empty or invalid ----
        cc_str = "" if pd.isna(cc) else str(cc).strip()
        if cc_str == "" or not re.fullmatch(r"[A-Za-z]{2}", cc_str):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        cc_upper = cc_str.upper()

        country_geom = union_geoms.get(cc_upper)
        if country_geom is None or country_geom.is_empty:
            # countryCode is well-formed but there is no boundary for it in the
            # sourceAuthority -> the coordinates cannot be within it.
            results.append("NOT_COMPLIANT")
            continue

        point_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(lon_f, lat_f)]}, crs="EPSG:4326"
        ).to_crs(metric_crs)
        point_metric = point_gdf.geometry.iloc[0]

        # COMPLIANT if the point is on/within the union boundary, or within
        # spatial_buffer_m of it; otherwise NOT_COMPLIANT.
        distance_m = country_geom.distance(point_metric)
        results.append("COMPLIANT" if distance_m <= spatial_buffer_m else "NOT_COMPLIANT")

    return pd.Series(results, index=idx)


def _stateprovince_union_geometries(gdf: gpd.GeoDataFrame, metric_crs: str = "EPSG:3857"):
    """
    Build, per lowercased/stripped dwc:stateProvince name, the metric-CRS
    geometry that is the spatial UNION of all admin-1 polygons in `gdf`
    sharing that name (a name can appear more than once, e.g. across
    countries, or as multipart polygons for the same province).
    Returns {name_lower: shapely geometry in metric_crs}.
    """
    gdf_metric = _prepare_gdf_for_buffering(gdf, metric_crs)
    names = gdf_metric["name"].astype(str).str.strip().str.lower()

    union_geoms = {}
    for name in set(names):
        if not name or name == "nan":
            continue
        geoms = list(gdf_metric.loc[names == name, "geometry"])
        if geoms:
            union_geoms[name] = unary_union(geoms)
    return union_geoms


def test_coordinates_stateprovince_consistent(
        lat_series: pd.Series,
        lon_series: pd.Series,
        state_series: pd.Series,
        gdf: gpd.GeoDataFrame = None,
        spatial_buffer_m: float = DEFAULT_SPATIAL_BUFFER_M,
) -> pd.Series:
    n = len(lat_series)
    idx = lat_series.index

    # ---- EXTERNAL_PREREQUISITES_NOT_MET: source authority unavailable ----
    if gdf is None or gdf.empty:
        return pd.Series(["EXTERNAL_PREREQUISITES_NOT_MET"] * n, index=idx)

    # Per-stateProvince-name UNION geometry, pre-computed in a metric CRS
    union_geoms = _stateprovince_union_geometries(gdf)
    metric_crs = "EPSG:3857"

    results = []
    for lat, lon, state in zip(lat_series, lon_series, state_series):
        # ---- INTERNAL_PREREQUISITES_NOT_MET: lat/lon empty or invalid ----
        try:
            if pd.isna(lat) or pd.isna(lon):
                raise ValueError
            lat_f = float(lat)
            lon_f = float(lon)
            if not (-90 <= lat_f <= 90) or not (-180 <= lon_f <= 180):
                raise ValueError
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        # ---- INTERNAL_PREREQUISITES_NOT_MET: stateProvince empty or not
        # found in the sourceAuthority ----
        state_str = "" if pd.isna(state) else str(state).strip()
        if state_str == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        state_geom = union_geoms.get(state_str.lower())
        if state_geom is None or state_geom.is_empty:
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        point_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(lon_f, lat_f)]}, crs="EPSG:4326"
        ).to_crs(metric_crs)
        point_metric = point_gdf.geometry.iloc[0]

        # COMPLIANT if the point is on/within the named boundary, or within
        # spatial_buffer_m of it; otherwise NOT_COMPLIANT.
        distance_m = state_geom.distance(point_metric)
        results.append("COMPLIANT" if distance_m <= spatial_buffer_m else "NOT_COMPLIANT")

    return pd.Series(results, index=idx)


def test_coordinates_not_zero(lat_series, lon_series):
    results = []
    for lat, lon in zip(lat_series, lon_series):
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(lat) or pd.isna(lon):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        results.append("COMPLIANT" if ((lat != 0) or (lon != 0)) else "NOT_COMPLIANT")

    return pd.Series(results, index=lat_series.index)


def test_coordinate_uncertainty_inrange(series):
    results = []
    for val in series:
        if pd.isna(val) or str(val).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            uncertainty = float(val)
            results.append("COMPLIANT" if 1 <= uncertainty <= 20037509 else "NOT_COMPLIANT")
        except (TypeError, ValueError):
            results.append("NOT_COMPLIANT")
    return pd.Series(results, index=series.index)


def test_countrycode_not_empty(country_code_series):
    results = []
    for  country_code in country_code_series:
        if pd.isna(country_code) or str(country_code).strip() == "":
             results.append("NOT_COMPLIANT")
        else:
            results.append("COMPLIANT")

    return pd.Series(results, index=country_code_series.index)
#Country code standard

# Practical fallback: a complete ISO-3166-1 alpha-2 list should live locally.
# Put your full list here OR load it from a CSV (recommended).
# (This short list is ONLY an example — replace with full set in production.)
FALLBACK_ISO_ALPHA2 = {
     "AD","AE","AF","AG","AI","AL","AM","AO","AQ","AR","AS","AT","AU","AW","AX","AZ",
    "BA","BB","BD","BE","BF","BG","BH","BI","BJ","BL","BM","BN","BO","BQ","BR","BS",
    "BT","BV","BW","BY","BZ",
    "CA","CC","CD","CF","CG","CH","CI","CK","CL","CM","CN","CO","CR","CU","CV","CW",
    "CX","CY","CZ",
    "DE","DJ","DK","DM","DO","DZ",
    "EC","EE","EG","EH","ER","ES","ET",
    "FI","FJ","FK","FM","FO","FR",
    "GA","GB","GD","GE","GF","GG","GH","GI","GL","GM","GN","GP","GQ","GR","GS","GT",
    "GU","GW","GY",
    "HK","HM","HN","HR","HT","HU",
    "ID","IE","IL","IM","IN","IO","IQ","IR","IS","IT",
    "JE","JM","JO","JP",
    "KE","KG","KH","KI","KM","KN","KP","KR","KW","KY","KZ",
    "LA","LB","LC","LI","LK","LR","LS","LT","LU","LV","LY",
    "MA","MC","MD","ME","MF","MG","MH","MK","ML","MM","MN","MO","MP","MQ","MR","MS",
    "MT","MU","MV","MW","MX","MY","MZ",
    "NA","NC","NE","NF","NG","NI","NL","NO","NP","NR","NU","NZ",
    "OM",
    "PA","PE","PF","PG","PH","PK","PL","PM","PN","PR","PS","PT","PW","PY",
    "QA",
    "RE","RO","RS","RU","RW",
    "SA","SB","SC","SD","SE","SG","SH","SI","SJ","SK","SL","SM","SN","SO","SR","SS",
    "ST","SV","SX","SY","SZ",
    "TC","TD","TF","TG","TH","TJ","TK","TL","TM","TN","TO","TR","TT","TV","TW","TZ",
    "UA","UG","UM","US","UY","UZ",
    "VA","VC","VE","VG","VI","VN","VU",
    "WF","WS",
    "YE","YT",
    "ZA","ZM","ZW"
}

def _normalize_authority_to_code_set(
    sourceAuthority: Union[set, list, tuple, dict, pd.DataFrame],
    authority_code_field: Optional[str] = None,
) -> Optional[set]:
    """Convert different authority shapes into a set of uppercase ISO alpha-2 codes."""
    if isinstance(sourceAuthority, (set, list, tuple)):
        codes = {str(c).strip().upper() for c in sourceAuthority if str(c).strip() != ""}
        return codes or None

    if isinstance(sourceAuthority, Mapping):
        codes = {str(k).strip().upper() for k in sourceAuthority.keys() if str(k).strip() != ""}
        return codes or None

    if isinstance(sourceAuthority, pd.DataFrame):
        candidate_cols = [authority_code_field] if authority_code_field else []
        candidate_cols += [
            "alpha2", "alpha_2", "alpha-2",
            "iso2", "iso_a2", "iso_2",
            "countryCode", "country_code",
            "code", "Code", "ISO2", "ISO_A2"
        ]
        col = next((c for c in candidate_cols if c and c in sourceAuthority.columns), None)
        if col is None:
            return None

        codes = {
            str(c).strip().upper()
            for c in sourceAuthority[col].dropna().astype(str).tolist()
            if str(c).strip() != ""
        }
        return codes or None

    return None


def _pycountry_iso_alpha2_codes() -> Optional[set]:
    """
    Real bdqval:sourceAuthority data for ISO 3166 Country Codes, via pycountry
    (a maintained mirror of the official ISO 3166-1 list, sourced from
    Debian's iso-codes project: `pip install pycountry`). Returns None if
    pycountry isn't installed or its data can't be read, so the caller can
    tell the difference between "authority genuinely unavailable" and
    "authority returned zero codes".
    """
    if pycountry is None:
        return None
    try:
        codes = {
            c.alpha_2.strip().upper()
            for c in pycountry.countries
            if getattr(c, "alpha_2", None)
        }
        return codes or None
    except Exception:
        return None


def _get_source_authority_codes(
    sourceAuthority: Optional[Union[set, list, tuple, dict, pd.DataFrame, str]] = None,
    authority_code_field: Optional[str] = None,
    allow_fallback: bool = False,
):
    """
    Returns (valid_codes_set, diagnostic_message).
    Accepts:
      - set/list/tuple/dict/DataFrame as before
      - str path to a CSV file (local)

    When sourceAuthority is None (the normal/default call pattern), the real
    ISO 3166-1 alpha-2 list is loaded from pycountry -- not a hand-typed
    table. This makes "source authority not available" a genuine, reachable
    condition (pycountry missing/broken) instead of dead code: previously,
    this branch always silently substituted a hardcoded local set, so
    EXTERNAL_PREREQUISITES_NOT_MET could never actually be produced by the
    app's real call site.

    allow_fallback=True is an explicit opt-in escape hatch: if pycountry is
    unavailable but the caller still wants *some* offline authority rather
    than an EXTERNAL_PREREQUISITES_NOT_MET result, passing allow_fallback=True
    uses the small static FALLBACK_ISO_ALPHA2 table instead. Off by default,
    since silently masking a missing/broken authority is exactly the
    behavior being fixed here.
    """
    if sourceAuthority is None:
        pycountry_codes = _pycountry_iso_alpha2_codes()
        if pycountry_codes:
            return (pycountry_codes, "Using pycountry's ISO 3166-1 database (real, maintained authority)")
        if allow_fallback and FALLBACK_ISO_ALPHA2:
            return (set(FALLBACK_ISO_ALPHA2), "pycountry unavailable; using FALLBACK_ISO_ALPHA2 (static offline authority)")
        return (
            None,
            "sourceAuthority unavailable: pycountry is not installed or unusable "
            "(`pip install pycountry`), and no fallback authority was permitted",
        )

    # If user passes a file path, try to load it as CSV
    if isinstance(sourceAuthority, str):
        try:
            df = pd.read_csv(sourceAuthority)
            codes = _normalize_authority_to_code_set(df, authority_code_field=authority_code_field)
            if codes:
                return (codes, f"Loaded authority from CSV: {sourceAuthority}")
            return (None, f"Loaded CSV but could not find a valid code column. Columns={list(df.columns)}")
        except Exception as e:
            if allow_fallback and FALLBACK_ISO_ALPHA2:
                return (set(FALLBACK_ISO_ALPHA2), f"Failed to load authority CSV ({e}); using fallback")
            return (None, f"Failed to load authority CSV: {e}")

    codes = _normalize_authority_to_code_set(sourceAuthority, authority_code_field=authority_code_field)
    if codes:
        return (codes, f"Authority normalized from {type(sourceAuthority).__name__} with {len(codes)} codes")

    # If normalization failed, provide a helpful message
    if isinstance(sourceAuthority, pd.DataFrame):
        return (None, f"Authority DataFrame has no recognized code column. Columns={list(sourceAuthority.columns)}")
    if isinstance(sourceAuthority, dict):
        return (None, "Authority dict provided, but keys may not be alpha-2 codes (expected keys like 'US', 'CD').")

    return (None, f"Authority type not supported or empty: {type(sourceAuthority).__name__}")


def test_countrycode_standard(
    country_code_series: pd.Series,
    sourceAuthority: Optional[Union[set, list, tuple, dict, pd.DataFrame, str]] = None,
    sourceAuthorityName: str = ISO_SOURCEAUTHORITY_NAME_DEFAULT,
    sourceAuthorityURL: str = ISO_SOURCEAUTHORITY_URL_DEFAULT,
    sourceAuthoritySearchURL: str = ISO_ALPHA2_SEARCH_URL_DEFAULT,
    authority_code_field: Optional[str] = None,
    return_comment: bool = False,
    allow_fallback: bool = False,  # opt-in only; see _get_source_authority_codes
):
    """
    BDQ test: Is dwc:countryCode a valid ISO 3166-1-alpha-2 code?

    Default sourceAuthority (None) resolves to pycountry's real ISO 3166-1
    database. If pycountry is unavailable, this correctly returns
    EXTERNAL_PREREQUISITES_NOT_MET for every row -- pass allow_fallback=True
    to instead fall back to the small static FALLBACK_ISO_ALPHA2 table.
    """

    valid_codes, diag = _get_source_authority_codes(
        sourceAuthority=sourceAuthority,
        authority_code_field=authority_code_field,
        allow_fallback=allow_fallback,
    )

    # External prerequisite
    if not valid_codes:
        result = pd.Series(
            ["EXTERNAL_PREREQUISITES_NOT_MET"] * len(country_code_series),
            index=country_code_series.index,
        )
        if not return_comment:
            return result
        return pd.DataFrame(
            {
                "result": result,
                "comment": [
                    f"bdq:sourceAuthority not available/unusable ({sourceAuthorityName}). {diag}. "
                    f"See {sourceAuthorityURL} / {sourceAuthoritySearchURL}"
                ] * len(country_code_series)
            },
            index=country_code_series.index,
        )

    results = []
    comments = []

    for val in country_code_series:
        # Internal prerequisite
        if pd.isna(val) or str(val).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            if return_comment:
                comments.append("dwc:countryCode is bdq:Empty")
            continue

        code_raw = str(val).strip()
        code = code_raw.upper()

        # Must be unambiguous alpha-2
        if not re.fullmatch(r"[A-Z]{2}", code):
            results.append("NOT_COMPLIANT")
            if return_comment:
                comments.append(
                    f'dwc:countryCode="{code_raw}" is NOT a valid ISO ({sourceAuthorityName}) value '
                    f"(must be 2-letter ISO 3166-1-alpha-2); see {sourceAuthoritySearchURL}"
                )
            continue

        if code in valid_codes:
            results.append("COMPLIANT")
            if return_comment:
                comments.append(
                    f'dwc:countryCode="{code}" is a valid ISO ({sourceAuthorityName}) value'
                )
        else:
            results.append("NOT_COMPLIANT")
            if return_comment:
                comments.append(
                    f'dwc:countryCode="{code_raw}" is NOT a valid ISO ({sourceAuthorityName}) value; '
                    f"see {sourceAuthoritySearchURL}"
                )

    result_series = pd.Series(results, index=country_code_series.index)
    if not return_comment:
        return result_series

    return pd.DataFrame({"result": result_series, "comment": comments}, index=country_code_series.index)

# ============================================================
# BDQ Source Authority (DEFAULT): Getty TGN -- via the Reconciliation API
#   bdq:sourceAuthority default = "The Getty Thesaurus of Geographic Names (TGN)"
#   Reconciliation service (replaces the old SPARQL endpoint, which was
#   returning HTTP 499 "Service temporarily degraded"):
#   https://services.getty.edu/vocab/reconcile/
# ============================================================

GETTY_RECONCILE = "https://services.getty.edu/vocab/reconcile/"

TGN_SOURCE_AUTHORITY_DEFAULT = {
    "name": "The Getty Thesaurus of Geographic Names (TGN)",
    "url": GETTY_RECONCILE,
    "endpoint": GETTY_RECONCILE,
}

# Reconciliation "type" id for TGN, as returned in this service's own manifest
# (GET https://services.getty.edu/vocab/reconcile/ -> defaultTypes[].id == "/tgn").
GETTY_RECONCILE_TGN_TYPE = "/tgn"

# Extend (data-extension) property ids, confirmed via
# GET https://services.getty.edu/vocab/reconcile/extend/properties?type=/tgn
GETTY_RECONCILE_PROP_PLACETYPES = "/tgn_placetypes"
GETTY_RECONCILE_PROP_HIER = "/tgn_hier"
GETTY_RECONCILE_PROP_COORDINATES = "/tgn_coordinates"

# Supplementary local reference: Getty's reconciliation API does not expose ISO
# 3166-1 alpha-2 codes as an extend property (only preferred/variant terms, notes,
# hierarchy, place types, and coordinates are available), so the code half of
# VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT is checked against this table once TGN has
# confirmed dwc:country is a real, unambiguous nation. This is a FAST PATH only
# -- a hand-curated table for the countries most common in this project's data,
# not the only source consulted. Anything TGN confirms as a real nation but that
# isn't in this table falls through to a pycountry-backed lookup instead of
# reporting EXTERNAL_PREREQUISITES_NOT_MET (see _pycountry_name_to_alpha2 below).
TGN_COUNTRY_ISO2 = {
    "US": "United States", "CA": "Canada", "MX": "Mexico", "BW": "Botswana", "ZA": "South Africa",
    "GB": "United Kingdom", "FR": "France", "DE": "Germany", "ES": "Spain", "IT": "Italy",
    "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden", "NO": "Norway", "PL": "Poland",
    "UA": "Ukraine", "GR": "Greece", "PT": "Portugal",
    "BR": "Brazil", "AR": "Argentina", "CO": "Colombia", "PE": "Peru", "CL": "Chile", "VE": "Venezuela",
    "AU": "Australia", "NZ": "New Zealand",
    "CN": "China", "IN": "India", "JP": "Japan", "ID": "Indonesia", "TH": "Thailand", "VN": "Vietnam",
    "PH": "Philippines", "MY": "Malaysia", "PK": "Pakistan", "BD": "Bangladesh", "SA": "Saudi Arabia",
    "TR": "Turkey", "IR": "Iran", "IQ": "Iraq",
    "KE": "Kenya", "NG": "Nigeria", "EG": "Egypt", "TZ": "Tanzania", "UG": "Uganda", "GH": "Ghana",
    "CD": "Democratic Republic of the Congo", "ET": "Ethiopia", "MA": "Morocco", "SD": "Sudan",
    "ZM": "Zambia", "ZW": "Zimbabwe", "NA": "Namibia", "MZ": "Mozambique", "AO": "Angola", "CM": "Cameroon",
    "RU": "Russia",
}


class TGNQueryError(Exception):
    pass


def _norm_text(x: object) -> str:
    return " ".join(str(x).strip().split()).lower()


def _norm_iso2(x: object) -> str:
    return str(x).strip().upper()


# {normalized country name: ISO alpha-2 code}, built once from TGN_COUNTRY_ISO2.
TGN_NAME_TO_ISO2 = {_norm_text(name): code for code, name in TGN_COUNTRY_ISO2.items()}


@lru_cache(maxsize=1)
def _pycountry_name_lookup() -> dict:
    """
    {normalized country name variant: alpha_2 code}, built once (cached) from
    every name pycountry exposes for each ISO 3166-1 country: the plain
    `name`, plus `common_name`/`official_name` when a country has them (e.g.
    Bolivia's `name` is "Bolivia, Plurinational State of" but its
    `common_name` is "Bolivia" -- both variants are indexed here). pycountry
    is a maintained mirror of the real ISO 3166-1 database (sourced from
    Debian's iso-codes project), not a hand-typed table -- this is what lets
    VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT verify the code for any TGN-confirmed
    nation, not just the ~57 in the TGN_COUNTRY_ISO2 fast-path table above.
    Returns {} if pycountry isn't installed.
    """
    lookup = {}
    if pycountry is None:
        return lookup
    for c in pycountry.countries:
        for attr in ("name", "common_name", "official_name"):
            val = getattr(c, attr, None)
            if val:
                lookup[_norm_text(val)] = c.alpha_2
    return lookup


def _pycountry_name_to_alpha2(name_norm: str) -> Optional[str]:
    """
    ISO 3166-1 alpha-2 code for a country name, via pycountry. `name_norm` is
    expected to already be a name TGN itself confirmed is a real, unambiguous
    nation (see _tgn_reconcile_nation_id) -- this function's only job is
    mapping that confirmed name to a code, not judging whether it's real.

    Tries an exact match against every name/common_name/official_name
    pycountry exposes first (covers the large majority of cases, since TGN's
    canonical English name usually matches one of pycountry's). If that
    misses -- e.g. a spelling difference between TGN's and pycountry's
    canonical forms -- falls back to pycountry's own fuzzy name search, but
    only accepts the result if it's unambiguous (exactly one match); an
    ambiguous fuzzy match isn't a safe basis for a COMPLIANT/NOT_COMPLIANT
    verdict, so those still fall through to EXTERNAL_PREREQUISITES_NOT_MET.

    Returns None (never raises) if pycountry isn't installed, or no
    unambiguous match is found.
    """
    if pycountry is None:
        return None
    exact = _pycountry_name_lookup().get(name_norm)
    if exact:
        return exact
    try:
        matches = pycountry.countries.search_fuzzy(name_norm)
    except LookupError:
        return None
    except Exception:
        return None
    if len(matches) == 1:
        return matches[0].alpha_2
    return None


@st.cache_data(ttl=300, show_spinner=False)
def tgn_health_check(source_authority: dict | None = None, timeout: int = 8) -> tuple[bool, str]:
    """
    External prerequisite check: is the Getty reconciliation service reachable and
    does it return its service manifest as JSON? Cached for 5 minutes so every row
    doesn't re-check the endpoint.

    Returns (ok, reason) -- `reason` is a short, human-readable diagnostic string
    (network/firewall/proxy vs. TLS vs. timeout vs. a bad HTTP status vs. an
    unexpected response body) instead of collapsing every failure into one boolean.
    """
    sa = source_authority or TGN_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")
    if not endpoint:
        return False, "No 'endpoint' configured in source_authority."
    try:
        r = requests.get(
            endpoint,
            headers={"Accept": "application/json", "User-Agent": "bdq-validator/1.0"},
            timeout=timeout,
        )
    except requests.exceptions.SSLError as e:
        return False, f"TLS/SSL error talking to {endpoint} -- often a corporate proxy/antivirus doing SSL " \
                       f"inspection with an untrusted cert, or an outdated CA bundle. ({e})"
    except requests.exceptions.ConnectTimeout:
        return False, f"Connection to {endpoint} timed out after {timeout}s while connecting -- usually a " \
                       f"firewall/proxy silently dropping the request rather than refusing it."
    except requests.exceptions.ReadTimeout:
        return False, f"{endpoint} accepted the connection but didn't respond within {timeout}s -- the Getty " \
                       f"reconciliation service itself may be slow or overloaded right now."
    except requests.exceptions.ConnectionError as e:
        return False, f"Could not connect to {endpoint} -- likely no outbound network access to this host " \
                       f"(school/corporate/VPN firewall blocking it, DNS not resolving, or you're offline). ({e})"
    except requests.exceptions.RequestException as e:
        return False, f"Request to {endpoint} failed: {e}"

    if r.status_code != 200:
        return False, f"{endpoint} responded with HTTP {r.status_code} instead of 200: {r.text[:200]!r}"

    try:
        js = r.json()
    except ValueError:
        return False, f"{endpoint} returned a 200 but the body wasn't valid JSON: {r.text[:200]!r}"

    if not (isinstance(js, dict) and ("name" in js or "identifierSpace" in js)):
        return False, f"{endpoint} returned JSON without the expected service-manifest fields: {str(js)[:200]!r}"

    return True, "OK"


def tgn_available(source_authority: dict | None = None, timeout: int = 8) -> bool:
    """Backward-compatible boolean wrapper around tgn_health_check()."""
    return tgn_health_check(source_authority, timeout)[0]


# ------------------------------------------------------------
# Core Getty Reconciliation API helpers
# ------------------------------------------------------------
def _reconcile_query(name: str, endpoint: str, type_id: str = GETTY_RECONCILE_TGN_TYPE,
                      limit: int = 15, timeout: int = 10) -> list[dict]:
    """
    Run one reconciliation query (POST, per the W3C/OpenRefine Reconciliation API --
    GET against this endpoint only returns the service manifest, confirmed live).
    Returns the "result" list of candidates for that single query.
    """
    payload = {"q0": {"query": name, "type": type_id, "limit": limit}}
    try:
        r = requests.post(
            endpoint,
            data={"queries": json.dumps(payload)},
            headers={"Accept": "application/json", "User-Agent": "bdq-validator/1.0"},
            timeout=timeout,
        )
        if r.status_code != 200:
            raise TGNQueryError(f"Getty reconcile HTTP {r.status_code}: {r.text[:200]}")
        js = r.json()
        return js.get("q0", {}).get("result", []) or []
    except (requests.RequestException, ValueError) as e:
        raise TGNQueryError(str(e)) from e


def _reconcile_extend(ids: list[str], property_ids: list[str], endpoint: str, timeout: int = 10) -> dict:
    """
    Data-extension request (POST): fetch extend properties (place types, parent
    hierarchy, coordinates, ...) for a batch of reconciliation candidate ids.
    Returns {id: {property_id: [value, ...]}} (the "rows" section of the response).
    """
    if not ids:
        return {}
    payload = {"ids": ids, "properties": [{"id": p} for p in property_ids]}
    try:
        r = requests.post(
            endpoint,
            data={"extend": json.dumps(payload)},
            headers={"Accept": "application/json", "User-Agent": "bdq-validator/1.0"},
            timeout=timeout,
        )
        if r.status_code != 200:
            raise TGNQueryError(f"Getty reconcile extend HTTP {r.status_code}: {r.text[:200]}")
        js = r.json()
        return js.get("rows", {}) or {}
    except (requests.RequestException, ValueError) as e:
        raise TGNQueryError(str(e)) from e


def _extend_value_strs(rows: dict, entity_id: str, property_id: str) -> list[str]:
    """Flatten one entity's extend property values (list of {"str"/"id": ...} dicts) to plain strings."""
    vals = (rows.get(entity_id) or {}).get(property_id) or []
    out = []
    for v in vals:
        if isinstance(v, dict):
            if v.get("str") is not None:
                out.append(str(v["str"]))
            elif v.get("id") is not None:
                out.append(str(v["id"]))
        elif v is not None:
            out.append(str(v))
    return out


@lru_cache(maxsize=4096)
def _tgn_reconcile_nation_id(country_name_norm: str, endpoint: str) -> str | None:
    """
    Resolve dwc:country -> UNIQUE Getty TGN place id via the reconciliation API,
    restricted to candidates whose returned name matches country_name_norm exactly
    (case-insensitive) AND whose /tgn_placetypes extend value contains "nation"
    (TGN's own place-type vocabulary uses "Nation" or "nations (nation-level
    political entities)" for country-level places). Returns the short id (e.g.
    "tgn/7000084") only if exactly one such candidate exists, else None.
    """
    candidates = _reconcile_query(country_name_norm, endpoint, type_id=GETTY_RECONCILE_TGN_TYPE)
    name_matches = [c for c in candidates if _norm_text(c.get("name", "")) == country_name_norm and c.get("id")]
    if not name_matches:
        return None

    ids = list(dict.fromkeys(c["id"] for c in name_matches))
    rows = _reconcile_extend(ids, [GETTY_RECONCILE_PROP_PLACETYPES], endpoint)

    nation_ids = [
        cid for cid in ids
        if any("nation" in v.lower() for v in _extend_value_strs(rows, cid, GETTY_RECONCILE_PROP_PLACETYPES))
    ]
    nation_ids = list(dict.fromkeys(nation_ids))
    return nation_ids[0] if len(nation_ids) == 1 else None


@lru_cache(maxsize=8192)
def _tgn_reconcile_state_id(nation_name_norm: str, state_name_norm: str, endpoint: str) -> str | None:
    """
    Resolve dwc:stateProvince -> UNIQUE TGN place id, restricted to candidates whose
    name matches state_name_norm exactly AND whose /tgn_hier (parent hierarchy)
    extend value mentions the country's name -- i.e. is actually located within
    that country. TGN's hierarchy display is a breadcrumb string (e.g.
    "Kenya, Eastern Africa, Africa, World"), so this checks substring containment
    rather than requiring an exact structural parent match. Returns the short id
    only if exactly one such candidate exists, else None.
    """
    candidates = _reconcile_query(state_name_norm, endpoint, type_id=GETTY_RECONCILE_TGN_TYPE)
    name_matches = [c for c in candidates if _norm_text(c.get("name", "")) == state_name_norm and c.get("id")]
    if not name_matches:
        return None

    ids = list(dict.fromkeys(c["id"] for c in name_matches))
    rows = _reconcile_extend(ids, [GETTY_RECONCILE_PROP_HIER], endpoint)

    in_country_ids = [
        cid for cid in ids
        if any(nation_name_norm in _norm_text(v) for v in _extend_value_strs(rows, cid, GETTY_RECONCILE_PROP_HIER))
    ]
    in_country_ids = list(dict.fromkeys(in_country_ids))
    return in_country_ids[0] if len(in_country_ids) == 1 else None


@lru_cache(maxsize=4096)
def _tgn_reconcile_coordinates(nation_id: str, endpoint: str) -> tuple | None:
    """
    Fetch (lat, lon) for a TGN place id via the /tgn_coordinates extend property.
    Getty's TGN pages display coordinates as decimal-degree lat/long text; this
    parses the first two signed decimal numbers found in that value. Returns None
    if the property is empty or doesn't parse into a valid (lat, lon) pair.
    """
    rows = _reconcile_extend([nation_id], [GETTY_RECONCILE_PROP_COORDINATES], endpoint)
    vals = _extend_value_strs(rows, nation_id, GETTY_RECONCILE_PROP_COORDINATES)
    if not vals:
        return None
    nums = re.findall(r"-?\d+\.\d+", vals[0])
    if len(nums) < 2:
        return None
    try:
        lat, lon = float(nums[0]), float(nums[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    return lat, lon


@lru_cache(maxsize=8192)
def _tgn_reconcile_state_found(state_name_norm: str, endpoint: str) -> bool:
    """
    True if dwc:stateProvince (name only -- no dwc:country needed) resolves via
    the Getty reconciliation API to at least one TGN place whose name matches
    state_name_norm exactly and whose /tgn_hier parent-hierarchy breadcrumb
    (e.g. "Kenya, Eastern Africa, Africa, World") names at least one entity
    TGN itself represents as an ISO-3166-country-like place. This is
    deliberately country-agnostic: it checks that dwc:stateProvince is *some*
    TGN administrative entity under *some* ISO 3166 country, independent of
    whatever (possibly wrong or missing) value dwc:country holds.

    "Represents an ISO 3166 country-like entity in bdqval:sourceAuthority" is
    checked two ways so this test is driven by TGN itself, not capped by the
    app's small local reference table:
      1) Fast path -- the breadcrumb component matches the app's local
         TGN_COUNTRY_ISO2 table (same table VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT /
         VALIDATION_COUNTRY_STATEPROVINCE_UNAMBIGUOUS use).
      2) Fallback -- for any breadcrumb component not in that local table,
         ask TGN itself (via _tgn_reconcile_nation_id) whether that name
         resolves, unambiguously, to a place TGN classifies with place type
         "nation". This covers ISO 3166 countries outside the local table
         instead of silently reporting NOT_COMPLIANT for them.
    """
    candidates = _reconcile_query(state_name_norm, endpoint, type_id=GETTY_RECONCILE_TGN_TYPE)
    name_matches = [c for c in candidates if _norm_text(c.get("name", "")) == state_name_norm and c.get("id")]
    if not name_matches:
        return False

    ids = list(dict.fromkeys(c["id"] for c in name_matches))
    rows = _reconcile_extend(ids, [GETTY_RECONCILE_PROP_HIER], endpoint)

    breadcrumb_components = set()
    for cid in ids:
        for breadcrumb in _extend_value_strs(rows, cid, GETTY_RECONCILE_PROP_HIER):
            for part in breadcrumb.split(","):
                part_norm = _norm_text(part)
                if part_norm:
                    breadcrumb_components.add(part_norm)

    if not breadcrumb_components:
        return False

    # 1) Fast path: the app's local ISO 3166 reference table.
    if any(p in TGN_NAME_TO_ISO2 for p in breadcrumb_components):
        return True

    # 2) Fallback: ask TGN itself whether any remaining breadcrumb component
    # is a real, unambiguous "nation"-type place.
    for part in breadcrumb_components:
        try:
            if _tgn_reconcile_nation_id(part, endpoint) is not None:
                return True
        except TGNQueryError:
            # A single breadcrumb-component lookup failing shouldn't sink the
            # whole check -- keep trying the other components.
            continue

    return False


# ============================================================
# TEST 1: VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT (TGN-backed)
# ============================================================
def test_country_countrycode_consistent(
    country_series: pd.Series,
    countrycode_series: pd.Series,
    source_authority: dict | None = None,
    source_authority_available: bool | None = None,
) -> pd.Series:
    """
    bdq:sourceAuthority default = Getty TGN (via GETTY_RECONCILE)
    COMPLIANT if:
      - dwc:country resolves unambiguously to ONE "nation" in TGN
      - that nation's ISO alpha-2 code -- checked first against the local
        TGN_COUNTRY_ISO2 fast-path table, then against pycountry's full ISO
        3166-1 database (see _pycountry_name_to_alpha2; TGN itself doesn't
        expose ISO codes, so neither of these queries TGN directly for the
        code) -- matches dwc:countryCode
    """
    sa = source_authority or TGN_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")

    if source_authority_available is None:
        source_authority_available = tgn_available(sa)

    out = []
    for country, code in zip(country_series, countrycode_series):
        if (not endpoint) or (not source_authority_available):
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(country) or str(country).strip() == "" or pd.isna(code) or str(code).strip() == "":
            out.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        country_norm = _norm_text(country)
        code_norm = _norm_iso2(code)

        try:
            nation_id = _tgn_reconcile_nation_id(country_norm, endpoint)
            if not nation_id:
                out.append("NOT_COMPLIANT")
                continue

            expected_code = TGN_NAME_TO_ISO2.get(country_norm)
            if expected_code is None:
                # Not in the small local fast-path table -- fall back to
                # pycountry's full ISO 3166-1 database rather than giving up.
                expected_code = _pycountry_name_to_alpha2(country_norm)
            if expected_code is None:
                # TGN confirms this is a real, unambiguous nation, but neither
                # the local table nor pycountry can unambiguously map its name
                # to an ISO code, so the code half genuinely can't be verified.
                out.append("EXTERNAL_PREREQUISITES_NOT_MET")
                continue

            out.append("COMPLIANT" if code_norm == expected_code else "NOT_COMPLIANT")

        except TGNQueryError:
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")

    return pd.Series(out, index=country_series.index)


# ============================================================
# TEST 2: VALIDATION_COUNTRY_STATEPROVINCE_UNAMBIGUOUS (TGN-backed)
# ============================================================
def test_country_stateprovince_unambiguous(
    country_series: pd.Series,
    state_series: pd.Series,
    source_authority: dict | None = None,
    source_authority_available: bool | None = None,
) -> pd.Series:
    """
    bdq:sourceAuthority default = Getty TGN (via GETTY_RECONCILE)
    COMPLIANT if:
      - dwc:country resolves unambiguously to ONE "nation" in TGN
      - dwc:stateProvince resolves unambiguously to ONE entity
      - and that state/province's TGN parent hierarchy names that country
    """
    sa = source_authority or TGN_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")

    if source_authority_available is None:
        source_authority_available = tgn_available(sa)

    out = []
    for country, state in zip(country_series, state_series):
        if (not endpoint) or (not source_authority_available):
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(country) or str(country).strip() == "" or pd.isna(state) or str(state).strip() == "":
            out.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        country_norm = _norm_text(country)
        state_norm = _norm_text(state)

        try:
            nation_id = _tgn_reconcile_nation_id(country_norm, endpoint)
            if not nation_id:
                out.append("NOT_COMPLIANT")
                continue

            state_id = _tgn_reconcile_state_id(country_norm, state_norm, endpoint)
            out.append("COMPLIANT" if state_id is not None else "NOT_COMPLIANT")

        except TGNQueryError:
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")

    return pd.Series(out, index=country_series.index)


# ============================================================
# VALIDATION_STATEPROVINCE_FOUND (TGN-backed)
#
# Expected Response:
#   EXTERNAL_PREREQUISITES_NOT_MET if the bdqval:sourceAuthority is not
#   available; INTERNAL_PREREQUISITES_NOT_MET if dwc:stateProvince is
#   bdqval:Empty; COMPLIANT if the value of dwc:stateProvince occurs as an
#   administrative entity that is a child to at least one entity
#   representing an ISO 3166 country-like entity in the
#   bdqval:sourceAuthority; otherwise NOT_COMPLIANT.
# Information Elements Acted Upon: dwc:stateProvince
# Parameters: bdqval:sourceAuthority
# Default Parameter Values:
#   bdqval:sourceAuthority default = "The Getty Thesaurus of Geographic
#   Names (TGN)" {https://services.getty.edu/vocab/reconcile/}
# ============================================================
def test_stateprovince_found(
    stateprovince_series: pd.Series,
    source_authority: dict | None = None,
    source_authority_available: bool | None = None,
) -> pd.Series:
    """
    bdq:sourceAuthority default = Getty TGN (via GETTY_RECONCILE)
    COMPLIANT if dwc:stateProvince resolves (by exact name match) to at
    least one TGN place whose parent hierarchy names an ISO 3166
    country-like entity -- see _tgn_reconcile_state_found() docstring.
    Unlike VALIDATION_COUNTRY_STATEPROVINCE_UNAMBIGUOUS, this does not require or
    use dwc:country at all.
    """
    sa = source_authority or TGN_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")

    if source_authority_available is None:
        source_authority_available = tgn_available(sa)

    out = []
    for state in stateprovince_series:
        if (not endpoint) or (not source_authority_available):
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(state) or str(state).strip() == "":
            out.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        state_norm = _norm_text(state)

        try:
            found = _tgn_reconcile_state_found(state_norm, endpoint)
            out.append("COMPLIANT" if found else "NOT_COMPLIANT")
        except TGNQueryError:
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")

    return pd.Series(out, index=stateprovince_series.index)


def test_country_found(
    country_series: pd.Series,
    source_authority: dict | None = None,
    source_authority_available: bool | None = None,
) -> pd.Series:
    """
    bdq:sourceAuthority default = Getty TGN (via GETTY_RECONCILE)
    COMPLIANT if dwc:country resolves unambiguously to ONE TGN place whose
    place type is equivalent to "nation" -- see _tgn_reconcile_nation_id()
    docstring (same resolver VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT and
    VALIDATION_COUNTRY_STATEPROVINCE_UNAMBIGUOUS use for the country half of their
    checks). Unlike a plain non-empty check, a value that's present but not a
    real, unambiguous nation-level place (a typo, a state/province name, a
    historical or disputed territory TGN doesn't classify as a nation, etc.)
    correctly comes back NOT_COMPLIANT here, not COMPLIANT.
    """
    sa = source_authority or TGN_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")

    if source_authority_available is None:
        source_authority_available = tgn_available(sa)

    out = []
    for country in country_series:
        if (not endpoint) or (not source_authority_available):
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        if pd.isna(country) or str(country).strip() == "":
            out.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        country_norm = _norm_text(country)

        try:
            nation_id = _tgn_reconcile_nation_id(country_norm, endpoint)
            out.append("COMPLIANT" if nation_id is not None else "NOT_COMPLIANT")
        except TGNQueryError:
            out.append("EXTERNAL_PREREQUISITES_NOT_MET")

    return pd.Series(out, index=country_series.index)


def test_decimallatitude_inrange(lat_series):
    results = []
    for lat in lat_series:
        # NaN (e.g. a blank cell from pd.read_csv) is a valid float and
        # would NOT raise below -- it must be caught explicitly as Empty
        # before the numeric range check, or it silently falls through as
        # NOT_COMPLIANT instead of INTERNAL_PREREQUISITES_NOT_MET.
        if pd.isna(lat):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            lat = float(lat)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append("COMPLIANT" if -90 <= lat <= 90 else "NOT_COMPLIANT")
    return pd.Series(results, index=lat_series.index)


def test_decimallatitude_notempty(lat_series):
    results = []
    for lat in lat_series:
        results.append("COMPLIANT" if (pd.notna(lat) and str(lat).strip() != "") else "NOT_COMPLIANT")
    return pd.Series(results, index=lat_series.index)


def test_decimallongitude_inrange(lon_series):
    results = []
    for lon in lon_series:
        # NaN (e.g. a blank cell from pd.read_csv) is a valid float and
        # would NOT raise below -- it must be caught explicitly as Empty
        # before the numeric range check, or it silently falls through as
        # NOT_COMPLIANT instead of INTERNAL_PREREQUISITES_NOT_MET.
        if pd.isna(lon):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            lon = float(lon)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append("COMPLIANT" if -180 <= lon <= 180 else "NOT_COMPLIANT")
    return pd.Series(results, index=lon_series.index)


def test_decimallongitude_notempty(lon_series):
    results = []
    for lon in lon_series:
        results.append("COMPLIANT" if (pd.notna(lon) and str(lon).strip() != "") else "NOT_COMPLIANT")
    return pd.Series(results, index=lon_series.index)


# ============================================================
# VALIDATION_GEODETICDATUM_NOTEMPTY
#
# Expected Response:
#   COMPLIANT if dwc:geodeticDatum is bdqval:NotEmpty; otherwise NOT_COMPLIANT.
# Information Elements Acted Upon: dwc:geodeticDatum
# ============================================================
def test_geodeticdatum_notempty(geodeticdatum_series):
    results = []
    for datum in geodeticdatum_series:
        results.append("COMPLIANT" if (pd.notna(datum) and str(datum).strip() != "") else "NOT_COMPLIANT")
    return pd.Series(results, index=geodeticdatum_series.index)


# ============================================================
# BDQ Source Authority: EPSG Geodetic Parameter Dataset
#   bdqval:sourceAuthority = "EPSG" {https://epsg.org}
#   {API for EPSG codes https://apps.epsg.org/api/swagger/ui/index}
#
# Confirmed live (via https://epsg.org/API_UG_E.html and
# https://epsg.org/API_UG_1-2.html, EPSG's own official API documentation):
#   - base URL: https://apps.epsg.org/api/v1
#   - endpoint pattern: {TypeName}/{code}  (documented worked example:
#     ProjectedCoordRefSystem/{code}, Extent/{code})
#   - JSON responses carry a "Kind" field identifying the object type
#   - no authentication / API token is required
#
# NOT individually execution-verified: every live request to apps.epsg.org
# from this environment -- including EPSG's own documented example URL --
# returned HTTP 403, which points to a bot/WAF block on the outbound
# connection rather than a real auth requirement or a wrong URL (the docs
# are explicit that no token is needed). The type names below beyond
# ProjectedCoordRefSystem are best-effort, same-family names; if EPSG's API
# shape ever changes, cross-check them against
# https://apps.epsg.org/api/swagger/ui/index.
# ============================================================

EPSG_SOURCE_AUTHORITY_DEFAULT = {
    "name": "EPSG Geodetic Parameter Dataset",
    "url": "https://epsg.org",
    "api_url": "https://apps.epsg.org/api/swagger/ui/index",
    "endpoint": "https://apps.epsg.org/api/v1",
}

# EPSG object "types" (as used in the {endpoint}/{TypeName}/{code} pattern)
# that can legitimately be the value of dwc:geodeticDatum per this test's
# spec ("... a valid code ... for a Datum, or ellipsoid, or for a CRS
# appropriate for a 2D geographic coordinate in degrees"). Tried in order
# until one returns a real object for the given numeric code.
EPSG_OBJECT_TYPES = ["GeodeticDatum", "Ellipsoid", "GeodeticCoordRefSystem"]

# Substrings (lower-cased) of the EPSG "Kind" field that identify an object
# as an allowed Datum / Ellipsoid / 2D-geographic-degrees CRS.
EPSG_ALLOWED_KIND_SUBSTRINGS = ("datum", "ellipsoid", "geogcrs", "geographic 2d")

# "Kind" substrings that must be rejected even if a broader match above
# would otherwise accept them -- e.g. a 3D or compound CRS is not
# "appropriate for a 2D geographic coordinate in degrees".
EPSG_EXCLUDED_KIND_SUBSTRINGS = (
    "3d", "compound", "projcrs", "projected", "vertcrs", "vertical", "engcrs", "engineering",
)

# dwc:geodeticDatum values meaning "the datum is genuinely unknown/not
# recorded" per this test's spec -- the literal value 'not recorded'.
EPSG_NOT_RECORDED_VALUES = {"not recorded"}

# Matches the "Authority:Number" form required by the spec, e.g. "EPSG:4326".
_EPSG_CODE_RE = re.compile(r"^\s*([A-Za-z]+)\s*:\s*(\d+)\s*$")


class EPSGQueryError(Exception):
    """Raised on a genuine network/API-level failure (not a clean 404)."""
    pass


@st.cache_data(ttl=300, show_spinner=False)
def epsg_health_check(source_authority: dict | None = None, timeout: int = 8) -> tuple[bool, str]:
    """
    External prerequisite check: is the EPSG REST API reachable and does a
    known-good lookup (EPSG:4326, WGS 84) return a usable JSON object?
    Cached for 5 minutes so every row doesn't re-check the endpoint.

    Returns (ok, reason) -- mirrors tgn_health_check()'s diagnostic style.
    """
    sa = source_authority or EPSG_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")
    if not endpoint:
        return False, "No 'endpoint' configured in source_authority."
    url = f"{endpoint}/GeodeticCoordRefSystem/4326/"
    try:
        r = requests.get(
            url,
            headers={"Accept": "application/json", "User-Agent": "bdq-validator/1.0"},
            timeout=timeout,
        )
    except requests.exceptions.SSLError as e:
        return False, f"TLS/SSL error talking to {url} -- often a corporate proxy/antivirus doing SSL " \
                       f"inspection with an untrusted cert, or an outdated CA bundle. ({e})"
    except requests.exceptions.ConnectTimeout:
        return False, f"Connection to {url} timed out after {timeout}s while connecting -- usually a " \
                       f"firewall/proxy silently dropping the request rather than refusing it."
    except requests.exceptions.ReadTimeout:
        return False, f"{url} accepted the connection but didn't respond within {timeout}s -- the EPSG " \
                       f"API itself may be slow or overloaded right now."
    except requests.exceptions.ConnectionError as e:
        return False, f"Could not connect to {url} -- likely no outbound network access to this host " \
                       f"(school/corporate/VPN firewall blocking it, DNS not resolving, or you're offline). ({e})"
    except requests.exceptions.RequestException as e:
        return False, f"Request to {url} failed: {e}"

    if r.status_code == 403:
        return False, f"{url} responded with HTTP 403 (Forbidden) -- EPSG's own docs say no API token is " \
                       f"required, so this is most likely a network/WAF block on this connection rather " \
                       f"than an authentication problem."
    if r.status_code != 200:
        return False, f"{url} responded with HTTP {r.status_code} instead of 200: {r.text[:200]!r}"

    try:
        js = r.json()
    except ValueError:
        return False, f"{url} returned a 200 but the body wasn't valid JSON: {r.text[:200]!r}"

    if not (isinstance(js, dict) and js):
        return False, f"{url} returned JSON without a usable EPSG object body: {str(js)[:200]!r}"

    return True, "OK"


def epsg_available(source_authority: dict | None = None, timeout: int = 8) -> bool:
    """Backward-compatible boolean wrapper around epsg_health_check()."""
    return epsg_health_check(source_authority, timeout)[0]


@lru_cache(maxsize=512)
def _epsg_lookup_code(code: str, endpoint: str, timeout: int = 10) -> dict | None:
    """
    Look up one EPSG numeric code against each of EPSG_OBJECT_TYPES in turn
    (GET {endpoint}/{TypeName}/{code}/) until one returns a real object.

    Returns the parsed JSON dict on success, or None if EPSG genuinely has
    no Datum/Ellipsoid/GeodeticCoordRefSystem object with that code (a
    clean "doesn't exist"). A genuine network/API-level failure raises
    EPSGQueryError instead, so the caller can report
    EXTERNAL_PREREQUISITES_NOT_MET rather than silently marking the row
    NOT_COMPLIANT.
    """
    last_network_error = None
    for type_name in EPSG_OBJECT_TYPES:
        url = f"{endpoint}/{type_name}/{code}/"
        try:
            r = requests.get(
                url,
                headers={"Accept": "application/json", "User-Agent": "bdq-validator/1.0"},
                timeout=timeout,
            )
        except requests.exceptions.RequestException as e:
            last_network_error = e
            continue

        if r.status_code == 404:
            continue
        if r.status_code != 200:
            last_network_error = f"HTTP {r.status_code} from {url}"
            continue

        try:
            js = r.json()
        except ValueError:
            last_network_error = f"Non-JSON response from {url}"
            continue

        if isinstance(js, dict) and js:
            return js

    if last_network_error is not None:
        raise EPSGQueryError(f"EPSG lookup for code {code} failed: {last_network_error}")
    return None


# ============================================================
# VALIDATION_GEODETICDATUM_STANDARD
#
# Expected Response:
#   EXTERNAL_PREREQUISITES_NOT_MET if the bdqval:sourceAuthority is not
#   available; INTERNAL_PREREQUISITES_NOT_MET if dwc:geodeticDatum is
#   bdqval:Empty; COMPLIANT if the value of dwc:geodeticDatum is a valid
#   code from the bdqval:sourceAuthority (in the form Authority:Number) for
#   a Datum, or ellipsoid, or for a CRS appropriate for a 2D geographic
#   coordinate in degrees, or is the value 'not recorded'; otherwise
#   NOT_COMPLIANT.
# Information Elements Acted Upon: dwc:geodeticDatum
# Source Authority: bdqval:sourceAuthority = "EPSG" {https://epsg.org}
#   {API for EPSG codes https://apps.epsg.org/api/swagger/ui/index}
# ============================================================
def test_geodeticdatum_standard(geodeticdatum_series, source_authority=None, source_authority_available=None):
    sa = source_authority or EPSG_SOURCE_AUTHORITY_DEFAULT
    endpoint = sa.get("endpoint")

    if source_authority_available is None:
        source_authority_available = epsg_available(sa)

    if not source_authority_available or not endpoint:
        return pd.Series(
            ["EXTERNAL_PREREQUISITES_NOT_MET"] * len(geodeticdatum_series),
            index=geodeticdatum_series.index,
        )

    results = []
    for datum in geodeticdatum_series:
        if pd.isna(datum) or str(datum).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue

        text = str(datum).strip()

        if text.lower() in EPSG_NOT_RECORDED_VALUES:
            results.append("COMPLIANT")
            continue

        m = _EPSG_CODE_RE.match(text)
        if not m or m.group(1).strip().upper() != "EPSG":
            results.append("NOT_COMPLIANT")
            continue

        code = m.group(2)
        try:
            obj = _epsg_lookup_code(code, endpoint)
        except EPSGQueryError:
            results.append("EXTERNAL_PREREQUISITES_NOT_MET")
            continue

        if obj is None:
            results.append("NOT_COMPLIANT")
            continue

        kind = str(obj.get("Kind", "")).lower()
        if any(bad in kind for bad in EPSG_EXCLUDED_KIND_SUBSTRINGS):
            results.append("NOT_COMPLIANT")
            continue
        if any(good in kind for good in EPSG_ALLOWED_KIND_SUBSTRINGS):
            results.append("COMPLIANT")
        else:
            results.append("NOT_COMPLIANT")

    return pd.Series(results, index=geodeticdatum_series.index)


# ============================================================
# VALIDATION_MAXIMUMELEVATIONINMETERS_INRANGE
#
# Expected Response:
#   INTERNAL_PREREQUISITES_NOT_MET if dwc:maximumElevationInMeters is
#   bdqval:Empty or the value cannot be interpreted as a number; COMPLIANT
#   if the value of dwc:maximumElevationInMeters is within the range of
#   bdqval:minimumValidElevationInMeters to bdqval:maximumValidElevationInMeters
#   inclusive; otherwise NOT_COMPLIANT.
# Information Elements Acted Upon: dwc:maximumElevationInMeters
# Parameters: bdqval:minimumValidElevationInMeters, bdqval:maximumValidElevationInMeters
# Default Parameter Values:
#   bdqval:minimumValidElevationInMeters default = "-430"
#   bdqval:maximumValidElevationInMeters default = "8850"
# ============================================================
DEFAULT_MIN_VALID_ELEVATION_M = -430
DEFAULT_MAX_VALID_ELEVATION_M = 8850


def test_maximumelevation_inrange(
    maxelevation_series,
    min_valid_elevation_m=DEFAULT_MIN_VALID_ELEVATION_M,
    max_valid_elevation_m=DEFAULT_MAX_VALID_ELEVATION_M,
):
    results = []
    for val in maxelevation_series:
        if pd.isna(val) or str(val).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            elevation = float(val)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append(
            "COMPLIANT" if min_valid_elevation_m <= elevation <= max_valid_elevation_m else "NOT_COMPLIANT"
        )
    return pd.Series(results, index=maxelevation_series.index)


# ============================================================
# VALIDATION_MINIMUMELEVATIONINMETERS_INRANGE
#
# Expected Response:
#   INTERNAL_PREREQUISITES_NOT_MET if dwc:minimumElevationInMeters is
#   bdqval:Empty or the value is not a number; COMPLIANT if the value of
#   dwc:minimumElevationInMeters is within the range of
#   bdqval:minimumValidElevationInMeters to bdqval:maximumValidElevationInMeters
#   inclusive; otherwise NOT_COMPLIANT.
# Information Elements Acted Upon: dwc:minimumElevationInMeters
# Parameters: bdqval:minimumValidElevationInMeters, bdqval:maximumValidElevationInMeters
# Default Parameter Values:
#   bdqval:minimumValidElevationInMeters default = "-430"
#   bdqval:maximumValidElevationInMeters default = "8850"
# (Same default valid-elevation range as VALIDATION_MAXIMUMELEVATIONINMETERS_INRANGE,
# reusing DEFAULT_MIN_VALID_ELEVATION_M / DEFAULT_MAX_VALID_ELEVATION_M.)
# ============================================================
def test_minimumelevation_inrange(
    minelevation_series,
    min_valid_elevation_m=DEFAULT_MIN_VALID_ELEVATION_M,
    max_valid_elevation_m=DEFAULT_MAX_VALID_ELEVATION_M,
):
    results = []
    for val in minelevation_series:
        if pd.isna(val) or str(val).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            elevation = float(val)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append(
            "COMPLIANT" if min_valid_elevation_m <= elevation <= max_valid_elevation_m else "NOT_COMPLIANT"
        )
    return pd.Series(results, index=minelevation_series.index)


# ============================================================
# VALIDATION_MINELEVATION_LESSTHAN_MAXELEVATION
#
# Expected Response:
#   INTERNAL_PREREQUISITES_NOT_MET if dwc:maximumElevationInMeters or
#   dwc:minimumElevationInMeters is bdqval:Empty, or if either is not a
#   number; COMPLIANT if the value of dwc:minimumElevationInMeters is a
#   number less than or equal to the value of the number
#   dwc:maximumElevationInMeters, otherwise NOT_COMPLIANT.
# Information Elements Acted Upon: dwc:minimumElevationInMeters, dwc:maximumElevationInMeters
# ============================================================
def test_minelevation_lessthan_maxelevation(minelevation_series, maxelevation_series):
    min_aligned, max_aligned = minelevation_series.align(maxelevation_series, join="outer")
    results = []
    for minval, maxval in zip(min_aligned, max_aligned):
        if pd.isna(minval) or str(minval).strip() == "" or pd.isna(maxval) or str(maxval).strip() == "":
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        try:
            min_elevation = float(minval)
            max_elevation = float(maxval)
        except (TypeError, ValueError):
            results.append("INTERNAL_PREREQUISITES_NOT_MET")
            continue
        results.append("COMPLIANT" if min_elevation <= max_elevation else "NOT_COMPLIANT")
    return pd.Series(results, index=min_aligned.index)


def test_location_notempty(df_row):
    location_columns = [
        "higherGeographyID", "higherGeography", "continent", "country",
        "countryCode", "stateProvince", "county", "municipality",
        "waterBody", "island", "islandGroup", "locality", "locationID",
        "verbatimLocality", "decimalLatitude", "decimalLongitude",
        "verbatimCoordinates", "verbatimLatitude", "verbatimLongitude",
        "footprintWKT",
    ]
    if any(pd.notna(df_row.get(col, pd.NA)) and str(df_row.get(col, "")).strip() != "" for col in location_columns):
        return "COMPLIANT"
    return "NOT_COMPLIANT"


# ============================================================
# Streamlit UI
# ============================================================
# ============================================================
# Official TDWG Biodiversity Data Quality (BDQ) TG2 test specifications
# for the tests implemented in this app -- sourced from the authoritative
# GitHub issue for each test at https://github.com/tdwg/bdq/issues
# (one issue per CORE test; the issue body carries the ratified Description,
# Dimension, Type, Darwin Core Class, Information Elements, and Expected
# Response text, which is quoted verbatim below).
#
# A few of this app's test columns don't map onto a distinct, separately
# numbered CORE test in the standard (confirmed=False below); for those the
# description reflects this app's own implementation instead, clearly
# labeled as such rather than presented as ratified BDQ wording.
# ============================================================
BDQ_TEST_SPECS = {
    "VALIDATION_COUNTRY_NOT_EMPTY": {
        "official_name": "VALIDATION_COUNTRY_NOTEMPTY",
        "description": "Is there a value in dwc:country?",
        "dimension": "Completeness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:country"],
        "expected_response": (
            'COMPLIANT if dwc:country is bdq:NotEmpty or dwc:countryCode has a value of "XZ" and '
            'either dwc:country is bdq:Empty or has a value of "High seas"; otherwise NOT_COMPLIANT'
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/42",
        "confirmed": True,
    },
    "VALIDATION_COORDINATES_COUNTRYCODE_CONSISTENT": {
        "official_name": "VALIDATION_COORDINATESCOUNTRYCODE_CONSISTENT",
        "description": (
            "Do the geographic coordinates fall on or within the boundaries of the territory given "
            "in dwc:countryCode or its Exclusive Economic Zone?"
        ),
        "dimension": "Consistency",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:countryCode", "dwc:decimalLatitude", "dwc:decimalLongitude"],
        "parameters": ["bdq:sourceAuthority", "bdq:spatialBufferInMeters"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:decimalLatitude, dwc:decimalLongitude or "
            "dwc:countryCode are bdq:Empty or not valid/interpretable; COMPLIANT if the coordinates "
            "fall within the boundary of the territory given by dwc:countryCode, or within its "
            "Exclusive Economic Zone, plus bdq:spatialBufferInMeters; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/50",
        "confirmed": True,
    },
    "VALIDATION_COORDINATES_STATEPROVINCE_CONSISTENT": {
        "official_name": "VALIDATION_COORDINATESSTATEPROVINCE_CONSISTENT",
        "description": (
            "Do the geographic coordinates fall on or within the boundary from the bdq:sourceAuthority "
            "for the given dwc:stateProvince, or within the distance given by "
            "bdq:spatialBufferInMeters outside that boundary?"
        ),
        "dimension": "Consistency",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:stateProvince", "dwc:decimalLatitude", "dwc:decimalLongitude"],
        "parameters": [
            'bdq:sourceAuthority (default: "10m-admin-1 boundaries")',
            'bdq:spatialBufferInMeters (default: "3000")',
        ],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if the coordinates or dwc:stateProvince are empty, "
            "invalid, or dwc:stateProvince is not found in the bdq:sourceAuthority; COMPLIANT if the "
            "coordinates fall within the dwc:stateProvince boundary (or within the specified buffer "
            "distance of it); otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/56",
        "confirmed": True,
    },
    "VALIDATION_COORDINATES_NOTZERO": {
        "official_name": "VALIDATION_COORDINATES_NOTZERO",
        "description": (
            "Are the values of either dwc:decimalLatitude or dwc:decimalLongitude numbers that are "
            "not equal to 0?"
        ),
        "dimension": "Likeliness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:decimalLatitude", "dwc:decimalLongitude"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:decimalLatitude is bdq:Empty or is not "
            "interpretable as a number, or dwc:decimalLongitude is bdq:Empty or is not interpretable "
            "as a number; COMPLIANT if either the value of dwc:decimalLatitude is not = 0 or the "
            "value of dwc:decimalLongitude is not = 0; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/87",
        "confirmed": True,
    },
    "VALIDATION_COORDINATEUNCERTAINTY_INRANGE": {
        "official_name": None,
        "description": (
            "Is the value of dwc:coordinateUncertaintyInMeters a number between 1 and 20,037,509 "
            "(half of the Earth's circumference, in meters) inclusive?"
        ),
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:coordinateUncertaintyInMeters"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:coordinateUncertaintyInMeters is bdq:Empty or is "
            "not interpretable as a number; COMPLIANT if the value is between 1 and 20037509 "
            "inclusive; otherwise NOT_COMPLIANT"
        ),
        "issue_url": None,
        "confirmed": False,
    },
    "VALIDATION_COUNTRYCODE_NOTEMPTY": {
        "official_name": None,
        "description": "Is there a value in dwc:countryCode?",
        "dimension": "Completeness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:countryCode"],
        "expected_response": "COMPLIANT if dwc:countryCode is bdq:NotEmpty; otherwise NOT_COMPLIANT",
        "issue_url": None,
        "confirmed": False,
    },
    "VALIDATION_COUNTRYCODE_STANDARD": {
        "official_name": "VALIDATION_COUNTRYCODE_STANDARD",
        "description": "Is the value of dwc:countryCode a valid ISO 3166-1-alpha-2 country code?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:countryCode"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if the dwc:countryCode is bdq:Empty; COMPLIANT if "
            "dwc:countryCode can be unambiguously interpreted as a valid ISO 3166-1-alpha-2 country "
            "code in the bdq:sourceAuthority; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/20",
        "confirmed": True,
    },
    "VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT": {
        "official_name": "VALIDATION_COUNTRYCOUNTRYCODE_CONSISTENT",
        "description": (
            "Does the ISO country code, determined from the value of dwc:country, equal the value of "
            "dwc:countryCode?"
        ),
        "dimension": "Consistency",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:country", "dwc:countryCode"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if either of the terms dwc:country or dwc:countryCode "
            "are bdq:Empty; COMPLIANT if the values of dwc:country and dwc:countryCode match the "
            "national-level country name and matching country code respectively in the "
            "bdq:sourceAuthority; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/62",
        "confirmed": True,
    },
    "VALIDATION_COUNTRY_STATEPROVINCE_UNAMBIGUOUS": {
        "official_name": None,
        "description": (
            "Is the value of dwc:stateProvince unambiguous (found in the bdq:sourceAuthority as a "
            "child administrative entity of the country given by dwc:country/dwc:countryCode, "
            "without matching more than one country)?"
        ),
        "dimension": "Consistency",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:country", "dwc:countryCode", "dwc:stateProvince"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:stateProvince and dwc:country/dwc:countryCode are "
            "empty; COMPLIANT if dwc:stateProvince resolves to exactly one place in the "
            "bdq:sourceAuthority consistent with dwc:country/dwc:countryCode; otherwise NOT_COMPLIANT"
        ),
        "issue_url": None,
        "confirmed": False,
    },
    "VALIDATION_STATEPROVINCE_FOUND": {
        "official_name": "VALIDATION_STATEPROVINCE_FOUND",
        "description": "Does the value of dwc:stateProvince occur in the bdq:sourceAuthority?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:stateProvince"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:stateProvince is bdq:Empty; COMPLIANT if the "
            "value of dwc:stateProvince occurs as an administrative entity that is a child to at "
            "least one entity representing an ISO 3166 country-like entity in the "
            "bdq:sourceAuthority; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/199",
        "confirmed": True,
    },
    "VALIDATION_COUNTRY_FOUND": {
        "official_name": "VALIDATION_COUNTRY_FOUND",
        "description": "Does the value of dwc:country occur in the bdq:sourceAuthority?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:country"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:country is bdq:Empty; COMPLIANT if the value of "
            "dwc:country is a place type equivalent to administrative entity of 'nation' in the "
            "bdq:sourceAuthority; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/21",
        "confirmed": True,
    },
    "VALIDATION_DECIMALLATITUDE_INRANGE": {
        "official_name": "VALIDATION_DECIMALLATITUDE_INRANGE",
        "description": "Is the value of dwc:decimalLatitude a number between -90 and 90 inclusive?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:decimalLatitude"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:decimalLatitude is bdq:Empty or the value is not "
            "a number; COMPLIANT if the value of dwc:decimalLatitude is between -90 and 90 degrees, "
            "inclusive; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/79",
        "confirmed": True,
    },
    "VALIDATION_DECIMALLATITUDE_NOTEMPTY": {
        "official_name": "VALIDATION_DECIMALLATITUDE_NOTEMPTY",
        "description": "Is there a value in dwc:decimalLatitude?",
        "dimension": "Completeness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:decimalLatitude"],
        "expected_response": "COMPLIANT if dwc:decimalLatitude is bdq:NotEmpty; otherwise NOT_COMPLIANT",
        "issue_url": "https://github.com/tdwg/bdq/issues/119",
        "confirmed": True,
    },
    "VALIDATION_DECIMALLONGITUDE_INRANGE": {
        "official_name": "VALIDATION_DECIMALLONGITUDE_INRANGE",
        "description": "Is the value of dwc:decimalLongitude a number between -180 and 180 inclusive?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:decimalLongitude"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:decimalLongitude is bdq:Empty or the value is "
            "not a number; COMPLIANT if the value of dwc:decimalLongitude is between -180 and 180 "
            "degrees, inclusive; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/30",
        "confirmed": True,
    },
    "VALIDATION_DECIMALLONGITUDE_NOTEMPTY": {
        "official_name": "VALIDATION_DECIMALLONGITUDE_NOTEMPTY",
        "description": "Is there a value in dwc:decimalLongitude?",
        "dimension": "Completeness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:decimalLongitude"],
        "expected_response": "COMPLIANT if dwc:decimalLongitude is bdq:NotEmpty; otherwise NOT_COMPLIANT",
        "issue_url": "https://github.com/tdwg/bdq/issues/96",
        "confirmed": True,
    },
    "VALIDATION_GEODETICDATUM_NOTEMPTY": {
        "official_name": "VALIDATION_GEODETICDATUM_NOTEMPTY",
        "description": "Is there a value in dwc:geodeticDatum?",
        "dimension": "Completeness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:geodeticDatum"],
        "expected_response": "COMPLIANT if dwc:geodeticDatum is bdq:NotEmpty; otherwise NOT_COMPLIANT",
        "issue_url": "https://github.com/tdwg/bdq/issues/78",
        "confirmed": True,
    },
    "VALIDATION_GEODETICDATUM_STANDARD": {
        "official_name": "VALIDATION_GEODETICDATUM_STANDARD",
        "description": (
            "Does the value of dwc:geodeticDatum occur as a valid geographic CRS, geodetic Datum or "
            "ellipsoid in bdq:sourceAuthority?"
        ),
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:geodeticDatum"],
        "expected_response": (
            "EXTERNAL_PREREQUISITES_NOT_MET if the bdq:sourceAuthority is not available; "
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:geodeticDatum is bdq:Empty; COMPLIANT if the "
            'value of dwc:geodeticDatum is a valid code from the bdq:sourceAuthority (in the form '
            'Authority:Number) for a Datum, ellipsoid, or a CRS appropriate for a 2D geographic '
            'coordinate in degrees, or is the value "not recorded"; otherwise NOT_COMPLIANT'
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/59",
        "confirmed": True,
    },
    "VALIDATION_MAXIMUMELEVATIONINMETERS_INRANGE": {
        "official_name": "VALIDATION_MAXELEVATION_INRANGE",
        "description": "Is the value of dwc:maximumElevationInMeters of a single record within a valid range?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:maximumElevationInMeters"],
        "parameters": ["bdq:minimumValidElevationInMeters", "bdq:maximumValidElevationInMeters"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:maximumElevationInMeters is bdq:Empty or the "
            "value cannot be interpreted as a number; COMPLIANT if the value of "
            "dwc:maximumElevationInMeters is within the range of bdq:minimumValidElevationInMeters "
            "to bdq:maximumValidElevationInMeters inclusive; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/112",
        "confirmed": True,
    },
    "VALIDATION_MINIMUMELEVATIONINMETERS_INRANGE": {
        "official_name": "VALIDATION_MINELEVATION_INRANGE",
        "description": "Is the value of dwc:minimumElevationInMeters within the parameter range?",
        "dimension": "Conformance",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:minimumElevationInMeters"],
        "parameters": ["bdq:minimumValidElevationInMeters", "bdq:maximumValidElevationInMeters"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if dwc:minimumElevationInMeters is bdq:Empty or the "
            "value is not a number; COMPLIANT if the value of dwc:minimumElevationInMeters is "
            "within the range of bdq:minimumValidElevationInMeters to "
            "bdq:maximumValidElevationInMeters inclusive; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/39",
        "confirmed": True,
    },
    "VALIDATION_MINELEVATION_LESSTHAN_MAXELEVATION": {
        "official_name": None,
        "description": (
            "Is the value of dwc:minimumElevationInMeters less than or equal to the value of "
            "dwc:maximumElevationInMeters? (modeled on the ratified depth equivalent, "
            "VALIDATION_MINDEPTH_LESSTHAN_MAXDEPTH)"
        ),
        "dimension": "Consistency",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": ["dwc:minimumElevationInMeters", "dwc:maximumElevationInMeters"],
        "expected_response": (
            "INTERNAL_PREREQUISITES_NOT_MET if either dwc:minimumElevationInMeters or "
            "dwc:maximumElevationInMeters is bdq:Empty or not interpretable as a number; COMPLIANT "
            "if the value of dwc:minimumElevationInMeters is less than or equal to the value of "
            "dwc:maximumElevationInMeters; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/24",
        "confirmed": False,
    },
    "VALIDATION_LOCATION_NOTEMPTY": {
        "official_name": "VALIDATION_LOCATION_NOTEMPTY",
        "description": "Is there a value in any of the Darwin Core spatial terms that could specify a location?",
        "dimension": "Completeness",
        "type": "Validation",
        "dwc_class": "dcterms:Location",
        "elements_acted_upon": [
            "dwc:higherGeographyID", "dwc:higherGeography", "dwc:continent", "dwc:country",
            "dwc:countryCode", "dwc:stateProvince", "dwc:county", "dwc:municipality",
            "dwc:waterBody", "dwc:island", "dwc:islandGroup", "dwc:locality", "dwc:locationID",
            "dwc:verbatimLocality", "dwc:decimalLatitude", "dwc:decimalLongitude",
            "dwc:verbatimCoordinates", "dwc:verbatimLatitude", "dwc:verbatimLongitude",
            "dwc:footprintWKT",
        ],
        "expected_response": (
            "COMPLIANT if at least one term needed to determine the location of the entity exists "
            "and is bdq:NotEmpty; otherwise NOT_COMPLIANT"
        ),
        "issue_url": "https://github.com/tdwg/bdq/issues/40",
        "confirmed": True,
    },
}


def render_bdq_test_details(test_col: str, row_counts: dict | None = None):
    """Renders one test's official BDQ Standard description/expected-response
    text (or, for the few tests without a confirmed distinct CORE
    specification, a clearly-labeled implementation-based description),
    plus this dataset's compliance counts for it."""
    spec = BDQ_TEST_SPECS.get(test_col)
    if spec is None:
        st.caption("No BDQ Standard description available for this test.")
        return

    if spec.get("confirmed"):
        st.markdown(f"**Official BDQ test name:** `{spec['official_name']}`")
    else:
        st.caption(
            "This app's test does not map onto a single, separately-numbered CORE test in the "
            "ratified BDQ standard -- the description below reflects this test's own "
            "implementation, written in the same Dimension/Type/Expected Response style the "
            "standard uses."
        )

    st.markdown(f"**Description:** {spec['description']}")
    st.markdown(f"**Data Quality Dimension:** {spec['dimension']}")
    st.markdown(f"**Type:** {spec['type']}")
    st.markdown(f"**Darwin Core Class:** {spec['dwc_class']}")
    st.markdown(f"**Information Element(s) Acted Upon:** {', '.join(spec['elements_acted_upon'])}")
    if spec.get("parameters"):
        st.markdown(f"**Parameters:** {', '.join(spec['parameters'])}")
    st.markdown("**Expected Response:**")
    st.info(spec["expected_response"])
    if spec.get("issue_url"):
        st.caption(f"Specification source: {spec['issue_url']}")

    if row_counts is not None:
        st.markdown("**This dataset's results for this test:**")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Compliant", int(row_counts.get("Compliant Count", 0)))
        cc2.metric("Not compliant", int(row_counts.get("Not Compliant Count", 0)))
        cc3.metric("Potential issue", int(row_counts.get("Potential Issue Count", 0)))
        cc4.metric("Prerequisite not met", int(row_counts.get("Prerequisite Not Met", 0)))


st.set_page_config(layout="wide")
st.title("AFRICAN TROPICAL PLANTS Geospatial Data Quality Validator")
st.caption("Upload a CSV file with geospatial data for BDQ-style validation.")

if not render_auth_gate():
    st.stop()

gdf_states = load_source_authority_gdf()
gdf_eez = load_eez_gdf()
tgn_is_up, tgn_status_reason = tgn_health_check()
epsg_is_up, epsg_status_reason = epsg_health_check()

# ---- Upload history: date of uploading a version + its status (ONLY) ----
if "upload_history" not in st.session_state:
    st.session_state["upload_history"] = []  # list of dicts: uploaded_at, version_id, status

def _file_version_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:12]

def _dataset_status_from_summary(summary_df: pd.DataFrame) -> str:
    """
    Single status for the uploaded dataset version.
    Priority:
      1) Any EXTERNAL/INTERNAL prereq not met -> PREREQUISITES_NOT_MET
      2) Any NOT_COMPLIANT or POTENTIAL_ISSUE -> ISSUES_FOUND
      3) Otherwise -> ALL_COMPLIANT
    """
    if summary_df.empty:
        return "UNKNOWN"

    prereq = summary_df["Prerequisite Not Met"].sum() if "Prerequisite Not Met" in summary_df.columns else 0
    not_ok = summary_df["Not Compliant Count"].sum() if "Not Compliant Count" in summary_df.columns else 0
    pot = summary_df["Potential Issue Count"].sum() if "Potential Issue Count" in summary_df.columns else 0

    if prereq > 0:
        return "PREREQUISITES_NOT_MET"
    if (not_ok + pot) > 0:
        return "ISSUES_FOUND"
    return "ALL_COMPLIANT"


# ------------------------------------------------------------
# Branding: African-Plants.org logo, embedded as a base64 PNG so this
# stays a single self-contained script (no separate image file needs to
# ship alongside this .py file). To swap in a different image, regenerate
# this string with:
#   import base64
#   base64.b64encode(open("logo.png", "rb").read()).decode("ascii")
# ------------------------------------------------------------
AFRICAN_PLANTS_LOGO_B64 = (
    "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"
    "TYcMGZI20EKn09HS0kJdXR0DBgxgyZIlbNy4UX2dUChEeXm5GkfeUdKf8nvlOnq9XhoaGlQZP/WQpdRvWywWtmzZwrXXXst///tf"
    "Lr30Us4880wViPfAJC1d0ieeAdhlWT5NEISgBsKaBP1rga8dmJME35gGvt1L0OXl5WpMT3FEgUBAdezRaBSHw0FdXR0nnngiK1as"
    "ULNUd1ZyPeKII9JG1pnNZr755hu8Xq862F4B4eOOO24rObqgoIBHH32U1157jf3331+ds2symdiwYQPLli1L+04KCCug0rX5xo6c"
    "bXcDGRKJhArAv4QBx2IxIpEIer1elZWVBLWMjAwyMzNxOp04nU6ysrLIzs4mOzubgoICJEnCaDTy7rvvsnTpUux2u3qfgsEgV1xx"
    "RVpiVWoMObUpicViYe+996asrExtlKLEgdetW7dVByqFdX766afq6yvX4aijjkq71tCZiKXEhBWVwev1UllZiSzLzJ07V+18psSR"
    "x44d220DlB0BsSRJ3H333WotcOqhIR6P09HR0W2y1qWXXso//vGPXeqm1gtNnwThY4A5sizbk75TwxgNgP8n8JWSC8gJzAXGa7Lz"
    "9q2srGyrHr1nnHGGGieTJIloNIrdbqe1tZWTTz6ZH374YadAWHHie++9d1r81mQyUV1dzeLFi9MAzmg08tJLL3H88cfT3NysgnAs"
    "FiMrK4vnnntObQChZNMGg0E++eSTrYCxu+zonT0w2O12lRmlguQvYcCpTUQsFguNjY20t7fT3t5OR0dH2sPlcuF2u/F4PHi9Xnw+"
    "Hy6XC71eTygUUtmvIAjo9XqCwSBDhw5l8uTJaRnZqTHk1KYkitrRtSe0wWCgurqatrY29WcKi1yzZg2rVq3CYrGo8nNBQQHjxo1T"
    "DznK9SstLSU/P199beUgV1VVhSAILFiwIA3IBUHYKfm5q7oiiiKLFy/mk08+wel0quAdiUQoLCzkww8/ZOrUqerwCOVAoLQb7dOn"
    "zy8+UPUSEI4lfeVcWZadWutKTYL+X8E3LstyNvAxsL8GvjsGm7KyMlWC1Ol0NDc3c+yxx3L00Udz5plnkpGRoYKwxWLB7XZz8skn"
    "8/rrrzN69OjtytEKiBuNRg499FBWrlyZFr/95JNP1Pit0v3KYDDw4osvcv755/Pee++Rl5enlu0oUq3CdBKJBCaTiQULFnDzzTf/"
    "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"
    "Vha33noroVCIW265RQVxpfvS8OHDVWkyFouxadMmvvnmG0477bQdMlmtacIfC8Q7wy4V5WXUqFEce+yxvPnmm2RnZ2M0GtX7Z7PZ"
    "OOKII7arfKSyX0mSWLRo0VbsNxAIMHjwYM4+++ydZr9KedKsWbNobm4mNzdXZblKX+qXX35ZHVX5ww8/cPvtt/P555+rk7JMJhPT"
    "pk3jyy+/VJPIenmZktI7+nRZll2CIFwmy7JOluW4IAh7lPy0JzJgBXzvBi5jDxmsoMhk999/PzNnzqSgoIBwOIwkSXg8Hvr168eb"
    "b77J+++/z7Rp0zjssMMYPnw4AwcOxGg07jSryM3NVVv5KaDd0NDAunXruP766zn88MN55plnAFRHmkgkaG1txWw2q2w2VSrOycnh"
    "rrvu4vbbb1edruI0DzvsMJWxGwwGpk6dytixY3fagWq2+5uSOf70009zzz33MGjQICKRCO3t7VRXVzN69GgGDhy4U/c8NfabynCV"
    "rle7wn5TX2/p0qVqDFz5eSQS4e9//ztDhgxR68733Xdf3n33Xc4880w6OjrUdqp1dXU89thjO1150AtM6R19aXKecAzY407EexTt"
    "l2VZLwhCVJblPwMPsYeUGikn6nXr1jFu3Di1A5TS2H7UqFG89tprOJ3OtMYOqc0bdsUmTpyojgyMx+Nqd6rW1lYyMjLShiN4PB6y"
    "srKYPHkyV111FZ988gnXXXdd2gQiRXZsaWnh5ptv5rbbblM/f1NTE6+++ip9+/blwAMPpLCwUEOsXm7xeJyffvqJBQsWqJJxQUFB"
    "WknQjtjvsccemwa04XCY0tJStWNZKrjuzN46+eSTWbBggcqolfX51VdfUVpaqu4jRcGJx+Mcd9xxLF68GKfTSVNTE1dccQX33Xff"
    "npCQlWqKD75OEISHFB+9p3z5PanDky4JvmemgK+0h3x3AJ544gm8Xi85OTnE43EikQi5ubk8//zzOJ1ONdv4f2WOijMqKiriu+++"
    "U0E8EokAnbN1obN3s9vtJjMzk0svvZQrrriC/v37A3DppZciCALXXXedmt2sONbs7GymT5/O4YcfztixY4nH4+Tn53PttdemfYb/"
    "ZXiCZj1jHStKjtKbuTs2uiO2+tBDD22T/SqHxl0FQKvVmpZzoMSTt2zZslVlgCJbP/zwwxx++OEEAgHy8/O5+uqrd3iI6I2KZNIX"
    "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"
    "t7C1stq1/n8MeQfnUooYzgAAAABJRU5ErkJggg=="
)


with st.sidebar:
    st.image(io.BytesIO(base64.b64decode(AFRICAN_PLANTS_LOGO_B64)), width="stretch")

    st.markdown("---")
    st.markdown(f"**Logged in as:** `{st.session_state.get('auth_username', '')}`")
    if st.button("Log out", key="btn_logout"):
        _log_out()
        st.rerun()
    if not st.session_state.get("_history_persistence_available", False):
        st.caption(
            "History persistence: session only "
            f"({st.session_state.get('_history_persistence_reason', 'GitHub not configured')})"
        )

    st.markdown("---")
    st.header("Source Authority Status")
    st.markdown(f"**Natural Earth Admin-1:** {'✅ Loaded' if gdf_states is not None else '❌ Not Loaded'}")
    st.markdown(f"**Marine Regions EEZ:** {'✅ Loaded' if gdf_eez is not None else '❌ Not Loaded'}")
    st.markdown(
        f"**Getty TGN (Reconciliation API):** {'✅ Available' if tgn_is_up else '❌ Not available'}  \n"
        f"{TGN_SOURCE_AUTHORITY_DEFAULT['url']}"
    )
    if not tgn_is_up:
        st.caption(f"Reason: {tgn_status_reason}")
    st.markdown(
        f"**EPSG Geodetic Parameter Dataset (API):** {'✅ Available' if epsg_is_up else '❌ Not available'}  \n"
        f"{EPSG_SOURCE_AUTHORITY_DEFAULT['url']}"
    )
    if not epsg_is_up:
        st.caption(f"Reason: {epsg_status_reason}")
    st.markdown(
        f"**ISO 3166 Country Codes (pycountry):** {'✅ Available' if pycountry is not None else '❌ Not available'}  \n"
        f"{ISO_SOURCEAUTHORITY_URL_DEFAULT}"
    )
    if pycountry is None:
        st.caption("Reason: pycountry is not installed (`pip install pycountry`)")
    st.markdown("---")
    st.caption("If TGN is not available, tests that depend on TGN will return EXTERNAL_PREREQUISITES_NOT_MET.")
    st.caption("If EPSG is not available, tests that depend on EPSG will return EXTERNAL_PREREQUISITES_NOT_MET.")
    st.caption("If ISO 3166 Country Codes is not available, COUNTRYCODE_STANDARD will return EXTERNAL_PREREQUISITES_NOT_MET.")

# -------------------------------------------------
# Load data (ROBUST encoding-safe version)
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin the data quality validation.")
    st.stop()

# -------------------------------------------------
# SAFE to read the file from here onward
# -------------------------------------------------
raw_bytes = uploaded_file.getvalue()

version_id = _file_version_id(raw_bytes)
st.session_state["_active_upload_version_id"] = version_id

df = None
for enc in ("utf-8", "utf-16", "latin1", "cp1252"):
    try:
        text = raw_bytes.decode(enc)
        df = pd.read_csv(StringIO(text))
        break
    except UnicodeDecodeError:
        continue

if df is None:
    st.error("Could not decode the uploaded file. Please save it as UTF-8.")
    st.stop()

st.subheader("Original Data Preview")
st.dataframe(df.head())

REQUIRED_COLS = [
    "country", "countryCode", "decimalLatitude", "decimalLongitude",
    "stateProvince", "coordinateUncertaintyInMeters", "geodeticDatum",
    "minimumElevationInMeters", "maximumElevationInMeters",
]
for col in REQUIRED_COLS:
    if col not in df.columns:
        df[col] = np.nan


@st.cache_data
def run_all_tests(data_df, _states_gdf, _tgn_is_available: bool, _eez_gdf=None,
                   _epsg_is_available: bool = False):
    df_result = data_df.copy()

    df_result["VALIDATION_COUNTRY_NOT_EMPTY"] = test_country_not_empty(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("countryCode", pd.Series(dtype="object")),
    )

    # ---- Natural Earth-ish polygon tests (mock) ----
    if _states_gdf is not None:
        df_result["VALIDATION_COORDINATES_COUNTRYCODE_CONSISTENT"] = test_coordinates_countrycode_consistent(
            df_result["decimalLatitude"],
            df_result["decimalLongitude"],
            df_result.get("countryCode", pd.Series(dtype="object")),
            gdf=_states_gdf,
            eez_gdf=_eez_gdf,
        )
        df_result["VALIDATION_COORDINATES_STATEPROVINCE_CONSISTENT"] = test_coordinates_stateprovince_consistent(
            df_result["decimalLatitude"],
            df_result["decimalLongitude"],
            df_result.get("stateProvince", pd.Series(dtype="object")),
            gdf=_states_gdf,
        )
    else:
        df_result["VALIDATION_COORDINATES_COUNTRYCODE_CONSISTENT"] = "EXTERNAL_PREREQUISITES_NOT_MET"
        df_result["VALIDATION_COORDINATES_STATEPROVINCE_CONSISTENT"] = "EXTERNAL_PREREQUISITES_NOT_MET"

    # ---- Non-TGN tests ----
    df_result["VALIDATION_COORDINATES_NOTZERO"] = test_coordinates_not_zero(
        df_result["decimalLatitude"],
        df_result["decimalLongitude"],
    )

    df_result["VALIDATION_COORDINATEUNCERTAINTY_INRANGE"] = test_coordinate_uncertainty_inrange(
        df_result.get("coordinateUncertaintyInMeters", pd.Series(dtype="object"))
    )

    df_result["VALIDATION_COUNTRYCODE_NOTEMPTY"] = test_countrycode_not_empty(
        df_result.get("countryCode", pd.Series(dtype="object")),
    )

    df_result["VALIDATION_COUNTRYCODE_STANDARD"] = test_countrycode_standard(
        df_result.get("countryCode", pd.Series(dtype="object"))
    )

    # ---- TGN-backed (bdq:sourceAuthority default) ----
    df_result["VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT"] = test_country_countrycode_consistent(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("countryCode", pd.Series(dtype="object")),
        source_authority=TGN_SOURCE_AUTHORITY_DEFAULT,
        source_authority_available=_tgn_is_available,
    )

    df_result["VALIDATION_COUNTRY_STATEPROVINCE_UNAMBIGUOUS"] = test_country_stateprovince_unambiguous(
        df_result.get("country", pd.Series(dtype="object")),
        df_result.get("stateProvince", pd.Series(dtype="object")),
        source_authority=TGN_SOURCE_AUTHORITY_DEFAULT,
        source_authority_available=_tgn_is_available,
    )

    df_result["VALIDATION_STATEPROVINCE_FOUND"] = test_stateprovince_found(
        df_result.get("stateProvince", pd.Series(dtype="object")),
        source_authority=TGN_SOURCE_AUTHORITY_DEFAULT,
        source_authority_available=_tgn_is_available,
    )

    # ---- Remaining tests ----
    df_result["VALIDATION_COUNTRY_FOUND"] = test_country_found(
        df_result.get("country", pd.Series(dtype="object")),
        source_authority=TGN_SOURCE_AUTHORITY_DEFAULT,
        source_authority_available=_tgn_is_available,
    )
    df_result["VALIDATION_DECIMALLATITUDE_INRANGE"] = test_decimallatitude_inrange(df_result["decimalLatitude"])
    df_result["VALIDATION_DECIMALLATITUDE_NOTEMPTY"] = test_decimallatitude_notempty(df_result["decimalLatitude"])
    df_result["VALIDATION_DECIMALLONGITUDE_INRANGE"] = test_decimallongitude_inrange(df_result["decimalLongitude"])
    df_result["VALIDATION_DECIMALLONGITUDE_NOTEMPTY"] = test_decimallongitude_notempty(df_result["decimalLongitude"])
    df_result["VALIDATION_GEODETICDATUM_NOTEMPTY"] = test_geodeticdatum_notempty(
        df_result.get("geodeticDatum", pd.Series(dtype="object"))
    )

    # ---- EPSG-backed (bdq:sourceAuthority default = EPSG) ----
    df_result["VALIDATION_GEODETICDATUM_STANDARD"] = test_geodeticdatum_standard(
        df_result.get("geodeticDatum", pd.Series(dtype="object")),
        source_authority=EPSG_SOURCE_AUTHORITY_DEFAULT,
        source_authority_available=_epsg_is_available,
    )

    df_result["VALIDATION_MAXIMUMELEVATIONINMETERS_INRANGE"] = test_maximumelevation_inrange(
        df_result.get("maximumElevationInMeters", pd.Series(dtype="object"))
    )

    df_result["VALIDATION_MINIMUMELEVATIONINMETERS_INRANGE"] = test_minimumelevation_inrange(
        df_result.get("minimumElevationInMeters", pd.Series(dtype="object"))
    )

    df_result["VALIDATION_MINELEVATION_LESSTHAN_MAXELEVATION"] = test_minelevation_lessthan_maxelevation(
        df_result.get("minimumElevationInMeters", pd.Series(dtype="object")),
        df_result.get("maximumElevationInMeters", pd.Series(dtype="object")),
    )

    df_result["VALIDATION_LOCATION_NOTEMPTY"] = df_result.apply(test_location_notempty, axis=1)

    return df_result

with st.spinner("Running Data Quality Validation..."):
    validated_df = run_all_tests(
        df, gdf_states, tgn_is_up, _eez_gdf=gdf_eez, _epsg_is_available=epsg_is_up
    )

# ============================================================
# Results + Summary
# ============================================================
st.markdown("---")
st.subheader("Data Quality Validation Results")

dq_cols = [c for c in validated_df.columns if c.startswith("VALIDATION_")]
summary = pd.DataFrame(
    {
        "Test": dq_cols,
        "Compliant Count": [validated_df[c].eq("COMPLIANT").sum() for c in dq_cols],
        "Not Compliant Count": [validated_df[c].eq("NOT_COMPLIANT").sum() for c in dq_cols],
        "Potential Issue Count": [validated_df[c].eq("POTENTIAL_ISSUE").sum() for c in dq_cols],
        "Prerequisite Not Met": [
            validated_df[c].isin(["INTERNAL_PREREQUISITES_NOT_MET", "EXTERNAL_PREREQUISITES_NOT_MET"]).sum()
            for c in dq_cols
        ],
    }
).set_index("Test")

st.dataframe(summary)

st.markdown("##### Click a test name to see its BDQ Standard description")
for _test_col in dq_cols:
    with st.expander(_test_col):
        render_bdq_test_details(_test_col, summary.loc[_test_col].to_dict())

# ============================================================
# Register this upload "version" in history (date + status only)
# ============================================================
uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
dataset_status = _dataset_status_from_summary(summary)

# Avoid duplicating identical file uploads
already = any(h.get("version_id") == version_id for h in st.session_state["upload_history"]) or _is_history_paused_for(version_id)
if not already:
    st.session_state["upload_history"].append(
        {"uploaded_at": uploaded_at, "version_id": version_id, "status": dataset_status}
    )

# Full results
st.markdown("### Full DataFrame with Validation Columns")
st.dataframe(validated_df)

def _clean_text_cell(x):
    """Remove problematic control chars + normalize unicode safely."""
    if pd.isna(x):
        return x
    if not isinstance(x, str):
        return x
    # Normalize unicode (turn weird forms into standard forms)
    x = unicodedata.normalize("NFKC", x)
    # Remove C0/C1 control chars that often break encoders
    x = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", x)
    return x


def sanitize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].map(_clean_text_cell)
    return out


@st.cache_data
def convert_df_utf8(df_: pd.DataFrame) -> bytes:
    df_clean = sanitize_for_export(df_)
    # utf-8-sig adds BOM so Excel opens it correctly
    return df_clean.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download Validated Data (.csv)",
    data=convert_df_utf8(validated_df),
    file_name="validated_data_quality_results.csv",
    mime="text/csv",
)

# ============================================================
# Stacked bar chart (COMPLETE)
# ============================================================
st.markdown("---")
st.markdown("### Stacked bar chart of BDQ test outcomes")

stack_cols = ["Compliant Count", "Not Compliant Count", "Potential Issue Count", "Prerequisite Not Met"]
all_tests = summary.index.tolist()
default_tests = all_tests[: min(25, len(all_tests))]

selected_tests_for_bar = st.multiselect(
    "Select tests to include in the stacked bar chart:",
    options=all_tests,
    default=default_tests,
    key="bar_tests",
)

plot_summary = summary.loc[selected_tests_for_bar, stack_cols].copy()
plot_long = plot_summary.reset_index().melt(id_vars="Test", var_name="Outcome", value_name="Count")

sort_mode = st.selectbox(
    "Sort tests by:",
    options=["Original order", "Total issues (Not Compliant + Potential Issue)", "Total records (All outcomes)"],
    index=1,
    key="bar_sort",
)

if sort_mode == "Total issues (Not Compliant + Potential Issue)":
    totals = summary["Not Compliant Count"] + summary["Potential Issue Count"]
    order = totals.loc[selected_tests_for_bar].sort_values(ascending=False).index.tolist()
    plot_long["Test"] = pd.Categorical(plot_long["Test"], categories=order, ordered=True)

elif sort_mode == "Total records (All outcomes)":
    totals = summary[stack_cols].sum(axis=1)
    order = totals.loc[selected_tests_for_bar].sort_values(ascending=False).index.tolist()
    plot_long["Test"] = pd.Categorical(plot_long["Test"], categories=order, ordered=True)

fig_bar = px.bar(
    plot_long,
    x="Test",
    y="Count",
    color="Outcome",
    barmode="stack",
    title="BDQ Outcomes by Test (Stacked Bar)",
)

fig_bar.update_layout(
    xaxis_title="Test",
    yaxis_title="Count",
    xaxis_tickangle=-45,
    height=650,
    margin={"r": 0, "t": 60, "l": 0, "b": 0},
)

st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# Interactive "dataCleaning-style" dashboard
# Requirement:
#   - show ONLY upload date (dataset version) + status of each test
#   - also provide a map of points colored by selected test status
#
# Assumptions / required session_state structure:
#   st.session_state["upload_history"] is a list of dicts like:
#     {
#       "version_id": "v20260126_221530",
#       "uploaded_at": "2026-01-26 22:15:30",   # or datetime
#       "test_status": {                         # status per test for this upload version
#           "VALIDATION_COUNTRY_NOT_EMPTY": "COMPLIANT",
#           "VALIDATION_COUNTRY_COUNTRYCODE_CONSISTENT": "NOT_COMPLIANT",
#           ...
#       }
#     }
#
# If you do not already create this, you MUST add the "Append to upload history"
# block shown below right after you compute validated_df and summary.
# ============================================================

# ============================
# (A) Append to upload history (counts per test per upload)
# ============================

import uuid
import pandas as pd

SUSPECT_STATUSES_DEFAULT = [
    "NOT_COMPLIANT",
    "POTENTIAL_ISSUE",
    "INTERNAL_PREREQUISITES_NOT_MET",
    "EXTERNAL_PREREQUISIT"]

# Ensure history is loaded from GitHub at app start
load_persisted_history()

# ---- Make sure each "load" creates a new version_id (do NOT keep it forever in session_state) ----
# If you currently set current_version_id only once per session, you will NOT get a new version each load.
# So: generate a fresh id when validated_df is created (or when user clicks "Run / Validate")

if "current_version_id" not in st.session_state or st.session_state.get("_new_load_event", False):
    st.session_state["current_version_id"] = f"v{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    st.session_state["current_uploaded_at"] = utc_now_iso()
    st.session_state["_new_load_event"] = False  # you can set this True when a new file is chosen

dq_cols_now = [c for c in validated_df.columns if c.startswith("VALIDATION_") and not c.startswith("VALIDATION_COMMENT_")]

test_counts = {}
for test_col in dq_cols_now:
    s = validated_df[test_col].astype(str)
    test_counts[test_col] = {
        "suspect_records": int(s.isin(SUSPECT_STATUSES_DEFAULT).sum()),
        "total_records": int(len(s)),
        "COMPLIANT": int((s == "COMPLIANT").sum()),
        "NOT_COMPLIANT": int((s == "NOT_COMPLIANT").sum()),
        "POTENTIAL_ISSUE": int((s == "POTENTIAL_ISSUE").sum()),
        "INTERNAL_PREREQUISITES_NOT_MET": int((s == "INTERNAL_PREREQUISITES_NOT_MET").sum()),
        "EXTERNAL_PREREQUISITES_NOT_MET": int((s == "EXTERNAL_PREREQUISITES_NOT_MET").sum()),
    }

# Avoid duplicates on rerun (same version_id)
vid = st.session_state["current_version_id"]
already = any(isinstance(h, dict) and h.get("version_id") == vid for h in st.session_state.get("upload_history", [])) or _is_history_paused_for(vid)

if not already:
    record = {
        "version_id": vid,
        "uploaded_at": st.session_state["current_uploaded_at"],
        "collection": st.session_state.get("collection", "AFR"),
        "test_counts": test_counts,
        # optional metadata:
        "n_rows": int(validated_df.shape[0]),
        "n_cols": int(validated_df.shape[1]),
    }

    # Persist to GitHub AND keep session in sync
    append_persisted_history(record)
# -------------------------------------------------
# DATA INGESTION (THIS IS WHERE THE FLAG GOES)
# -------------------------------------------------
def read_csv_safe(uploaded_file):
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV with utf-8, latin1, or cp1252")

df_raw = read_csv_safe(uploaded_file)


    # -------------------------------------------------
    # VERSIONING (AFTER data exists)
    # -------------------------------------------------
if "validated_df" in locals():
    if (
            "current_version_id" not in st.session_state
            or st.session_state.get("_new_load_event", False)
    ):
        st.session_state["current_version_id"] = (
            f"v{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        st.session_state["current_uploaded_at"] = utc_now_iso()
        st.session_state["_new_load_event"] = False

# ============================================================
# Make the dashboard look like your screenshot:
#   - small-multiple “cards” (2 columns) with line+markers
#   - y-axis = Records (count of suspect records for that test in each upload)
#   - x-axis = upload date (each dataset version)
#
# IMPORTANT:
# Your current upload_history only stores per-test "dominant status".
# To draw curves like your screenshot, you must also store per-test COUNTS
# per upload (e.g., number of suspect records for that test).
#
# 1) ADD/REPLACE the "append to upload_history" block right after validated_df is created.
# 2) REPLACE your Section (B) with the dashboard code below.
# ============================================================


# ============================================================
# (1) ADD THIS RIGHT AFTER you compute `validated_df`
#     (this will store per-test counts per upload version)
# ============================================================

# Define which outcomes are considered "suspect" (like dataCleaning)
SUSPECT_STATUSES_DEFAULT = [
    "NOT_COMPLIANT",
    "POTENTIAL_ISSUE",
    "INTERNAL_PREREQUISITES_NOT_MET",
    "EXTERNAL_PREREQUISITES_NOT_MET",
]

# Make sure these exist
if "upload_history" not in st.session_state:
    st.session_state["upload_history"] = []

if "current_version_id" not in st.session_state:
    st.session_state["current_version_id"] = pd.Timestamp.now().strftime("v%Y%m%d_%H%M%S")

if "current_uploaded_at" not in st.session_state:
    st.session_state["current_uploaded_at"] = pd.Timestamp.now()

# Compute per-test suspect counts for THIS upload
dq_cols_now = [
    c for c in validated_df.columns
    if c.startswith("VALIDATION_") and not c.startswith("VALIDATION_COMMENT_")
]

test_counts = {}
for test_col in dq_cols_now:
    s = validated_df[test_col].astype(str)
    test_counts[test_col] = {
        "suspect_records": int(s.isin(SUSPECT_STATUSES_DEFAULT).sum()),
        "total_records": int(len(s)),
        # (optional) store breakdown if you want later
        "COMPLIANT": int((s == "COMPLIANT").sum()),
        "NOT_COMPLIANT": int((s == "NOT_COMPLIANT").sum()),
        "POTENTIAL_ISSUE": int((s == "POTENTIAL_ISSUE").sum()),
        "INTERNAL_PREREQUISITES_NOT_MET": int((s == "INTERNAL_PREREQUISITES_NOT_MET").sum()),
        "EXTERNAL_PREREQUISITES_NOT_MET": int((s == "EXTERNAL_PREREQUISITES_NOT_MET").sum()),
    }

# Avoid duplicates on rerun
already = any(h.get("version_id") == st.session_state["current_version_id"] for h in st.session_state["upload_history"]) or _is_history_paused_for(st.session_state["current_version_id"])
if not already:
    st.session_state["upload_history"].append({
        "version_id": st.session_state["current_version_id"],
        "uploaded_at": st.session_state["current_uploaded_at"],
        # optional label like in the screenshot
        "collection": st.session_state.get("collection", "AFR"),
        "test_counts": test_counts,  # <-- THIS enables the curves
    })


# ============================================================
# (2) REPLACE YOUR SECTION (B) WITH THIS DASHBOARD
# ============================================================

st.markdown("---")
# If you have a “collection” label, show it like the screenshot
collection_label = st.session_state.get("collection", "")
st.markdown(f"**collection: {collection_label}**")

st.header("dataCleaning — Suspect records over upload dates")
# ============================================================
# One-time migration: add test_counts to older history entries
# ============================================================
if "upload_history" in st.session_state and isinstance(st.session_state["upload_history"], list):
    migrated = 0

    for h in st.session_state["upload_history"]:
        # If entry already has test_counts, skip
        if isinstance(h, dict) and "test_counts" in h:
            continue

        # If entry has per-test status dict, convert it to count-like data
        # (Suspect = NOT_COMPLIANT, POTENTIAL_ISSUE, INTERNAL..., EXTERNAL...)
        ts = h.get("test_status", None) if isinstance(h, dict) else None
        if isinstance(ts, dict) and len(ts) > 0:
            suspect_set = {
                "NOT_COMPLIANT",
                "POTENTIAL_ISSUE",
                "INTERNAL_PREREQUISITES_NOT_MET",
                "EXTERNAL_PREREQUISITES_NOT_MET",
            }

            # We do NOT have record-level counts for old versions.
            # So we approximate: suspect_records = 1 if that upload's status is suspect, else 0.
            test_counts = {}
            for test, status in ts.items():
                status = str(status)
                test_counts[str(test)] = {
                    "suspect_records": 1 if status in suspect_set else 0,
                    "total_records": 1,
                    "COMPLIANT": 1 if status == "COMPLIANT" else 0,
                    "NOT_COMPLIANT": 1 if status == "NOT_COMPLIANT" else 0,
                    "POTENTIAL_ISSUE": 1 if status == "POTENTIAL_ISSUE" else 0,
                    "INTERNAL_PREREQUISITES_NOT_MET": 1 if status == "INTERNAL_PREREQUISITES_NOT_MET" else 0,
                    "EXTERNAL_PREREQUISITES_NOT_MET": 1 if status == "EXTERNAL_PREREQUISITES_NOT_MET" else 0,
                }

            h["test_counts"] = test_counts
            migrated += 1

    if migrated > 0:
        st.info(f"Migrated {migrated} historical uploads to include approximate test_counts.")



st.subheader("History Controls")
st.caption("Applies to the dataset/version history shown in this Panels section below.")
hc1, hc2 = st.columns(2)
with hc1:
    if st.button("🆕 Start New History Session", key="btn_start_new_history"):
        start_new_history_session()
        st.success("Started a new history session. Your next upload will be tracked as a fresh entry.")
        st.rerun()
with hc2:
    if st.session_state.get("_confirm_clear_history", False):
        st.warning("Delete ALL of your saved dataset/version history? This cannot be undone.")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Yes, clear it", key="btn_confirm_clear_history"):
                clear_user_history()
                st.session_state["_confirm_clear_history"] = False
                st.success("History cleared. It will start tracking again once you upload a new dataset.")
                st.rerun()
        with cc2:
            if st.button("Cancel", key="btn_cancel_clear_history"):
                st.session_state["_confirm_clear_history"] = False
                st.rerun()
    else:
        if st.button("🗑️ Clear My History", key="btn_clear_history"):
            st.session_state["_confirm_clear_history"] = True
            st.rerun()

history = st.session_state.get("upload_history", [])
history_df = pd.DataFrame(history)

if history_df.empty:
    st.info("No uploads recorded yet. Upload a dataset above to start tracking.")
    st.stop()

# Parse timestamps
history_df["uploaded_at"] = pd.to_datetime(history_df["uploaded_at"], errors="coerce")
history_df = history_df.dropna(subset=["uploaded_at"]).sort_values("uploaded_at")

# Expand test_counts to long format: (uploaded_at, version_id, test, suspect_records)
long_rows = []
for _, r in history_df.iterrows():
    tc = r.get("test_counts", None)

    # HARDENING
    if tc is None or isinstance(tc, float) or not isinstance(tc, dict):
        continue

    for test, metrics in tc.items():
        if not isinstance(metrics, dict):
            continue
        long_rows.append({
            "uploaded_at": r["uploaded_at"],
            "version_id": r.get("version_id", ""),
            "test": str(test),
            "suspect_records": int(metrics.get("suspect_records", 0)),
            "total_records": int(metrics.get("total_records", 0)),
        })

long_df = pd.DataFrame(long_rows)

if long_df.empty:
    st.warning(
        "Upload history exists, but contains no per-test count data. "
        "Confirm you added the `test_counts` storage block right after `validated_df`."
    )
    st.stop()

# ---------------------------
# Optional: compact table (like a version log)
# ---------------------------
with st.expander("Show upload versions (date + dataset version only)"):
    tbl = history_df[["uploaded_at", "version_id"]].copy()
    tbl.rename(columns={"uploaded_at": "Upload date", "version_id": "Dataset version"}, inplace=True)
    st.dataframe(tbl, use_container_width=True)

# ---------------------------
# Controls (which tests to show)
# ---------------------------
st.subheader("Panels")
all_tests = sorted(long_df["test"].dropna().unique().tolist())

# Default: first 6 tests
default_tests = all_tests[: min(6, len(all_tests))]

c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
with c1:
    panel_tests = st.multiselect(
        "Select tests to display as panels:",
        options=all_tests,
        default=default_tests,
        key="dc_panel_tests",
    )
with c2:
    show_cumulative = st.checkbox("Cumulative", value=False, key="dc_cumulative")
with c3:
    use_log = st.checkbox("Log y-axis", value=False, key="dc_logy")

if len(panel_tests) == 0:
    st.info("Select at least one test to display.")
    st.stop()

# ---------------------------
# Styling to resemble the screenshot “cards”
# ---------------------------
st.markdown(
    """
    <style>
      .dc-card {
        border: 1px solid #d9d9d9;
        background: #f6f6f6;
        padding: 8px 10px 0px 10px;
        border-radius: 2px;
        margin-bottom: 14px;
      }
      .dc-title {
        font-weight: 600;
        text-align: center;
        margin: 2px 0 6px 0;
        color: #3a3a3a;
        text-transform: lowercase;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper: build one panel chart (orange line + open circle markers)
# ---------------------------
def make_panel_chart(df_in: pd.DataFrame, test_name: str, cumulative: bool, logy: bool):
    d = df_in[df_in["test"] == test_name].sort_values("uploaded_at").copy()

    if d.empty:
        # Return an empty figure with a helpful message rather than erroring
        fig = go.Figure()
        fig.update_layout(
            height=270,
            margin=dict(l=40, r=10, t=10, b=45),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f2efee",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No data for this test.", x=0.5, y=0.5, showarrow=False)],
        )
        return fig

    if cumulative:
        d["y"] = d["suspect_records"].cumsum()
    else:
        d["y"] = d["suspect_records"]

    line_color = "#d9792d"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["uploaded_at"],
            y=d["y"],
            mode="lines+markers",
            line=dict(width=3, color=line_color),
            marker=dict(size=7, symbol="circle-open", line=dict(width=2, color=line_color)),
            hovertemplate=(
                "Upload: %{x|%Y-%m-%d %H:%M}<br>"
                "Suspect records: %{y}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=270,
        margin=dict(l=40, r=10, t=10, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f2efee",
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#eadbd4",
            tickangle=-55,
            tickformat="%d-%m-%Y",
        ),
        yaxis=dict(
            title="Records",
            showgrid=True,
            gridcolor="#eadbd4",
            type="log" if logy else "linear",
        ),
        showlegend=False,
    )
    return fig

# ---------------------------
# Render as 2-column grid like screenshot
# ---------------------------
cols = st.columns(2, vertical_alignment="top")

for i, t in enumerate(panel_tests):
    title = t.replace("VALIDATION_", "").replace("_", " ").lower()

    with cols[i % 2]:
        st.markdown(
            f'<div class="dc-card"><div class="dc-title">{title}</div>',
            unsafe_allow_html=True
        )

        fig = make_panel_chart(long_df, t, cumulative=show_cumulative, logy=use_log)

        # ✅ UNIQUE KEY per panel, stable across reruns and UI toggles
        panel_key = f"dc_panel_plot_{t}_{i}_cum{int(show_cumulative)}_log{int(use_log)}"
        st.plotly_chart(fig, use_container_width=True, key=panel_key)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# (3) Map of points with different colours based on test status
# ============================================================
st.markdown("---")
st.header("Map — Points colored by selected test status")

dq_cols_for_map = [
    c for c in validated_df.columns
    if c.startswith("VALIDATION_") and not c.startswith("VALIDATION_COMMENT_")
]

m1, m2 = st.columns([1, 2], vertical_alignment="top")

with m1:
    selected_test_map = st.selectbox(
        "Select a test for map coloring:",
        options=dq_cols_for_map,
        index=0,
        key="dc_map_selected_test"
    )

    # Consistent status ordering (legend + filtering)
    status_order = [
        "COMPLIANT",
        "POTENTIAL_ISSUE",
        "NOT_COMPLIANT",
        "INTERNAL_PREREQUISITES_NOT_MET",
        "EXTERNAL_PREREQUISITES_NOT_MET",
        "NOT_ISSUE",
    ]

    present_statuses = validated_df[selected_test_map].astype(str).unique().tolist()
    map_statuses = [s for s in status_order if s in present_statuses]
    if len(map_statuses) == 0:
        map_statuses = sorted(present_statuses)

    selected_map_statuses = st.multiselect(
        "Filter statuses to show on map:",
        options=map_statuses,
        default=map_statuses,
        key="dc_map_statuses"
    )

    jitter = st.checkbox("Jitter overlapping points (recommended)", value=True, key="dc_map_jitter")
    jitter_m = st.slider("Jitter strength (meters)", 0, 300, 80, step=10, key="dc_map_jitter_m")

with m2:
    if not {"decimalLatitude", "decimalLongitude"}.issubset(validated_df.columns):
        st.error("Map requires columns: decimalLatitude and decimalLongitude.")
    else:
        map_df = validated_df.copy()

        # --- numeric conversion
        map_df["decimalLatitude"] = pd.to_numeric(map_df["decimalLatitude"], errors="coerce")
        map_df["decimalLongitude"] = pd.to_numeric(map_df["decimalLongitude"], errors="coerce")

        # --- drop missing and invalid ranges (prevents silent Plotly weirdness)
        map_df = map_df.dropna(subset=["decimalLatitude", "decimalLongitude"])
        map_df = map_df[
            map_df["decimalLatitude"].between(-90, 90) &
            map_df["decimalLongitude"].between(-180, 180)
        ]

        # --- apply status filtering
        if selected_map_statuses:
            map_df = map_df[map_df[selected_test_map].astype(str).isin(selected_map_statuses)]

        if map_df.empty:
            st.info("No points to display after filtering (or coordinates are missing/invalid).")
        else:
            # Optional jitter so coincident points become visible
            # Convert meters -> degrees roughly (good enough for visualization)
            if jitter and jitter_m > 0:
                rng = np.random.default_rng(42)
                lat_j = (rng.normal(0, 1, size=len(map_df)) * (jitter_m / 111_000.0))
                # longitude degrees shrink with latitude
                lon_scale = np.cos(np.deg2rad(map_df["decimalLatitude"].to_numpy()))
                lon_scale[lon_scale == 0] = 1e-6
                lon_j = (rng.normal(0, 1, size=len(map_df)) * (jitter_m / (111_000.0 * lon_scale)))

                map_df = map_df.copy()
                map_df["lat_plot"] = map_df["decimalLatitude"] + lat_j
                map_df["lon_plot"] = map_df["decimalLongitude"] + lon_j
                lat_col, lon_col = "lat_plot", "lon_plot"
            else:
                lat_col, lon_col = "decimalLatitude", "decimalLongitude"

            # Consistent discrete colors (your mapping)
            color_map = {
                "COMPLIANT": "#2ecc71",
                "NOT_COMPLIANT": "#e74c3c",
                "POTENTIAL_ISSUE": "#f1c40f",
                "INTERNAL_PREREQUISITES_NOT_MET": "#95a5a6",
                "EXTERNAL_PREREQUISITES_NOT_MET": "#34495e",
                "NOT_ISSUE": "#3498db",
            }

            # Optional comment column for hover (if it exists)
            test_short = selected_test_map.replace("VALIDATION_", "")
            comment_col = f"VALIDATION_COMMENT_{test_short}"

            hover_cols = [selected_test_map, "decimalLatitude", "decimalLongitude"]
            if "country" in map_df.columns:
                hover_cols.append("country")
            if "stateProvince" in map_df.columns:
                hover_cols.append("stateProvince")
            if comment_col in map_df.columns:
                hover_cols.append(comment_col)

            # Keep legend order stable
            map_df[selected_test_map] = pd.Categorical(
                map_df[selected_test_map].astype(str),
                categories=status_order,
                ordered=True
            )

            fig_map = px.scatter_mapbox(
                map_df,
                lat=lat_col,
                lon=lon_col,
                color=selected_test_map,
                color_discrete_map=color_map,
                hover_data=hover_cols,
                height=700,
                zoom=1,  # will be overridden by fitbounds
                title=f"Points colored by: {selected_test_map}",
                category_orders={selected_test_map: status_order},
            )

            # ✅ key part: automatically fit to your points
            fig_map.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(
                    # Fit to data bounds (best way to “show points appropriately”)
                    # Works with scatter_mapbox in plotly express.
                ),
                margin={"r": 0, "t": 55, "l": 0, "b": 0},
                legend_title_text="Status",
            )
            fig_map.update_traces(marker=dict(size=8, opacity=0.75))


            def center_zoom_from_points(lats, lons, padding=0.10):
                lats = np.asarray(lats, dtype=float)
                lons = np.asarray(lons, dtype=float)
                mask = np.isfinite(lats) & np.isfinite(lons)
                lats, lons = lats[mask], lons[mask]

                if len(lats) == 0:
                    return {"lat": 0, "lon": 0}, 1

                lat_min, lat_max = lats.min(), lats.max()
                lon_min, lon_max = lons.min(), lons.max()

                # Add padding
                lat_pad = (lat_max - lat_min) * padding if lat_max > lat_min else 0.01
                lon_pad = (lon_max - lon_min) * padding if lon_max > lon_min else 0.01
                lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad
                lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad

                center = {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}

                # Rough zoom heuristic (works well enough for dashboards)
                max_range = max(lat_max - lat_min, lon_max - lon_min)
                if max_range < 0.01:
                    zoom = 14
                elif max_range < 0.05:
                    zoom = 12
                elif max_range < 0.2:
                    zoom = 10
                elif max_range < 1:
                    zoom = 7
                elif max_range < 5:
                    zoom = 5
                elif max_range < 15:
                    zoom = 4
                else:
                    zoom = 2

                return center, zoom
            center, zoom = center_zoom_from_points(df["decimalLatitude"], df["decimalLongitude"])
            fig_map.update_layout(mapbox_center=center, mapbox_zoom=zoom)

            # ✅ stable unique key
            statuses_key_part = "_".join(selected_map_statuses) if selected_map_statuses else "ALL"
            map_key = f"dc_map_{selected_test_map}_{statuses_key_part}_jit{int(jitter)}_{jitter_m}"
            st.plotly_chart(fig_map, use_container_width=True, key=map_key)