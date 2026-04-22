"""OpenSky Network data fetcher for historical airport arrivals and departures.

This is the CORE historical data source for the project.
Fetches flight movements by airport ICAO code in 7-day chunks
(max allowed by the OpenSky REST API).

TWO MODES:
1. Historical flights (requires OpenSky account - free signup at opensky-network.org)
2. State vectors fallback (works anonymously - polls live aircraft positions)

API docs: https://openskynetwork.github.io/opensky-api/rest.html
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    AIRPORT_ICAO,
    AIRPORT_LAT,
    AIRPORT_LON,
    BOUNDING_BOX,
    DATA_END_DATE,
    DATA_START_DATE,
    OPENSKY_PASSWORD,
    OPENSKY_USERNAME,
    get_path,
)
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_parquet
from src.utils.time_utils import to_unix

logger = get_logger(__name__)

BASE_URL = "https://opensky-network.org/api"
MAX_CHUNK_DAYS = 7  # OpenSky max query window
REQUEST_DELAY = 1.5  # seconds between requests to respect rate limits
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


def _get_auth() -> tuple[str, str] | None:
    """Return HTTP basic auth tuple if credentials are configured."""
    if OPENSKY_USERNAME and OPENSKY_PASSWORD:
        return (OPENSKY_USERNAME, OPENSKY_PASSWORD)
    logger.warning("No OpenSky credentials configured. Using anonymous access (lower rate limits).")
    return None


def _fetch_flights(
    endpoint: str,
    airport: str,
    begin_ts: int,
    end_ts: int,
) -> list[dict] | None:
    """Fetch flights from an OpenSky endpoint with retry logic.

    Args:
        endpoint: 'arrival' or 'departure'.
        airport: ICAO airport code.
        begin_ts: Start Unix timestamp.
        end_ts: End Unix timestamp.

    Returns:
        List of flight dicts, or None on failure.
    """
    url = f"{BASE_URL}/flights/{endpoint}"
    params = {"airport": airport, "begin": begin_ts, "end": end_ts}
    auth = _get_auth()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, auth=auth, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data if data else []

            if response.status_code == 404:
                logger.info(f"No {endpoint} data for {airport} in this window (404)")
                return []

            if response.status_code == 429:
                wait = RETRY_DELAY * attempt
                logger.warning(f"Rate limited (429). Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            logger.warning(
                f"OpenSky {endpoint} returned {response.status_code}: "
                f"{response.text[:200]} (attempt {attempt}/{MAX_RETRIES})"
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e} (attempt {attempt}/{MAX_RETRIES})")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    logger.error(f"Failed to fetch {endpoint} for {airport} after {MAX_RETRIES} attempts")
    return None


def fetch_arrivals(
    airport: str = AIRPORT_ICAO,
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    output_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Fetch historical arrivals in weekly chunks and save as daily parquet files.

    Args:
        airport: ICAO airport code (default: LEMD).
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
        output_dir: Directory to save parquet files. Defaults to config path.
        skip_existing: Skip chunks whose output file already exists.

    Returns:
        Combined DataFrame of all arrivals.
    """
    return _fetch_by_type("arrival", airport, start_date, end_date, output_dir, skip_existing)


def fetch_departures(
    airport: str = AIRPORT_ICAO,
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    output_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Fetch historical departures in weekly chunks and save as daily parquet files.

    Args:
        airport: ICAO airport code (default: LEMD).
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
        output_dir: Directory to save parquet files. Defaults to config path.
        skip_existing: Skip chunks whose output file already exists.

    Returns:
        Combined DataFrame of all departures.
    """
    return _fetch_by_type("departure", airport, start_date, end_date, output_dir, skip_existing)


def _fetch_by_type(
    flight_type: str,
    airport: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path | None,
    skip_existing: bool,
) -> pd.DataFrame:
    """Internal: fetch arrivals or departures in weekly chunks."""
    if output_dir is None:
        output_dir = Path(get_path("raw_opensky"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Build weekly chunks
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=MAX_CHUNK_DAYS - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)

    logger.info(
        f"Fetching {flight_type}s for {airport}: {start_date} to {end_date} "
        f"({len(chunks)} chunks)"
    )

    all_frames = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_label = f"{chunk_start.strftime('%Y_%m_%d')}_to_{chunk_end.strftime('%Y_%m_%d')}"
        filename = f"{flight_type}s_{chunk_label}.parquet"
        filepath = output_dir / filename

        # Skip if already fetched
        if skip_existing and filepath.exists():
            logger.info(f"  [{i+1}/{len(chunks)}] Cached: {filename}")
            df_chunk = pd.read_parquet(filepath)
            all_frames.append(df_chunk)
            continue

        # Fetch from API
        begin_ts = to_unix(datetime(chunk_start.year, chunk_start.month, chunk_start.day, tzinfo=timezone.utc))
        end_ts = to_unix(
            datetime(chunk_end.year, chunk_end.month, chunk_end.day, 23, 59, 59, tzinfo=timezone.utc)
        )

        data = _fetch_flights(flight_type, airport, begin_ts, end_ts)

        if data is None:
            logger.warning(f"  [{i+1}/{len(chunks)}] FAILED: {chunk_label}")
            continue

        if len(data) == 0:
            logger.info(f"  [{i+1}/{len(chunks)}] Empty: {chunk_label}")
            # Save empty frame to mark as fetched
            df_chunk = pd.DataFrame()
            save_parquet(df_chunk, filepath, f"empty {flight_type}s {chunk_label}")
            continue

        df_chunk = pd.DataFrame(data)
        save_parquet(df_chunk, filepath, f"{flight_type}s {chunk_label}")
        all_frames.append(df_chunk)

        logger.info(f"  [{i+1}/{len(chunks)}] Fetched {len(df_chunk)} {flight_type}s: {chunk_label}")

        # Rate limit delay
        if i < len(chunks) - 1:
            time.sleep(REQUEST_DELAY)

    if not all_frames:
        logger.warning(f"No {flight_type} data collected for {airport}")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total {flight_type}s collected: {len(combined)}")

    # Save combined file
    combined_path = output_dir / f"all_{flight_type}s_{airport}.parquet"
    save_parquet(combined, combined_path, f"all {flight_type}s combined")

    return combined


def load_all_flights(airport: str = AIRPORT_ICAO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all cached arrivals and departures.

    Returns:
        Tuple of (arrivals_df, departures_df).
    """
    raw_dir = Path(get_path("raw_opensky"))

    arr_path = raw_dir / f"all_arrivals_{airport}.parquet"
    dep_path = raw_dir / f"all_departures_{airport}.parquet"

    arrivals = pd.read_parquet(arr_path) if arr_path.exists() else pd.DataFrame()
    departures = pd.read_parquet(dep_path) if dep_path.exists() else pd.DataFrame()

    logger.info(f"Loaded {len(arrivals)} arrivals, {len(departures)} departures for {airport}")
    return arrivals, departures


def fetch_state_vectors(
    output_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Fetch current state vectors (live aircraft positions) near the airport.

    This works WITHOUT authentication and can be used as a fallback
    when the historical flights endpoint returns 403.

    Returns:
        DataFrame with current aircraft positions, or None on failure.
    """
    if output_dir is None:
        output_dir = Path(get_path("raw_opensky"))
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/states/all"
    params = {
        "lamin": BOUNDING_BOX["south"],
        "lomin": BOUNDING_BOX["west"],
        "lamax": BOUNDING_BOX["north"],
        "lomax": BOUNDING_BOX["east"],
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"State vectors request failed: {e}")
        return None

    data = response.json()
    states = data.get("states", [])

    if not states:
        logger.info("No aircraft in bounding box")
        return pd.DataFrame()

    columns = [
        "icao24", "callsign", "origin_country", "time_position", "last_contact",
        "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
        "true_track", "vertical_rate", "sensors", "geo_altitude",
        "squawk", "spi", "position_source",
    ]

    df = pd.DataFrame(states, columns=columns[: len(states[0])])
    df["snapshot_time"] = pd.Timestamp.now(tz="UTC")

    # Save snapshot
    ts_str = df["snapshot_time"].iloc[0].strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"state_vectors_{ts_str}.parquet"
    save_parquet(df, filepath, f"state vectors snapshot ({len(df)} aircraft)")

    logger.info(f"State vectors: {len(df)} aircraft near {AIRPORT_ICAO}")
    return df


def check_auth_status() -> dict:
    """Check OpenSky authentication and endpoint availability.

    Returns:
        Dict with status of each endpoint.
    """
    status = {}
    auth = _get_auth()

    # Test state vectors (anonymous)
    try:
        r = requests.get(f"{BASE_URL}/states/all", params={
            "lamin": BOUNDING_BOX["south"], "lomin": BOUNDING_BOX["west"],
            "lamax": BOUNDING_BOX["north"], "lomax": BOUNDING_BOX["east"],
        }, timeout=15)
        status["state_vectors"] = {"code": r.status_code, "works": r.status_code == 200}
    except Exception as e:
        status["state_vectors"] = {"code": None, "works": False, "error": str(e)}

    # Test flights endpoint
    test_begin = int(pd.Timestamp("2025-03-01", tz="UTC").timestamp())
    test_end = int(pd.Timestamp("2025-03-02", tz="UTC").timestamp())
    try:
        r = requests.get(f"{BASE_URL}/flights/arrival", params={
            "airport": AIRPORT_ICAO, "begin": test_begin, "end": test_end
        }, auth=auth, timeout=15)
        status["flights_arrival"] = {
            "code": r.status_code,
            "works": r.status_code == 200,
            "auth_used": auth is not None,
        }
    except Exception as e:
        status["flights_arrival"] = {"code": None, "works": False, "error": str(e)}

    return status


if __name__ == "__main__":
    logger.info("Checking OpenSky API access...")
    status = check_auth_status()
    for endpoint, info in status.items():
        logger.info(f"  {endpoint}: {info}")

    if status.get("flights_arrival", {}).get("works"):
        logger.info("\nFlights endpoint accessible. Starting historical collection...")
        arr = fetch_arrivals()
        dep = fetch_departures()
        logger.info(f"Done. Arrivals: {len(arr)}, Departures: {len(dep)}")
    else:
        logger.warning(
            "\nFlights endpoint returned 403. You need OpenSky credentials.\n"
            "1. Sign up at https://opensky-network.org/\n"
            "2. Add OPENSKY_USERNAME and OPENSKY_PASSWORD to your .env file\n"
            "3. Re-run this script\n"
            "\nMeanwhile, collecting a state vectors snapshot..."
        )
        sv = fetch_state_vectors()
        if sv is not None:
            logger.info(f"State vectors snapshot: {len(sv)} aircraft")
