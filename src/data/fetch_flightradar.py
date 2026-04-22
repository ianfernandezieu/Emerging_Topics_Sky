"""FlightRadarAPI live snapshot collector (OPTIONAL enrichment).

Uses the unofficial FlightRadarAPI educational wrapper to collect
live flight snapshots around the airport.

IMPORTANT:
- This is NOT the official Flightradar24 API
- This is for educational purposes only
- The project must work WITHOUT this module
- Controlled by FLIGHTRADAR_ENABLED env var

Docs: https://jeanextreme002.github.io/FlightRadarAPI/
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    AIRPORT_LAT,
    AIRPORT_LON,
    BOUNDING_BOX,
    FLIGHTRADAR_ENABLED,
    get_path,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _check_enabled() -> bool:
    """Check if FlightRadarAPI is enabled and importable."""
    if not FLIGHTRADAR_ENABLED:
        logger.info("FlightRadarAPI disabled via FLIGHTRADAR_ENABLED=false")
        return False

    try:
        import FlightRadar24  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "FlightRadarAPI not installed. Install with: pip install FlightRadarAPI\n"
            "This is optional - the project works without it."
        )
        return False


def fetch_live_snapshot(
    output_dir: str | Path | None = None,
) -> dict | None:
    """Collect a single live snapshot of flights near the airport.

    Returns:
        Dict with snapshot data, or None if disabled/failed.
    """
    if not _check_enabled():
        return None

    if output_dir is None:
        output_dir = Path(get_path("raw_flightradar"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from FlightRadar24 import FlightRadar24API

        fr = FlightRadar24API()

        # Get flights in bounding box
        bounds = f"{BOUNDING_BOX['north']},{BOUNDING_BOX['south']},{BOUNDING_BOX['west']},{BOUNDING_BOX['east']}"
        flights = fr.get_flights(bounds=bounds)

        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Extract flight data
        flight_data = []
        for flight in flights:
            try:
                flight_data.append({
                    "id": flight.id,
                    "callsign": getattr(flight, "callsign", None),
                    "latitude": flight.latitude,
                    "longitude": flight.longitude,
                    "altitude": getattr(flight, "altitude", None),
                    "ground_speed": getattr(flight, "ground_speed", None),
                    "heading": getattr(flight, "heading", None),
                    "origin_airport_iata": getattr(flight, "origin_airport_iata", None),
                    "destination_airport_iata": getattr(flight, "destination_airport_iata", None),
                    "airline_icao": getattr(flight, "airline_icao", None),
                })
            except Exception:
                continue

        snapshot = {
            "timestamp_utc": timestamp.isoformat(),
            "airport_lat": AIRPORT_LAT,
            "airport_lon": AIRPORT_LON,
            "bounding_box": BOUNDING_BOX,
            "flight_count": len(flight_data),
            "flights": flight_data,
        }

        # Save snapshot
        filepath = output_dir / f"snapshot_{timestamp_str}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)

        logger.info(f"Live snapshot: {len(flight_data)} flights near airport at {timestamp_str}")
        return snapshot

    except Exception as e:
        logger.error(f"FlightRadarAPI snapshot failed: {e}")
        logger.info("This is expected if the unofficial wrapper is unavailable. Project continues without it.")
        return None


def collect_multiple_snapshots(
    count: int = 5,
    interval_seconds: int = 60,
) -> list[dict]:
    """Collect multiple live snapshots with delay between each.

    Args:
        count: Number of snapshots to collect.
        interval_seconds: Seconds between snapshots.

    Returns:
        List of snapshot dicts.
    """
    import time

    if not _check_enabled():
        return []

    snapshots = []
    for i in range(count):
        logger.info(f"Snapshot {i+1}/{count}")
        snap = fetch_live_snapshot()
        if snap:
            snapshots.append(snap)
        if i < count - 1:
            time.sleep(interval_seconds)

    logger.info(f"Collected {len(snapshots)} snapshots")
    return snapshots


def load_snapshots() -> list[dict]:
    """Load all cached live snapshots.

    Returns:
        List of snapshot dicts sorted by timestamp.
    """
    snap_dir = Path(get_path("raw_flightradar"))
    if not snap_dir.exists():
        return []

    snapshots = []
    for filepath in sorted(snap_dir.glob("snapshot_*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            snapshots.append(json.load(f))

    logger.info(f"Loaded {len(snapshots)} cached snapshots")
    return snapshots


def fetch_airport_board(
    airport_code: str = "MAD",
    max_pages: int = 15,
    output_dir: str | Path | None = None,
) -> tuple:
    """Fetch the full airport flight board (arrivals + departures).

    Paginates through all available pages of the airport board.
    This provides real scheduled/actual flight data for the current ~36h window.

    Args:
        airport_code: IATA airport code.
        max_pages: Maximum pages to fetch.
        output_dir: Output directory for parquet files.

    Returns:
        Tuple of (arrivals_df, departures_df) or (None, None) on failure.
    """
    if not _check_enabled():
        return None, None

    if output_dir is None:
        output_dir = Path(get_path("raw_flightradar"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import time
        import pandas as pd
        from FlightRadar24 import FlightRadar24API

        fr = FlightRadar24API()
        all_arrivals = []
        all_departures = []

        logger.info(f"Fetching airport board for {airport_code} (up to {max_pages} pages)...")

        for page in range(1, max_pages + 1):
            details = fr.get_airport_details(airport_code, flight_limit=100, page=page)
            schedule = details.get("airport", {}).get("pluginData", {}).get("schedule", {})

            arr = schedule.get("arrivals", {}).get("data", [])
            dep = schedule.get("departures", {}).get("data", [])

            if not arr and not dep:
                logger.info(f"  Page {page}: empty - stopping")
                break

            for flight in arr:
                f = flight.get("flight", {})
                times = f.get("time", {})
                ident = f.get("identification", {})
                origin = f.get("airport", {}).get("origin", {})
                aircraft = f.get("aircraft", {})
                all_arrivals.append({
                    "flight_number": ident.get("number", {}).get("default", ""),
                    "callsign": ident.get("callsign", ""),
                    "origin_iata": origin.get("code", {}).get("iata", "") if origin else "",
                    "origin_icao": origin.get("code", {}).get("icao", "") if origin else "",
                    "origin_name": origin.get("name", "") if origin else "",
                    "scheduled_arrival": times.get("scheduled", {}).get("arrival"),
                    "estimated_arrival": times.get("estimated", {}).get("arrival"),
                    "real_arrival": times.get("real", {}).get("arrival"),
                    "scheduled_departure": times.get("scheduled", {}).get("departure"),
                    "real_departure": times.get("real", {}).get("departure"),
                    "status_text": f.get("status", {}).get("text", ""),
                    "aircraft_model": aircraft.get("model", {}).get("text", "") if aircraft else "",
                    "aircraft_reg": aircraft.get("registration", "") if aircraft else "",
                })

            for flight in dep:
                f = flight.get("flight", {})
                times = f.get("time", {})
                ident = f.get("identification", {})
                dest = f.get("airport", {}).get("destination", {})
                aircraft = f.get("aircraft", {})
                all_departures.append({
                    "flight_number": ident.get("number", {}).get("default", ""),
                    "callsign": ident.get("callsign", ""),
                    "dest_iata": dest.get("code", {}).get("iata", "") if dest else "",
                    "dest_icao": dest.get("code", {}).get("icao", "") if dest else "",
                    "dest_name": dest.get("name", "") if dest else "",
                    "scheduled_departure": times.get("scheduled", {}).get("departure"),
                    "estimated_departure": times.get("estimated", {}).get("departure"),
                    "real_departure": times.get("real", {}).get("departure"),
                    "scheduled_arrival": times.get("scheduled", {}).get("arrival"),
                    "real_arrival": times.get("real", {}).get("arrival"),
                    "status_text": f.get("status", {}).get("text", ""),
                    "aircraft_model": aircraft.get("model", {}).get("text", "") if aircraft else "",
                    "aircraft_reg": aircraft.get("registration", "") if aircraft else "",
                })

            logger.info(f"  Page {page}: +{len(arr)} arr, +{len(dep)} dep")
            if page < max_pages:
                time.sleep(0.8)

        arr_df = pd.DataFrame(all_arrivals) if all_arrivals else pd.DataFrame()
        dep_df = pd.DataFrame(all_departures) if all_departures else pd.DataFrame()

        # Save
        if not arr_df.empty:
            arr_df.to_parquet(output_dir / "fr24_arrivals_board.parquet", index=False)
        if not dep_df.empty:
            dep_df.to_parquet(output_dir / "fr24_departures_board.parquet", index=False)

        logger.info(f"Airport board: {len(arr_df)} arrivals, {len(dep_df)} departures")
        return arr_df, dep_df

    except Exception as e:
        logger.error(f"Airport board collection failed: {e}")
        return None, None


if __name__ == "__main__":
    # Collect airport board data
    arr_df, dep_df = fetch_airport_board()
    if arr_df is not None:
        logger.info(f"Board: {len(arr_df)} arrivals, {len(dep_df)} departures")

    # Collect live snapshot
    snap = fetch_live_snapshot()
    if snap:
        logger.info(f"Live snapshot: {snap['flight_count']} flights captured.")
    else:
        logger.info("No snapshot collected (disabled or unavailable).")
