"""Airport metadata fetcher from OurAirports open data.

Downloads the airports.csv file and extracts metadata for our target airport.
Static file - only needs to be fetched once.

Source: https://ourairports.com/data/
"""

from pathlib import Path

import pandas as pd
import requests

from src.config import AIRPORT_ICAO, get_path
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_parquet

logger = get_logger(__name__)

AIRPORTS_CSV_URL = "https://ourairports.com/data/airports.csv"


def fetch_airport_metadata(
    airport_icao: str = AIRPORT_ICAO,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download OurAirports data and extract target airport metadata.

    Args:
        airport_icao: ICAO code of the target airport.
        output_dir: Output directory. Defaults to config path.

    Returns:
        DataFrame with airport metadata.
    """
    if output_dir is None:
        output_dir = Path(get_path("raw_airports"))
    else:
        output_dir = Path(output_dir)

    csv_path = output_dir / "airports.csv"
    parquet_path = output_dir / "airport_reference.parquet"

    # Check cache
    if parquet_path.exists():
        logger.info(f"Airport metadata cached: {parquet_path}")
        return pd.read_parquet(parquet_path)

    # Download CSV if needed
    if not csv_path.exists():
        logger.info(f"Downloading OurAirports data from {AIRPORTS_CSV_URL}")
        try:
            response = requests.get(AIRPORTS_CSV_URL, timeout=30)
            response.raise_for_status()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Saved airports.csv ({len(response.content)} bytes)")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download airport data: {e}")
            raise

    # Load and filter
    df_all = pd.read_csv(csv_path)
    logger.info(f"OurAirports: {len(df_all)} airports total")

    # Find target airport
    df_airport = df_all[df_all["ident"] == airport_icao].copy()

    if df_airport.empty:
        logger.error(f"Airport {airport_icao} not found in OurAirports data")
        raise ValueError(f"Airport {airport_icao} not found")

    # Keep useful columns
    cols = [
        "ident", "type", "name", "latitude_deg", "longitude_deg",
        "elevation_ft", "continent", "iso_country", "iso_region",
        "municipality", "iata_code",
    ]
    existing_cols = [c for c in cols if c in df_airport.columns]
    df_airport = df_airport[existing_cols].reset_index(drop=True)

    save_parquet(df_airport, parquet_path, f"airport metadata for {airport_icao}")

    logger.info(f"Airport: {df_airport.iloc[0]['name']} ({airport_icao})")
    return df_airport


def load_airport_metadata(airport_icao: str = AIRPORT_ICAO) -> pd.DataFrame:
    """Load cached airport metadata."""
    raw_dir = Path(get_path("raw_airports"))
    filepath = raw_dir / "airport_reference.parquet"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Airport metadata not found at {filepath}. Run fetch_airport_metadata() first."
        )

    return pd.read_parquet(filepath)


if __name__ == "__main__":
    df = fetch_airport_metadata()
    logger.info("Done.")
    print(df.to_string(index=False))
