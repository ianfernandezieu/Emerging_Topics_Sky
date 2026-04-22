"""Spanish public holiday fetcher using Nager.Date API.

Fetches national holidays for Spain. Used for calendar features.
No API key needed.

API docs: https://date.nager.at/swagger/index.html
"""

from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_START_DATE, DATA_END_DATE, get_path
from src.utils.logging_utils import get_logger
from src.utils.io_utils import save_json, load_json

logger = get_logger(__name__)

BASE_URL = "https://date.nager.at/api/v3/PublicHolidays"
COUNTRY_CODE = "ES"


def fetch_holidays(
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    country_code: str = COUNTRY_CODE,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch Spanish public holidays for the date range.

    Args:
        start_date: Start date 'YYYY-MM-DD'.
        end_date: End date 'YYYY-MM-DD'.
        country_code: Country code (default 'ES' for Spain).
        output_dir: Output directory. Defaults to config path.

    Returns:
        DataFrame with holiday dates and names.
    """
    if output_dir is None:
        output_dir = Path(get_path("raw_holidays"))
    else:
        output_dir = Path(output_dir)

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    all_holidays = []

    for year in range(start_year, end_year + 1):
        cache_path = output_dir / f"spain_holidays_{year}.json"

        if cache_path.exists():
            logger.info(f"Holidays {year} cached: {cache_path}")
            data = load_json(cache_path)
        else:
            url = f"{BASE_URL}/{year}/{country_code}"
            logger.info(f"Fetching holidays for {year}: {url}")

            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                save_json(data, cache_path)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch holidays for {year}: {e}")
                continue

        all_holidays.extend(data)

    if not all_holidays:
        logger.warning("No holidays fetched")
        return pd.DataFrame(columns=["date", "name", "local_name", "country_code", "types"])

    df = pd.DataFrame(all_holidays)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to date range
    df = df[
        (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    ].reset_index(drop=True)

    # Keep useful columns
    cols = ["date", "name", "localName", "countryCode", "types"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    logger.info(f"Holidays: {len(df)} holidays in range {start_date} to {end_date}")

    return df


def get_holiday_dates(
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
) -> set[str]:
    """Get a set of holiday date strings for quick lookup.

    Returns:
        Set of 'YYYY-MM-DD' strings.
    """
    df = fetch_holidays(start_date, end_date)
    return set(df["date"].dt.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    df = fetch_holidays()
    logger.info(f"Done. {len(df)} holidays")
    print(df.to_string(index=False))
