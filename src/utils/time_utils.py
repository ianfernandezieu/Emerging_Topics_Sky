"""Time and date utilities for the project."""

from datetime import datetime, timezone


def to_unix(dt: datetime) -> int:
    """Convert a datetime to Unix timestamp (seconds)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def from_unix(ts: int) -> datetime:
    """Convert a Unix timestamp to UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def date_range_days(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Generate a list of (start, end) date string pairs for each day.

    Args:
        start_date: Start date as 'YYYY-MM-DD'.
        end_date: End date as 'YYYY-MM-DD'.

    Returns:
        List of (day_start, day_end) tuples as 'YYYY-MM-DD' strings.
    """
    import pandas as pd

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return [(d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")) for d in dates]


def date_range_chunks(start_date: str, end_date: str, chunk_days: int = 7) -> list[tuple[str, str]]:
    """Generate date range chunks of N days for batched API calls.

    Args:
        start_date: Start date as 'YYYY-MM-DD'.
        end_date: End date as 'YYYY-MM-DD'.
        chunk_days: Number of days per chunk (default 7 for OpenSky max).

    Returns:
        List of (chunk_start, chunk_end) tuples as 'YYYY-MM-DD' strings.
    """
    import pandas as pd

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []

    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + pd.Timedelta(days=1)

    return chunks
