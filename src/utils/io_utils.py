"""I/O utilities for reading and writing data files."""

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_parquet(df: pd.DataFrame, filepath: str | Path, description: str = "") -> None:
    """Save a DataFrame to parquet with logging.

    Args:
        df: DataFrame to save.
        filepath: Output file path.
        description: Human-readable description for the log message.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    desc = f" ({description})" if description else ""
    logger.info(f"Saved {len(df)} rows to {filepath}{desc}")


def load_parquet(filepath: str | Path, description: str = "") -> pd.DataFrame:
    """Load a parquet file with logging.

    Args:
        filepath: Input file path.
        description: Human-readable description for the log message.

    Returns:
        Loaded DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_parquet(filepath)
    desc = f" ({description})" if description else ""
    logger.info(f"Loaded {len(df)} rows from {filepath}{desc}")
    return df


def save_json(data: dict | list, filepath: str | Path) -> None:
    """Save data as JSON."""
    import json

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str | Path) -> dict | list:
    """Load a JSON file."""
    import json

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
