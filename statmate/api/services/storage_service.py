"""Storage service for file operations.

Handles uploading, reading, and managing dataset and result files.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for managing file storage operations."""

    @staticmethod
    def generate_filename(original_filename: str) -> str:
        """Generate a unique filename for storage.

        Args:
            original_filename: Original filename from user

        Returns:
            Unique filename with timestamp
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        stem = Path(original_filename).stem
        return f'{stem}_{timestamp}.parquet'

    @staticmethod
    def save_upload(file: BinaryIO, filename: str) -> Path:
        """Save an uploaded file.

        Args:
            file: File object to save
            filename: Target filename

        Returns:
            Path to saved file
        """
        file_path = settings.get_upload_path(filename)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)
        logger.info(f'Saved uploaded file: {file_path}')
        return file_path

    @staticmethod
    def read_dataset(file_path: Path) -> pd.DataFrame:
        """Read a dataset file into a DataFrame.

        Args:
            file_path: Path to dataset file

        Returns:
            DataFrame containing the dataset

        Raises:
            ValueError: If file format is not supported
        """
        suffix = file_path.suffix.lower()

        if suffix == '.parquet':
            return pd.read_parquet(file_path)
        if suffix == '.csv':
            return pd.read_csv(file_path)
        if suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        msg = f'Unsupported file format: {suffix}'
        raise ValueError(msg)

    @staticmethod
    def save_dataset(df: pd.DataFrame, filename: str) -> Path:
        """Save a DataFrame to parquet format.

        Args:
            df: DataFrame to save
            filename: Target filename

        Returns:
            Path to saved file
        """
        file_path = settings.get_upload_path(filename)
        df.to_parquet(file_path, index=False)
        logger.info(f'Saved dataset: {file_path}')
        return file_path

    @staticmethod
    def delete_upload(filename: str) -> None:
        """Delete an uploaded file.

        Args:
            filename: Filename to delete
        """
        file_path = settings.get_upload_path(filename)
        if file_path.exists():
            file_path.unlink()
            logger.info(f'Deleted file: {file_path}')

    @staticmethod
    def delete_results(analysis_id: str) -> None:
        """Delete stored results for an analysis."""
        results_dir = settings.get_results_path(analysis_id)
        if results_dir.exists():
            shutil.rmtree(results_dir, ignore_errors=True)
            logger.info('Deleted results directory: %s', results_dir)

    @staticmethod
    def delete_log(analysis_id: str) -> None:
        """Delete stored log file for an analysis."""
        log_file = settings.get_log_path(analysis_id)
        if log_file.exists():
            log_file.unlink()
            logger.info('Deleted log file: %s', log_file)

    @staticmethod
    def save_results(analysis_id: str, results_data: dict) -> Path:
        """Save analysis results to JSON file.

        Args:
            analysis_id: UUID of the analysis
            results_data: Results dictionary to save

        Returns:
            Path to saved results file
        """
        results_dir = settings.get_results_path(analysis_id)
        results_file = results_dir / 'summary.json'

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f'Saved results: {results_file}')
        return results_file

    @staticmethod
    def read_results(analysis_id: str) -> dict | None:
        """Read analysis results from file.

        Args:
            analysis_id: UUID of the analysis

        Returns:
            Results dictionary or None if not found
        """
        results_dir = settings.get_results_path(analysis_id)
        results_file = results_dir / 'summary.json'

        if not results_file.exists():
            return None

        with open(results_file) as f:
            return json.load(f)

    @staticmethod
    def save_log(analysis_id: str, log_content: str) -> Path:
        """Save analysis execution log.

        Args:
            analysis_id: UUID of the analysis
            log_content: Log content to save

        Returns:
            Path to saved log file
        """
        log_file = settings.get_log_path(analysis_id)

        with open(log_file, 'w') as f:
            f.write(log_content)

        logger.info(f'Saved log: {log_file}')
        return log_file

    @staticmethod
    def read_log(analysis_id: str) -> str | None:
        """Read analysis execution log.

        Args:
            analysis_id: UUID of the analysis

        Returns:
            Log content or None if not found
        """
        log_file = settings.get_log_path(analysis_id)

        if not log_file.exists():
            return None

        with open(log_file) as f:
            return f.read()

    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes
        """
        return file_path.stat().st_size if file_path.exists() else 0
