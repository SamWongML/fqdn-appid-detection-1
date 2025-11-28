"""
Data Loading Module

Provides unified data loading from multiple sources:
- PostgreSQL databases (using ConnectorX for speed)
- CSV files (using Polars)

All data is returned as Polars DataFrames for efficient processing.
"""



import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import polars as pl

from src.config.settings import (
    CSVSettings,
    DatabaseSettings,
    DataSettings,
    get_settings,
)
from src.utils.helpers import set_seed, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_labeled(self) -> pl.DataFrame:
        """Load labeled data (appId != 0)."""
        pass

    @abstractmethod
    def load_unlabeled(self) -> pl.DataFrame:
        """Load unlabeled/orphan data (appId = 0)."""
        pass

    @abstractmethod
    def load_all(self) -> pl.DataFrame:
        """Load all data."""
        pass

    def split_data(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: str | None = None,
        seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split data into train/validation/test sets.

        Args:
            df: DataFrame to split
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
            stratify_column: Column for stratified splitting
            seed: Random seed

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        set_seed(seed)

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        if stratify_column and stratify_column in df.columns:
            # Stratified split
            return self._stratified_split(
                df, train_ratio, val_ratio, stratify_column, seed
            )
        else:
            # Random split
            return self._random_split(df, train_ratio, val_ratio, seed)

    def _random_split(
        self,
        df: pl.DataFrame,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Perform random train/val/test split."""
        n = len(df)

        # Shuffle
        df = df.sample(fraction=1.0, seed=seed, shuffle=True)

        # Calculate split indices
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = df.slice(0, train_end)
        val_df = df.slice(train_end, val_end - train_end)
        test_df = df.slice(val_end, n - val_end)

        logger.info(
            f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def _stratified_split(
        self,
        df: pl.DataFrame,
        train_ratio: float,
        val_ratio: float,
        stratify_column: str,
        seed: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Perform stratified train/val/test split."""
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Split each class
        for class_value in df[stratify_column].unique().to_list():
            class_df = df.filter(pl.col(stratify_column) == class_value)

            if len(class_df) < 3:
                # Too few samples, put all in training
                train_dfs.append(class_df)
                continue

            train, val, test = self._random_split(
                class_df, train_ratio, val_ratio, seed
            )

            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)

        train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
        val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame()
        test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame()

        # Shuffle again
        train_df = train_df.sample(fraction=1.0, seed=seed, shuffle=True)
        val_df = val_df.sample(fraction=1.0, seed=seed + 1, shuffle=True)
        test_df = test_df.sample(fraction=1.0, seed=seed + 2, shuffle=True)

        logger.info(
            f"Stratified split on '{stratify_column}': "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df


class PostgresDataLoader(DataLoader):
    """
    Data loader for PostgreSQL databases.

    Uses ConnectorX for fast data transfer to Polars DataFrames.
    Falls back to psycopg2 if ConnectorX is unavailable.
    """

    def __init__(
        self,
        settings: DatabaseSettings | None = None,
        target_column: str = "appid",
        orphan_value: int = 0,
    ):
        """
        Initialize PostgreSQL data loader.

        Args:
            settings: Database connection settings
            target_column: Name of target column
            orphan_value: Value indicating orphan records
        """
        if settings is None:
            settings = get_settings().data.postgres

        self.settings = settings
        self.target_column = target_column
        self.orphan_value = orphan_value

        # Build connection URI
        self.connection_uri = settings.connectorx_uri
        self.schema = settings.schema_name
        self.table = settings.table
        self.chunk_size = settings.chunk_size

    def _build_query(
        self,
        columns: list[str | None] = None,
        where_clause: str | None = None,
        limit: int | None = None,
    ) -> str:
        """
        Build SQL query.

        Args:
            columns: Columns to select (None for all)
            where_clause: WHERE clause condition
            limit: Maximum rows to return

        Returns:
            SQL query string
        """
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM {self.schema}.{self.table}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        return query

    def _execute_query(self, query: str) -> pl.DataFrame:
        """
        Execute query and return Polars DataFrame.

        Uses ConnectorX for fast transfer if available.

        Args:
            query: SQL query

        Returns:
            Polars DataFrame
        """
        try:
            import connectorx as cx

            logger.debug(f"Executing query with ConnectorX: {query[:100]}...")
            df = cx.read_sql(
                self.connection_uri,
                query,
                return_type="polars",
            )
            return df

        except ImportError:
            logger.warning("ConnectorX not available, using psycopg2")
            return self._execute_with_psycopg2(query)

    def _execute_with_psycopg2(self, query: str) -> pl.DataFrame:
        """Fallback execution using psycopg2."""
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=self.settings.host,
            port=self.settings.port,
            database=self.settings.database,
            user=self.settings.user,
            password=self.settings.password,
        )

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()

            # Convert to Polars
            if rows:
                df = pl.DataFrame(rows)
            else:
                df = pl.DataFrame()

            return df
        finally:
            conn.close()

    @timer("Loading labeled data from PostgreSQL")
    def load_labeled(
        self,
        columns: list[str | None] = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Load labeled data (records with valid appId).

        Args:
            columns: Specific columns to load
            limit: Maximum rows

        Returns:
            Polars DataFrame with labeled data
        """
        where = f"{self.target_column} != {self.orphan_value}"
        query = self._build_query(columns, where, limit)

        df = self._execute_query(query)
        logger.info(f"Loaded {len(df)} labeled records from PostgreSQL")

        return df

    @timer("Loading unlabeled data from PostgreSQL")
    def load_unlabeled(
        self,
        columns: list[str | None] = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Load unlabeled/orphan data (records with appId = 0).

        Args:
            columns: Specific columns to load
            limit: Maximum rows

        Returns:
            Polars DataFrame with orphan data
        """
        where = f"{self.target_column} = {self.orphan_value}"
        query = self._build_query(columns, where, limit)

        df = self._execute_query(query)
        logger.info(f"Loaded {len(df)} orphan records from PostgreSQL")

        return df

    @timer("Loading all data from PostgreSQL")
    def load_all(
        self,
        columns: list[str | None] = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """
        Load all data.

        Args:
            columns: Specific columns to load
            limit: Maximum rows

        Returns:
            Polars DataFrame with all data
        """
        query = self._build_query(columns, limit=limit)

        df = self._execute_query(query)
        logger.info(f"Loaded {len(df)} total records from PostgreSQL")

        return df

    def load_chunked(
        self,
        chunk_size: int | None = None,
        columns: list[str | None] = None,
        where_clause: str | None = None,
    ):
        """
        Load data in chunks (generator).

        Args:
            chunk_size: Rows per chunk
            columns: Columns to select
            where_clause: WHERE clause

        Yields:
            Polars DataFrames
        """
        chunk_size = chunk_size or self.chunk_size
        offset = 0

        while True:
            query = self._build_query(columns, where_clause)
            query += f" OFFSET {offset} LIMIT {chunk_size}"

            df = self._execute_query(query)

            if len(df) == 0:
                break

            yield df

            if len(df) < chunk_size:
                break

            offset += chunk_size

    def get_class_distribution(self) -> pl.DataFrame:
        """
        Get distribution of classes in the dataset.

        Returns:
            DataFrame with class counts
        """
        query = f"""
            SELECT {self.target_column}, COUNT(*) as count
            FROM {self.schema}.{self.table}
            GROUP BY {self.target_column}
            ORDER BY count DESC
        """
        return self._execute_query(query)


class CSVDataLoader(DataLoader):
    """
    Data loader for CSV files using Polars.

    Optimized for large files with lazy evaluation and streaming.
    """

    def __init__(
        self,
        settings: CSVSettings | None = None,
        labeled_path: str | Path | None = None,
        unlabeled_path: str | Path | None = None,
        target_column: str = "appid",
        orphan_value: int = 0,
    ):
        """
        Initialize CSV data loader.

        Args:
            settings: CSV reading settings
            labeled_path: Path to labeled data CSV (overrides settings)
            unlabeled_path: Path to unlabeled data CSV (overrides settings)
            target_column: Name of target column
            orphan_value: Value indicating orphan records
        """
        if settings is None:
            settings = get_settings().data.csv

        self.settings = settings
        self.target_column = target_column
        self.orphan_value = orphan_value

        # Set paths
        self.labeled_path = Path(labeled_path or settings.labeled_data)
        self.unlabeled_path = Path(unlabeled_path or settings.unlabeled_data)

        # Parse null values
        self.null_values = settings.null_values

    def _read_csv(
        self,
        path: Path,
        columns: list[str | None] = None,
        n_rows: int | None = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Read CSV file with Polars.

        Args:
            path: Path to CSV file
            columns: Columns to select
            n_rows: Number of rows to read
            lazy: Whether to return LazyFrame

        Returns:
            DataFrame or LazyFrame
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Build read arguments
        read_args = {
            "separator": self.settings.separator,
            "has_header": self.settings.has_header,
            "encoding": self.settings.encoding,
            "null_values": self.null_values,
            "ignore_errors": self.settings.ignore_errors,
            "n_rows": n_rows,
        }

        if lazy:
            df = pl.scan_csv(str(path), **read_args)
            if columns:
                df = df.select(columns)
        else:
            if columns:
                read_args["columns"] = columns
            df = pl.read_csv(str(path), **read_args)

        return df

    @timer("Loading labeled data from CSV")
    def load_labeled(
        self,
        columns: list[str | None] = None,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        """
        Load labeled data from CSV.

        If a separate labeled file exists, load it directly.
        Otherwise, filter from combined file.

        Args:
            columns: Specific columns to load
            n_rows: Maximum rows

        Returns:
            Polars DataFrame with labeled data
        """
        if self.labeled_path.exists():
            df = self._read_csv(self.labeled_path, columns, n_rows)
            # Filter out any orphans that might be in the file
            if self.target_column in df.columns:
                df = df.filter(pl.col(self.target_column) != self.orphan_value)
        else:
            # Try combined file approach
            df = self.load_all(columns, n_rows)
            df = df.filter(pl.col(self.target_column) != self.orphan_value)

        logger.info(f"Loaded {len(df)} labeled records from CSV")
        return df

    @timer("Loading unlabeled data from CSV")
    def load_unlabeled(
        self,
        columns: list[str | None] = None,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        """
        Load unlabeled/orphan data from CSV.

        Args:
            columns: Specific columns to load
            n_rows: Maximum rows

        Returns:
            Polars DataFrame with orphan data
        """
        if self.unlabeled_path.exists():
            df = self._read_csv(self.unlabeled_path, columns, n_rows)
        else:
            # Try loading from labeled file and filtering
            df = self.load_all(columns, n_rows)
            df = df.filter(pl.col(self.target_column) == self.orphan_value)

        logger.info(f"Loaded {len(df)} orphan records from CSV")
        return df

    @timer("Loading all data from CSV")
    def load_all(
        self,
        columns: list[str | None] = None,
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        """
        Load all data from CSV files.

        Args:
            columns: Specific columns to load
            n_rows: Maximum rows

        Returns:
            Polars DataFrame with all data
        """
        dfs = []

        if self.labeled_path.exists():
            df = self._read_csv(self.labeled_path, columns, n_rows)
            dfs.append(df)

        if self.unlabeled_path.exists():
            remaining_rows = n_rows - len(dfs[0]) if n_rows and dfs else n_rows
            df = self._read_csv(self.unlabeled_path, columns, remaining_rows)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No data files found at {self.labeled_path} or {self.unlabeled_path}"
            )

        result = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
        logger.info(f"Loaded {len(result)} total records from CSV")

        return result

    def load_lazy(self, path: Path | None = None) -> pl.LazyFrame:
        """
        Load data as LazyFrame for memory-efficient processing.

        Args:
            path: Path to CSV (uses labeled_path if None)

        Returns:
            Polars LazyFrame
        """
        path = path or self.labeled_path
        return self._read_csv(path, lazy=True)

    def get_schema(self, path: Path | None = None) -> dict[str, pl.DataType]:
        """
        Get schema of CSV file.

        Args:
            path: Path to CSV (uses labeled_path if None)

        Returns:
            Dictionary mapping column names to data types
        """
        path = path or self.labeled_path

        # Read a small sample to infer schema
        df = self._read_csv(path, n_rows=100)
        return dict(zip(df.columns, df.dtypes))

    def get_class_distribution(
        self,
        path: Path | None = None,
    ) -> pl.DataFrame:
        """
        Get distribution of classes.

        Args:
            path: Path to CSV (uses labeled_path if None)

        Returns:
            DataFrame with class counts
        """
        path = path or self.labeled_path

        df = self._read_csv(path, columns=[self.target_column])

        return df.group_by(self.target_column).count().sort("count", descending=True)


class CombinedDataLoader(DataLoader):
    """
    Combined data loader that can load from either PostgreSQL or CSV.

    Automatically selects the appropriate loader based on configuration.
    """

    def __init__(
        self,
        settings: DataSettings | None = None,
        source: Literal["postgres", "csv", "auto"] | None = "auto",
    ):
        """
        Initialize combined data loader.

        Args:
            settings: Data settings
            source: Data source to use
        """
        if settings is None:
            settings = get_settings().data

        self.settings = settings

        # Determine source
        if source == "auto":
            source = settings.primary_source

        self.source = source

        # Initialize appropriate loader
        if source == "postgres":
            self._loader = PostgresDataLoader(
                settings=settings.postgres,
                target_column=get_settings().target.column,
                orphan_value=get_settings().target.orphan_value,
            )
        else:
            self._loader = CSVDataLoader(
                settings=settings.csv,
                target_column=get_settings().target.column,
                orphan_value=get_settings().target.orphan_value,
            )

    def load_labeled(self, **kwargs) -> pl.DataFrame:
        """Load labeled data."""
        return self._loader.load_labeled(**kwargs)

    def load_unlabeled(self, **kwargs) -> pl.DataFrame:
        """Load unlabeled data."""
        return self._loader.load_unlabeled(**kwargs)

    def load_all(self, **kwargs) -> pl.DataFrame:
        """Load all data."""
        return self._loader.load_all(**kwargs)

    def get_class_distribution(self) -> pl.DataFrame:
        """Get class distribution."""
        return self._loader.get_class_distribution()


def create_data_loader(
    source: Literal["postgres", "csv", "auto"] | None = "auto",
    settings: DataSettings | None = None,
) -> DataLoader:
    """
    Factory function to create appropriate data loader.

    Args:
        source: Data source type
        settings: Data settings

    Returns:
        DataLoader instance
    """
    return CombinedDataLoader(settings, source)
