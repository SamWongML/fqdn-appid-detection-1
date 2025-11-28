"""
Feature Engineering Module

Comprehensive feature extraction for FQDN classification including:
- FQDN structural features
- Text vectorization (TF-IDF, n-grams)
- Categorical encoding
- Domain pattern extraction
"""



import re
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.helpers import timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FQDNFeatureExtractor:
    """
    Extract structural and pattern features from FQDN strings.
    """

    ENV_PATTERNS = {
        "is_dev": r"(dev|development)",
        "is_staging": r"(stg|stage|staging)",
        "is_uat": r"(uat|qa|test)",
        "is_prod": r"(prod|production|prd)",
        "is_dr": r"(dr|disaster|recovery)",
    }

    SERVICE_PATTERNS = {
        "is_api": r"(api|rest|graphql|gateway)",
        "is_web": r"(www|web|portal|site)",
        "is_cdn": r"(cdn|static|assets|media)",
        "is_email": r"(mail|smtp|imap|pop3|mx)",
        "is_database": r"(db|database|mysql|postgres|mongo|redis)",
        "is_admin": r"(admin|manage|console|dashboard)",
        "is_auth": r"(auth|login|sso|identity|oauth)",
        "is_internal": r"(internal|intranet|corp|private)",
    }

    REGION_PATTERNS = {
        "region_uk": r"(uk|gb|london|ldn)",
        "region_us": r"(us|usa|america|nyc|sfo|chi)",
        "region_hk": r"(hk|hongkong|hong-kong)",
        "region_sg": r"(sg|singapore)",
        "region_eu": r"(eu|europe|fra|ams|dub)",
        "region_apac": r"(apac|asia|sydney|tokyo|mumbai)",
    }

    CLOUD_PATTERNS = {
        "is_aws": r"(amazonaws|awsglobalaccelerator|cloudfront)",
        "is_azure": r"(azure|azurewebsites|trafficmanager\.net)",
        "is_gcp": r"(googleapis|appspot|cloudfunctions)",
        "is_cloudflare": r"(cloudflare|cdn-cgi)",
    }

    def __init__(self):
        self._tld_extractor = None
        try:
            import tldextract

            self._tld_extractor = tldextract.TLDExtract(cache_dir=".tld_cache")
        except ImportError:
            logger.warning("tldextract not available, using basic domain parsing")

    def extract_features(
        self, df: pl.DataFrame, fqdn_column: str = "fqdn"
    ) -> pl.DataFrame:
        if fqdn_column not in df.columns:
            return df

        logger.info(f"Extracting FQDN features from {len(df)} records...")
        df = self._extract_structural_features(df, fqdn_column)
        df = self._extract_domain_components(df, fqdn_column)
        df = self._extract_pattern_features(df, fqdn_column)
        return df

    def _extract_structural_features(
        self, df: pl.DataFrame, fqdn_column: str
    ) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.col(fqdn_column).str.len_chars().alias("fqdn_length"),
                pl.col(fqdn_column).str.count_matches(r"\.").alias("fqdn_num_dots"),
                pl.col(fqdn_column).str.count_matches(r"-").alias("fqdn_num_hyphens"),
                pl.col(fqdn_column).str.count_matches(r"\d").alias("fqdn_num_digits"),
                pl.col(fqdn_column)
                .str.starts_with("www.")
                .cast(pl.Int8)
                .alias("fqdn_has_www"),
                (pl.col(fqdn_column).str.count_matches(r"\.") + 1).alias(
                    "fqdn_num_labels"
                ),
                pl.col(fqdn_column)
                .str.split(".")
                .list.eval(pl.element().str.len_chars())
                .list.max()
                .alias("fqdn_max_label_length"),
                pl.col(fqdn_column)
                .str.split(".")
                .list.eval(pl.element().str.len_chars())
                .list.mean()
                .alias("fqdn_avg_label_length"),
                pl.col(fqdn_column)
                .str.contains(r"\.\d+\.")
                .cast(pl.Int8)
                .alias("fqdn_has_numeric_label"),
                pl.col(fqdn_column)
                .str.split(".")
                .list.first()
                .alias("fqdn_first_label"),
            ]
        )

    def _extract_domain_components(
        self, df: pl.DataFrame, fqdn_column: str
    ) -> pl.DataFrame:
        if self._tld_extractor:
            fqdns = df[fqdn_column].to_list()
            tlds, domains, subdomains = [], [], []
            for fqdn in fqdns:
                if fqdn:
                    try:
                        extracted = self._tld_extractor(fqdn)
                        tlds.append(extracted.suffix or "")
                        domains.append(extracted.domain or "")
                        subdomains.append(extracted.subdomain or "")
                    except Exception:
                        tlds.append("")
                        domains.append("")
                        subdomains.append("")
                else:
                    tlds.append("")
                    domains.append("")
                    subdomains.append("")

            df = df.with_columns(
                [
                    pl.Series("fqdn_tld", tlds),
                    pl.Series("fqdn_domain", domains),
                    pl.Series("fqdn_subdomain", subdomains),
                    pl.Series(
                        "fqdn_subdomain_depth",
                        [len(s.split(".")) if s else 0 for s in subdomains],
                    ),
                ]
            )
        else:
            df = df.with_columns(
                [
                    pl.col(fqdn_column).str.split(".").list.last().alias("fqdn_tld"),
                    pl.col(fqdn_column)
                    .str.split(".")
                    .list.get(-2, null_on_oob=True)
                    .alias("fqdn_domain"),
                ]
            )
        return df

    def _extract_pattern_features(
        self, df: pl.DataFrame, fqdn_column: str
    ) -> pl.DataFrame:
        all_patterns = {
            **self.ENV_PATTERNS,
            **self.SERVICE_PATTERNS,
            **self.REGION_PATTERNS,
            **self.CLOUD_PATTERNS,
        }
        for name, pattern in all_patterns.items():
            df = df.with_columns(
                pl.col(fqdn_column)
                .str.contains(f"(?i){pattern}")
                .cast(pl.Int8)
                .alias(f"fqdn_{name}")
            )
        return df


class TextVectorizer:
    """Text vectorization using TF-IDF."""

    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: tuple[int, int] = (1, 3),
        analyzer: str = "char_wb",
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.sublinear_tf = sublinear_tf
        self._vectorizer: TfidfVectorizer | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        return text.replace(".", " ").replace("-", " ").lower()

    def fit(self, texts: list[str], column_name: str = "text") -> "TextVectorizer":
        processed = [self._preprocess_text(t) for t in texts]
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            sublinear_tf=self.sublinear_tf,
        )
        self._vectorizer.fit(processed)
        vocab = self._vectorizer.get_feature_names_out()
        self._feature_names = [f"{column_name}_tfidf_{v}" for v in vocab]
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted")
        processed = [self._preprocess_text(t) for t in texts]
        return self._vectorizer.transform(processed).toarray()

    def fit_transform(self, texts: list[str], column_name: str = "text") -> np.ndarray:
        self.fit(texts, column_name)
        return self.transform(texts)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class CategoricalEncoder:
    """Categorical feature encoder with multiple strategies."""

    def __init__(
        self,
        strategy: str = "label",
        handle_unknown: str = "use_default",
        min_frequency: int | None = None,
        max_categories: int | None = None,
    ):
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self._category_maps: dict[str, dict[Any, int]] = {}
        self._target_means: dict[str, dict[Any, float]] = {}
        self._frequencies: dict[str, dict[Any, float]] = {}
        self._default_values: dict[str, Any] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    def fit(
        self, df: pl.DataFrame, columns: list[str], target: pl.Series | None = None
    ) -> "CategoricalEncoder":
        for col in columns:
            if col not in df.columns:
                continue
            values = df[col].to_list()

            if self.strategy == "label":
                value_counts = {}
                for v in values:
                    if v is not None:
                        value_counts[v] = value_counts.get(v, 0) + 1
                if self.min_frequency:
                    value_counts = {
                        k: v for k, v in value_counts.items() if v >= self.min_frequency
                    }
                if self.max_categories and len(value_counts) > self.max_categories:
                    sorted_items = sorted(value_counts.items(), key=lambda x: -x[1])
                    value_counts = dict(sorted_items[: self.max_categories])
                categories = sorted(value_counts.keys())
                self._category_maps[col] = {
                    cat: idx for idx, cat in enumerate(categories)
                }
                self._default_values[col] = len(categories)
                self._feature_names.append(col)

            elif self.strategy == "target" and target is not None:
                category_targets: dict[Any, list[float]] = {}
                for v, t in zip(values, target.to_list()):
                    if v is not None:
                        category_targets.setdefault(v, []).append(float(t))
                global_mean = np.mean(target.to_list())
                smoothing = 10
                self._target_means[col] = {}
                for cat, targets in category_targets.items():
                    n = len(targets)
                    cat_mean = np.mean(targets)
                    smoothed = (cat_mean * n + global_mean * smoothing) / (
                        n + smoothing
                    )
                    self._target_means[col][cat] = smoothed
                self._default_values[col] = global_mean
                self._feature_names.append(f"{col}_target_encoded")

            elif self.strategy == "frequency":
                total = len(values)
                value_counts = {}
                for v in values:
                    if v is not None:
                        value_counts[v] = value_counts.get(v, 0) + 1
                self._frequencies[col] = {k: v / total for k, v in value_counts.items()}
                self._default_values[col] = 0.0
                self._feature_names.append(f"{col}_freq_encoded")

        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame, columns: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted")
        results = []
        for col in columns:
            if col not in df.columns:
                continue
            values = df[col].to_list()
            if self.strategy == "label":
                mapping = self._category_maps.get(col, {})
                default = self._default_values.get(col, -1)
                encoded = np.array([mapping.get(v, default) for v in values])
            elif self.strategy == "target":
                means = self._target_means.get(col, {})
                default = self._default_values.get(col, 0.0)
                encoded = np.array([means.get(v, default) for v in values])
            elif self.strategy == "frequency":
                freqs = self._frequencies.get(col, {})
                default = self._default_values.get(col, 0.0)
                encoded = np.array([freqs.get(v, default) for v in values])
            else:
                continue
            results.append(encoded.reshape(-1, 1))

        return np.hstack(results) if results else np.array([]).reshape(len(df), 0)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class FeatureEngineer:
    """Main feature engineering class combining all extractors."""

    def __init__(
        self,
        fqdn_column: str = "fqdn",
        text_columns: list[str | None] = None,
        categorical_columns: list[str | None] = None,
        numerical_columns: list[str | None] = None,
        tfidf_max_features: int = 500,
        categorical_strategy: str = "label",
    ):
        self.fqdn_column = fqdn_column
        self.text_columns = text_columns or [fqdn_column]
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.tfidf_max_features = tfidf_max_features
        self.categorical_strategy = categorical_strategy

        self.fqdn_extractor = FQDNFeatureExtractor()
        self.text_vectorizers: dict[str, TextVectorizer] = {}
        self.categorical_encoder: CategoricalEncoder | None = None
        self.scaler: StandardScaler | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    def fit(
        self, df: pl.DataFrame, target: pl.Series | None = None
    ) -> "FeatureEngineer":
        logger.info("Fitting feature engineer...")
        df = self.fqdn_extractor.extract_features(df, self.fqdn_column)

        for col in self.text_columns:
            if col in df.columns:
                vectorizer = TextVectorizer(
                    max_features=self.tfidf_max_features,
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                )
                texts = df[col].fill_null("").to_list()
                vectorizer.fit(texts, column_name=col)
                self.text_vectorizers[col] = vectorizer

        if self.categorical_columns:
            actual_cat_cols = [c for c in self.categorical_columns if c in df.columns]
            if actual_cat_cols:
                self.categorical_encoder = CategoricalEncoder(
                    strategy=self.categorical_strategy
                )
                self.categorical_encoder.fit(df, actual_cat_cols, target)

        if self.numerical_columns:
            actual_num_cols = [c for c in self.numerical_columns if c in df.columns]
            if actual_num_cols:
                num_data = df.select(actual_num_cols).to_numpy()
                self.scaler = StandardScaler()
                self.scaler.fit(num_data)

        self._compile_feature_names(df)
        self._fitted = True
        logger.info(f"Feature engineer fitted with {len(self._feature_names)} features")
        return self

    def transform(self, df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
        if not self._fitted:
            raise RuntimeError("Feature engineer not fitted")

        df = self.fqdn_extractor.extract_features(df, self.fqdn_column)
        feature_arrays = []

        # FQDN numerical features
        fqdn_num_cols = [
            "fqdn_length",
            "fqdn_num_dots",
            "fqdn_num_hyphens",
            "fqdn_num_digits",
            "fqdn_has_www",
            "fqdn_num_labels",
            "fqdn_max_label_length",
            "fqdn_avg_label_length",
            "fqdn_has_numeric_label",
            "fqdn_subdomain_depth",
        ]
        existing_fqdn_cols = [c for c in fqdn_num_cols if c in df.columns]
        if existing_fqdn_cols:
            feature_arrays.append(df.select(existing_fqdn_cols).fill_null(0).to_numpy())

        # Pattern features
        pattern_cols = [
            c
            for c in df.columns
            if c.startswith("fqdn_is_") or c.startswith("fqdn_region_")
        ]
        if pattern_cols:
            feature_arrays.append(df.select(pattern_cols).fill_null(0).to_numpy())

        # Text features
        for col, vectorizer in self.text_vectorizers.items():
            if col in df.columns:
                texts = df[col].fill_null("").to_list()
                feature_arrays.append(vectorizer.transform(texts))

        # Categorical features
        if self.categorical_encoder and self.categorical_columns:
            actual_cat_cols = [c for c in self.categorical_columns if c in df.columns]
            if actual_cat_cols:
                feature_arrays.append(
                    self.categorical_encoder.transform(df, actual_cat_cols)
                )

        # Numerical features
        if self.scaler and self.numerical_columns:
            actual_num_cols = [c for c in self.numerical_columns if c in df.columns]
            if actual_num_cols:
                num_data = df.select(actual_num_cols).fill_null(0).to_numpy()
                feature_arrays.append(self.scaler.transform(num_data))

        X = (
            np.hstack(feature_arrays)
            if feature_arrays
            else np.array([]).reshape(len(df), 0)
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, self._feature_names

    def fit_transform(
        self, df: pl.DataFrame, target: pl.Series | None = None
    ) -> tuple[np.ndarray, list[str]]:
        self.fit(df, target)
        return self.transform(df)

    def _compile_feature_names(self, df: pl.DataFrame) -> None:
        self._feature_names = []
        fqdn_num_cols = [
            "fqdn_length",
            "fqdn_num_dots",
            "fqdn_num_hyphens",
            "fqdn_num_digits",
            "fqdn_has_www",
            "fqdn_num_labels",
            "fqdn_max_label_length",
            "fqdn_avg_label_length",
            "fqdn_has_numeric_label",
            "fqdn_subdomain_depth",
        ]
        self._feature_names.extend([c for c in fqdn_num_cols if c in df.columns])
        pattern_cols = [
            c
            for c in df.columns
            if c.startswith("fqdn_is_") or c.startswith("fqdn_region_")
        ]
        self._feature_names.extend(sorted(pattern_cols))
        for vectorizer in self.text_vectorizers.values():
            self._feature_names.extend(vectorizer.feature_names)
        if self.categorical_encoder:
            self._feature_names.extend(self.categorical_encoder.feature_names)
        if self.numerical_columns:
            self._feature_names.extend(
                [c for c in self.numerical_columns if c in df.columns]
            )

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        import joblib

        return joblib.load(path)
