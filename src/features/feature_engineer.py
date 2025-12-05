"""
Feature Engineering Module

Comprehensive feature extraction for FQDN classification including:
- FQDN structural features
- Text vectorization (TF-IDF, n-grams)
- Categorical encoding
- Domain pattern extraction
"""

import re
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AppDescriptionVectorizer:
    """
    TF-IDF vectorization of app descriptions for business context.

    Extracts semantic features from the 'appdesc' field to capture
    business domain information like "email gateway", "trading platform", etc.
    """

    # Key business keywords to detect as binary features
    BUSINESS_KEYWORDS = [
        "trading",
        "payment",
        "customer",
        "portal",
        "gateway",
        "email",
        "security",
        "risk",
        "compliance",
        "authentication",
        "mobile",
        "website",
        "api",
        "internal",
        "external",
        "banking",
        "retail",
        "corporate",
        "wealth",
        "insurance",
        "loan",
        "credit",
        "deposit",
        "transaction",
        "transfer",
    ]

    def __init__(
        self,
        max_features: int = 100,
        min_df: int = 2,
        max_df: float = 0.9,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self._vectorizer: TfidfVectorizer | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    def _preprocess_text(self, text: str) -> str:
        """Preprocess app description text."""
        if not text:
            return ""
        # Remove special characters, lowercase
        text = re.sub(r"<CR>", " ", text)  # Remove carriage return markers
        text = re.sub(r"[^\w\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(
        self, texts: list[str], column_name: str = "appdesc"
    ) -> "AppDescriptionVectorizer":
        """Fit the TF-IDF vectorizer on app descriptions."""
        processed = [self._preprocess_text(t) for t in texts]

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),
            analyzer="word",
            sublinear_tf=True,
            stop_words="english",
        )
        self._vectorizer.fit(processed)

        vocab = self._vectorizer.get_feature_names_out()
        self._feature_names = [f"{column_name}_tfidf_{v}" for v in vocab]
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform app descriptions to TF-IDF features."""
        if not self._fitted:
            raise RuntimeError("AppDescriptionVectorizer not fitted")
        processed = [self._preprocess_text(t) for t in texts]
        return self._vectorizer.transform(processed).toarray()

    def fit_transform(
        self, texts: list[str], column_name: str = "appdesc"
    ) -> np.ndarray:
        """Fit and transform app descriptions."""
        self.fit(texts, column_name)
        return self.transform(texts)

    def extract_keyword_features(self, texts: list[str]) -> np.ndarray:
        """Extract binary keyword presence features."""
        results = []
        for text in texts:
            processed = self._preprocess_text(text)
            row = [1 if kw in processed else 0 for kw in self.BUSINESS_KEYWORDS]
            results.append(row)
        return np.array(results, dtype=np.int8)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def keyword_feature_names(self) -> list[str]:
        return [f"appdesc_has_{kw}" for kw in self.BUSINESS_KEYWORDS]


class ScanResultExtractor:
    """
    Extract features from scan_result field.

    The scan_result field contains an array of classification tags like:
    - "cname_ip_present"
    - "rxdomain"
    - "no_host_records"
    - "http_unreachable"
    """

    KNOWN_SCAN_TAGS = [
        "cname_ip_present",
        "rxdomain",
        "no_host_records",
        "http_unreachable",
        "https_unreachable",
        "ssl_cert_error",
        "timeout",
        "dns_error",
    ]

    def extract_features(
        self, df: pl.DataFrame, scan_result_column: str = "scan_result"
    ) -> pl.DataFrame:
        """
        Extract binary features from scan_result array.

        Args:
            df: Input DataFrame
            scan_result_column: Name of the scan result column

        Returns:
            DataFrame with binary scan result features
        """
        if scan_result_column not in df.columns:
            # Add zero-filled columns if scan_result not present
            for tag in self.KNOWN_SCAN_TAGS:
                df = df.with_columns(
                    pl.lit(0).cast(pl.Int8).alias(f"scan_{tag.replace('-', '_')}")
                )
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("scan_count"))
            return df

        scan_values = df[scan_result_column].to_list()

        # Create feature columns for each known tag
        tag_features = {tag: [] for tag in self.KNOWN_SCAN_TAGS}
        scan_counts = []

        for scan_result in scan_values:
            if scan_result is None:
                scan_counts.append(0)
                for tag in self.KNOWN_SCAN_TAGS:
                    tag_features[tag].append(0)
            elif isinstance(scan_result, (list, tuple)):
                scan_counts.append(len(scan_result))
                for tag in self.KNOWN_SCAN_TAGS:
                    # Check if tag is present in the array
                    found = any(tag in str(item).lower() for item in scan_result)
                    tag_features[tag].append(1 if found else 0)
            elif isinstance(scan_result, str):
                # Handle string representation of array
                scan_counts.append(1 if scan_result else 0)
                for tag in self.KNOWN_SCAN_TAGS:
                    tag_features[tag].append(1 if tag in scan_result.lower() else 0)
            else:
                scan_counts.append(0)
                for tag in self.KNOWN_SCAN_TAGS:
                    tag_features[tag].append(0)

        # Add all feature columns
        for tag in self.KNOWN_SCAN_TAGS:
            safe_name = tag.replace("-", "_")
            df = df.with_columns(
                pl.Series(f"scan_{safe_name}", tag_features[tag]).cast(pl.Int8)
            )

        df = df.with_columns(pl.Series("scan_count", scan_counts).cast(pl.Int8))

        return df

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor generates."""
        features = [f"scan_{tag.replace('-', '_')}" for tag in self.KNOWN_SCAN_TAGS]
        features.append("scan_count")
        return features


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
        # New feature extraction options
        enable_record_data_features: bool = True,
        enable_appdesc_features: bool = True,
        enable_scan_result_features: bool = True,
        appdesc_max_features: int = 100,
    ):
        self.fqdn_column = fqdn_column
        self.text_columns = text_columns or [fqdn_column]
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.tfidf_max_features = tfidf_max_features
        self.categorical_strategy = categorical_strategy

        # New feature options
        self.enable_record_data_features = enable_record_data_features
        self.enable_appdesc_features = enable_appdesc_features
        self.enable_scan_result_features = enable_scan_result_features
        self.appdesc_max_features = appdesc_max_features

        # Extractors
        self.fqdn_extractor = FQDNFeatureExtractor()
        self.text_vectorizers: dict[str, TextVectorizer] = {}
        self.categorical_encoder: CategoricalEncoder | None = None
        self.scaler: StandardScaler | None = None
        self._feature_names: list[str] = []
        self._fitted = False

        # New extractors (lazy-loaded when needed)
        self._record_data_extractor = None
        self._appdesc_vectorizer: AppDescriptionVectorizer | None = None
        self._scan_result_extractor = None

    def fit(
        self, df: pl.DataFrame, target: pl.Series | None = None
    ) -> "FeatureEngineer":
        logger.info("Fitting feature engineer...")
        df = self.fqdn_extractor.extract_features(df, self.fqdn_column)

        # Apply record data extractor if enabled
        if self.enable_record_data_features and "data" in df.columns:
            from src.features.record_data_extractor import RecordDataExtractor

            self._record_data_extractor = RecordDataExtractor()
            df = self._record_data_extractor.extract_features(df)

        # Apply scan result extractor if enabled
        if self.enable_scan_result_features and "scan_result" in df.columns:
            self._scan_result_extractor = ScanResultExtractor()
            df = self._scan_result_extractor.extract_features(df)

        # Fit app description vectorizer if enabled
        if self.enable_appdesc_features and "appdesc" in df.columns:
            self._appdesc_vectorizer = AppDescriptionVectorizer(
                max_features=self.appdesc_max_features
            )
            appdesc_texts = df["appdesc"].fill_null("").to_list()
            self._appdesc_vectorizer.fit(appdesc_texts, column_name="appdesc")

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

        # Apply record data extractor if enabled
        if self._record_data_extractor and "data" in df.columns:
            df = self._record_data_extractor.extract_features(df)

        # Apply scan result extractor if enabled
        if self._scan_result_extractor and "scan_result" in df.columns:
            df = self._scan_result_extractor.extract_features(df)

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

        # Record data features (from new extractor)
        if self._record_data_extractor:
            record_data_cols = [c for c in df.columns if c.startswith("data_")]
            if record_data_cols:
                feature_arrays.append(
                    df.select(record_data_cols).fill_null(0).to_numpy()
                )

        # Scan result features
        if self._scan_result_extractor:
            scan_cols = [c for c in df.columns if c.startswith("scan_")]
            if scan_cols:
                feature_arrays.append(df.select(scan_cols).fill_null(0).to_numpy())

        # App description TF-IDF features
        if self._appdesc_vectorizer and "appdesc" in df.columns:
            appdesc_texts = df["appdesc"].fill_null("").to_list()
            feature_arrays.append(self._appdesc_vectorizer.transform(appdesc_texts))
            # Also add keyword features
            feature_arrays.append(
                self._appdesc_vectorizer.extract_keyword_features(appdesc_texts)
            )

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

        # Record data features
        if self._record_data_extractor:
            self._feature_names.extend(self._record_data_extractor.feature_names)

        # Scan result features
        if self._scan_result_extractor:
            self._feature_names.extend(self._scan_result_extractor.feature_names)

        # App description features
        if self._appdesc_vectorizer:
            self._feature_names.extend(self._appdesc_vectorizer.feature_names)
            self._feature_names.extend(self._appdesc_vectorizer.keyword_feature_names)

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
