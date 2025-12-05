"""
Record Data Feature Extractor Module

Extracts features from DNS record data (CNAME targets, A record IPs).
Provides infrastructure and cloud provider detection from the 'data' field.
"""

import ipaddress

import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecordDataExtractor:
    """
    Extract features from DNS record data field.

    The 'data' field in FQDN records contains:
    - CNAME targets (e.g., "api.gslb-uk1.hsbc.com")
    - A record IPs (e.g., "193.108.75.128")
    - TXT record values (e.g., DMARC policies)
    """

    # Cloud provider patterns
    CLOUD_PATTERNS = {
        "aws": [
            r"\.amazonaws\.com",
            r"\.awsglobalaccelerator\.com",
            r"\.cloudfront\.net",
            r"\.elb\.[a-z]+-[a-z]+-\d+\.amazonaws\.com",
            r"ap-east-\d+\.amazonaws",
            r"us-east-\d+\.amazonaws",
            r"eu-west-\d+\.amazonaws",
        ],
        "azure": [
            r"\.azure\.com",
            r"\.azurewebsites\.net",
            r"\.trafficmanager\.net",
            r"\.cloudapp\.net",
        ],
        "gcp": [
            r"\.googleapis\.com",
            r"\.appspot\.com",
            r"\.cloudfunctions\.net",
            r"\.run\.app",
        ],
        "akamai": [
            r"\.akamai\.net",
            r"\.akamaiedge\.net",
            r"\.akamaitechnologies\.com",
        ],
        "cloudflare": [
            r"\.cloudflare\.com",
            r"\.cloudflaressl\.com",
        ],
    }

    # Infrastructure patterns
    INFRA_PATTERNS = {
        "gslb": r"\.gslb[^\.]*\.",  # Global Server Load Balancer
        "dynp": r"\.dynp\.",  # Dynamic DNS/Proxy
        "wv": r"\.wv\d+\.",  # Internal routing (wv1865, etc.)
        "cloud1": r"\.cloud1\.",  # Cloud platform indicator
        "nlb": r"-nlb-",  # Network Load Balancer
        "alb": r"-alb-",  # Application Load Balancer
        "cdn": r"\.cdn\.",  # CDN
    }

    # DMARC/Email patterns
    EMAIL_PATTERNS = {
        "dmarc": r"v=DMARC",
        "spf": r"v=spf",
        "dkim": r"v=DKIM",
    }

    # Private IP ranges
    PRIVATE_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
    ]

    def __init__(self):
        """Initialize the record data extractor."""
        self._tld_extractor = None
        try:
            import tldextract

            self._tld_extractor = tldextract.TLDExtract(cache_dir=".tld_cache")
        except ImportError:
            logger.warning("tldextract not available for record data parsing")

    def extract_features(
        self,
        df: pl.DataFrame,
        data_column: str = "data",
        record_type_column: str = "record_type",
    ) -> pl.DataFrame:
        """
        Extract features from the record data column.

        Args:
            df: Input DataFrame
            data_column: Name of the record data column (default: "data")
            record_type_column: Name of the record type column

        Returns:
            DataFrame with new record data features
        """
        if data_column not in df.columns:
            logger.warning(
                f"Column '{data_column}' not found, skipping record data features"
            )
            return df

        logger.info(f"Extracting record data features from {len(df)} records...")

        # Extract all features
        df = self._extract_cloud_provider_features(df, data_column)
        df = self._extract_infrastructure_features(df, data_column)
        df = self._extract_ip_features(df, data_column, record_type_column)
        df = self._extract_structural_features(df, data_column)
        df = self._extract_email_features(df, data_column, record_type_column)

        return df

    def _extract_cloud_provider_features(
        self, df: pl.DataFrame, data_column: str
    ) -> pl.DataFrame:
        """Extract cloud provider detection features."""
        for provider, patterns in self.CLOUD_PATTERNS.items():
            combined_pattern = "|".join(patterns)
            df = df.with_columns(
                pl.col(data_column)
                .fill_null("")
                .str.contains(f"(?i){combined_pattern}")
                .cast(pl.Int8)
                .alias(f"data_is_{provider}")
            )

        # Combined cloud indicator
        cloud_cols = [f"data_is_{p}" for p in self.CLOUD_PATTERNS.keys()]
        df = df.with_columns(
            pl.max_horizontal(*[pl.col(c) for c in cloud_cols]).alias("data_is_cloud")
        )

        return df

    def _extract_infrastructure_features(
        self, df: pl.DataFrame, data_column: str
    ) -> pl.DataFrame:
        """Extract infrastructure pattern features."""
        for name, pattern in self.INFRA_PATTERNS.items():
            df = df.with_columns(
                pl.col(data_column)
                .fill_null("")
                .str.contains(f"(?i){pattern}")
                .cast(pl.Int8)
                .alias(f"data_has_{name}")
            )
        return df

    def _extract_ip_features(
        self, df: pl.DataFrame, data_column: str, record_type_column: str
    ) -> pl.DataFrame:
        """Extract IP address features for A records."""
        data_values = df[data_column].fill_null("").to_list()
        record_types = (
            df[record_type_column].fill_null("").to_list()
            if record_type_column in df.columns
            else [""] * len(df)
        )

        is_private_ip = []
        is_valid_ipv4 = []
        ip_first_octet = []

        for data, rtype in zip(data_values, record_types):
            if rtype == "A" and data:
                try:
                    ip = ipaddress.ip_address(data)
                    is_valid_ipv4.append(1)
                    is_private = any(ip in net for net in self.PRIVATE_RANGES)
                    is_private_ip.append(1 if is_private else 0)
                    ip_first_octet.append(int(str(ip).split(".")[0]))
                except ValueError:
                    is_valid_ipv4.append(0)
                    is_private_ip.append(0)
                    ip_first_octet.append(0)
            else:
                is_valid_ipv4.append(0)
                is_private_ip.append(0)
                ip_first_octet.append(0)

        df = df.with_columns(
            [
                pl.Series("data_is_private_ip", is_private_ip).cast(pl.Int8),
                pl.Series("data_is_valid_ipv4", is_valid_ipv4).cast(pl.Int8),
                pl.Series("data_ip_first_octet", ip_first_octet).cast(pl.Int32),
            ]
        )

        return df

    def _extract_structural_features(
        self, df: pl.DataFrame, data_column: str
    ) -> pl.DataFrame:
        """Extract structural features from record data."""
        # Subdomain depth of target
        df = df.with_columns(
            (pl.col(data_column).fill_null("").str.count_matches(r"\.") + 1).alias(
                "data_subdomain_depth"
            )
        )

        # Length of data value
        df = df.with_columns(
            pl.col(data_column).fill_null("").str.len_chars().alias("data_length")
        )

        # Check if data contains same domain as FQDN (for CNAME)
        if "fqdn" in df.columns:
            fqdns = df["fqdn"].fill_null("").to_list()
            data_values = df[data_column].fill_null("").to_list()

            same_domain = []
            for fqdn, data in zip(fqdns, data_values):
                if self._tld_extractor and fqdn and data:
                    try:
                        fqdn_ext = self._tld_extractor(fqdn)
                        data_ext = self._tld_extractor(data)
                        same = 1 if fqdn_ext.domain == data_ext.domain else 0
                        same_domain.append(same)
                    except Exception:
                        same_domain.append(0)
                else:
                    same_domain.append(0)

            df = df.with_columns(
                pl.Series("data_same_domain", same_domain).cast(pl.Int8)
            )

        return df

    def _extract_email_features(
        self, df: pl.DataFrame, data_column: str, record_type_column: str
    ) -> pl.DataFrame:
        """Extract email-related features from TXT records."""
        for name, pattern in self.EMAIL_PATTERNS.items():
            df = df.with_columns(
                pl.when(pl.col(record_type_column).fill_null("") == "TXT")
                .then(
                    pl.col(data_column)
                    .fill_null("")
                    .str.contains(f"(?i){pattern}")
                    .cast(pl.Int8)
                )
                .otherwise(pl.lit(0))
                .alias(f"data_is_{name}")
            )
        return df

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor generates."""
        features = []

        # Cloud provider features
        for provider in self.CLOUD_PATTERNS.keys():
            features.append(f"data_is_{provider}")
        features.append("data_is_cloud")

        # Infrastructure features
        for name in self.INFRA_PATTERNS.keys():
            features.append(f"data_has_{name}")

        # IP features
        features.extend(
            [
                "data_is_private_ip",
                "data_is_valid_ipv4",
                "data_ip_first_octet",
            ]
        )

        # Structural features
        features.extend(
            [
                "data_subdomain_depth",
                "data_length",
                "data_same_domain",
            ]
        )

        # Email features
        for name in self.EMAIL_PATTERNS.keys():
            features.append(f"data_is_{name}")

        return features
