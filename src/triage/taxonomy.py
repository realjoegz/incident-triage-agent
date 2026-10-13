"""The closed set of root-cause categories.

A category names the mechanism that failed, not what set it off: a flash sale that runs into an
acquirer's limit is `rate_limiting`, and a failover that leaves dead connections in a pool is
`connection_pool_exhaustion`. Root-cause accuracy is an exact match on this value, so the set is
kept small enough for each incident in the catalogue to have one defensible answer.
"""

from enum import StrEnum


class Category(StrEnum):
    CONNECTION_POOL_EXHAUSTION = "connection_pool_exhaustion"
    CERTIFICATE_EXPIRY = "certificate_expiry"
    CONFIG_CHANGE = "config_change"
    CODE_REGRESSION = "code_regression"
    DISK_FULL = "disk_full"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DNS_FAILURE = "dns_failure"
    LOCK_CONTENTION = "lock_contention"
    QUEUE_CONSUMER_STALL = "queue_consumer_stall"
    RATE_LIMITING = "rate_limiting"
    UNKNOWN = "unknown"


DEFINITIONS: dict[Category, str] = {
    Category.CONNECTION_POOL_EXHAUSTION: (
        "a client ran out of pooled connections to a dependency, so requests waited for one"
    ),
    Category.CERTIFICATE_EXPIRY: (
        "a TLS certificate or a signing key passed its expiry date or was not rotated in time"
    ),
    Category.CONFIG_CHANGE: (
        "a change to configuration, a feature flag or reference data, made outside a code release"
    ),
    Category.CODE_REGRESSION: "a code release introduced the defect",
    Category.DISK_FULL: "a volume or a disk watermark filled up",
    Category.MEMORY_EXHAUSTION: "a process ran out of memory or heap",
    Category.DNS_FAILURE: "a name could not be resolved",
    Category.LOCK_CONTENTION: "work was blocked or aborted by a database lock or a deadlock",
    Category.QUEUE_CONSUMER_STALL: (
        "a queue or stream consumer stopped or fell behind, so messages backed up"
    ),
    Category.RATE_LIMITING: "an upstream provider throttled requests above a quota",
    Category.UNKNOWN: "the evidence does not establish a cause",
}

KNOWN_CATEGORIES: tuple[Category, ...] = tuple(c for c in Category if c is not Category.UNKNOWN)
