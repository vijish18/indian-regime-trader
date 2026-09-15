"""Exception types shared by every ``broker.base.Broker`` implementation.

One hierarchy for every adapter -- paper, and any future live one -- so
callers never need to know which concrete broker raised a given failure,
only what kind of failure it was.
"""

from __future__ import annotations


class BrokerError(RuntimeError):
    """Base class for every broker-adapter failure."""


class BrokerAuthenticationError(BrokerError):
    """``Broker.authenticate`` could not establish a session -- missing or
    invalid credentials, or the broker rejected them."""


class BrokerSessionExpiredError(BrokerError):
    """A previously valid session was rejected by the broker (token
    expiry, forced logout elsewhere) -- re-authentication is required
    before any further call can succeed."""


class BrokerCapabilityError(BrokerError):
    """The requested operation is not available from this adapter --
    e.g. market-data streaming with no transport configured, or any
    order-placing call while live trading is disabled. Check
    ``Broker.capabilities()`` before assuming an operation is possible.
    """


class BrokerRequestError(BrokerError):
    """The broker rejected a request outright (bad parameters,
    insufficient funds/holdings, an internal broker-side failure) --
    carries the broker's own error classification where the adapter has
    one to report, via ``error_type``.
    """

    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


class BrokerRateLimitError(BrokerRequestError):
    """The broker's own rate limit was exceeded -- distinct from a
    plain rejection so a caller can choose to back off and retry."""
