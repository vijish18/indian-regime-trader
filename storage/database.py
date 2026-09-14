"""Database connection and session management (PostgreSQL, per
docs/SPECIFICATION.md section 3). Reads ``DATABASE_URL`` from the
environment (see ``config.loader.load_environment``), never from
``settings.yaml``.

Not implemented yet (Phase 2/3, once an ORM/migration tool is chosen).
"""

from __future__ import annotations

from contextlib import AbstractContextManager


class DatabaseSession(AbstractContextManager["DatabaseSession"]):
    """Placeholder for a real DB session context manager (e.g. SQLAlchemy).

    Backward-compatible migrations are required for every schema change
    (docs/SPECIFICATION.md section 23) -- see storage/migrations/.
    """

    def __enter__(self) -> DatabaseSession:
        raise NotImplementedError("Phase 2/3: database session management is not implemented yet.")

    def __exit__(self, *exc_info: object) -> None:
        raise NotImplementedError("Phase 2/3: database session management is not implemented yet.")


def get_engine(database_url: str, pool_size: int) -> object:
    raise NotImplementedError("Phase 2/3: database engine setup is not implemented yet.")
