"""SQLModel + SQLite. Swap DB_URL for Postgres and nothing else changes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from sqlalchemy.pool import StaticPool
from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_connect_args: dict[str, Any] = {}
_kwargs: dict[str, Any] = {}
if settings.DB_URL.startswith("sqlite"):
    # check_same_thread=False lets the websocket + request threads share the engine.
    _connect_args = {"check_same_thread": False}
    if ":memory:" in settings.DB_URL:
        _kwargs["poolclass"] = StaticPool

engine = create_engine(settings.DB_URL, connect_args=_connect_args, echo=False, **_kwargs)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    name: str = ""
    role: str = "user"          # user | admin | reviewer -- rename per PS
    password_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class Item(SQLModel, table=True):
    """The generic domain object. Rename to Complaint / Crop / Patient / Shipment
    on hackathon morning -- the API, UI table and charts all follow automatically."""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = ""
    body: str = ""
    category: str = Field(default="unclassified", index=True)
    status: str = Field(default="open", index=True)   # open | in_progress | resolved
    severity: int = Field(default=1, index=True)      # 1..5
    score: float = 0.0                                # model confidence / risk
    lat: Optional[float] = None
    lon: Optional[float] = None
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)


class Event(SQLModel, table=True):
    """Append-only audit log. Judges love a timeline -- this powers it for free."""
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str = Field(index=True)
    item_id: Optional[int] = Field(default=None, index=True)
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def log_event(session: Session, kind: str, item_id: int | None = None, **payload: Any) -> Event:
    ev = Event(kind=kind, item_id=item_id, payload=payload)
    session.add(ev)
    session.commit()
    session.refresh(ev)
    return ev
