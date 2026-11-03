"""Generic CRUD + filtering + aggregates for the core domain object.

Rename Item -> your domain noun in core/db.py and this whole file keeps working.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, func, select

from core.db import Item, User, get_session, log_event, utcnow
from core.security import optional_user
from core.ws import fire

router = APIRouter(prefix="/items", tags=["items"])


class ItemIn(BaseModel):
    title: str = ""
    body: str = ""
    category: str = "unclassified"
    status: str = "open"
    severity: int = 1
    score: float = 0.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    meta: dict[str, Any] = {}


class ItemPatch(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[int] = None
    score: Optional[float] = None
    meta: Optional[dict[str, Any]] = None


@router.get("")
def list_items(
    session: Session = Depends(get_session),
    q: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    min_severity: Optional[int] = None,
    sort: str = Query("created_at", pattern="^(created_at|severity|score|status|category)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    stmt = select(Item)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(Item.title.like(like) | Item.body.like(like))
    if category:
        stmt = stmt.where(Item.category == category)
    if status:
        stmt = stmt.where(Item.status == status)
    if min_severity is not None:
        stmt = stmt.where(Item.severity >= min_severity)

    total = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    col = getattr(Item, sort)
    stmt = stmt.order_by(col.desc() if order == "desc" else col.asc()).limit(limit).offset(offset)
    return {"total": total, "items": session.exec(stmt).all(), "limit": limit, "offset": offset}


@router.get("/stats")
def stats(session: Session = Depends(get_session)):
    """Everything the dashboard needs in one round trip."""
    def group(col):
        rows = session.exec(select(col, func.count()).group_by(col)).all()
        return [{"key": str(k), "count": int(c)} for k, c in rows]

    total = session.exec(select(func.count()).select_from(Item)).one()
    avg_sev = session.exec(select(func.avg(Item.severity))).one() or 0
    resolved = session.exec(select(func.count()).select_from(Item).where(Item.status == "resolved")).one()
    return {
        "total": int(total),
        "resolved": int(resolved),
        "resolution_rate": round(resolved / total, 3) if total else 0.0,
        "avg_severity": round(float(avg_sev), 2),
        "by_category": group(Item.category),
        "by_status": group(Item.status),
        "by_severity": group(Item.severity),
        "geo": [
            {"lat": i.lat, "lon": i.lon, "severity": i.severity, "category": i.category, "id": i.id}
            for i in session.exec(select(Item).where(Item.lat.is_not(None)).limit(500)).all()
        ],
    }


@router.get("/{item_id}")
def get_item(item_id: int, session: Session = Depends(get_session)):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(404, "Not found")
    return item


@router.post("", status_code=201)
def create_item(body: ItemIn, session: Session = Depends(get_session), user: Optional[User] = Depends(optional_user)):
    item = Item(**body.model_dump(), owner_id=user.id if user else None)
    session.add(item); session.commit(); session.refresh(item)
    log_event(session, "item.created", item.id, title=item.title)
    fire("item.created", {"id": item.id, "title": item.title, "category": item.category, "severity": item.severity})
    return item


@router.patch("/{item_id}")
def patch_item(item_id: int, body: ItemPatch, session: Session = Depends(get_session)):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(404, "Not found")
    changes = body.model_dump(exclude_unset=True)
    for k, v in changes.items():
        setattr(item, k, v)
    item.updated_at = utcnow()
    session.add(item); session.commit(); session.refresh(item)
    log_event(session, "item.updated", item.id, **changes)
    fire("item.updated", {"id": item.id, **changes})
    return item


@router.delete("/{item_id}", status_code=204)
def delete_item(item_id: int, session: Session = Depends(get_session)):
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(404, "Not found")
    session.delete(item); session.commit()
    log_event(session, "item.deleted", item_id)
    fire("item.deleted", {"id": item_id})
