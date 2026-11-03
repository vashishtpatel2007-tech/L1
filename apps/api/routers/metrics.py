"""Impact maths for the pitch. Put real assumptions in, get a defensible number out.

Judges reward teams that can say 'this saves X hours / Y rupees at Z scale'
AND show the arithmetic. Edit ASSUMPTIONS once you know the domain.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, func, select

from core.db import Event, Item, get_session
from core.ws import hub

router = APIRouter(prefix="/metrics", tags=["metrics"])

ASSUMPTIONS: dict[str, Any] = {
    "manual_minutes_per_case": 18,      # officer time to triage + route one case by hand
    "automated_minutes_per_case": 2,    # time with our system in the loop
    "officer_cost_per_hour_inr": 420,   # loaded hourly cost
    "cases_per_district_per_month": 9000,
    "districts_in_scope": 766,          # all districts in India -- adjust to your scope
    "triage_accuracy": 0.91,            # replace with your real eval number
    "baseline_accuracy": 0.64,          # human/legacy baseline you are beating
}


class AssumptionsIn(BaseModel):
    overrides: dict[str, float] = {}


@router.get("/impact")
def impact(session: Session = Depends(get_session)):
    a = ASSUMPTIONS
    processed = session.exec(select(func.count()).select_from(Item)).one()
    saved_min_each = a["manual_minutes_per_case"] - a["automated_minutes_per_case"]

    demo_hours = processed * saved_min_each / 60
    monthly_cases = a["cases_per_district_per_month"] * a["districts_in_scope"]
    monthly_hours = monthly_cases * saved_min_each / 60

    return {
        "assumptions": a,
        "demo": {
            "cases_processed": int(processed),
            "officer_hours_saved": round(demo_hours, 1),
            "cost_saved_inr": round(demo_hours * a["officer_cost_per_hour_inr"]),
        },
        "at_national_scale": {
            "cases_per_month": monthly_cases,
            "officer_hours_saved_per_month": round(monthly_hours),
            "cost_saved_per_month_inr": round(monthly_hours * a["officer_cost_per_hour_inr"]),
            "cost_saved_per_year_cr": round(monthly_hours * a["officer_cost_per_hour_inr"] * 12 / 1e7, 1),
        },
        "quality": {
            "accuracy": a["triage_accuracy"],
            "baseline": a["baseline_accuracy"],
            "relative_error_reduction": round((a["triage_accuracy"] - a["baseline_accuracy"]) / max(1 - a["baseline_accuracy"], 1e-6), 3),
        },
        "note": "Replace triage_accuracy with your measured eval before you present. Never quote a number you cannot derive on stage.",
    }


@router.post("/impact")
def impact_override(body: AssumptionsIn, session: Session = Depends(get_session)):
    ASSUMPTIONS.update(body.overrides)
    return impact(session)


@router.get("/system")
def system(session: Session = Depends(get_session)):
    """Live system telemetry -- cheap credibility on screen."""
    return {
        "items": int(session.exec(select(func.count()).select_from(Item)).one()),
        "events": int(session.exec(select(func.count()).select_from(Event)).one()),
        "ws_clients": hub.count,
        "recent_events": [
            {"kind": e.kind, "item_id": e.item_id, "at": e.created_at.isoformat()}
            for e in session.exec(select(Event).order_by(Event.created_at.desc()).limit(15)).all()
        ],
    }
