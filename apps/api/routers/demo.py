"""Demo control surface: seed, reset, simulate live traffic, toggle offline.

Never walk into a judging round with an empty database.
"""
from __future__ import annotations

import asyncio
import random
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, delete, select

from core.config import settings
from core.db import Event, Item, User, get_session, log_event
from core.security import hash_password
from core.ws import fire, hub

router = APIRouter(prefix="/demo", tags=["demo"])

CATEGORIES = ["infrastructure", "health", "education", "safety", "environment"]

TITLES = {
    "infrastructure": ["Streetlight dead for weeks", "Pothole flooding the lane", "Drain overflowing near market",
                       "Water supply cut since Monday", "Transformer sparking on the corner", "Footpath dug up and left open"],
    "health": ["No stock of basic medicines", "Ambulance took 50 minutes", "Clinic shut during posted hours",
               "Mosquito breeding behind the ward", "Sewage mixing into the drinking line", "Only one doctor for the block"],
    "education": ["Two teachers for six classes", "Midday meal skipped twice", "Broken classroom roof",
                  "No textbooks for class 7", "Toilet block unusable for girls", "Anganwadi closed since the holidays"],
    "safety": ["Unlit stretch near the bridge", "Stray dog pack at the bus stop", "Open transformer box",
               "Rash driving outside the school", "No barricade at the excavation", "Street cameras not working"],
    "environment": ["Garbage uncollected for days", "Factory discharge in the canal", "Tree fell across the road",
                    "Burning waste at night", "Construction dust over the colony", "Dead fish in the pond"],
}

# Varied phrasing matters: identical bodies make retrieval return duplicate
# hits, which looks broken on stage even though the index is correct.
DETAILS = {
    "infrastructure": ["Residents have raised this at two ward meetings with no action.",
                       "The contractor started work and abandoned the site midway.",
                       "Water is pooling and the lane is impassable for two-wheelers.",
                       "An elderly resident fell here last week because of the dark."],
    "health": ["Three children in the building now have diarrhoea.",
               "A pregnant woman had to be taken to a private hospital instead.",
               "Staff say the supply chain has not delivered for a month.",
               "People are buying medicines out of pocket at four times the price."],
    "education": ["Parents are pulling children out and sending them to a private school.",
                  "Students are sitting on the floor because benches were never replaced.",
                  "The roof leaks directly onto the only working blackboard.",
                  "Attendance has dropped sharply since the facility broke down."],
    "safety": ["Two near-misses were reported by shopkeepers this week.",
               "Children walk this stretch to school every morning.",
               "The exposed wiring is at waist height next to a tea stall.",
               "Nobody has responded despite repeated calls to the control room."],
    "environment": ["The smell is unbearable and windows cannot be opened.",
                    "Cattle are feeding on the uncollected waste pile.",
                    "The discharge turns the water black every evening after 7pm.",
                    "Residents with asthma say symptoms have worsened this month."],
}
CHANNELS = ["web", "whatsapp", "ivr", "field_officer"]
WARDS = [f"Ward {n}" for n in range(1, 25)]

# Delhi-ish bounding box; change to your demo city.
LAT, LON = 28.61, 77.21


def _body(cat: str, rnd: random.Random) -> str:
    """Compose a varied, realistic complaint body."""
    days = rnd.choice([2, 3, 5, 7, 10, 14, 21, 30, 45])
    unit = "days" if days < 14 else "weeks"
    n = days if days < 14 else days // 7
    lead = rnd.choice([
        f"Unresolved for {n} {unit} now.",
        f"This has been going on for {n} {unit}.",
        f"Reported {n} {unit} ago and still open.",
    ])
    return f"{lead} {rnd.choice(DETAILS[cat])} {rnd.choice(DETAILS[cat])}"


class SeedIn(BaseModel):
    n: int = 120
    reset: bool = True


@router.post("/seed")
def seed(body: SeedIn, session: Session = Depends(get_session)):
    if body.reset:
        session.exec(delete(Event)); session.exec(delete(Item)); session.commit()

    if not session.exec(select(User).where(User.email == "admin@demo.in")).first():
        session.add(User(email="admin@demo.in", name="Demo Admin", role="admin", password_hash=hash_password("demo1234")))
        session.add(User(email="user@demo.in", name="Demo Citizen", role="user", password_hash=hash_password("demo1234")))
        session.commit()

    rnd = random.Random(42)   # deterministic: the demo looks identical every run
    made = 0
    for _ in range(body.n):
        cat = rnd.choice(CATEGORIES)
        sev = rnd.choices([1, 2, 3, 4, 5], weights=[25, 30, 22, 15, 8])[0]
        status = rnd.choices(["open", "in_progress", "resolved"], weights=[45, 25, 30])[0]
        session.add(Item(
            title=rnd.choice(TITLES[cat]),
            body=_body(cat, rnd),
            category=cat, status=status, severity=sev,
            score=round(rnd.uniform(0.55, 0.98), 3),
            lat=round(LAT + rnd.uniform(-0.09, 0.09), 5),
            lon=round(LON + rnd.uniform(-0.09, 0.09), 5),
            meta={"source": rnd.choice(CHANNELS), "ward": rnd.choice(WARDS)},
        ))
        made += 1
    session.commit()
    return {"seeded": made, "logins": [{"email": "admin@demo.in", "password": "demo1234"}, {"email": "user@demo.in", "password": "demo1234"}]}


@router.post("/reset")
def reset(session: Session = Depends(get_session)):
    session.exec(delete(Event)); session.exec(delete(Item)); session.commit()
    return {"ok": True}


class SimIn(BaseModel):
    seconds: int = 20
    per_second: float = 1.5


@router.post("/simulate")
async def simulate(body: SimIn, session: Session = Depends(get_session)):
    """Pushes live items over the websocket. Run this while the judge watches
    the dashboard -- numbers moving on their own reads as a real system."""
    async def run():
        for _ in range(int(body.seconds * body.per_second)):
            cat = random.choice(CATEGORIES)
            sev = random.choices([1, 2, 3, 4, 5], weights=[20, 25, 25, 20, 10])[0]
            with Session(session.get_bind()) as s:
                item = Item(
                    title=random.choice(TITLES[cat]), body=_body(cat, random.Random()),
                    category=cat, severity=sev, score=round(random.uniform(0.6, 0.99), 3),
                    lat=round(LAT + random.uniform(-0.09, 0.09), 5),
                    lon=round(LON + random.uniform(-0.09, 0.09), 5),
                    meta={"source": "simulator"},
                )
                s.add(item); s.commit(); s.refresh(item)
                await hub.broadcast("item.created", {"id": item.id, "title": item.title, "category": cat, "severity": sev})
            await asyncio.sleep(1 / max(body.per_second, 0.1))

    asyncio.create_task(run())
    return {"started": True, "will_emit": int(body.seconds * body.per_second), "ws_clients": hub.count}


class OfflineIn(BaseModel):
    offline: bool


@router.post("/offline")
def set_offline(body: OfflineIn):
    """Panic switch. Venue wifi dies -> flip this -> every AI path uses local fallbacks."""
    settings.OFFLINE = body.offline
    return {"offline": settings.OFFLINE}
