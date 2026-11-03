"""All AI endpoints. Every one degrades instead of failing."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from ai import llm, rag, registry, rules, speech, tabular, vision
from core.config import settings
from core.db import Item, get_session, log_event
from core.ws import fire

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/health")
def health():
    return registry.health()


# ---------------- generic text triage ----------------
class TriageIn(BaseModel):
    text: str
    categories: list[str] = ["infrastructure", "health", "education", "safety", "environment", "other"]
    item_id: Optional[int] = None


@router.post("/triage")
def triage(body: TriageIn, session: Session = Depends(get_session)):
    """Free text -> structured decision. The workhorse for any 'intake' PS:
    citizen grievances, support tickets, patient symptoms, field reports."""
    out = llm.complete_json(
        f"Classify this report.\nAllowed categories: {', '.join(body.categories)}\n\nReport:\n{body.text}",
        schema_hint='{"category": str, "severity": 1-5, "summary": str, "suggested_action": str, "department": str, "confidence": 0-1}',
        system="You are a triage officer for a public service system. Be decisive and specific.",
    )
    if not out or not out.get("category"):
        # Offline / parse failure -> deterministic rules engine. Still defensible.
        out = rules.triage(body.text, body.categories)
    out.setdefault("engine", "llm")

    if body.item_id:
        item = session.get(Item, body.item_id)
        if item:
            item.category = str(out.get("category", item.category))
            item.severity = int(out.get("severity", item.severity) or item.severity)
            item.score = float(out.get("confidence", 0) or 0)
            item.meta = {**(item.meta or {}), "triage": out}
            session.add(item); session.commit()
            log_event(session, "ai.triage", item.id, **out)
            fire("ai.triage", {"id": item.id, **out})
    return out


class SummariseIn(BaseModel):
    text: str
    style: str = "3 bullet points for a government officer"


@router.post("/summarise")
def summarise(body: SummariseIn):
    return {"summary": llm.complete(body.text, system=f"Summarise as {body.style}. Be specific, no preamble.")}


# ---------------- RAG ----------------
class RagDoc(BaseModel):
    id: str
    text: str
    meta: dict[str, Any] = {}


class RagIndexIn(BaseModel):
    docs: list[RagDoc]


@router.post("/rag/index")
def rag_index(body: RagIndexIn):
    return rag.build([d.model_dump() for d in body.docs])


@router.post("/rag/index-from-items")
def rag_index_from_items(session: Session = Depends(get_session)):
    """Index whatever is already in the DB -- zero-effort knowledge base."""
    items = session.exec(select(Item)).all()
    docs = [{"id": str(i.id), "text": f"{i.title}\n{i.body}", "meta": {"category": i.category, "severity": i.severity}} for i in items]
    return rag.build(docs)


class AskIn(BaseModel):
    query: str
    k: int = 5


@router.post("/rag/ask")
def rag_ask(body: AskIn):
    return rag.answer(body.query, body.k)


@router.post("/rag/search")
def rag_search(body: AskIn):
    return {"hits": rag.search(body.query, body.k)}


# ---------------- vision ----------------
@router.post("/vision/predict")
async def vision_predict(file: UploadFile = File(...), topk: int = Form(3), explain: bool = Form(False)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    return vision.predict(data, topk=topk, explain=explain)


@router.get("/vision/info")
def vision_info():
    return vision.load()


# ---------------- tabular ----------------
class TrainIn(BaseModel):
    csv_path: str
    target: str
    task: str = "auto"


@router.post("/tabular/train")
def tabular_train(body: TrainIn):
    try:
        return tabular.train(body.csv_path, body.target, body.task)
    except Exception as e:
        raise HTTPException(400, str(e))


class PredictIn(BaseModel):
    rows: list[dict[str, Any]]


@router.post("/tabular/predict")
def tabular_predict(body: PredictIn):
    return tabular.predict(body.rows)


@router.post("/tabular/upload")
async def tabular_upload(file: UploadFile = File(...)):
    dest = settings.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'data.csv').name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    cols: list[str] = []
    try:
        import pandas as pd

        cols = list(pd.read_csv(dest, nrows=5).columns)
    except Exception:
        pass
    return {"path": str(dest), "columns": cols, "rows_preview": len(cols)}


# ---------------- speech ----------------
@router.post("/speech/transcribe")
async def speech_transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    dest = settings.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'audio.wav').name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return speech.transcribe(str(dest), language)
