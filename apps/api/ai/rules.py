"""Deterministic triage without any model.

This is not a toy. It is the path your demo runs on when the venue wifi dies,
so it is tuned to be defensible: weighted severity signals, category scoring,
and an explicit reason trail you can show on screen.
"""
from __future__ import annotations

import re
from typing import Any

# Severity signals -> weight. Tune these to your PS vocabulary in the morning.
SEVERITY_SIGNALS: dict[str, int] = {
    # life-threatening
    "death": 5, "died": 5, "fatal": 5, "collapse": 5, "collapsed": 5, "fire": 5,
    "electrocut": 5, "drowned": 5, "gas leak": 5, "blood": 4, "unconscious": 5,
    # health harm in progress
    "sewage": 4, "contaminat": 4, "diarrhoea": 4, "diarrhea": 4, "cholera": 5,
    "dengue": 4, "malaria": 4, "poison": 5, "infection": 3, "outbreak": 5,
    "children": 2, "infant": 3, "pregnant": 3, "elderly": 2, "disabled": 2,
    # urgency language
    "urgent": 3, "emergency": 4, "immediately": 2, "danger": 3, "risk": 2,
    "accident": 4, "injur": 4, "trapped": 5, "flood": 4,
    # neglect / repetition (escalation signal)
    "no response": 2, "nobody came": 2, "despite": 2, "repeated": 2,
    "weeks": 2, "months": 3, "years": 3, "again": 1, "still not": 2,
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "health": ["hospital", "clinic", "medicine", "doctor", "nurse", "ambulance", "patient",
               "disease", "sewage", "diarrhoea", "diarrhea", "dengue", "vaccine", "sick", "infection"],
    "infrastructure": ["road", "pothole", "streetlight", "street light", "drain", "water supply",
                       "pipeline", "bridge", "electricity", "power cut", "transformer", "tank", "sewer"],
    "education": ["school", "teacher", "classroom", "textbook", "midday meal", "student",
                  "exam", "scholarship", "dropout", "anganwadi"],
    "safety": ["police", "theft", "harass", "assault", "accident", "unlit", "stray dog",
               "rash driving", "crime", "unsafe", "violence"],
    "environment": ["garbage", "waste", "pollution", "burning", "tree", "canal", "river",
                    "discharge", "smoke", "dump", "plastic", "air quality"],
}

DEPARTMENTS = {
    "health": "Health & Family Welfare",
    "infrastructure": "Public Works / Municipal Engineering",
    "education": "School Education",
    "safety": "Police / Traffic",
    "environment": "Pollution Control / Sanitation",
    "other": "General Administration",
}


def triage(text: str, categories: list[str] | None = None) -> dict[str, Any]:
    low = text.lower()
    allowed = categories or list(CATEGORY_KEYWORDS)

    # --- category by weighted keyword hits ---
    scores: dict[str, int] = {}
    for cat in allowed:
        kws = CATEGORY_KEYWORDS.get(cat, [cat])
        scores[cat] = sum(2 if k in low else 0 for k in kws)
    best = max(scores, key=scores.get) if scores else "other"
    category = best if scores.get(best, 0) > 0 else ("other" if "other" in allowed else best)

    # --- severity by weighted signals ---
    matched: list[tuple[str, int]] = [(sig, w) for sig, w in SEVERITY_SIGNALS.items() if sig in low]
    peak = max((w for _, w in matched), default=0)
    breadth = min(len(matched), 6)
    # peak signal dominates; breadth nudges upward. 1..5
    severity = max(1, min(5, peak if peak else 1))
    if peak and breadth >= 3 and severity < 5:
        severity += 1

    # --- duration escalation: "5 days", "3 weeks" ---
    dur = re.search(r"(\d+)\s*(day|week|month|year)", low)
    if dur:
        n, unit = int(dur.group(1)), dur.group(2)
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        if days >= 30 and severity < 5:
            severity += 1
        elif days >= 7 and severity < 4:
            severity += 1

    confidence = round(min(0.9, 0.35 + 0.08 * breadth + (0.15 if scores.get(best, 0) >= 4 else 0)), 2)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    summary = " ".join(sentences[:2])[:260] or text[:260]

    action = {
        5: "Dispatch field team now; notify district officer within 1 hour",
        4: "Assign to department today; SLA 24 hours",
        3: "Assign to department; SLA 3 working days",
        2: "Queue for standard review; SLA 7 working days",
        1: "Log and batch with the weekly ward review",
    }[severity]

    return {
        "category": category,
        "severity": int(severity),
        "summary": summary,
        "suggested_action": action,
        "department": DEPARTMENTS.get(category, "General Administration"),
        "confidence": confidence,
        "engine": "rules",
        "reasons": [{"signal": s, "weight": w} for s, w in sorted(matched, key=lambda t: -t[1])[:6]],
        "duration_detected": dur.group(0) if dur else None,
    }
