# BACKEND PROMPT — paste into the AI builder

Fill the five `{{...}}` placeholders first. Everything else is already correct;
do not soften the constraints — they are what make the merge work.

> **Recommended tool:** Replit Agent, Bolt.new, or Cursor. **Avoid Lovable/v0 for this** —
> they are frontend-first and will insist on generating React + Supabase, which is
> exactly what we do not want and the single most likely cause of a failed merge.

---

```
You are a senior backend engineer. Build a production-quality Python backend.
Follow this specification exactly. Do not improvise, do not add features, do not
substitute technologies. I will run an automated contract test against your output
and reject it if any detail below is wrong.

=====================================================================
HARD CONSTRAINTS — violating any of these makes the work unusable
=====================================================================
1. Backend ONLY. Do NOT generate a frontend, React, HTML pages, or any UI.
2. Stack is fixed: Python 3.11+, FastAPI, SQLModel, SQLite, Uvicorn.
   Do NOT use Supabase, Firebase, Prisma, Django, Flask, Express, or Next.js.
   Do NOT use an external database. SQLite file only.
3. Do NOT use Docker. Do NOT add a docker-compose file.
4. Do NOT add Alembic migrations. Use SQLModel.metadata.create_all().
5. Every dependency must be in requirements.txt. No system packages.
6. The app MUST start with exactly: uvicorn main:app --port 8000
7. No endpoint may ever return an HTML error page. All errors are JSON: {"detail": "..."}.
8. CORS must allow all origins, all methods, all headers.

=====================================================================
DOMAIN
=====================================================================
The product is: {{ONE_SENTENCE_DESCRIPTION_OF_THE_PRODUCT}}

The core database object is called "{{OBJECT}}" (plural "{{OBJECT_PLURAL}}").
It represents: {{WHAT_ONE_RECORD_IS}}

Valid categories: {{CATEGORY_LIST}}
Valid statuses:   open, in_progress, resolved
Severity is an integer 1-5 where 5 is most urgent.

=====================================================================
FILE LAYOUT — produce exactly these files
=====================================================================
main.py              FastAPI app, CORS, startup, websocket, router registration
db.py                SQLModel models + engine + session dependency
security.py          password hashing + JWT (stdlib hashlib/hmac only, no pyjwt/passlib)
rules.py             deterministic triage engine (NO external API calls)
routers/items.py     CRUD + filtering + stats
routers/ai.py        triage + summarise + health
routers/demo.py      seed + offline toggle
routers/auth.py      signup + login
requirements.txt

=====================================================================
DATA MODEL — field names and types are exact
=====================================================================
Item (table name "{{OBJECT}}"):
  id          int, primary key, autoincrement
  title       str, default ""
  body        str, default ""
  category    str, default "unclassified", indexed
  status      str, default "open", indexed
  severity    int, default 1, indexed
  score       float, default 0.0
  lat         float, nullable
  lon         float, nullable
  meta        dict, stored as a JSON column, default {}
  created_at  datetime UTC, default now, indexed
  updated_at  datetime UTC, default now

User:
  id, email (unique, indexed), name, role (default "user"),
  password_hash, created_at

Event (append-only audit log):
  id, kind (str, indexed), item_id (int, nullable), payload (JSON), created_at

=====================================================================
ENDPOINTS — response shapes are a contract. Field names must match exactly.
=====================================================================

GET /health
  -> {"ok": true, "app": str, "uptime_s": float, "offline_mode": bool}

GET /items
  Query params (all optional): q, category, status, min_severity,
    sort (one of: created_at|severity|score|status|category, default created_at),
    order (asc|desc, default desc), limit (default 100, max 1000), offset (default 0)
  q performs a LIKE search across title and body.
  -> {"total": int, "items": [Item], "limit": int, "offset": int}
  NOTE: "total" is the count BEFORE limit/offset is applied, and it MUST be an
  integer, never a string.

GET /items/stats
  *** CRITICAL: register this route BEFORE GET /items/{id} in the router, or
  FastAPI will match the literal string "stats" as an id and this breaks. ***
  -> {
       "total": int, "resolved": int, "resolution_rate": float, "avg_severity": float,
       "by_category": [{"key": str, "count": int}],
       "by_status":   [{"key": str, "count": int}],
       "by_severity": [{"key": str, "count": int}],
       "geo": [{"lat": float, "lon": float, "severity": int, "category": str, "id": int}]
     }
  resolution_rate is resolved/total, or 0.0 when total is 0 (never divide by zero).
  geo contains only records with a non-null lat, capped at 500.

GET /items/{id}        -> Item, or 404 {"detail": "Not found"}
POST /items            -> HTTP 201, returns the created Item
PATCH /items/{id}      -> partial update, returns the updated Item
DELETE /items/{id}     -> HTTP 204, empty body

POST /ai/triage
  Body: {"text": str, "categories": [str] (optional), "item_id": int (optional)}
  -> {"category": str, "severity": int, "summary": str, "suggested_action": str,
      "department": str, "confidence": float, "engine": str,
      "reasons": [{"signal": str, "weight": int}], "duration_detected": str|null}

  *** THIS IS THE MOST IMPORTANT ENDPOINT. IT MUST NEVER FAIL. ***
  Implement it in rules.py with NO external API calls:
    - Score severity from weighted keyword signals. Life-threatening words
      (death, fire, collapse, electrocution, drowning, gas leak) score 5.
      Health-harm words (sewage, contamination, outbreak, poison) score 4.
      Vulnerability words (children, infant, pregnant, elderly) score 2-3.
      Neglect/repetition words (despite, repeated, no response, still not) score 2.
    - Pick the category by weighted keyword matching against the category list.
    - Detect an unresolved duration with a regex like (\d+)\s*(day|week|month|year)
      and escalate severity by 1 if >= 7 days, by 1 more if >= 30 days. Cap at 5.
    - Return "reasons" listing every signal that matched and its weight. This
      powers the explainability UI and is not optional.
    - "engine" is the string "rules".
    - If item_id is supplied, update that record's category/severity and append
      an Event row.

POST /ai/summarise
  Body: {"text": str, "style": str (optional)}
  -> {"summary": str}
  Implement extractively with no external API: split into sentences, score each
  by the frequency of its non-stopword terms, return the top 3 as bullet points.

GET /ai/health
  -> {"offline_mode": bool, "capabilities": {<name>: {"ok": bool, ...}}}

POST /demo/seed
  Body: {"n": int (default 120), "reset": bool (default true)}
  -> {"seeded": int, "logins": [{"email": str, "password": str}]}
  *** MUST BE DETERMINISTIC: use random.Random(42), not the global random module.
  Two runs with the same n must produce byte-identical records, because the demo
  has to look the same every single time. ***
  Create demo users admin@demo.in / demo1234 and user@demo.in / demo1234.
  Generate VARIED titles and bodies — at least 6 title templates per category and
  at least 4 body fragments, combined randomly. Identical bodies make search and
  retrieval return duplicate-looking results, which looks broken on screen.
  Scatter lat/lon within about +/-0.09 degrees of {{DEMO_LAT_LON}}.

POST /demo/reset       -> {"ok": true}   (deletes all Items and Events)

POST /demo/offline
  Body: {"offline": bool}  -> {"offline": bool}
  Sets a module-level flag that forces every AI path to use local logic.

GET /metrics/impact
  -> {"assumptions": {...}, "demo": {...}, "at_national_scale": {...}, "quality": {...}}
  Keep a module-level ASSUMPTIONS dict (minutes saved per case, cost per hour,
  cases per month, districts in scope, accuracy, baseline accuracy) and DERIVE
  every output number from it. Do not hardcode results.

POST /auth/signup  Body {"email","password","name"} -> {"token": str, "user": {...}}
POST /auth/login   Body {"email","password"}        -> {"token": str, "user": {...}}
GET  /auth/me      Bearer token                     -> the user object
  Hash passwords with hashlib.scrypt plus a random per-user salt. Sign JWTs with
  hmac + hashlib.sha256 from the standard library. Do NOT add pyjwt or passlib.

WEBSOCKET /ws
  On connect, send {"kind": "hello", "data": {"app": <app name>}}.
  Broadcast {"kind": "item.created", "data": {"id","title","category","severity"}}
  whenever an Item is created. The client sends arbitrary text as a keepalive —
  receive and ignore it, and never let a disconnect raise an unhandled exception.
  Maintain the client set safely and drop dead sockets on send failure.

POST /demo/simulate
  Body: {"seconds": int, "per_second": float}
  -> {"started": true, "will_emit": int}
  Spawns a background task that creates records and pushes them over the
  websocket at the given rate. Must return immediately, not block.

=====================================================================
NON-NEGOTIABLE BEHAVIOURS — these are tested automatically
=====================================================================
A. GET /items/stats must work. If it returns 404 or a validation error, your
   route ordering is wrong — move it above /items/{id}.
B. POST /ai/triage must return HTTP 200 with a non-empty category and severity
   even with no network access at all. It performs no external calls by design.
C. POST /demo/seed must be deterministic.
D. A request for a non-existent id returns JSON {"detail": ...}, never HTML.
E. "total" in GET /items must be a JSON integer.
F. The server must start cleanly with no warnings about deprecated event handlers
   (use the FastAPI lifespan context manager, not @app.on_event).

=====================================================================
DELIVERABLE
=====================================================================
Output every file in full. No placeholders, no "...", no "implement this later",
no TODO comments. The code must run immediately after:

    pip install -r requirements.txt
    uvicorn main:app --port 8000

Then confirm for me, explicitly:
  1. That GET /items/stats is registered before GET /items/{id}.
  2. That POST /ai/triage makes zero network calls.
  3. That POST /demo/seed uses a fixed random seed.
```

---

## After your friend sends you the code

Do not read it. Run the gate:

```bash
python scripts/verify_contract.py --base http://127.0.0.1:8000
```

Exit code 0 → merge. Anything else → paste the failures back to the AI builder
verbatim and ask it to fix exactly those. Repeat until it is 0. Never merge on
a human's opinion that it "looks fine".
