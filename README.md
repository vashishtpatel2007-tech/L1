# Hackathon Kit

A pre-built, pre-tested full-stack + AI foundation so that at hour 0 you are
**adapting a working system**, not creating one. Every piece below has been run
and verified on this machine.

---

## Start here (hour 0)

```bash
./dev.sh
```

- UI: http://localhost:5173
- API docs: http://localhost:8000/docs
- Logins: `admin@demo.in` / `demo1234`

If it is a fresh machine, run `./setup.sh` once first.

Then read **`PLAYBOOK/01_PS_SELECTION.md`** before you write any code.

---

## What is already built and working

| Capability | Endpoint | State |
|---|---|---|
| Auth (JWT, scrypt) | `/auth/*` | Working, no external crypto deps |
| CRUD + filters + aggregates | `/items/*` | Working |
| Realtime push | `ws://…/ws` | Verified: live dashboard updates |
| **Text triage** | `/ai/triage` | LLM **with an explainable rules fallback** |
| **Image classification** | `/ai/vision/predict` | Fine-tuned CNN → ImageNet → heuristic |
| **Grad-CAM explainability** | same, `explain=true` | Verified, renders a heatmap |
| **RAG with citations** | `/ai/rag/*` | Dense MiniLM embeddings, cached locally |
| **Tabular ML** | `/ai/tabular/*` | Train from CSV, returns feature importance |
| Speech → text | `/ai/speech/transcribe` | Needs `uv pip install faster-whisper` |
| Demo control | `/demo/seed`, `/demo/simulate` | Deterministic seed + live traffic sim |
| Impact calculator | `/metrics/impact` | Editable assumptions, derived numbers |
| Capability health | `/ai/health` | Reports its own degraded modes |

### The design principle

**Nothing hard-fails.** Every AI path degrades to something that still works:

```
LLM      -> rules engine        -> extractive summary
CNN      -> ImageNet pretrained -> colour heuristic
RAG      -> dense embeddings    -> TF-IDF -> keyword
```

Venue wifi dying cannot kill your demo. There is a live offline toggle in the
System tab — use it as a feature, not an excuse.

---

## Layout

```
apps/api/          FastAPI backend
  core/            config, db, auth, websocket hub
  ai/              llm, rules, rag, vision, tabular, speech, registry
  routers/         auth, items, ai, demo, metrics
apps/web/          React + Vite + Tailwind
  src/pages/       Dashboard, Triage, Vision, Knowledge, Impact, System
ml/                train_cnn.py, sample data generators
scripts/           setup, prefetch, tunnel, preflight
PLAYBOOK/          strategy: PS choice, timeline, roles, AI recipes, pitch
data/              db, models, index, uploads  (gitignored)
```

---

## Adapting it to your PS (about 30 minutes)

1. **Rename the domain object.** `apps/api/core/db.py` → `Item` becomes
   `Complaint` / `Patient` / `Crop` / `Shipment`. Adjust the fields.
2. **Rewrite the seeder.** `apps/api/routers/demo.py` → `TITLES`, `DETAILS`,
   `CATEGORIES`, and the `LAT`/`LON` for your city.
3. **Retune the rules engine.** `apps/api/ai/rules.py` → `SEVERITY_SIGNALS` and
   `CATEGORY_KEYWORDS` in your domain's vocabulary. 20 minutes, big payoff.
4. **Set your identity.** `.env` → `APP_NAME`, `TAGLINE`.
5. **Delete the tabs you do not need.** `apps/web/src/App.tsx` → the `TABS` array.
6. **Fix the impact assumptions.** `apps/api/routers/metrics.py` → `ASSUMPTIONS`.

---

## Training a model

```bash
# images: one folder per class under data/raw/images/
python ml/train_cnn.py --data data/raw/images --epochs 8
# -> writes data/models/cnn.pt; restart the API and it serves automatically

# tabular
curl -X POST localhost:8000/ai/tabular/train -H 'Content-Type: application/json' \
  -d '{"csv_path":"data/raw/yourdata.csv","target":"label_column"}'
```

No data yet? Generate realistic stand-ins and rehearse the whole pipeline now:

```bash
python ml/make_sample_images.py     # 270 images, 3 classes
python ml/make_sample_csv.py        # 3000 rows with a real learnable signal
```

Full guidance in `PLAYBOOK/03_AI_RECIPES.md`.

---

## Before you present

```bash
bash scripts/preflight.sh      # 9 checks; must be 9/9
bash scripts/tunnel.sh         # public HTTPS URL for judges' phones
```

Then pull the wifi and run preflight again. Everything except the LLM must still
pass. Drills in `PLAYBOOK/06_FAILURE_DRILLS.md`.

---

## Verified on this machine

- Apple M4, 16 GB, Python 3.13, Node 25, **Metal/MPS active (2.6 TFLOPS)**
- CNN transfer learning: 3 classes, 270 images, **85.2% val acc, ~1 s/epoch**
- Tabular: 3000 rows, trains in ~2 s, returns ranked feature importance
- RAG: MiniLM 384-dim dense retrieval, **weights cached locally** (87 MB)
- Realtime: websocket push verified end to end
- Frontend: typechecks clean, builds in ~1 s, served single-port from the API
