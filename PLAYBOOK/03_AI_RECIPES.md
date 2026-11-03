# AI recipes — what to train, when, and how

## The decision tree

```
Is the input an image?            -> CNN transfer learning      (Recipe A)
Is it a spreadsheet / numbers?    -> Gradient boosting          (Recipe B)
Is it free text to classify?      -> Rules first, then LLM      (Recipe C)
Is it a question over documents?  -> RAG                        (Recipe D)
Is it audio?                      -> Whisper                    (Recipe E)
Is it "generate something"?       -> LLM API, no training       (Recipe F)
```

**Default answer: do not train anything.** Training is only worth it when you
can show a number that beats a baseline. If you cannot measure it, an API call
or the rules engine is strictly better use of your 24 hours.

---

## The single most important idea

> **Always build a dumb baseline first, measure it, then beat it.**

The kit ships `apps/api/ai/rules.py` for exactly this. It is a real, explainable
triage engine with zero training. Measure it on 100 labelled examples. Whatever
you build next must beat it, and *that delta is your pitch*:

> "A keyword baseline gets 64%. Our model gets 91%. That is a 75% reduction in
> misrouted cases."

A team with 91% and a baseline beats a team with 94% and no baseline, every time.

---

## Recipe A — Image classification (CNN)

**When:** crop disease, waste segregation, damage assessment, X-ray triage,
document type, defect detection.

**Do not train from scratch.** You will have 50–500 images. Transfer learning
freezes an ImageNet backbone and retrains only the final layer.

```
data/raw/images/
    class_a/   img1.jpg img2.jpg ...
    class_b/   ...
```

```bash
python ml/train_cnn.py --data data/raw/images --epochs 8
```

Verified on this machine: **3 classes, 270 images, 5 epochs, ~1 s/epoch on MPS,
85% validation accuracy.** You can retrain between demo runs.

Then restart the API — `ai/vision.py` picks up `data/models/cnn.pt`
automatically and `/ai/vision/predict` serves it. No code change.

**Getting images fast:**
- Kaggle (PlantVillage, garbage classification, medical sets) — best option.
- Roboflow Universe — pre-labelled, downloads as folders already.
- Your own phone — 40 photos per class is genuinely enough for 2–3 classes.

**Tuning, in order of payoff:**
1. More images per class, especially of the *hard* cases.
2. `--unfreeze` (fine-tunes `layer4` too) — usually +3–8%.
3. More epochs (12–20). Watch for val accuracy flattening while train climbs:
   that is overfitting, stop there.
4. `--arch resnet34` or `resnet50` — only if you have >500 images/class.

**Report `val_acc` and per-class accuracy. Never report train accuracy.**
`data/models/cnn_report.json` has both, ready to screenshot.

**Grad-CAM is already wired.** Pass `explain=true` to the endpoint and you get a
heatmap overlay showing which pixels drove the decision. This is your Tier-1 USP
for any vision PS — use it.

---

## Recipe B — Tabular prediction

**When:** risk scoring, demand forecast, triage priority, fraud, dropout, churn,
no-show prediction, yield. A very large share of PS statements are this in disguise.

```bash
curl -X POST localhost:8000/ai/tabular/train \
  -H 'Content-Type: application/json' \
  -d '{"csv_path":"data/raw/yourdata.csv","target":"your_label_column"}'
```

It auto-detects classification vs regression, handles missing values and
categorical columns, trains gradient boosting, and returns metrics **plus feature
importance** — which is your explainability story for free.

Verified: 3000 rows, 8 features, trains in ~2 s, returns accuracy + ranked
feature weights.

**Make the feature importance a screen.** "Understaffing is the strongest
predictor of late resolution" is an *insight*, and insights score higher than
predictions.

No CSV yet? `python ml/make_sample_csv.py` generates one with a real learnable
signal so you can build the whole flow before your data arrives.

---

## Recipe C — Text classification

**Hour 0–6: use the rules engine.** `ai/rules.py`. Edit `SEVERITY_SIGNALS` and
`CATEGORY_KEYWORDS` to your domain vocabulary — that is 20 minutes of work and
gives you a working, explainable classifier immediately.

**Hour 6+: add the LLM** via `/ai/triage`, which falls back to the rules engine
automatically when the LLM is unreachable. Both paths return the same shape, so
the frontend never knows the difference.

**Only fine-tune a transformer if** you have >1000 labelled examples and the LLM
is genuinely wrong on your domain. In 24 hours this is almost never the right call.

---

## Recipe D — RAG (questions over documents)

**When:** policy/legal/scheme Q&A, manuals, FAQ, "what does this circular say".

```bash
# index whatever is in the database
curl -X POST localhost:8000/ai/rag/index-from-items

# or index your own documents
curl -X POST localhost:8000/ai/rag/index -H 'Content-Type: application/json' \
  -d '{"docs":[{"id":"scheme1","text":"full text here...","meta":{"src":"PDF p.3"}}]}'

curl -X POST localhost:8000/ai/rag/ask -H 'Content-Type: application/json' \
  -d '{"query":"who is eligible?","k":5}'
```

Dense embeddings (MiniLM, 384-dim) are cached locally and verified working —
semantic matching, not keywords. Falls back to TF-IDF if the encoder is missing.

**Always show the retrieved sources on screen.** An answer with citations is
trustworthy; an answer without them is a chatbot. The Knowledge tab already
renders them with similarity scores.

---

## Recipe E — Speech

```bash
uv pip install faster-whisper
```
Endpoint `/ai/speech/transcribe` is already written. Use model size `tiny` or
`base` — `large` is far too slow for a live demo and the accuracy difference
does not show on stage.

Strongest use: **voice → transcribe → triage → route**, in an Indian language.
That full chain is about 90 minutes of work and is a Tier-1 differentiator.

---

## Recipe F — Generation (no training)

Summaries, drafted replies, explanations, translation. Use `/ai/summarise` or
`llm.complete()`. Set any one API key in `.env`.

**Groq is the right choice for a hackathon**: free tier, very fast, OpenAI-compatible.
Get a key at console.groq.com and put it in `GROQ_API_KEY`.

Never let an LLM call sit in the critical path of your demo without a fallback.
The kit already handles this — `llm.complete()` never raises and never returns
empty; it degrades to extractive summarisation.

---

## Evaluation — the part that wins

Spend 45 minutes on this. It is worth more than 4 hours of model tuning.

1. Hand-label 100 examples. Four people, 25 each, 20 minutes.
2. Run the baseline. Record the number.
3. Run your model. Record the number.
4. Build a confusion matrix. Find the worst class.
5. Put all of it on one slide.

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

**Know your model's worst failure case and say it before a judge finds it.**
"Our weakest class is X at 74%, because we only had 31 training examples; here
is how we would fix it with a week" is a *strong* answer. Being caught not
knowing is fatal.
