# USP bank — pick exactly ONE

Each of these is buildable in 2–4 hours on top of this kit, and each one
visibly separates you from the other teams. Choose for *demo impact*, not for
technical taste. One, done perfectly, beats three half-finished.

---

## Tier 1 — highest impact per hour

### 1. "Why" on every decision
Already shipped in the kit: `/ai/triage` returns `reasons` with weights, and the
Triage screen renders them as chips. For vision, Grad-CAM is already wired.
**Why it wins:** every other team shows a prediction. You show the evidence.
Judges in health/finance/governance domains care about this more than accuracy.

### 2. Offline mode as a feature, not a fallback
The System tab has a live toggle. On stage: *"Rural deployment, 2G or no
network — watch."* Flip it. Everything keeps working.
**Why it wins:** nobody else does it, it is visibly hard, and it directly
answers the "will this work in a real Indian village" question before it is asked.

### 3. The live simulator
`/demo/simulate` pushes records over websockets while you talk. Numbers move on
their own.
**Why it wins:** a dashboard that updates by itself reads as a production
system, not a hackathon project. Costs you one button.

### 4. Derivable impact numbers
The Impact tab lets you edit assumptions live and recompute.
**Why it wins:** when a judge says "that number seems optimistic", you change it
in front of them and show the result still holds. That single move has won
arguments and events.

---

## Tier 2 — strong, domain-dependent

### 5. Voice input in an Indian language
`pip install faster-whisper`, endpoint already exists at `/ai/speech/transcribe`.
Record 10 seconds of Hindi/Gujarati/Tamil describing a problem → it transcribes
→ triages → routes.
**Why it wins:** accessibility for non-English, non-literate users is the single
most reliable emotional lever with Indian judging panels. Use `tiny` or `base`
model size; `large` is too slow for a live demo.

### 6. WhatsApp-shaped intake
You do not need the real WhatsApp Business API. Build a chat-styled web page
that posts to `/items` and `/ai/triage`.
**Why it wins:** "the citizen does not install anything" removes the biggest
real-world adoption objection, and it is 2 hours of frontend.

### 7. Human-in-the-loop override
Let the officer correct the model's category, store the correction in `Event`,
and show a "model improved from your corrections" counter.
**Why it wins:** it answers "what if the AI is wrong" with a mechanism instead
of a promise, and it hints at a real production feedback loop.

### 8. A map that works offline
Already shipped: `GeoScatter` in `components/ui.tsx` is a plain SVG, no tiles,
no network. Point it at your geography.
**Why it wins:** every other team's Leaflet map goes grey when the venue wifi
dies. Yours will not.

---

## Tier 3 — only if you are ahead at hour 14

9. **Anomaly detection over time** — flag wards/regions whose rate jumped this week.
10. **Duplicate clustering** — "these 14 reports are the same incident", using the
    embeddings you already compute for RAG. Cheap and looks smart.
11. **Priority queue with SLA countdown** — visible timers turning red. Pure
    frontend, surprisingly persuasive.
12. **Multilingual output** — translate the officer's action text into the
    citizen's language for the reply.

---

## What NOT to add

- A login/signup flow (already there if you need it; nobody scores it).
- An admin settings page. Zero marks, four hours.
- A second ML model to "ensemble". You will not have time to evaluate it, and an
  unevaluated ensemble is worse than one measured model.
- Dark/light theme toggle. Ship dark. It photographs better.
