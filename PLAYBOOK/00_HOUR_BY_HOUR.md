# The 24-hour timeline

Times assume the PS drops at **hour 0**. Adjust the clock, keep the ratios.

---

## Hour 0–1 — Choose and commit
- Apply the scoring grid in `01_PS_SELECTION.md`. **30 minutes maximum.**
- Write the scope contract (demo sentence / must / nice / will-not-build).
- `cp .env.example .env`, set `APP_NAME` and `TAGLINE`. Takes 60 seconds and
  makes every screenshot look intentional from hour 1.

## Hour 1–2 — Skeleton with fake data
- **A**: rename `Item` in `apps/api/core/db.py` to your domain noun. Adjust
  fields. Rewrite the seeder lists in `routers/demo.py` to your domain.
- **B**: find the dataset. Kaggle, data.gov.in, HuggingFace. If none exists,
  decide *now* to generate synthetic data and say so openly in the pitch.
- **C**: `./dev.sh`, delete the tabs you do not need, rename the rest.
- **D**: repo, README, and the one-line pitch. Start the slide outline.

> At hour 2 you should already have a running app with your domain's data in it.
> The kit exists so that this is true. If it is not, stop and fix that first.

## Hour 2–6 — First vertical slice
Pick the **single most impressive path** and make it work end to end, ugly.

- Input → model/logic → stored decision → visible on screen.
- Use the rules engine (`ai/rules.py`) as your day-one "model". It works
  immediately, it is explainable, and it is a real baseline to beat.
- **Gate at hour 6: the slice works with fake data.** If not, cut scope now.

## Hour 6–12 — Make it real
- **B**: train the actual model. `ml/train_cnn.py` or `/ai/tabular/train`.
  **Measure against the rules baseline.** That delta is your headline number.
- **A**: real ingestion, real aggregates, the audit trail.
- **C**: charts, realtime feed, the explainability view. This is what judges see.
- **D**: write the demo script as a numbered list of clicks. Yes, already.

> **Gate at hour 12: real data through the real model.** Whatever accuracy you
> have at hour 12 is roughly what you will present. Improving a model from 86%
> to 89% in hour 18 is worth far less than a demo that never stutters.

## Hour 12–16 — Depth and the USP
- Add exactly **one** thing from `05_USP_BANK.md`. One. Chosen for how it looks.
- Fill the empty states. A screen that says "No data yet" during judging is a
  self-inflicted wound.
- Sleep shifts begin. Two awake at all times.

## Hour 16 — FEATURE FREEZE
Nothing new after this point. Everything from here is polish, stability, rehearsal.

## Hour 16–20 — Make it unbreakable
- `bash scripts/preflight.sh` — must be 9/9.
- **Pull the wifi and run it again.** Everything except the LLM must still pass.
  Flip offline mode in the System tab and check the demo path still works.
- Seed the exact data you will demo with. Deterministic seed: same every run.
- Fix every console error. A red error in devtools during judging reads as "broken".
- **Record the backup video.** Full demo, screen recording, with audio. If the
  laptop dies, you still present. Teams have won on video.

## Hour 20–23 — Pitch
- Slides: 6 of them, structure in `04_PITCH.md`.
- **Rehearse three times, out loud, with a timer.** Not in your head.
- Rehearsal 2: someone unplugs the network mid-demo. Recover gracefully.
- Rehearsal 3: someone asks the three hardest questions. Answer in under 30s each.
- Write the final numbers on the impact page and make sure you can derive each
  one aloud from its assumptions.

## Hour 23–24 — Stop
- `git push`. Tag it. Export the backup video to a phone as well as the laptop.
- Close every unrelated tab, silence notifications, set the screen to the
  opening frame of the demo.
- Eat something. Drink water. Do not touch the code.

---

## Rules that save events

1. **The demo runs locally.** Tunnels are for judges' phones, never your primary.
2. **Deterministic seed data.** Your demo must look identical every single run.
3. **One person drives the laptop.** Not whoever is talking. The driver rehearses.
4. **Never apologise for what is missing.** Judges only know what you tell them.
5. **If something breaks mid-demo, keep talking and move to the next screen.**
   Silence is the only unrecoverable failure.
