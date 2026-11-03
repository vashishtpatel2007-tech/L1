# The pitch

Most teams build well and pitch badly. The pitch is a separate artefact that
deserves its own 3 hours. Treat it that way.

---

## The structure (6 slides, no more)

| # | Slide | Time | Content |
|---|---|---|---|
| 1 | **The person** | 20s | One named human with a concrete problem. Not statistics yet. |
| 2 | **The scale** | 20s | Now the number. "This happens to 40 million people." |
| 3 | **What we built** | 15s | One sentence + one screenshot. Then stop talking and demo. |
| 4 | **DEMO** | 3 min | Live. The whole case rests here. |
| 5 | **How it works** | 45s | Architecture diagram + your accuracy vs baseline. |
| 6 | **Impact + what's next** | 30s | Derived numbers, honest limitations, the 3-month path. |

Total: about 5 minutes 30. Check your event's limit and cut slide 5 first if needed.

---

## Slide 1 is the whole game

Do not open with "India generates 62 million tonnes of waste annually."
Every team opens that way and judges stop listening by the second clause.

Open with a person:

> "Lakshmi runs a tea stall in Ward 14. The drain outside it has been overflowing
> for three weeks. She has called the municipal helpline four times. Nobody came,
> because her complaint was filed under 'general' and sits at position 4,000 in a
> queue nobody sorts. This is not a people problem. It is a routing problem."

Then the number. Then what you built. The emotional hook *precedes* the statistic
and makes the statistic land.

---

## The demo — the only part that really matters

### Rules

1. **Live, local, rehearsed.** No tunnel as primary, no slides pretending to be
   a product, no "imagine if".
2. **One driver**, who has rehearsed the exact click path. Not whoever is talking.
3. **Narrate the outcome, not the UI.** Say "it has routed this to the Health
   department and flagged it critical", not "now I'm clicking on the triage tab".
4. **Deterministic data.** Same seed every run. No surprises on stage.
5. **If something breaks, keep talking and move on.** Never debug in front of
   judges. The backup video exists for exactly this.

### A demo path that works (adapt to your PS)

1. **The problem, as data.** Dashboard. 158 open cases, 16 critical, unsorted.
   *"This is what the officer sees today. There is no order to it."*
2. **The magic moment.** Paste Lakshmi's actual complaint into Triage. Hit run.
   Structured decision appears: category, severity 5, routed department, action.
   *"Four seconds. That took a human officer eighteen minutes."*
3. **Why you can trust it.** Point at the reason chips. *"It is not a black box.
   It escalated because of 'sewage', 'children', and because the complaint has
   been open five days."*
4. **Scale it.** Hit Simulate. Records stream in live, dashboard updates itself.
   *"This is 500 complaints an hour. No extra staff."*
5. **The hard question, pre-answered.** Flip offline mode. *"Ward offices have
   2G. Watch what happens when the network dies."* Everything keeps working.
6. **The money.** Impact tab. *"Eighteen minutes saved per case, 766 districts,
   ₹926 crore a year — and every one of these assumptions is editable. Tell me
   which one you doubt and I'll change it right now."*

That last line is a flex, and it works, because almost nobody can do it.

---

## The three questions you will be asked

Prepare 30-second answers. Rehearse them out loud.

**1. "How is this different from what exists?"**
Name the incumbent honestly, then name the one thing it cannot do.
> "The existing CPGRAMS portal accepts complaints but does not prioritise them —
> routing is manual and keyword-based. We rank by urgency and route automatically,
> and we work offline, which matters because 60% of these complaints originate
> from areas with unreliable connectivity."

**2. "Will this actually work at scale / in the real world?"**
Answer with a number and a constraint, never with optimism.
> "The model is 12MB and runs on CPU in 40ms, so it deploys on existing district
> hardware without GPUs. The real constraint isn't technical — it's that each
> state uses different category taxonomies, so onboarding a new state is about a
> week of mapping work. That's the honest blocker."

**3. "What doesn't work yet?"**
Answer this *honestly and specifically*. It is a trust test, not a trap.
> "Three things. Our weakest class is 'education' at 74% because we only had 31
> labelled examples. We don't handle code-mixed Hindi-English well yet. And we
> have no feedback loop — an officer can override the model, but we don't retrain
> on those corrections. That's the first thing we'd build next."

**Never say "nothing" or "we thought of everything". It reads as inexperience.**

---

## Slide 5 — how it works

One diagram. Boxes and arrows. Show: input channels → your pipeline → model →
decision → interface. Mark clearly which parts run offline.

Under it, one table:

| | Baseline | Ours |
|---|---|---|
| Accuracy | 64% | 91% |
| Time per case | 18 min | 2 min |
| Works offline | No | Yes |

That table does more work than any amount of architecture detail.

---

## Delivery

- **Two speakers maximum.** One for story and impact, one for the demo. More
  than two looks disorganised and wastes handover seconds.
- **Stand up.** Both of you. Energy is contagious and judges are on their ninth pitch.
- **Rehearse three times with a timer.** Running over is a visible failure of
  judgement before you have said anything about your work.
- **Nobody says "basically", "simply", or "just a".** You are diminishing your
  own 24 hours of work with filler.
- **End with the ask, not a thank-you.** "We need access to one district's real
  complaint data to validate this at scale. That's what we want next."

---

## The final 15 minutes before you present

- Close Slack, mail, notifications. Full screen. Do Not Disturb on.
- Charger plugged in. Brightness up.
- `bash scripts/preflight.sh` one last time. 9/9.
- Backup video open in a second tab, and on a phone.
- Browser on the opening frame of the demo, already seeded.
- One person holds the laptop open at the demo screen while the other speaks.
