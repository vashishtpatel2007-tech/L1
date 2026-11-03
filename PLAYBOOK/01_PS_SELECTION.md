# Picking the Problem Statement

You will have 30–60 minutes to choose. Choosing badly costs you the whole event,
and no amount of execution recovers from it. Spend 30 minutes, not 3 hours.

---

## The 60-second filter (apply to every PS first)

Kill any PS where you answer "no" to either:

1. **Can we show a working slice in 18 hours?** Not the full system — one
   complete path from real input to a decision a human would act on.
2. **Can we get or fabricate credible data today?** If the PS needs proprietary
   hospital records or live government APIs you cannot access, it is a trap.

Whatever survives, score it.

---

## The scoring grid

Score 1–5 on each. Weights reflect how judges actually allocate marks.

| Criterion | Weight | What a 5 looks like |
|---|---|---|
| **Demo-ability** | ×3 | A judge sees the magic in 90 seconds without explanation |
| **Problem pain** | ×3 | Real people lose money, time, health or safety today |
| **Our unfair advantage** | ×2 | Your team already knows this domain or stack |
| **Data availability** | ×2 | Public dataset exists, or synthetic data is credible |
| **Technical defensibility** | ×2 | There is a real model/algorithm, not just CRUD + an API call |
| **Scope realism** | ×2 | Core loop buildable in 12h, leaving 6h for polish |
| **Competition density** | ×1 | Most teams will NOT pick this (score high if it's unpopular) |

Highest weighted total wins. **If two are within 10%, pick the one with better
demo-ability.** Judges score what they see, not what you describe.

---

## Patterns that reliably win Indian national hackathons

1. **Accessibility / vernacular.** Voice-first in an Indian language, for users
   who cannot type English. Judging panels respond to inclusion with real force.
2. **Explainable decisions in high-stakes domains.** Health, credit, admissions,
   welfare. Showing *why* beats showing *what*.
3. **Offline-first / low-bandwidth.** Works on a 2G phone in a village. Almost
   no team does this and it is visibly hard.
4. **Closing a loop nobody closes.** Not "detect the problem" — detect, route,
   assign, verify resolution, and show the audit trail.
5. **A measurable baseline you beat.** "Humans do this at 64%, we do 91%" is the
   single most persuasive sentence available to you.

## Patterns that reliably lose

- Generic "AI chatbot for X" — every event has nine of them.
- Blockchain where a database would do; judges now read this as a red flag.
- Anything needing a hardware component you do not already own.
- Dashboards with no decision attached. Pretty charts, zero stakes.
- A wrapper around one LLM API call with no engineering underneath.

---

## When your team cannot agree

Do not vote. Voting picks the blandest option. Instead:

1. Each member writes the **demo script** for their preferred PS — the literal
   90 seconds they would show a judge — in 10 minutes flat.
2. Read them aloud. The one that sounds most impressive when *spoken* is the one
   that will actually win, because that is the medium judges consume.
3. If still tied, pick the one with better **failure tolerance**: which demo
   still works if the wifi dies and the LLM is unreachable?

Then commit. Changing the PS after hour 3 is how teams finish with nothing.

---

## Immediately after choosing: the 30-minute scope contract

Write these four lines on paper before anyone writes code. Everyone signs it.

```
DEMO SENTENCE : "We show a <user> doing <action> and getting <outcome> in <time>."
MUST HAVE     : the 3 things without which there is no demo
NICE TO HAVE  : the 3 things we build only if we are ahead at hour 14
WILL NOT BUILD: auth, admin panels, settings, onboarding, payments, mobile app
```

That last line matters most. **Every hour spent on a login screen is an hour
stolen from the thing that wins.** The kit already ships auth if you truly need
it; do not build a second one.
