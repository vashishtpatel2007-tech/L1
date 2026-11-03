# Team split for 24 hours

## The principle

Parallelism dies at integration. The most common failure is four people building
four things that meet for the first time at hour 20 and do not fit.

So: **integrate at hour 6, not hour 20.** Build the skeleton end to end first
with fake data, then each person deepens their layer behind a frozen interface.

---

## Roles (4-person team)

| Role | Owns | Hour 0–6 | Hour 6–16 | Hour 16–24 |
|---|---|---|---|---|
| **A — Backend / Data** | API, DB, data pipeline | Freeze the schema, seed realistic data | Real endpoints, ingestion, aggregates | Harden, cache, kill N+1s |
| **B — ML / AI** | Models, eval, explainability | Find the dataset, label 200 rows, baseline | Train, measure, beat the baseline | Freeze weights, write the eval slide |
| **C — Frontend** | The thing judges look at | Nav + mock screens off fake JSON | Wire to real API, charts, realtime | Polish, empty states, mobile width |
| **D — Integration / Pitch** | Glue, demo, story, slides | Scope contract, repo, deploy path | Fills whichever lane is behind | Demo script, slides, rehearsal |

**Role D is the one teams skip and the one that wins events.** They own the
deck, the demo script, the recorded backup video, and they are the floating
resource who rescues whichever lane is drowning.

### 3-person team
Merge A and D. The backend owner also owns the pitch. Cut one AI capability.

### 5-person team
Split C into "screens" and "data-viz + realtime". Do **not** add a second ML
person — two people on one model is slower than one, not faster.

---

## Hard rules

1. **Interfaces are frozen at hour 6.** The API response shapes in
   `apps/api/routers/` are a contract. Change one → announce it out loud.
2. **No one works more than 2 hours without pushing.** Long-lived branches
   are how you lose four hours at 4am.
3. **B never blocks C.** The frontend codes against the mock response from hour
   0. When the real model lands, it is a drop-in swap.
4. **Everyone commits to `main`.** Small team, 24 hours. Branch protection and
   PR review cost you more than they save.
5. **Sleep in shifts, 90 minutes each, starting hour 12.** Two people awake at
   all times. A team that stays up for 24 straight hours ships worse code from
   hour 16 onward and pitches badly. This is not softness; it is throughput.

---

## Communication

- One shared terminal doing `./dev.sh`, one person responsible for it being green.
- Status in one line every 2 hours: **done / doing / blocked**. Nothing else.
- "Blocked" is said out loud immediately, never queued for the next check-in.
- Disagreement lasting more than 5 minutes → Role D decides, work continues.

---

## The integration checkpoints

| Hour | Gate | If you fail it |
|---|---|---|
| 6 | End-to-end skeleton works with fake data | Cut a feature now, not later |
| 12 | Real data flows through the real model | Freeze model quality, ship what you have |
| 16 | **Feature freeze.** Nothing new after this. | — |
| 18 | Full demo rehearsed once, start to finish | Cut whatever broke |
| 20 | Backup video recorded | — |
| 22 | Rehearsed 3 times, slides done | — |
| 24 | Present | — |

**Hour 16 feature freeze is non-negotiable.** The last 8 hours are for making
what exists work perfectly, not for adding more. Teams that add features at
hour 20 demo a crash.
