# Merge protocol

Two people building separately is fine. Merging on vibes is not. Follow this exactly.

---

## Before anyone writes code (hour 0, 10 minutes)

1. Open `contract/api_contract.json`. Change **only** the `domain` block:
   `object`, `object_plural`, `categories`. Leave every response shape alone.
2. Commit it and push: `git add contract/ && git commit -m "freeze contract" && git push`
3. **Say out loud: "contract is frozen."** From here, changing it requires
   telling both people. A silent contract change is the one thing that will
   cost you the event.
4. Paste `PROMPTS/BACKEND_PROMPT.md` (filled in) to your friend.
5. Paste `PROMPTS/FRONTEND_PROMPT.md` (filled in) to me.

---

## While building

**Branches:**
```bash
git checkout -b backend     # your friend
git checkout -b frontend    # you + me
```

**The frontend is never blocked.** The kit's backend is already running and
already satisfies the contract, so I build against a real, working API from
minute one. When your friend's backend lands, it is a base-URL swap and nothing else.

---

## The merge (hour N — budget 20 minutes, it usually takes 5)

### Step 1 — test their backend ALONE, before any git operation

```bash
# put their code in a scratch folder, install, run it on a different port
cd /tmp/friend-backend && pip install -r requirements.txt
uvicorn main:app --port 9000
```

```bash
python scripts/verify_contract.py --base http://127.0.0.1:9000
```

- **Exit code 0** → proceed to step 2.
- **Anything else** → do NOT merge. Copy the failure lines verbatim, paste them
  back into the AI builder, and say: *"Fix exactly these and change nothing
  else."* Re-run. Repeat until it is 0.

> This is the whole point of the gate. You are never arguing about whether the
> backend is right — a script decides, in about four seconds.

### Step 2 — merge

```bash
git checkout main
git merge backend     # no frontend files touched -> no conflicts
git merge frontend    # no backend files touched -> no conflicts
```

Conflicts should be impossible: the two branches touch disjoint directories
(`apps/api/**` vs `apps/web/**`). **If git reports a conflict, someone edited
outside their lane — stop and find out what before resolving anything.**

### Step 3 — verify the merged whole

```bash
./dev.sh
python scripts/verify_contract.py    # against the merged backend on :8000
bash scripts/preflight.sh            # 9/9
```

Then open the UI and click through every screen. **The contract test proves the
shapes are right; only your eyes prove the product works.**

### Step 4 — tag the known-good state

```bash
git tag good-1 && git push --tags
```

Do this after **every** successful merge. When something breaks at 4am,
`git reset --hard good-N` is a 3-second recovery instead of a 40-minute panic.

---

## The rule that makes this work

**Lane discipline.** One person owns `apps/api/**`, the other owns `apps/web/**`.
Nobody edits the other's lane, ever — not even "a quick fix".

`contract/api_contract.json` is owned by **both**, which means it is changed by
**neither** without the other being told out loud.

---

## If you are short on time

Honestly: you already have a backend that passes 22/22 and is tested end to end.
If the merge is not green within 20 minutes, **abandon the parallel backend and
extend the existing one.** Sunk cost is not a reason to ship a broken demo.

Keep your friend productive on something that cannot collide:
the demo dataset, the labelled eval set for the accuracy number, the slides, or
the backup video. All of those are genuinely valuable and carry zero merge risk.
