# Failure drills

Run each of these at least once before hour 20. The ones you have rehearsed
cannot hurt you.

| Failure | Drill | Your response on stage |
|---|---|---|
| **Venue wifi dies** | Turn wifi off, run `scripts/preflight.sh`. Everything but the LLM must pass. | Flip offline mode in the System tab. *"This is the rural deployment mode."* Turn it into the USP. |
| **LLM API down / rate-limited / key dead** | Set a bogus key, run the demo. | Nothing happens — `llm.complete()` falls back to the rules engine silently. The response shape is identical. |
| **Laptop dies** | Present from the backup video for 60 seconds. | *"I'll walk you through the recording while we get this back."* Never stop talking. |
| **Port 8000 already in use** | `lsof -ti:8000 \| xargs kill -9` | Rehearse the command. Know it by heart. |
| **Database corrupted / empty** | `rm data/app.db`, restart, POST `/demo/seed`. | 10 seconds. Practise it. |
| **Frontend white screen** | Open devtools, check console. | Serve the built bundle from the API instead: `cd apps/web && npm run build`, then just `:8000`. One process, fewer failure modes. |
| **Model file missing** | `mv data/models/cnn.pt /tmp/` and hit the endpoint. | Degrades to ImageNet tier, then to the heuristic. Never 500s. |
| **Someone's commit breaks main** | `git stash`, `git reset --hard <last good>`. | Know the last good SHA at all times. Tag it: `git tag good-1`. |
| **A judge asks for the code** | Have the repo open in a second window. | Show `ai/rules.py` or `ml/train_cnn.py` — readable, commented files, not minified bundles. |

---

## The two commands to memorise

```bash
bash scripts/preflight.sh                  # am I safe to present?
lsof -ti:8000 | xargs kill -9              # port is stuck
```

---

## Golden rule

**Every failure mode above is survivable except silence.** If the screen freezes,
keep narrating what the judge would be seeing. You are being scored on the idea,
the engineering and your composure — and you control all three even when the
laptop does not.
