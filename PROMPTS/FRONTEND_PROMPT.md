# FRONTEND PROMPT — paste this to me (Claude Code)

Fill the `{{...}}` placeholders. Everything below is already tuned; the
constraints are what keep it mergeable with the independently-built backend.

---

```
You are a senior frontend engineer building the judge-facing interface for a
24-hour national hackathon. This screen IS the product as far as scoring goes.
Build it to be demoed live, on a projector, by a nervous person, at hour 23.

=====================================================================
CONTEXT
=====================================================================
Product: {{ONE_SENTENCE_DESCRIPTION}}
The user on stage is: {{WHO_USES_IT — e.g. a ward officer triaging complaints}}
The judge must understand what they are seeing within 10 seconds.

The backend is being built in parallel by someone else against the frozen
contract in contract/api_contract.json. Read that file FIRST and treat it as
the only source of truth for every response shape. Do not invent fields. If you
need a field that is not in the contract, say so instead of inventing it —
changing the contract requires telling the backend owner.

=====================================================================
HARD CONSTRAINTS
=====================================================================
1. Stack is fixed: React 18 + TypeScript + Vite + Tailwind v4 + Recharts.
   Do NOT add Next.js, a component library, Redux, React Query, or a router
   package. Hash-based routing only (location.hash) so it survives a refresh
   and can deep-link a single tab mid-demo.
2. All API calls go through src/lib/api.ts. No fetch() anywhere else.
3. Dev talks to /api (Vite proxies to :8000). Production is same-origin, because
   the API serves the built bundle. The same code must work in both.
4. Dark theme only. No theme toggle. It photographs better on a projector.
5. Must render correctly at 1280x720 (typical projector) AND at 390px wide
   (a judge opening the tunnel URL on their phone). Test both.
6. Zero console errors or React warnings. A red error in devtools during
   judging reads as "this is broken".
7. npx tsc --noEmit must pass clean. No `any` in component props.

=====================================================================
SCREENS — build exactly these, in this order of priority
=====================================================================

1. DASHBOARD  (the opening frame of the demo — it must look alive)
   - Four KPI tiles across the top: {{FOUR_HEADLINE_NUMBERS}}
     Each tile: label, large coloured number, one line of context beneath.
   - Bar chart of records by category. Pie/donut of records by status.
   - A geographic view of the records.
     *** Use an inline SVG scatter plot, NOT Leaflet or any map-tile library. ***
     Tiles require the network; when venue wifi dies a tile map renders as a grey
     square on stage. The SVG version always works. Size and colour each point by
     severity.
   - A live feed panel driven by the websocket at /ws, newest first, capped at 12,
     with a brief highlight animation on arrival.
   - Buttons: "Simulate live intake" (POST /demo/simulate) and "Reseed demo data"
     (POST /demo/seed). These are demo controls and must be one click each.
   - A connection pill showing Live / Reconnecting.
   All of this must come from GET /items/stats in ONE request. Do not fan out.

2. {{PRIMARY_AI_SCREEN — e.g. TRIAGE}}
   Left: a large textarea pre-filled with a realistic sample input, plus a
   "Run" button and a "Reset sample" button. The sample must be pre-filled —
   nobody types a paragraph on stage.
   Right: the structured decision from POST /ai/triage, rendered as:
     - severity chip, category chip, confidence percentage
     - "Routed to" department, "Action", "Summary"
     - *** A "Why this severity" row rendering the `reasons` array as chips
       showing each matched signal and its weight. This explainability view is
       the single highest-value element on the screen — it is what separates us
       from every other team. Do not drop it, do not bury it. ***
     - If duration_detected is non-null, show "Unresolved duration: X -> escalated".
   Show the `engine` field as a small badge ("LLM" vs "rules engine") — being
   transparent about which path ran reads as engineering maturity, not weakness.

3. {{SECONDARY_SCREEN — e.g. KNOWLEDGE / VISION / UPLOAD}}
   {{WHAT_IT_DOES}}
   If it shows model output, always render the supporting evidence alongside it
   (retrieved passages with similarity scores, or an attention heatmap).
   An answer with citations is trustworthy; an answer without them is a chatbot.

4. IMPACT
   KPI tiles for the derived impact numbers from GET /metrics/impact, plus an
   editable form of the underlying assumptions with a "Recompute" button that
   POSTs the overrides and re-renders.
   This exists so that when a judge says "that number seems optimistic", we change
   it in front of them and show the conclusion still holds. Make the inputs
   obviously editable.

5. SYSTEM
   Render GET /ai/health as a grid of capability cards, each showing ready/error
   and its detail. Include a prominent "Offline mode" toggle hitting
   POST /demo/offline. Poll every 4 seconds.
   This screen is a planned demo beat: we flip offline mode on stage to show the
   system degrading gracefully instead of dying. Make the toggle easy to hit.

=====================================================================
RESILIENCE — assume everything fails at the worst moment
=====================================================================
- Every API call wrapped in try/catch. A failed call shows an inline message in
  that panel only. One dead endpoint must never blank the whole page.
- The websocket must auto-reconnect with a delay. A dropped socket must heal
  itself without anyone touching the keyboard.
- Every list and chart needs a real empty state with a next action
  ("No records yet — hit Reseed demo data"). A bare "No data" during judging is
  a self-inflicted wound.
- Loading states on every async action. Buttons disable while in flight so a
  nervous double-click cannot fire two requests.
- Never render raw error objects or stack traces to the screen.

=====================================================================
VISUAL DIRECTION
=====================================================================
- Dark navy background (#080b14), panels one step lighter with a subtle border
  and ~14px radius. Accent blue, success teal, warning amber, danger red.
- Numbers are the hero. KPI values large and coloured; labels small and muted.
- Generous spacing. A dense screen reads as a prototype; a calm one reads as a
  product.
- Severity colour scale consistent everywhere: 1 teal, 2 blue, 3 amber,
  4 orange, 5 red. Never use colour alone — always pair it with the number or label.
- Smooth but fast transitions. Nothing over 200ms; slow animation wastes demo seconds.

=====================================================================
DELIVERABLE
=====================================================================
- Complete, running code. No TODOs, no placeholder components.
- npx tsc --noEmit passes.
- npm run build succeeds.
- Then actually open it in a browser, screenshot each screen, and verify it
  renders correctly before telling me it is done. Do not report success from the
  fact that the build command exited zero.
- Finally, run: python scripts/verify_contract.py
  and confirm the backend still satisfies the contract.
```

---

## Why this prompt is shaped this way

- It points me at `contract/api_contract.json` rather than describing shapes
  inline, so the frontend and backend cannot drift.
- It forbids Leaflet explicitly, because a tile map is the most common way a
  hackathon dashboard dies on venue wifi.
- It names the explainability row as non-droppable, because that is the element
  judges reward and the one most likely to get cut under time pressure.
- It demands I verify visually, not just that the build exited zero.
