# Installed skills (from ECC v2.2.3)

Source: https://github.com/affaan-m/ECC (MIT)

## What was installed

17 skills, project-scoped under `.claude/skills/`. They are committed, so every
teammate gets them automatically on `git clone` — nobody has to install anything.

| Area | Skills |
|---|---|
| Backend | `fastapi-patterns` `api-design` `backend-patterns` `python-patterns` `python-testing` |
| Frontend | `react-patterns` `react-performance` `typescript-patterns` `frontend-patterns` `frontend-design-direction` `design-system` `frontend-a11y` |
| ML | `pytorch-patterns` `mle-workflow` `ai-regression-testing` |
| Workflow | `git-workflow` `deep-research` |

These map 1:1 onto the kit's actual stack (FastAPI + SQLModel backend,
React/Vite/TS frontend, PyTorch transfer learning).

## Why not the full install

ECC ships 314 skills, 71 agents and 98 commands. The official installer's
`--skills` flag does not install only what you ask for: requesting 17 resolves
core modules and produces **185 skills, 71 agents and 98 commands** (729 file
operations), including `bengali-nlp`, `homelab-pihole-dns`, `cisco-ios-patterns`,
HIPAA compliance and Perl tooling.

None of that applies to this project, and a crowded skill namespace makes skill
selection noisier exactly when speed matters. So the 17 relevant skill
directories were copied directly instead. 268 KB, no agents, no commands,
no hooks, no rules tree.

## Safety audit performed

- Cloned to an isolated directory and read before anything ran.
- `install.sh` and `scripts/install-apply.js` reviewed: no destructive
  operations, no network calls in hooks.
- Hook runtime deliberately NOT installed — automatic hooks firing on tool calls
  is not a risk worth taking mid-hackathon.
- All 17 skills confirmed self-contained (no references outside their own folder).
- Only one bundled executable found, `frontend-design-direction/scripts/check-evidence.js`:
  read in full. It is a local JSON validator with explicit path-traversal guards,
  no network and no exec. Kept.
- Existing global config backed up to `~/.claude-backup-<timestamp>/` before any change.
- Global `~/.claude/` was left untouched; this install is project-scoped only.

## To add more later

```bash
git clone --depth 1 https://github.com/affaan-m/ECC.git /tmp/ecc
cp -R /tmp/ecc/skills/<skill-name> .claude/skills/
```

Read the skill's `SKILL.md` first, and check for bundled scripts:

```bash
find .claude/skills/<skill-name> -type f ! -name "*.md"
```
