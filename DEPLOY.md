# Deploying DisasterMind AI

This repo ships two ways to run the full stack (backend + frontend) with
Docker. Both were validated in this environment as far as it's possible to
without a Docker daemon or a real server: the frontend production build was
run and passed (`npm run build`), the backend's 91 tests pass, and both
compose files were YAML-validated and cross-checked against the actual
routes/env vars the code expects. **`docker build` / `docker compose up`
itself has not been run** (no Docker available here) -- do a smoke test on
your VPS before pointing DNS at it.

## Option A -- quick HTTP test (no domain needed)

```bash
git clone <your-repo-url> disastermind-ai && cd disastermind-ai
cp .env.example .env        # edit secrets if you're not using DEMO_MODE
docker compose up -d --build
```

Visit `http://<server-ip>/`. This runs:
- `api` -- the FastAPI backend (PPO + QR-DQN, not exposed to the host directly)
- `web` -- nginx serving the built frontend on port 80, reverse-proxying
  `/api/*`, `/ws/*`, `/health`, `/ready` to `api` (see `frontend/nginx.conf`)

Check both are healthy:
```bash
docker compose ps                 # both should show "healthy"
curl http://localhost/health       # backend health, via the nginx proxy
```

## Option B -- real domain with HTTPS

```bash
export DOMAIN=disastermind.example.com
docker compose -f docker-compose.prod.yml up -d --build
```

This adds a `caddy` service in front of `web`/`api` that automatically
gets and renews a free Let's Encrypt certificate for `$DOMAIN`. Point your
domain's A/AAAA record at the server's IP *before* starting Caddy, or
certificate issuance will fail (Let's Encrypt needs to reach port 80 on
that hostname).

Requirements: DNS already pointing at the server, and ports 80+443 open.

## Updating a running deployment

```bash
git pull
docker compose up -d --build      # or -f docker-compose.prod.yml
```

## What "live" means here

- The PPO and QR-DQN checkpoints ship in `rl/checkpoints/` and load on
  container start -- no training required to go live, but they're the
  same weights checked into the repo, not further-trained for production.
- `DEMO_MODE=true` (the `docker-compose.yml` default) skips requiring a
  database; the Postgres-backed history/analytics features need a real
  `DATABASE_URL` in `.env` (see `docker-compose.postgres.yml` for a local
  Postgres you can adapt) and `DEMO_MODE=false`.
- `AUTH_SECRET` / `ADMIN_EMAIL` / `ADMIN_PASSWORD` in `.env.example` are
  placeholders -- replace them before any real deployment; the repo will
  run with the placeholders, but anyone who reads this file could log in
  as admin if you don't change them.
- Nothing here provisions the VPS itself (firewall, non-root user, fail2ban,
  automatic OS updates) -- that's standard server hardening independent of
  this app and out of scope for what I can verify from this sandbox.

## Cloud Run (already partially set up)

`deploy/cloudrun/service.yaml` targets the backend only and predates this
compose setup -- it hasn't been reconciled with the QR-DQN/env additions
above. If you want Cloud Run specifically (backend + a static frontend
bucket/CDN instead of the nginx container), say so and I'll update it
rather than leave two half-matching configs.
