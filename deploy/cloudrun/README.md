# Relief-RL API — Cloud Run deployment

This directory contains a parameterized Cloud Run service manifest. It intentionally uses placeholders rather than embedding project IDs or secrets.

## 1. Build the image

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
gcloud builds submit \
  --tag REGION-docker.pkg.dev/PROJECT_ID/relief-rl/relief-rl-api:latest \
  .
```

Replace `REGION` and `PROJECT_ID` with your Google Cloud values.

## 2. Create secrets

Store the production PostgreSQL URL and Google Maps API key in Secret Manager. Do not commit either value to Git.

```bash
echo -n 'YOUR_DATABASE_URL' | gcloud secrets create relief-rl-database-url --data-file=-
echo -n 'YOUR_GOOGLE_MAPS_KEY' | gcloud secrets create relief-rl-google-maps-key --data-file=-
```

If the secrets already exist, add a new version instead of recreating them.

## 3. Configure the manifest

Replace these placeholders in `service.yaml`:

- `REGION`
- `PROJECT_ID`

Grant the Cloud Run runtime service account access to both secrets.

## 4. Deploy

```bash
gcloud run services replace deploy/cloudrun/service.yaml \
  --region REGION
```

The application listens on port 8000 and exposes:

- `/health` — liveness
- `/ready` — readiness/database check
- `/docs` — FastAPI OpenAPI UI

## Production notes

- Restrict the Google Maps key by API and server-side usage where possible.
- Use a managed PostgreSQL provider with SSL/TLS.
- Keep `DEMO_MODE=false` in production.
- Do not expose PostgreSQL directly to the public internet.
- Review Cloud Run IAM before choosing public unauthenticated access.
- The included manifest is a starting point; verify quotas, CPU/memory, autoscaling, logging, and emergency-service requirements before production use.
