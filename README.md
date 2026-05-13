# FixerFlip AI Renovation API

FastAPI service for estimating renovation scope, cost, and timeline from property facts plus an optional listing photo.

The current codebase focuses on a renovation workflow:

- analyze one property image for visible condition issues
- estimate renovation class, cost range, and timeline
- generate a renovated preview image as a side effect when image editing is enabled
- expose simple health checks and OpenAPI docs for local/dev and deployment use

## What This Service Does

The main endpoint is `POST /api/v1/renovation/estimate`.

You send:

- property facts such as `sqft`, `beds`, `baths`, `year_built`, and `zip_code`
- either an `image_url` or a manual `condition_score`
- optional market context like `avg_area_price_per_sqft`
- optional renovation preferences like `desired_quality_level`, `renovation_elements`, and `user_inputs`

The service returns:

- `renovation_class`
- `estimated_renovation_range`
- `estimated_timeline`
- `suggested_work_items`
- `confidence_score`
- `explanation_summary`
- `room_type`
- `condition_score` (0–100: from vision when `image_url` is sent, otherwise the request’s `condition_score`)
- `renovated_image_url`

There is also a lighter endpoint, `POST /api/v1/renovation/image-condition`, for image-only condition scoring.

## How The Pipeline Works

1. Request validation and normalization happen in `app/services/renovation_payload_validator.py`.
2. If `image_url` is present, the service runs two async tasks in parallel:
   - OpenAI vision analysis to detect room type and visible issues
   - OpenAI image editing to create a renovated preview
3. The estimate engine converts the condition result plus property inputs into:
   - renovation class (`Cosmetic`, `Moderate`, `Heavy`, `Full Gut`)
   - cost line items
   - total min/max estimate
   - timeline estimate
   - work item suggestions
   - confidence score
4. The public response mapper trims the rich internal estimate into the production API contract (including the condition score used for the estimate).
5. If storage is configured, the edited image is uploaded to an S3-compatible bucket and the public response returns it as `renovated_image_url` when available.

If no `image_url` is supplied, the service requires a manual `condition_score` and skips vision analysis.

## Tech Stack

- Python 3.11
- FastAPI
- Pydantic v2
- Uvicorn / Gunicorn
- OpenAI Responses API for vision
- OpenAI Images API for renovation previews
- Redis for optional image download caching
- S3-compatible object storage for optional preview uploads

## Project Layout

```text
app/
  api/v1/                  Versioned routes
  core/                    Settings, logging, error handlers, cache helpers
  engine/renovation_engine/Condition scoring, cost estimation, image edit logic
  middleware/              Request ID and security headers
  prompts/                 Prompt templates for vision and image edits
  schemas/                 Request/response models
  services/                Orchestration, payload validation, response mapping, storage
tests/                     API contract tests
ui/                        Simple static tester
```

## Local Development

### Option 1: Cross-platform manual setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/api/v1/health`

### Option 2: Use the included dev script

`run.dev` bootstraps `.venv`, installs dependencies, creates a minimal `.env`, and starts Uvicorn with reload.

```bash
bash run.dev
```

This script is Unix-oriented. On Windows, use Git Bash/WSL or the manual setup above.

## Environment Variables

### Core

| Variable | Purpose | Default |
| --- | --- | --- |
| `PROJECT_NAME` | FastAPI app title | `FastAPI Production App` |
| `ENVIRONMENT` | Controls production behavior | `local` |
| `DEBUG` | Enables debug-friendly behavior | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_JSON` | Emit JSON logs | `false` |
| `ENABLE_OPENAPI` | Expose `/docs` and OpenAPI | `true` |
| `BACKEND_CORS_ORIGINS` | Production CORS allowlist | `[]` |
| `TRUSTED_HOSTS` | Optional host header allowlist | `[]` |

### OpenAI

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for vision and image editing |
| `OPENAI_VISION_ENABLED` | Enables image condition analysis |
| `OPENAI_VISION_MODEL` | Primary vision model |
| `OPENAI_MODEL` | Secondary/fallback model hint |
| `OPENAI_IMAGE_EDIT_MODEL` | Model for edited renovation previews |

### Caching and image downloads

| Variable | Purpose |
| --- | --- |
| `REDIS_URL` | Optional Redis cache |
| `REDIS_CACHE_TTL_SECONDS` | TTL for cached image downloads |
| `IMAGE_DOWNLOAD_REFERER` | Override referer for MLS/CDN image hosts that block generic requests |

### Storage upload

| Variable | Purpose |
| --- | --- |
| `STORAGE_ENDPOINT_URL` | S3-compatible endpoint |
| `STORAGE_REGION` | Storage region |
| `STORAGE_BUCKET_NAME` | Target bucket |
| `STORAGE_ACCESS_KEY_ID` | Access key |
| `STORAGE_SECRET_ACCESS_KEY` | Secret key |
| `STORAGE_PUBLIC_BASE_URL` | Optional public base URL |
| `STORAGE_RENOVATED_IMAGE_PREFIX` | Object key prefix for generated previews |
| `STORAGE_PRESIGNED_URL_TTL_SECONDS` | Presigned URL lifetime |

## API Endpoints

### `GET /`

Minimal root endpoint:

```json
{"status":"ok"}
```

### `GET /health`

Root-level health endpoint used by deployment platforms such as Railway.

Example response:

```json
{
  "status": "ok",
  "service": "FastAPI Production App",
  "environment": "local"
}
```

### `GET /api/v1/health`

Versioned health endpoint with the same response shape.

### `POST /api/v1/renovation/image-condition`

Single-image condition analysis. In the current implementation, `image_url` is accepted as a query parameter.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/renovation/image-condition?image_url=https://example.com/property.jpg"
```

Typical response:

```json
{
  "condition_score": 72,
  "issues": ["outdated cabinets", "paint wear", "floor damage"],
  "room_type": "kitchen",
  "issue_details": [
    {
      "type": "outdated cabinets",
      "severity": "moderate",
      "confidence": 0.88
    }
  ]
}
```

If vision is disabled or unavailable, the service falls back to a conservative default score.

### `POST /api/v1/renovation/estimate`

Primary estimation endpoint.

Required fields:

- `sqft`
- `beds`
- `baths`
- one of:
  - `image_url`
  - `condition_score`

Useful optional fields:

- `year_built`
- `zip_code`
- `listing_price`
- `listing_description`
- `days_on_market`
- `avg_area_price_per_sqft`
- `years_since_last_sale`
- `permit_years_since_last`
- `desired_quality_level`: `cosmetic`, `standard`, `premium`, `luxury`
- `renovation_elements`
- `user_inputs`
- `reference_image_url`

Manual fallback example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/renovation/estimate" \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 1600,
    "beds": 3,
    "baths": 2,
    "zip_code": "94103",
    "condition_score": 62,
    "issues": ["old tiles", "paint wear", "roof damage"],
    "room_type": "kitchen,exterior",
    "labor_index": 1.10,
    "material_index": 1.05,
    "desired_quality_level": "standard"
  }'
```

Image-based example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/renovation/estimate" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/property.jpg",
    "sqft": 1400,
    "beds": 3,
    "baths": 2,
    "zip_code": "78704",
    "desired_quality_level": "premium",
    "renovation_elements": ["kitchen", "paint"],
    "user_inputs": "Keep the layout but modernize the kitchen and repaint walls."
  }'
```

Typical response:

```json
{
  "renovation_class": "Moderate",
  "estimated_renovation_range": "$45,000 - $72,000",
  "estimated_timeline": "6-10 weeks",
  "suggested_work_items": [
    "paint",
    "flooring",
    "kitchen update"
  ],
  "confidence_score": "82%",
  "explanation_summary": "Property is classified as Moderate rehab based on kitchen condition and detected issues.",
  "room_type": "kitchen",
  "condition_score": 72,
  "renovated_image_url": "https://example.com/renovated.png"
}
```

Validation errors return HTTP `422` with a `VALIDATION_ERROR` code.

## Estimation Logic Notes

The cost engine is rule-based and deterministic after inputs are prepared. It uses:

- issue severity and issue count
- requested quality level
- labor and material indices
- room type and suggested scope categories
- market gap signal (`listing_price` vs area price-per-sqft context)
- age signal (`year_built`, years since sale, years since permit)
- selected renovation elements or user-entered scope hints

The public response is intentionally compact even though the internal estimate model contains richer details such as line items, assumptions, impacted elements, and confidence labels.

## Simple UI

A lightweight tester is included at:

- `ui/renovation_v1_simple.html`

When the app is running, it is also served from:

- `http://127.0.0.1:8000/ui/renovation_v1_simple.html`

## Tests

Run the test suite with:

```bash
pytest
```

Current tests focus on the renovation estimate response contract and validation error shape.

## Deployment

This repo already includes:

- `Dockerfile`
- `Procfile`
- `railway.json`
- `nixpacks.toml`

Deployment details reflected in the codebase:

- the container entrypoint runs `gunicorn` with `uvicorn.workers.UvicornWorker`
- Railway health checks target `/health`
- in non-production environments, CORS is wide open for easier local UI testing
- in production/staging, CORS and trusted hosts are restricted through settings

## Notes And Current Boundaries

- This repo currently documents and implements renovation endpoints only.
- The previous README mentioned a valuation endpoint, but that endpoint is not present in the current codebase.
- The image-condition route currently accepts `image_url` as a query parameter, not a JSON body.
- Renovated preview uploads happen as part of the estimate flow when image editing and storage are configured, and the uploaded URL is returned as `renovated_image_url`.
