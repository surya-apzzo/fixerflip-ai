# FixerFlip Renovation Image Condition API

This repo contains the **Renovation Image Condition** module only.

## Run

```bash
./run.dev
```

Open docs:
- `http://127.0.0.1:8000/docs`

## Endpoint

### Image condition score (0–100)
`POST /api/v1/renovation/image-condition`

- **Input**: JSON with a single `image_url` (from ATTOM / MLS / RealEstateAPI)
- **Output**:

```json
{
  "condition_score": 72,
  "issues": ["outdated kitchen cabinets", "wall stains", "old carpet"],
  "room_type": "kitchen"
}
```

Example `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/renovation/image-condition" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example-cdn.com/property/photo1.jpg"
  }'
```

### Renovation estimate (cost + timeline)

`POST /api/v1/renovation/estimate`

- **Input**: property facts + either a single `image_url` OR manual fallback `condition_score`.
- **Output**: image condition + renovation class + low/mid/high estimate + timeline + suggested work items + confidence + explanation.
- **Optional market context (no DB required)**:
  - `avg_area_price_per_sqft`
  - `years_since_last_sale`
  - `permit_years_since_last`
  - These sharpen confidence/timeline and assumptions if your backend already has those values from external APIs.

Example with manual fallback (no image call):

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
    "material_index": 1.05
  }'
```

### Valuation (ARV + ROI)

`POST /api/v1/valuation/analyze`

- Uses comp-based ARV first, with renovation-level adjustment (`standard`, `premium`, `luxury`)
- Calculates estimated profit, ROI %, margin, and 3-point sensitivity view

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/valuation/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "purchase_price": 350000,
    "sqft": 1800,
    "lot_size": 5200,
    "beds": 3,
    "baths": 2,
    "year_built": 1988,
    "property_type": "SFR",
    "neighborhood_price_per_sqft": 250,
    "renovation_level": "standard",
    "market_trend_adjustment": 0.01,
    "estimated_renovation_cost": 70000,
    "estimated_holding_cost": 10000,
    "closing_cost": 5000,
    "financing_cost": 8000,
    "selling_cost": 15000,
    "comparable_renovated_sales": [
      {"comp_id":"comp-1","sold_price":470000,"sqft":1820,"distance_miles":0.4,"days_ago":45,"renovated":true}
    ]
  }'
```

## Enable OpenAI Vision (optional)

By default, the service returns a conservative fallback score when vision is disabled.

In `.env`:

```env
OPENAI_VISION_ENABLED=true
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

The prompt used for vision extraction is:
- `app/prompts/renovation_image_condition_prompt.txt`

## Simple UI tester

There is a basic test UI at:
- `ui/renovation_v1_simple.html`

Open it in your browser and:
- set API URL (default is `http://127.0.0.1:8000/api/v1/renovation/image-condition`)
- paste one image URL
- click **Analyze**

It will show response JSON on screen.

If opening `ui/renovation_v1_simple.html` directly causes browser fetch/CORS issues, serve the UI locally:

```bash
python -m http.server 8080 -d ui
```

Then open:
- `http://localhost:8080`

