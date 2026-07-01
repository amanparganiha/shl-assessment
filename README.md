# Conversational SHL Assessment Recommender

A stateless FastAPI agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a **grounded shortlist of real SHL assessments**
through dialogue. It clarifies, recommends, refines, compares, and refuses
out-of-scope requests — and it can only ever return assessments that exist in the
SHL Individual Test Solutions catalog.

Built for the SHL Labs AI Intern take-home.

## Architecture

```
POST /chat (full history, stateless)
      │
      ▼
 build retrieval query  ──►  Hybrid Retriever (BM25 + dense embeddings, RRF)
 (cumulative user intent)         │   over 377 catalog items  + always-on staples
      │                           ▼
      │                    top-K candidates (with metadata)
      ▼                           │
 System prompt (rules) + candidates + conversation  ──►  LLM (gpt-4o-mini, JSON out)
      │                                                        │
      │   returns: {action, reply, recommended_ids, end}       │
      ▼                                                        ▼
 Server maps ids ──► exact catalog {name, url, test_type}  (URLs cannot be hallucinated)
      │
      ▼
 {reply, recommendations[], end_of_conversation}
```

**Grounding guarantee:** the model only ever *selects candidate ids*; the server
maps each id back to the catalog record. A fabricated name or URL is therefore
structurally impossible.

## Repo layout

```
app/            FastAPI service, agent, retriever, catalog, LLM client, prompts
scripts/        build_index.py  (offline: normalize catalog + compute embeddings)
data/           raw catalog JSON, sample traces, and data/processed/ (generated)
eval/           trace parser, Recall@10 simulator, behavior probes
static/         index.html test UI
render.yaml     Render deployment blueprint
```

## Quickstart (local)

```bash
# 1) Install
pip install -r requirements.txt

# 2) Add your key
cp .env.example .env          # then edit .env: OPENAI_API_KEY=sk-...

# 3) Build the processed catalog + embeddings (one-time, ~10s)
python scripts/build_index.py

# 4) Run
uvicorn app.main:app --reload
# open http://localhost:8000  (test UI)  and  http://localhost:8000/docs
```

## API

`GET /health` → `{"status":"ok"}` (HTTP 200)

`POST /chat` (stateless — send the full history every call):

```json
{ "messages": [ {"role":"user","content":"Hiring a mid-level Java developer"} ] }
```
```json
{
  "reply": "Here are assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name":"Core Java (Advanced Level) (New)","url":"https://www.shl.com/...","test_type":"K"}
  ],
  "end_of_conversation": false
}
```
`recommendations` is empty while clarifying or refusing, and holds 1–10 items once
the agent commits to a shortlist.

## Evaluation

```bash
python eval/run_eval.py                 # Mean Recall@10 via LLM user-simulator
python eval/run_eval.py --mode deterministic   # cheaper, feeds trace turns in order
python eval/probes.py                   # behavior probes pass-rate
```

## Deploy (Render)

1. Push this repo to GitHub (processed data is committed, so no build-time API calls).
2. Render → New + → Blueprint → select the repo (`render.yaml` is detected).
3. Set `OPENAI_API_KEY` in the dashboard.
4. `/health` and `/chat` are then publicly reachable (first cold start ≤ ~1 min).

See `approach.md` for design rationale, retrieval/prompt details, and evaluation.
