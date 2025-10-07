# Lab 5 — Observe vLLM + RAG + Chat API

## Goal

* Export **system + model metrics**:

  * vLLM: TTFT, latency histograms, tokens/sec, request counts (already exposed when `--enable-metrics` is on).

  * Chat API: end-to-end latency, prompt/response token counts, request/5xx counts.

  * Retriever: retrieval latency and top-k stats.

* Scrape with **Prometheus (kube-prometheus-stack)**.

* Visualize with **Grafana**.

> From Lab 0 we installed kube-prometheus-stack with
> `prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false` — so **any** ServiceMonitor is scraped (no special labels needed).

---

## What’s new in this lab

```
atharva-dental-assistant/
├─ serving/
│  ├─ chat_api.py                 # UPDATED: Prometheus metrics (/metrics)
│  └─ prompt_templates.py
├─ rag/
│  └─ retriever.py                # UPDATED: Prometheus metrics (/metrics)
├─ k8s/
│  └─ 50-observability/
│     ├─ sm-vllm.yaml            # ServiceMonitor for vLLM
│     ├─ sm-chat-api.yaml        # ServiceMonitor for Chat API
│     ├─ sm-retriever.yaml       # ServiceMonitor for Retriever
│     ├─ pr-alerts.yaml          # (optional) basic alert rules
│     └─ cm-grafana-dashboard.yaml
└─ scripts/
   └─ deploy_observability.sh
```

---

## 0) Setup Prometheus

Setup prometheus stack along with grafana with 

```
helm upgrade --install kps -n monitoring \
  prometheus-community/kube-prometheus-stack \
  --create-namespace \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=30400 \
  --set prometheus.service.type=NodePort \
  --set prometheus.service.nodePort=30500 \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

```

validate 

```
kubectl rollout status -n monitoring deploy/kps-grafana --timeout=300s 
kubectl get pods -n monitoring
```

* Grafana : http://localhost:30400/
  * user: `admin`
  * pass: `prom-operator`
* Prometheus : http://localhost:30500/


## 1) Instrument the services

### A) Chat API — add Prometheus metrics (`serving/chat_api.py`)

Replace your file with this version (differences: imports, metrics, `/metrics` endpoint, timings):

```
import os
import time
import httpx
from fastapi import FastAPI, Query, Response
from pydantic import BaseModel
from typing import List, Dict, Any

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from prompt_templates import build_messages

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://atharva-retriever.atharva-ml.svc.cluster.local:8001")
VLLM_URL      = os.getenv("VLLM_URL",      "http://atharva-vllm.atharva-ml.svc.cluster.local:8000")
MODEL_NAME    = os.getenv("MODEL_NAME",    "smollm2-135m-atharva")

MAX_CTX_SNIPPETS = int(os.getenv("MAX_CTX_SNIPPETS", "3"))
MAX_CTX_CHARS    = int(os.getenv("MAX_CTX_CHARS", "2400"))

app = FastAPI()

# -----------------------------
# Prometheus metrics
# -----------------------------
REQS = Counter("chat_requests_total", "Total Chat API requests", ["route"])
ERRS = Counter("chat_errors_total", "Total Chat API errors", ["stage"])
E2E_LAT = Histogram(
    "chat_end_to_end_latency_seconds",
    "End-to-end /chat latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13),
)
RAG_LAT = Histogram(
    "rag_retrieval_latency_seconds",
    "Retriever call latency in seconds",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3),
)
VLLM_LAT = Histogram(
    "vllm_request_latency_seconds",
    "vLLM chat/completions call latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13),
)

TOK_PROMPT = Gauge("chat_prompt_tokens", "Prompt tokens for the last completed /chat request")
TOK_COMPLETION = Gauge("chat_completion_tokens", "Completion tokens for the last completed /chat request")
TOK_TOTAL = Gauge("chat_total_tokens", "Total tokens for the last completed /chat request")


class ChatRequest(BaseModel):
    question: str
    k: int = 4
    max_tokens: int = 200
    temperature: float = 0.1
    debug: bool = False  # when true, include prompt/messages in response


def _label(meta: Dict[str, Any]) -> str:
    did = (meta or {}).get("doc_id")
    sec = (meta or {}).get("section")
    if not did:
        return "unknown"
    return f"{did}#{sec}" if sec and sec != "full" else did


def _collect_citations(hits: List[Dict[str, Any]]) -> List[str]:
    seen, out = set(), []
    for h in hits:
        lab = _label(h.get("meta"))
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


def _normalize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Drop recent_queries from grounding context (they’re noisy)
    filt = []
    for h in hits:
        did = ((h.get("meta") or {}).get("doc_id") or "").lower()
        if did.startswith("recent_queries.jsonl"):
            continue
        filt.append(h)

    # Prefer those with text first
    filt.sort(key=lambda h: (h.get("text") is None), reverse=False)

    # Dedup by label
    seen, dedup = set(), []
    for h in filt:
        lab = _label(h.get("meta"))
        if lab in seen:
            continue
        seen.add(lab)
        dedup.append(h)

    # Trim by count and char budget
    total = 0
    trimmed = []
    for h in dedup:
        txt = h.get("text") or (h.get("meta") or {}).get("text") or ""
        if len(trimmed) < MAX_CTX_SNIPPETS and total + len(txt) <= MAX_CTX_CHARS:
            trimmed.append(h)
            total += len(txt)
        if len(trimmed) >= MAX_CTX_SNIPPETS:
            break

    return trimmed


def _strip_existing_source(txt: str) -> str:
    lines = txt.rstrip().splitlines()
    kept = [ln for ln in lines if not ln.strip().lower().startswith("source:")]
    return "\n".join(kept).rstrip()


@app.get("/health")
def health():
    return {"ok": True, "retriever": RETRIEVER_URL, "vllm": VLLM_URL, "model": MODEL_NAME}


@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dryrun")
def dryrun(q: str = Query(..., alias="question"), k: int = 4):
    """Build exactly what /chat would send to vLLM, but don’t call vLLM."""
    REQS.labels(route="/dryrun").inc()
    with httpx.Client(timeout=30) as cx:
        t_r0 = time.time()
        try:
            r = cx.post(f"{RETRIEVER_URL}/search", json={"query": q, "k": k})
            r.raise_for_status()
            raw_hits = r.json().get("hits", [])
        except Exception:
            ERRS.labels(stage="retriever").inc()
            raise
        finally:
            RAG_LAT.observe(time.time() - t_r0)

    ctx_hits   = _normalize_hits(raw_hits)
    citations  = _collect_citations(ctx_hits)
    messages   = build_messages(q, ctx_hits)

    # Also surface the precise snippets we used (label + text)
    used_snippets = []
    for h in ctx_hits:
        meta = h.get("meta") or {}
        used_snippets.append({
            "label": _label(meta),
            "text": h.get("text") or meta.get("text") or ""
        })

    return {
        "question": q,
        "citations": citations,
        "used_snippets": used_snippets,   # what the model will actually see
        "messages": messages,             # the exact OpenAI Chat payload
        "note": "This is a dry run; no LLM call was made."
    }


@app.post("/chat")
def chat(req: ChatRequest):
    REQS.labels(route="/chat").inc()
    t0 = time.time()

    # 1) retrieve
    with httpx.Client(timeout=30) as cx:
        t_r0 = time.time()
        try:
            r = cx.post(f"{RETRIEVER_URL}/search", json={"query": req.question, "k": req.k})
            r.raise_for_status()
            raw_hits = r.json().get("hits", [])
        except Exception:
            ERRS.labels(stage="retriever").inc()
            raise
        finally:
            RAG_LAT.observe(time.time() - t_r0)

    # 2) normalize + citations
    ctx_hits  = _normalize_hits(raw_hits)
    citations = _collect_citations(ctx_hits)

    # 3) build messages with actual snippet text
    messages = build_messages(req.question, ctx_hits)

    # 4) call vLLM (OpenAI-compatible)
    temperature = max(0.0, min(req.temperature, 0.5))
    max_tokens  = min(req.max_tokens, 256)
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }

    with httpx.Client(timeout=120) as cx:
        t_llm0 = time.time()
        try:
            rr = cx.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
            rr.raise_for_status()
            data = rr.json()
        except Exception:
            ERRS.labels(stage="vllm").inc()
            raise
        finally:
            VLLM_LAT.observe(time.time() - t_llm0)

    content = data["choices"][0]["message"]["content"]
    usage   = data.get("usage", {})
    dt      = time.time() - t0

    content = _strip_existing_source(content)
    content = content + ("\nSource: " + "; ".join(citations) if citations else "\nSource: (none)")

    # Update token gauges (best-effort)
    try:
        TOK_PROMPT.set(float(usage.get("prompt_tokens", 0) or 0))
        TOK_COMPLETION.set(float(usage.get("completion_tokens", 0) or 0))
        TOK_TOTAL.set(float(usage.get("total_tokens", 0) or 0))
    except Exception:
        # Avoid failing the request if usage is missing or malformed
        pass

    # Observe end-to-end latency
    E2E_LAT.observe(dt)

    resp = {
        "answer": content,
        "citations": citations,
        "latency_seconds": round(dt, 3),
        "usage": usage,
    }

    # 5) optional debug payload so you can inspect exactly what was sent
    if req.debug:
        used_snippets = []
        for h in ctx_hits:
            meta = h.get("meta") or {}
            used_snippets.append({
                "label": _label(meta),
                "text": h.get("text") or meta.get("text") or ""
            })
        resp["debug"] = {
            "messages": messages,         # exact system+user messages sent
            "used_snippets": used_snippets,
            "raw_hits": raw_hits[:10],    # original retriever output (trimmed)
            "payload_model": MODEL_NAME,
            "payload_temperature": temperature,
            "payload_max_tokens": max_tokens,
        }

    return resp
```

Update  deployment spec to install `prometheus-client>=0.20.0`.
 
File: `k8s/40-serve/deploy-chat-api.yaml`

```
        command: ["bash","-lc"]
        args:
          - |
            pip install --no-cache-dir fastapi==0.112.2 uvicorn==0.30.6 httpx==0.27.2  prometheus-client>=0.20.0
            uvicorn chat_api:app --host 0.0.0.0 --port 8080 --proxy-headers
```

Redeploy chat api as 

```
kubectl delete -f k8s/40-serve/deploy-retriever.yaml
kubectl apply -f k8s/40-serve/deploy-retriever.yaml
```
---

### B) Retriever — add Prometheus metrics (`rag/retriever.py`)

Replace your retriever with this version (adds `/metrics`, latency histogram, request counter):

```
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

# --- Prometheus (only new dependency) ---
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

BACKEND    = os.getenv("BACKEND", "dense")  # "sparse" or "dense"
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/index.faiss"))
META_PATH  = Path(os.getenv("META_PATH",  "/mnt/project/atharva-dental-assistant/artifacts/rag/meta.json"))
MODEL_DIR  = os.getenv("MODEL_DIR")  # optional for dense
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title=f"Atharva Retriever ({BACKEND})")

class SearchRequest(BaseModel):
    query: str
    k: int = 4

_ready_reason = "starting"
_model = None; _index = None; _meta: List[dict] = []
_vec = None; _X = None  # sparse objects

# ------------------ Prometheus metrics (added) ------------------
REQS_TOTAL = Counter("retriever_requests_total", "Total /search requests")
ERRS_TOTAL = Counter("retriever_errors_total", "Total retriever errors", ["stage"])
E2E_LAT = Histogram(
    "retriever_search_latency_seconds",
    "End-to-end /search latency (s)",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)
# Dense sub-steps
ENC_LAT = Histogram(
    "retriever_dense_encode_latency_seconds",
    "SentenceTransformer.encode latency (s)",
    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
)
FAISS_LAT = Histogram(
    "retriever_dense_faiss_latency_seconds",
    "FAISS search latency (s)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
# Sparse sub-steps
VEC_LAT = Histogram(
    "retriever_sparse_vectorize_latency_seconds",
    "TF-IDF vectorizer.transform latency (s)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
DOT_LAT = Histogram(
    "retriever_sparse_dot_latency_seconds",
    "Sparse dot/matmul latency (s)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
# Load-time & sizes
MODEL_LOAD_SEC = Gauge("retriever_model_load_seconds", "Dense model load time (s)")
INDEX_LOAD_SEC = Gauge("retriever_index_load_seconds", "FAISS index load time (s)")
SPARSE_VEC_LOAD_SEC = Gauge("retriever_sparse_vectorizer_load_seconds", "TF-IDF vectorizer load time (s)")
SPARSE_MAT_LOAD_SEC = Gauge("retriever_sparse_matrix_load_seconds", "TF-IDF matrix load time (s)")
INDEX_ITEMS = Gauge("retriever_index_items", "Items in index/matrix")
META_ITEMS = Gauge("retriever_meta_items", "Number of meta records")

# ------------------ Utils ------------------

def _normalize_meta_loaded(data: Any) -> List[dict]:
    """
    Accepts various shapes of meta.json and returns a list of entries.
    Supported:
      - list[dict]
      - {"items": [...]}  (common pattern)
      - {"hits": [...]}   (fallback)
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
            # keep original behavior
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "hits" in data and isinstance(data["hits"], list):
            return data["hits"]
    raise ValueError("META_PATH must contain a list or a dict with 'items'/'hits'.")

def _parse_doc_and_section(path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Parse labels from meta.path:
      - 'treatments.json#0' -> ('treatments.json', '0')
      - 'faq.md'            -> ('faq.md', None)
      - 'policies/emergency.md' -> ('policies/emergency.md', None)
    """
    if not path:
        return "unknown", None
    if "#" in path:
        d, s = path.split("#", 1)
        return d, s
    return path, None

def _extract_text(m: dict) -> Optional[str]:
    """Try common keys for stored chunk text."""
    return m.get("text") or m.get("chunk") or m.get("content")

def _enrich_hit(idx: int, score: float) -> dict:
    """
    Build a single enriched hit from meta[idx].
    """
    if idx < 0 or idx >= len(_meta):
        doc_id, section, path, typ, txt = "unknown", None, None, None, None
    else:
        m   = _meta[idx] or {}
        path = m.get("path")
        typ  = m.get("type")
        doc_id, section = _parse_doc_and_section(path)
        txt = _extract_text(m)

    hit = {
        "score": float(score),
        "meta": {
            "doc_id": doc_id,
            "section": section,
            "path": path,
            "type": typ,
        },
    }
    if txt:
        hit["text"] = txt
    return hit

# ------------------ Loaders (unchanged behavior; just timed gauges) ------------------

def _load_dense():
    global _model, _index, _meta
    try:
        import time as _t
        import faiss
        from sentence_transformers import SentenceTransformer

        t0 = _t.time()
        _model = SentenceTransformer(MODEL_DIR) if (MODEL_DIR and Path(MODEL_DIR).exists()) else SentenceTransformer(MODEL_NAME)
        MODEL_LOAD_SEC.set(_t.time() - t0)

        t1 = _t.time()
        _index = faiss.read_index(str(INDEX_PATH))
        INDEX_LOAD_SEC.set(_t.time() - t1)

        _meta = _normalize_meta_loaded(json.loads(META_PATH.read_text(encoding="utf-8")))
        META_ITEMS.set(len(_meta) if isinstance(_meta, list) else 0)

        # best-effort size (for FAISS)
        try:
            INDEX_ITEMS.set(int(getattr(_index, "ntotal", len(_meta))))
        except Exception:
            INDEX_ITEMS.set(len(_meta))

        return None
    except Exception as e:
        return f"dense load error: {e}"

def _load_sparse():
    global _vec, _X, _meta
    try:
        import time as _t
        import joblib
        from scipy import sparse

        vec_p = Path(os.getenv("VEC_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/tfidf_vectorizer.joblib"))
        X_p   = Path(os.getenv("MAT_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/tfidf_matrix.npz"))

        t0 = _t.time()
        _vec = joblib.load(vec_p)
        SPARSE_VEC_LOAD_SEC.set(_t.time() - t0)

        t1 = _t.time()
        _X = sparse.load_npz(X_p)  # assume rows L2-normalized; dot == cosine
        SPARSE_MAT_LOAD_SEC.set(_t.time() - t1)

        _meta = _normalize_meta_loaded(json.loads(META_PATH.read_text(encoding="utf-8")))
        META_ITEMS.set(len(_meta) if isinstance(_meta, list) else 0)

        try:
            INDEX_ITEMS.set(int(getattr(_X, "shape", (0, 0))[0]))
        except Exception:
            INDEX_ITEMS.set(len(_meta))

        return None
    except Exception as e:
        return f"sparse load error: {e}"

@app.on_event("startup")
def startup():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()

# ------------------ Endpoints (original behavior preserved) ------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    return {"ready": _ready_reason is None, "reason": _ready_reason}

@app.post("/reload")
def reload_index():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()
    if _ready_reason is not None:
        raise HTTPException(status_code=503, detail=_ready_reason)
    return {"reloaded": True}

# --- /metrics added (Prometheus text format) ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/search")
def search(req: SearchRequest):
    if _ready_reason is not None:
        raise HTTPException(status_code=503, detail=_ready_reason)

    import time as _t
    t0 = _t.time()
    REQS_TOTAL.inc()

    k = max(1, min(int(req.k), 20))

    if BACKEND == "sparse":
        try:
            import numpy as np
            t_vec0 = _t.time()
            q = _vec.transform([req.query])
        except Exception:
            ERRS_TOTAL.labels(stage="sparse_vectorize").inc()
            raise
        finally:
            VEC_LAT.observe(_t.time() - t_vec0)

        try:
            t_dot0 = _t.time()
            scores = (_X @ q.T).toarray().ravel()  # cosine since rows normalized
        except Exception:
            ERRS_TOTAL.labels(stage="sparse_dot").inc()
            raise
        finally:
            DOT_LAT.observe(_t.time() - t_dot0)

        if scores.size == 0:
            E2E_LAT.observe(_t.time() - t0)
            return {"hits": []}
        # get top-k indices by score desc
        k_eff = min(k, scores.size)
        top = np.argpartition(-scores, range(k_eff))[:k_eff]
        top = top[np.argsort(-scores[top])]
        hits = [
            _enrich_hit(int(i), float(scores[int(i)]))
            for i in top
            if scores[int(i)] > 0
        ]
        E2E_LAT.observe(_t.time() - t0)
        return {"hits": hits}

    # dense (unchanged behavior; with timing)
    try:
        import faiss
        import numpy as np
        t_enc0 = _t.time()
        v = _model.encode([req.query], normalize_embeddings=True)  # IP ~ cosine
    except Exception:
        ERRS_TOTAL.labels(stage="dense_encode").inc()
        raise
    finally:
        ENC_LAT.observe(_t.time() - t_enc0)

    try:
        t_faiss0 = _t.time()
        D, I = _index.search(v.astype("float32"), k)
    except Exception:
        ERRS_TOTAL.labels(stage="dense_faiss_search").inc()
        raise
    finally:
        FAISS_LAT.observe(_t.time() - t_faiss0)

    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        hits.append(_enrich_hit(int(idx), float(score)))

    E2E_LAT.observe(_t.time() - t0)
    return {"hits": hits}
```

Update  deployment spec to install `prometheus-client>=0.20.0`.

File: `k8s/40-serve/deploy-retriever.yaml`
```
            python -m pip install --upgrade pip
            if [ "${BACKEND:-sparse}" = "sparse" ]; then
              # Pure sparse stack (forces wheels only; will fail if a wheel is unavailable for your arch)
              python -m pip install --only-binary=:all: \
                "numpy==1.26.4" "scipy==1.10.1" "scikit-learn==1.3.2" joblib==1.4.2 \
                fastapi==0.112.2 uvicorn==0.30.6 prometheus-client>=0.20.0
```

Redeploy chat api as 
```
kubectl delete -f k8s/40-serve/deploy-chat-api.yaml
kubectl apply -f k8s/40-serve/deploy-chat-api.yaml
```

> No rebuild needed—these run from the mounted repo (`/mnt/project`).

---

## 2) ServiceMonitors

> These tell Prometheus which Services to scrape. We’ll scrape:

* **vLLM** on `atharva-ml/atharva-vllm:8000/metrics`

* **Chat API** on `atharva-app/atharva-chat-api:8080/metrics`

* **Retriever** on `atharva-ml/atharva-retriever:8001/metrics`

Create these under `k8s/50-observability/`.

```
cd project/atharva-dental-assistant
mkdir k8s/50-observability/
```

### `k8s/50-observability/sm-vllm.yaml`

```
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sm-atharva-vllm
  namespace: monitoring
  labels:
    release: kps   # <-- REQUIRED if your Prometheus CR uses selector.matchLabels.release
spec:
  namespaceSelector:
    matchNames:
      - atharva-ml
  selector:
    matchLabels:
      app: isvc.atharva-vllm-predictor              # <-- must match your Service’s label
  endpoints:
    - port: http1                    # <-- must match the Service port *name*
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s             # optional, good practice

```

### `k8s/50-observability/sm-chat-api.yaml`

```
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sm-atharva-chat-api
  namespace: monitoring
  labels:
    release: kps   # optional, add if your Prometheus selects by this
spec:
  namespaceSelector:
    matchNames:
      - atharva-app
  selector:
    matchLabels:
      app: atharva-chat-api          # <-- must match your Service’s label
  endpoints:
    - port: http                     # <-- must match the Service port *name*
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s             # optional but recommended
```

### `k8s/50-observability/sm-retriever.yaml`

```
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sm-atharva-retriever
  namespace: monitoring
  labels:
    release: kps   # optional; include if your Prometheus uses this selector
spec:
  namespaceSelector:
    matchNames:
      - atharva-ml
  selector:
    matchLabels:
      app: retriever         # <-- must match your Service’s label
  endpoints:
    - port: http    # <-- must match the port *name* in Service
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s             # optional, safe default
```

*(Your Services from previous labs used port names `http`. If you changed them, reflect the right name here.)*

---

## 3) (Optional) Alerts (PrometheusRule)

Basic examples to show how you’d alert on high error rate or slow E2E latency.

### `k8s/50-observability/pr-alerts.yaml`

```
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: atharva-alerts
  namespace: monitoring
spec:
  groups:
  - name: atharva-chat
    rules:
    - alert: ChatApiHighErrorRate
      expr: sum(rate(chat_errors_total[5m])) by (stage) / sum(rate(chat_requests_total[5m])) by () > 0.05
      for: 5m
      labels: { severity: warning }
      annotations:
        summary: "Chat API error rate > 5% (stage={{ $labels.stage }})"
    - alert: ChatApiLatencyHighP95
      expr: histogram_quantile(0.95, sum(rate(chat_end_to_end_latency_seconds_bucket[5m])) by (le)) > 3
      for: 10m
      labels: { severity: warning }
      annotations:
        summary: "Chat API p95 latency > 3s"
```

---

## 4) Grafana dashboard (ConfigMap)

A concise starter dashboard with:

* E2E latency p50/p95

* Requests & errors

* Tokens (prompt/comp/total)

* Retriever latency p95

* vLLM tokens/sec (if exported by your vLLM build)

### `k8s/50-observability/cm-grafana-dashboard.yaml`

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: atharva-llmops-dashboard2
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  atharva-llmops.json: |
    {
      "title": "Atharva LLMOps - Overview",
      "timezone": "browser",
      "schemaVersion": 39,
      "version": 2,
      "panels": [
        { "type": "stat", "title": "Chat RPS",
          "targets": [ { "expr": "sum(rate(chat_requests_total[1m]))" } ],
          "gridPos": { "x": 0, "y": 0, "w": 6, "h": 4 }
        },
        { "type": "graph", "title": "Chat Errors by Stage (/s)",
          "targets": [ { "expr": "sum by (stage) (rate(chat_errors_total[1m]))" } ],
          "legend": { "show": true },
          "gridPos": { "x": 6, "y": 0, "w": 18, "h": 6 }
        },

        { "type": "graph", "title": "Chat E2E Latency (p50/p95)",
          "targets": [
            { "expr": "histogram_quantile(0.50, sum by (le) (rate(chat_end_to_end_latency_seconds_bucket[5m])))", "legendFormat": "p50" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(chat_end_to_end_latency_seconds_bucket[5m])))", "legendFormat": "p95" }
          ],
          "gridPos": { "x": 0, "y": 4, "w": 12, "h": 7 }
        },
        { "type": "graph", "title": "Chat Sub-steps p95 (Retriever & vLLM)",
          "targets": [
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(rag_retrieval_latency_seconds_bucket[5m])))", "legendFormat": "retriever (inside chat)" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm_request_latency_seconds_bucket[5m])))", "legendFormat": "vLLM (inside chat)" }
          ],
          "legend": { "show": true },
          "gridPos": { "x": 12, "y": 4, "w": 12, "h": 7 }
        },

        { "type": "stat", "title": "Retriever RPS",
          "targets": [ { "expr": "sum(rate(retriever_requests_total[1m]))" } ],
          "gridPos": { "x": 0, "y": 11, "w": 6, "h": 4 }
        },
        { "type": "graph", "title": "Retriever Errors by Stage (/s)",
          "targets": [ { "expr": "sum by (stage) (rate(retriever_errors_total[1m]))" } ],
          "legend": { "show": true },
          "gridPos": { "x": 6, "y": 11, "w": 18, "h": 6 }
        },
        { "type": "graph", "title": "Retriever Search Latency (p50/p95)",
          "targets": [
            { "expr": "histogram_quantile(0.50, sum by (le) (rate(retriever_search_latency_seconds_bucket[5m])))", "legendFormat": "p50" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(retriever_search_latency_seconds_bucket[5m])))", "legendFormat": "p95" }
          ],
          "gridPos": { "x": 0, "y": 17, "w": 12, "h": 7 }
        },
        { "type": "graph", "title": "Retriever Sub-steps p95 (Dense/Sparse)",
          "targets": [
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(retriever_dense_encode_latency_seconds_bucket[5m])))", "legendFormat": "dense encode" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(retriever_dense_faiss_latency_seconds_bucket[5m])))", "legendFormat": "dense faiss" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(retriever_sparse_vectorize_latency_seconds_bucket[5m])))", "legendFormat": "sparse vectorize" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(retriever_sparse_dot_latency_seconds_bucket[5m])))", "legendFormat": "sparse dot" }
          ],
          "legend": { "show": true },
          "gridPos": { "x": 12, "y": 17, "w": 12, "h": 7 }
        },

        { "type": "stat", "title": "Prompt Tokens (last)",
          "targets": [ { "expr": "chat_prompt_tokens" } ],
          "gridPos": { "x": 0, "y": 24, "w": 6, "h": 4 }
        },
        { "type": "stat", "title": "Completion Tokens (last)",
          "targets": [ { "expr": "chat_completion_tokens" } ],
          "gridPos": { "x": 6, "y": 24, "w": 6, "h": 4 }
        },
        { "type": "stat", "title": "Total Tokens (last)",
          "targets": [ { "expr": "chat_total_tokens" } ],
          "gridPos": { "x": 12, "y": 24, "w": 6, "h": 4 }
        },

        { "type": "graph", "title": "vLLM Tokens/sec",
          "targets": [
            { "expr": "sum(rate(vllm:prompt_tokens_total[5m]))", "legendFormat": "prompt t/s" },
            { "expr": "sum(rate(vllm:generation_tokens_total[5m]))", "legendFormat": "generation t/s" }
          ],
          "legend": { "show": true },
          "gridPos": { "x": 18, "y": 24, "w": 6, "h": 6 }
        },
        { "type": "graph", "title": "vLLM Latencies (p95)",
          "targets": [
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))", "legendFormat": "TTFT p95" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))", "legendFormat": "E2E p95" },
            { "expr": "histogram_quantile(0.95, sum by (le) (rate(vllm:request_inference_time_seconds_bucket[5m])))", "legendFormat": "inference p95" }
          ],
          "legend": { "show": true },
          "gridPos": { "x": 0, "y": 28, "w": 12, "h": 7 }
        },
        { "type": "stat", "title": "vLLM Queue / Running",
          "targets": [
            { "expr": "sum(vllm:num_requests_waiting)", "legendFormat": "waiting" },
            { "expr": "sum(vllm:num_requests_running)", "legendFormat": "running" }
          ],
          "gridPos": { "x": 12, "y": 28, "w": 6, "h": 4 }
        },

        { "type": "table", "title": "Retriever Index & Meta Sizes",
          "targets": [
            { "expr": "retriever_index_items", "legendFormat": "index_items" },
            { "expr": "retriever_meta_items", "legendFormat": "meta_items" }
          ],
          "gridPos": { "x": 18, "y": 30, "w": 6, "h": 5 }
        }
      ]
    }
```

Also update the existing service spec to add relevant labels which are then used by prometheus service monitor to scrape the metrics from 

File : `k8s/40-serve/svc-retriever.yaml`
```
apiVersion: v1
kind: Service
metadata:
  name: atharva-retriever
  namespace: atharva-ml
  labels:
    app: retriever
spec:
  type: NodePort
  selector: { app: retriever }
  ports:
  - name: http
    port: 8001
    targetPort: 8001
    nodePort: 30100
```

File : `k8s/40-serve/svc-chat-api.yaml`
```
apiVersion: v1
kind: Service
metadata:
  name: atharva-chat-api
  namespace: atharva-app
  labels:
    app: atharva-chat-api
spec:
  type: NodePort
  selector: { app: atharva-chat-api }
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    nodePort: 30300
```

Apply service changes 

```
kubectl apply -f k8s/40-serve/svc-retriever.yaml
kubectl apply -f k8s/40-serve/svc-chat-api.yaml
```
---

## 5) Deploy all observability bits

### `scripts/deploy_observability.sh`

```
#!/usr/bin/env bash
set -euo pipefail

# Apply ServiceMonitors + Alerts + Dashboard
kubectl apply -f k8s/50-observability/sm-vllm.yaml
kubectl apply -f k8s/50-observability/sm-chat-api.yaml
kubectl apply -f k8s/50-observability/sm-retriever.yaml
kubectl apply -f k8s/50-observability/cm-grafana-dashboard.yaml
kubectl apply -f k8s/50-observability/pr-alerts.yaml || true

echo "Waiting a bit for Prometheus to pick up targets..."
sleep 10

echo "List ServiceMonitors:"
kubectl -n monitoring get servicemonitors
echo "Prometheus targets (check Up status via UI)."

echo -e "Access Prometheus and Grafana using\n\
  * Prometheus : http://localhost:30500/\n\
  * Grafana : http://localhost:30400/\n\
    * user: admin\n\
    * pass: prom-operator"

```

---

## 6) Run the lab

```
# Make sure Lab 4 services are running:
kubectl -n atharva-ml get svc atharva-vllm atharva-retriever
kubectl -n atharva-app get svc atharva-chat-api

# Deploy observability
bash scripts/deploy_observability.sh
```

Open Grafana at [http://127.0.0.1:3000](http://127.0.0.1:3000/)
(Default admin/admin unless you changed it via Helm values.)

* Find dashboard: **“Atharva LLMOps - Overview”**.

* Generate some traffic (reuse Lab 4’s `smoke_e2e.sh` a few times) and watch:

  * **Chat RPS**, **errors by stage**,

  * **E2E p50/p95**, **Retriever p95**,

  * **Tokens gauges**,

  * (If present) **vLLM tokens/sec**.

---

## Lab Summary 

This is what we accomplished in this lab

* Exported **custom metrics** from Chat API & Retriever.

* Scraped **vLLM** metrics alongside app metrics via **ServiceMonitors**.

* Visualized a consolidated **LLMOps dashboard** in Grafana.

* (Optional) Added **alerts** on error rate and latency.


#courses/llmops/labs/v1
