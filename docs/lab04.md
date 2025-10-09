# Lab 4 — vLLM Serving (KServe RawDeployment) + RAG-aware Chat API

## Goal

* Serve the **merged SmolLM2-135M** from Lab 2 via **vLLM**.

* Mount the model snapshot with **ImageVolumes**.

* Deploy a **Chat API** (FastAPI) that:

  1. calls the **Retriever** (Lab 1) for top-k context,

  2. composes a safe system prompt (Pune/INR, clinic context),

  3. calls **vLLM’s OpenAI-compatible** endpoint,

  4. returns answer + citations + basic timing/token stats.

> Namespaces:

* **atharva-ml**: vLLM server

* **atharva-app**: Chat API (or keep all in atharva-ml if you prefer)

---

## Repo additions (this lab)

```
atharva-dental-assistant/
├─ serving/
│  ├─ chat_api.py
│  └─ prompt_templates.py
├─ k8s/
│  └─ 40-serve/
│     ├─ rawdeployment-vllm.yaml          # KServe RawDeployment (preferred)
│     ├─ svc-vllm.yaml
│     ├─ deploy-chat-api.yaml
│     ├─ svc-chat-api.yaml
│     └─ cm-chat-api.yaml
└─ scripts/
   ├─ deploy_vllm.sh
   ├─ deploy_chat_api.sh
   └─ smoke_e2e.sh
```

> Assumes you already built & pushed your model image to the local registry in **Lab 3** as:
> `kind-registry:5001/atharva/smollm2-135m-merged:v1`

---

```
cd project/atharva-dental-assistant

# Pull a vLLM image for CPU Inference
docker image pull schoolofdevops/vllm-cpu-nonuma:0.9.1

# Load it onto KinD cluster
kind load docker-image --name llmops-kind schoolofdevops/vllm-cpu-nonuma:0.9.1 --nodes llmops-kind-worker

```

## 1) vLLM with ImageVolume (KServe RawDeployment)

### `k8s/40-serve/rawdeployment-vllm.yaml`

```
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: atharva-vllm
  namespace: atharva-ml
  annotations:
    autoscaling.knative.dev/metric: "concurrency"
    autoscaling.knative.dev/target: "1"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    containerConcurrency: 1
    containers:
      - name: vllm
        image: schoolofdevops/vllm-cpu-nonuma:0.9.1
        args:
          - --model=/models/model
          - --host=0.0.0.0
          - --port=8000               # KServe/Knative-friendly
          - --max-model-len=2048
          - --served-model-name=smollm2-135m-atharva
          - --dtype=float16           # keeps RAM lower on CPU for this tiny model
          - --disable-frontend-multiprocessing
          - --max-num-seqs=1          # clamp engine concurrency (OOM guard)
          - --swap-space=0.5          # GiB reserved for CPU KV cache (fits small pod)
        env:
          - name: VLLM_TARGET_DEVICE
            value: "cpu"
          - name: VLLM_CPU_KVCACHE_SPACE
            value: "1"
          - name: OMP_NUM_THREADS
            value: "2"
          - name: OPENBLAS_NUM_THREADS
            value: "1"
          - name: MKL_NUM_THREADS
            value: "1"
          - name: VLLM_CPU_OMP_THREADS_BIND
            value: "0-1"              # avoid NUMA auto-binding path
        ports:
          - name: http1
            containerPort: 8000
        resources:
          requests:
            cpu: "2"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "3Gi"             # bump to 4Gi if you still see OOM
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        volumeMounts:
          - name: model
            mountPath: /models
            readOnly: true
    # Pinning to a node is fine; nodeSelector is usually nicer than nodeName:
    # nodeSelector:
    #   kubernetes.io/hostname: llmops-kind-worker
    nodeName: llmops-kind-worker
    volumes:
      - name: model
        image:
          reference: initcron/smollm2-135m-merged:v3
          pullPolicy: IfNotPresent
```
where, replace `initcron/smollm2-135m-merged:v2` with actual tag. 


> If your cluster CRD doesn’t support `RawDeployment`, you can temporarily deploy a **plain Deployment** (fallback) with the **same container/volume spec**. Keep going with the rest of the lab; the Chat API doesn’t care how vLLM is deployed as long as the Service is up.

### `k8s/40-serve/svc-vllm.yaml`

```
apiVersion: v1
kind: Service
metadata:
  name: atharva-vllm
  namespace: atharva-ml
spec:
  type: NodePort
  selector:
    # KServe RawDeployment pods are labeled with app=<deployment name> by default
    app: isvc.atharva-vllm-predictor
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    nodePort: 30200
```

---

## 2) Chat API (RAG → prompt → vLLM)

```
mkdir serving
```

### `serving/prompt_templates.py`

```
SYSTEM_PROMPT = (
  "You are Atharva Dental Clinic assistant based in Pune, India. "
  "Respond in concise steps, use INR as currency for any prices or costs, be safety-minded, first try to find the answer from the context provided here."
  "ask for missing info when necessary, and ALWAYS include a final 'Source:' line "
  "citing file#section for facts derived from context.\n"
  "If the question indicates emergency red flags (uncontrolled bleeding, facial swelling, high fever, trauma), "
  "urge immediate contact with the clinic's emergency number.\n"
)

def _label(meta: dict) -> str:
    did = (meta or {}).get("doc_id")
    sec = (meta or {}).get("section")
    if not did:
        return "unknown"
    return f"{did}#{sec}" if sec and sec != "full" else did

def _render_context_block(retrieved_hits: list[dict]) -> str:
    """
    Render only label + text, no Python dicts.
    """
    blocks: list[str] = []
    for h in retrieved_hits:
        meta = h.get("meta") or {}
        label = _label(meta)
        text = (h.get("text") or meta.get("text") or "").strip()
        if not text:
            continue
        blocks.append(f"### {label}\n{text}")
    return "\n\n".join(blocks).strip()

def build_messages(user_q: str, retrieved_hits: list[dict]) -> list[dict]:
    context_block = _render_context_block(retrieved_hits)
    system = SYSTEM_PROMPT + "\nContext snippets:\n" + (context_block if context_block else "(none)")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_q.strip()},
    ]
```

### `serving/chat_api.py`

```
import os
import time
import httpx
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any

from prompt_templates import build_messages

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://atharva-retriever.atharva-ml.svc.cluster.local:8001")
VLLM_URL      = os.getenv("VLLM_URL",      "http://atharva-vllm.atharva-ml.svc.cluster.local:8000")
MODEL_NAME    = os.getenv("MODEL_NAME",    "smollm2-135m-atharva")

MAX_CTX_SNIPPETS = int(os.getenv("MAX_CTX_SNIPPETS", "3"))
MAX_CTX_CHARS    = int(os.getenv("MAX_CTX_CHARS", "2400"))

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    k: int = 4
    max_tokens: int = 200
    temperature: float = 0.1
    debug: bool = False  # <— when true, include prompt/messages in response


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
            seen.add(lab); out.append(lab)
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
        seen.add(lab); dedup.append(h)

    # Trim by count and char budget
    total = 0; trimmed = []
    for h in dedup:
        txt = h.get("text") or (h.get("meta") or {}).get("text") or ""
        if len(trimmed) < MAX_CTX_SNIPPETS and total + len(txt) <= MAX_CTX_CHARS:
            trimmed.append(h); total += len(txt)
        if len(trimmed) >= MAX_CTX_SNIPPETS:
            break

    return trimmed

def _strip_existing_source(txt: str) -> str:
    lines = txt.rstrip().splitlines()
    kept = [ln for ln in lines if not ln.strip().lower().startswith("source:")]
    return "\n".join(kept).rstrip()

@app.get("/health")
def health():
    return {"ok": True, "retriever": RETRIEVER_URL, "vllm": VLLM_URL}

@app.get("/dryrun")
def dryrun(q: str = Query(..., alias="question"), k: int = 4):
    """Build exactly what /chat would send to vLLM, but don’t call vLLM."""
    with httpx.Client(timeout=30) as cx:
        r = cx.post(f"{RETRIEVER_URL}/search", json={"query": q, "k": k})
        r.raise_for_status()
        raw_hits = r.json().get("hits", [])

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
    t0 = time.time()

    # 1) retrieve
    with httpx.Client(timeout=30) as cx:
        r = cx.post(f"{RETRIEVER_URL}/search", json={"query": req.question, "k": req.k})
        r.raise_for_status()
        raw_hits = r.json().get("hits", [])

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
        rr = cx.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
        rr.raise_for_status()
        data = rr.json()

    content = data["choices"][0]["message"]["content"]
    usage   = data.get("usage", {})
    dt      = time.time() - t0

    content = _strip_existing_source(content)
    content = content + ("\nSource: " + "; ".join(citations) if citations else "\nSource: (none)")

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

---

## 3) K8s manifests for Chat API

### `k8s/40-serve/cm-chat-api.yaml`

```
apiVersion: v1
kind: ConfigMap
metadata:
  name: chat-api-config
  namespace: atharva-app
data:
  RETRIEVER_URL: "http://atharva-retriever.atharva-ml.svc.cluster.local:8001"
  VLLM_URL: "http://atharva-vllm.atharva-ml.svc.cluster.local:8000"
  MODEL_NAME: "smollm2-135m-atharva"
```

### `k8s/40-serve/deploy-chat-api.yaml`

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atharva-chat-api
  namespace: atharva-app
spec:
  replicas: 1
  selector:
    matchLabels: { app: atharva-chat-api }
  template:
    metadata:
      labels: { app: atharva-chat-api }
    spec:
      containers:
      - name: api
        image: python:3.11-slim
        workingDir: /workspace
        command: ["bash","-lc"]
        args:
          - |
            pip install --no-cache-dir fastapi==0.112.2 uvicorn==0.30.6 httpx==0.27.2
            uvicorn chat_api:app --host 0.0.0.0 --port 8080 --proxy-headers
        envFrom:
        - configMapRef: { name: chat-api-config }
        volumeMounts:
        - name: host
          mountPath: /workspace
          subPath: atharva-dental-assistant/serving
        ports: [{ containerPort: 8080 }]
        readinessProbe:
          httpGet: { path: /health, port: 8080 }
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: host
        hostPath: { path: /mnt/project, type: Directory }
```

### `k8s/40-serve/svc-chat-api.yaml`

```
apiVersion: v1
kind: Service
metadata:
  name: atharva-chat-api
  namespace: atharva-app
spec:
  type: NodePort
  selector: { app: atharva-chat-api }
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    nodePort: 30300
```

*(Optional)* You can add an **Ingress** later; for local dev we’ll use `NodePort`.

---

## 4) Helper scripts

### `scripts/deploy_vllm.sh`

```
#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/40-serve/rawdeployment-vllm.yaml
kubectl apply -f k8s/40-serve/svc-vllm.yaml
echo "Waiting for vLLM Service endpoints..."
kubectl -n atharva-ml rollout status deploy/atharva-vllm-predictor --timeout=300s || true
kubectl -n atharva-ml get pods -l app=vllm -o wide
kubectl -n atharva-ml get svc atharva-vllm
```

> If the `rollout status` line errors (RawDeployment creates the Deployment; sometimes the generated name differs), don’t worry—check pods with the label `app=vllm`. If your cluster / KServe version behaves differently, just `kubectl -n atharva-ml get deploy,pod` to see the actual names.

### `scripts/deploy_chat_api.sh`

```
#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/40-serve/cm-chat-api.yaml
kubectl apply -f k8s/40-serve/deploy-chat-api.yaml
kubectl apply -f k8s/40-serve/svc-chat-api.yaml
kubectl -n atharva-app rollout status deploy/atharva-chat-api --timeout=180s
kubectl -n atharva-app get svc atharva-chat-api
```

### `scripts/smoke_e2e.sh`

```
#!/usr/bin/env bash
set -euo pipefail

CHAT_HOST="${CHAT_HOST:-127.0.0.1}"
CHAT_PORT="${CHAT_PORT:-30300}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-30200}"

chat_url="http://${CHAT_HOST}:${CHAT_PORT}"
vllm_url="http://${VLLM_HOST}:${VLLM_PORT}"

echo "=== End-to-End Test: Chat API -> Retriever -> vLLM ==="
echo "Chat API: ${chat_url}    vLLM: ${vllm_url}"
echo

# 1) Health checks
echo "[1/4] Health: Chat API"
curl -sf "${chat_url}/health" | jq . || { echo "Chat API health failed"; exit 1; }
echo

echo "[2/4] Health: vLLM (OpenAI models)"
curl -sf "${vllm_url}/v1/models" | jq '.data[0] // {}' || { echo "vLLM models failed"; exit 1; }
echo

# 2) Helper to run a chat and extract key fields
ask() {
  local q="$1"; local k="${2:-4}"; local max_tokens="${3:-256}"; local temp="${4:-0.3}"
  echo "Q: $q"
  resp="$(curl -s -X POST "${chat_url}/chat" \
    -H 'content-type: application/json' \
    -d "{\"question\":\"${q}\",\"k\":${k},\"max_tokens\":${max_tokens},\"temperature\":${temp}}")"

  # Pretty summary
  echo "$resp" | jq -r '
    . as $r |
    "─ Answer ─\n" +
    ($r.answer // "<no answer>") + "\n\n" +
    "─ Citations ─\n" + ((($r.citations // [])|join("\n")) // "<none>") + "\n\n" +
    "─ Stats ─\n" +
    ("latency_seconds: " + (($r.latency_seconds // 0)|tostring)) + "\n" +
    ("prompt_tokens: "   + (($r.usage.prompt_tokens // 0)|tostring)) + "\n" +
    ("completion_tokens:" + (($r.usage.completion_tokens // 0)|tostring)) + "\n"
  '
  echo "-------------------------------------------"
}

echo "[3/4] Functional E2E prompts"
ask "Are you open on Sundays ?"
ask "How long does scaling take and what aftercare is needed?"
ask "What is the typical cost range for a root canal and crown?" 4 256 0.2
ask "My face is badly swollen and I have a high fever after an extraction. What should I do?" 4 192 0.1

# 3) Optional: short latency/tokens smoke loop
echo
echo "[4/4] Throughput smoke (3 quick runs)"
for i in 1 2 3; do
  curl -s -X POST "${chat_url}/chat" \
    -H 'content-type: application/json' \
    -d '{"question":"Is next-day pain after RCT normal? Suggest aftercare.","k":3,"max_tokens":192}' \
    | jq -r '"run=\($i) lat=\(.latency_seconds)s tokens=(p:\(.usage.prompt_tokens // 0), c:\(.usage.completion_tokens // 0))"' --arg i "$i"
done

echo
echo "✅ E2E complete."
```

---

## 5) Install Kserve 

Install and validate kserve using 

```

# Setup Cert Manager 
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.0/cert-manager.yaml

# Validate cert-manager components are running. 
kubectl get pods -n cert-manager -w

# wait till cert-manager pods are up

# Install Kserver CRDs
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.15.2 --namespace kserve --create-namespace

# Validate CRDs are deployed 
helm list -A

# Install Kserver 
helm install kserve oci://ghcr.io/kserve/charts/kserve --version v0.15.2 \
  --namespace kserve \
  --set kserve.controller.deploymentMode=RawDeployment

# Validate kserver is deployed 
helm list -A

kubectl wait --for=condition=Available -n kserve deploy/kserve-controller-manager --timeout=300s

kubectl get pods -n kserve

```
---

## 5) Run the lab

From repo root:

```
# 1) Deploy vLLM (KServe RawDeployment) that mounts the model via ImageVolume
bash scripts/deploy_vllm.sh

# 2) Deploy the Chat API (RAG → prompt → vLLM)
bash scripts/deploy_chat_api.sh

# 3) End-to-end smoke test
bash scripts/smoke_e2e.sh
```


### Quick local validation

### 1) Model list

```
curl -s http://127.0.0.1:30200/v1/models | jq .
```

### 2) Simple chat completion

```
curl -s -X POST http://127.0.0.1:30300/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model":"smollm2-135m-atharva",
        "messages":[{"role":"user","content":"Say hello in 5 words."}],
        "max_tokens": 32,
        "temperature": 0.2
      }' | jq .
```


```
curl -s -X POST http://127.0.0.1:30300/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model":"smollm2-135m-atharva",
        "messages":[{"role":"user","content":"Say hello in 5 words."}],
        "max_tokens": 32,
        "temperature": 0.2
      }' | jq .
```

### 3) Legacy completions (optional)

```
curl -s -X POST http://127.0.0.1:30300/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model":"smollm2-135m-atharva",
        "prompt":"Write a haiku about dentists:",
        "max_tokens": 40
      }' | jq .
```


### 4) chat API

```
curl -s -X POST http://127.0.0.1:30300/chat -H "content-type: application/json" \
  -d '{"question":"Are you open on Sundays?","k":4}' | jq .

curl -s -X POST http://127.0.0.1:30300/chat -H "content-type: application/json" \
  -d '{"question":"Whats the price range for Root Canal Therapy?","k":4}' | jq .

```

### 5) To see whats being sent to the LLM 

```
curl -s -X POST http://127.0.0.1:30300/chat \\n  -H 'content-type: application/json' \\n  -d '{"question":"How long does scaling take and what aftercare is needed?","k":4,"debug":true}' \\n  | jq -r '.debug.messages[0].content'

curl -s -X POST http://127.0.0.1:30300/chat \\n  -H 'content-type: application/json' \\n  -d '{"question":"Whats the price range for Root Canal Therapy?","k":4,"debug":true}' \\n  | jq -r '.debug.messages[0].content'

```

> Expect answers in Atharva’s **concise, safety-minded** style with a trailing **`Source:`** line and citations like `treatments.json#TX-RCT-01`.

---

## Lab Summary 

This is what we accomplished in this lab

* Served your **fine-tuned SmolLM2-135M** using **vLLM** with model weights mounted from an **ImageVolume**.

* Exposed vLLM via a **Service** and wired a **Chat API** that performs **RAG → prompt → LLM**.

* Verified end-to-end responses and citations.



#courses/llmops/labs/v1
