# Lab 8 — Lightweight pipelines with Argo Workflows

## Goals

* Install **Argo Workflows** (minimal footprint) on KIND.
* Define a **WorkflowTemplate** that chains our steps in one DAG.
* Run it with one command (or wire it into ArgoCD later if you want GitOps for workflows too).

> Assumptions: You already have Labs 0–7 in place; your KIND nodes mount your repo at `/mnt/project`, and your repo path is `/mnt/project/atharva-dental-assistant`.

---

## What you’ll add

```
atharva-dental-assistant/
├─ k8s/
│  └─ 80-workflows/
│     ├─ rbac-argo-wf.yaml
│     ├─ workflowtemplate-atharva-train.yaml
│     └─ workflow-atharva-run.yaml          # (optional) direct Workflow using the template
└─ scripts/
   ├─ install_argo_workflows.sh
   └─ run_workflow.sh
```

---

## 1) Install Argo Workflows (helm, minimal)

### `scripts/install_argo_workflows.sh`

```
#!/usr/bin/env bash
set -euo pipefail

# Create Argo Workflows namespace
kubectl create namespace argo-workflows

# Install via Helm (lightweight defaults)
helm repo add argo https://argoproj.github.io/argo-helm >/dev/null
helm repo update >/dev/null
helm upgrade --install argo-workflows argo/argo-workflows \
  --namespace argo-workflows \
  --set server.enabled=true \
  --set server.authModes\[0\]=server \
  --set workflow.rbac.create=true \
  --set server.serviceType=NodePort \
  --set server.serviceNodePort=30600 

# Validate 
kubectl get all -n argo-workflows

# Print the NodePort for the UI
echo "Argo Workflows UI : http://127.0.0.1:30600"

```

> You can also expose the UI via Ingress later; NodePort is perfect for lab.

---

## 2) Allow workflows to run in `atharva-ml` with access to pods/workflows

```
# Run this from project/atharva-dental-assistant
mkdir k8s/80-workflows
```

### `k8s/80-workflows/rbac-argo-wf.yaml`

```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: wf-runner
  namespace: atharva-ml
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: wf-runner-role
  namespace: atharva-ml
rules:
- apiGroups: [""]
  resources: ["pods","pods/log","pods/exec","secrets","configmaps","persistentvolumeclaims","events"]
  verbs: ["create","get","list","watch","update","patch","delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create","get","list","watch","delete"]
- apiGroups: ["argoproj.io"]
  resources: ["workflowtaskresults","workflowtasksets"]
  verbs: ["create","get","list","watch","update","patch","delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: wf-runner-rb
  namespace: atharva-ml
subjects:
- kind: ServiceAccount
  name: wf-runner
  namespace: atharva-ml
roleRef:
  kind: Role
  name: wf-runner-role
  apiGroup: rbac.authorization.k8s.io
```

Apply it:

```
kubectl apply -f k8s/80-workflows/rbac-argo-wf.yaml
```

---

## 3) The WorkflowTemplate (our 4-step DAG)

### `k8s/80-workflows/workflowtemplate-atharva-train.yaml`

```
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: atharva-train-merge
  namespace: atharva-ml
spec:
  entrypoint: dag
  serviceAccountName: wf-runner
  podGC:
    strategy: OnWorkflowSuccess

  templates:
  - name: dag
    dag:
      tasks:
      - name: generate-data
        template: generate-data
      - name: build-index
        dependencies: [generate-data]
        template: build-index
      - name: train-lora
        dependencies: [build-index]
        template: train-lora
        arguments:
          parameters:
          - name: max-steps
            value: "{{workflow.parameters.max-steps}}"
      - name: merge-model
        dependencies: [train-lora]
        template: merge-model

  # --- Step 1: Generate synthetic training data (no heavy deps) ---
  - name: generate-data
    nodeSelector:
      kubernetes.io/hostname: llmops-kind-worker
    container:
      image: python:3.11-slim
      imagePullPolicy: IfNotPresent
      command: ["bash","-lc"]
      args:
        - |
          set -euo pipefail
          python /mnt/project/atharva-dental-assistant/tools/synth_data.py \
            --clinic Pune --currency INR \
            --treatments /mnt/project/atharva-dental-assistant/datasets/clinic/treatments.json \
            --policies /mnt/project/atharva-dental-assistant/datasets/clinic/policies/*.md \
            --faq /mnt/project/atharva-dental-assistant/datasets/clinic/faq.md \
            --recent /mnt/project/atharva-dental-assistant/datasets/clinic/recent_queries.jsonl \
            --out /mnt/project/atharva-dental-assistant/datasets/training
      volumeMounts:
      - name: host
        mountPath: /mnt/project
      resources:
        requests:
          cpu: "250m"
          memory: "512Mi"
    volumes:
    - name: host
      hostPath:
        path: /mnt/project
        type: Directory

  # --- Step 2: Build sparse TF-IDF index (lightweight, wheels-only) ---
  - name: build-index
    nodeSelector:
      kubernetes.io/hostname: llmops-kind-worker
    container:
      image: python:3.11-slim
      imagePullPolicy: IfNotPresent
      command: ["bash","-lc"]
      args:
        - |
          set -euo pipefail
          export HOME=/mnt/project
          VENV="$HOME/.venv-build"
          ROOT="$HOME/atharva-dental-assistant/datasets/clinic"
          OUT="$HOME/atharva-dental-assistant/artifacts/rag"

          python -m venv "$VENV"
          . "$VENV/bin/activate"
          python -m pip install --upgrade pip
          pip install --only-binary=:all: \
            numpy==1.26.4 scipy==1.10.1 scikit-learn==1.3.2 joblib==1.3.2

          mkdir -p "$OUT"
          python /mnt/project/atharva-dental-assistant/rag/build_index.py \
            --root "$ROOT" \
            --outdir "$OUT" \
            --backend sparse

          ls -lah "$OUT" && (wc -c "$OUT"/meta.json || true)
      volumeMounts:
      - name: host
        mountPath: /mnt/project
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
    volumes:
    - name: host
      hostPath:
        path: /mnt/project
        type: Directory

  # --- Step 3: LoRA training (matches job-train-lora.yaml) ---
  - name: train-lora
    inputs:
      parameters:
      - name: max-steps
        value: "400"
    nodeSelector:
      kubernetes.io/hostname: llmops-kind-worker
    container:
      image: schoolofdevops/lora-build-python:3.11-slim
      imagePullPolicy: IfNotPresent
      command: ["bash","-lc"]
      args:
        - |
          set -euo pipefail
          # MAX_STEPS is read by your training script from env
          export MAX_STEPS={{inputs.parameters.max-steps}}
          python /mnt/project/atharva-dental-assistant/training/train_lora.py
          # record last run id for the next step
          RUN_DIR="$(ls -1dt /mnt/project/atharva-dental-assistant/artifacts/train/*/ | head -n 1)"
          RUN_ID="$(basename "$RUN_DIR")"
          echo "$RUN_ID" > /mnt/project/atharva-dental-assistant/artifacts/train/LAST_RUN_ID.txt
      env:
      - name: BASE_MODEL
        value: "HuggingFaceTB/SmolLM2-135M-Instruct"
      - name: MAX_SEQ_LEN
        value: "256"
      - name: LORA_R
        value: "4"
      - name: LORA_ALPHA
        value: "8"
      - name: LORA_DROPOUT
        value: "0.05"
      - name: LR
        value: "2e-4"
      - name: WARMUP_RATIO
        value: "0.02"
      - name: BATCH_SIZE
        value: "1"
      - name: GRAD_ACCUM
        value: "1"
      - name: MAX_STEPS
        value: "80"                 # default; overridden by parameter above
      - name: DEMO_MAX_TRAIN_SAMPLES
        value: "0"
      - name: DEMO_MAX_VAL_SAMPLES
        value: "0"
      - name: HF_HOME
        value: "/cache/hf"
      - name: HF_HUB_DISABLE_TELEMETRY
        value: "1"
      - name: TOKENIZERS_PARALLELISM
        value: "true"
      - name: OMP_NUM_THREADS
        value: "4"
      - name: MKL_NUM_THREADS
        value: "4"
      - name: NUMEXPR_MAX_THREADS
        value: "4"
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
          ephemeral-storage: "5Gi"
        limits:
          cpu: "4"
          memory: "6Gi"
          ephemeral-storage: "20Gi"
      volumeMounts:
      - name: host
        mountPath: /mnt/project
      - name: hf-cache
        mountPath: /cache/hf
    volumes:
    - name: host
      hostPath:
        path: /mnt/project
        type: Directory
    - name: hf-cache
      hostPath:
        path: /mnt/hf-cache
        type: DirectoryOrCreate

  # --- Step 4: Merge LoRA into base model (matches job-merge-model.yaml) ---
  - name: merge-model
    nodeSelector:
      kubernetes.io/hostname: llmops-kind-worker
    container:
      image: schoolofdevops/lora-build-python:3.11-slim
      imagePullPolicy: IfNotPresent
      command: ["bash","-lc"]
      args:
        - |
          set -euo pipefail
          # If RUN_ID not provided, read the last run id produced by previous step
          export RUN_ID="${RUN_ID:-$(tr -d ' \n' </mnt/project/atharva-dental-assistant/artifacts/train/LAST_RUN_ID.txt)}"
          echo "Merging RUN_ID=$RUN_ID"
          python /mnt/project/atharva-dental-assistant/training/merge_lora.py
      env:
      - name: BASE_MODEL
        value: "HuggingFaceTB/SmolLM2-135M-Instruct"
      # Optional: allow overriding RUN_ID from workflow params if ever needed
      # - name: RUN_ID
      #   value: "REPLACE_WITH_RUN_ID"
      volumeMounts:
      - name: host
        mountPath: /mnt/project
      resources:
        requests:
          cpu: "500m"
          memory: "2Gi"
    volumes:
    - name: host
      hostPath:
        path: /mnt/project
        type: Directory

  arguments:
    parameters:
    - name: max-steps
      value: "100"
```

---

## 5) A concrete Workflow that uses the template (optional)

### `k8s/80-workflows/workflow-atharva-run.yaml`

```
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: atharva-run-
  namespace: atharva-ml
spec:
  serviceAccountName: wf-runner
  workflowTemplateRef:
    name: atharva-train-merge
  arguments:
    parameters:
    - name: max-steps
      value: "100"
```

---

## 6) Run it

```
# Install Argo Workflows (UI will be port-forwarded on 2746)
bash scripts/install_argo_workflows.sh
# (Use a new terminal if you want to keep the UI running)

# Create RBAC + PV/PVC + Template
kubectl apply -f k8s/80-workflows/rbac-argo-wf.yaml
kubectl apply -f k8s/80-workflows/pv-pvc-project.yaml
kubectl apply -f k8s/80-workflows/workflowtemplate-atharva-train.yaml

# Start a run
kubectl create -f k8s/80-workflows/workflow-atharva-run.yaml

# Watch progress
kubectl -n atharva-ml get wf
kubectl -n atharva-ml get pods -w
```

Or trigger from the **Argo Workflows UI** at [http://127.0.0.1:30600/](http://127.0.0.1:30600/) (submit from template, optionally set `max-steps` param).

When the workflow finishes, you’ll have:

```
artifacts/
  rag/...
  train/<RUN_ID>/
    lora_adapter/
    tokenizer/
    merged-model/
    model.tgz
    run.json
```

…ready for **Lab 3** (build OCI model image) and **Lab 4** (serve via vLLM) if you want to automate promotion later.

---

## Optional: GitOps this with ArgoCD

Since you already have ArgoCD managing serving/obs/scale, you can also let ArgoCD manage:

* the **WorkflowTemplate** YAML (declarative), and
* a **CronWorkflow** (if you want scheduled retraining; not included above to keep it minimal).

Example ArgoCD child app path: `k8s/80-workflows/` with sync-wave **1** (before serving) if you prefer.

---

## Why this is lightweight (in comparison to kubeflow ? )

* Argo Workflows adds a small controller + optional UI service (we enabled the server).
* No MinIO/MySQL pair like full Kubeflow Pipelines; all artifacts are simply **hostPath** on disk.
* Each step is a tiny **python:3.11-slim** container running the same scripts from your repo—no code duplication, no special SDK.
* You can swap images or add Kaniko/BuildKit step later if you want to bake the **model image** inside the workflow.

---

## Lab Summary 

This is what we accomplished in this lab

* Installed **Argo Workflows** and created a **WorkflowTemplate** to orchestrate data → index → train → merge.
* Reused `/mnt/project` via a PVC so steps share files seamlessly.
* Triggered a full run either via **kubectl** or the **Argo UI**.
* Kept the lab footprint **light** and aligned with your **ArgoCD** stack.


