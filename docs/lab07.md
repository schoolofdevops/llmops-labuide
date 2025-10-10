# Lab 7 — GitOps with ArgoCD

## Goal

* Run ArgoCD in-cluster.

* Manage all Atharva components as **declarative Applications** driven from Git.

* Order deploys via **sync waves**.

* Promote models by editing **one line** (ImageVolume `reference:` tag) and letting Argo roll it out.

* Roll back with a Git revert.

> Assumptions

* Your repo is at `https://github.com/<you>/atharva-dental-assistant.git` (replace everywhere below).

* Labs 1–6 manifests already live under `atharva-dental-assistant/k8s/**`.

---

## 0) Repo layout (GitOps view)

Add this under `k8s/70-gitops/`:

```
k8s/
  70-gitops/
    app-of-apps.yaml           # the “root” Application
    projects/
      atharva-project.yaml     # (optional) ArgoCD project
    apps/
      app-data.yaml            # data synth + FAISS index jobs (Lab 1)
      app-train.yaml           # LoRA train + merge (Lab 2)
      app-model.yaml           # model mount check (Lab 3) or future registry checks
      app-serve.yaml           # retriever + vLLM + chat API (Labs 1 & 4)
      app-observability.yaml   # SMs, alerts, grafana dashboard (Lab 5)
      app-autoscale.yaml       # HPA/VPA/KEDA (Lab 6)
```

Each Application points at an existing **path** in your repo. No refactor required.

```
cd project/atharva-dental-assistant
mkdir k8s/70-gitops
mkdir k8s/70-gitops/projects
mkdir k8s/70-gitops/apps
touch .gitignore
```

Also prepare your repo to be revision controlled so that you could deploy using GitOps. 

Add `.gitignore` file to atharva-dental-assistant
```
# ============================
# Python
# ============================
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.pdb
*.so
*.orig
*.log
*.tmp
*.swp
.env
.venv
venv/
env/
pip-wheel-metadata/
*.egg-info/
.eggs/
dist/
build/

# ============================
# Datasets & Artifacts
# ============================
artifacts/
datasets/training/*
!datasets/clinic/**

# Keep clinic dataset (source knowledge)
# Ignore training data (sensitive or large)

# ============================
# Model Checkpoints / Outputs
# ============================
*.ckpt
*.pt
*.bin
*.onnx
*.safetensors
*.h5
*.tar.gz
*.zip
*.npz

# ============================
# Jupyter & Dev Artifacts
# ============================
.ipynb_checkpoints/
*.ipynb~
*.DS_Store
*.idea/
.vscode/
*.code-workspace

# ============================
# Kubernetes & Deployment Junk
# ============================
# Ignore backups or temp yaml files
k8s/**/*.orig
k8s/**/*.bak
k8s/**/*.tmp
# Keep main manifests
!k8s/**/*.yaml

# ============================
# Logs & Results
# ============================
logs/
*.log
*.out
*.err
nohup.out
results/
outputs/

# ============================
# Temporary Scripts & Cache
# ============================
tmp/
.cache/
.tox/
pytest_cache/
__snapshots__/
.coverage
coverage.xml
htmlcov/

# ============================
# Local Configs & Misc
# ============================
*.local
*.env
.envrc
.env.*
*.cfg
*.ini
*.toml
*.jsonl.backup

# ============================
# OS / Editor
# ============================
.DS_Store
Thumbs.db
.swp
*.bak
*.tmp
```

Create a new repo [https://github.com/new](https://github.com/new) 

e.g.  `xxxxxx/atharva-dental-assistant`
![](Screenshot%202025-10-06%20at%202.18.06%E2%80%AFPM.png)

Initialise the local repo and prepate to push the changes 

```
# make sure you are in the right path
cd project/atharva-dental-assistant

git init
git add * 
git commit -am "importing llmops code"

# Replace xxxxxx with your github id
git remote add origin https://github.com/xxxxxx/atharva-dental-assistant.git

git push -u origin main
```

---

## 1) Install ArgoCD

Create namespace & install:

```
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml


# scale down optional components
# kubectl -n argocd scale deploy argocd-dex-server argocd-notifications-controller argocd-redis argocd-applicationset-controller --replicas=0

# Wait & get admin password
kubectl -n argocd rollout status deploy/argocd-server --timeout=300s || true
kubectl -n argocd get pods
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d; echo

# Patch and expose ArgoCD sevice on port 32100
kubectl patch svc argocd-server -n argocd --patch \
  '{"spec": { "type": "NodePort", "ports": [ { "nodePort": 32100, "port": 443, "protocol": "TCP", "targetPort": 8080 } ] } }'

kubectl get svc -n argocd
```

Login by visiting http://localhost:32100

Login in another shell:

```
argocd login 127.0.0.1:32100 --username admin --password <printed-password> --insecure
```

> If you don’t have the `argocd` CLI, you can do everything from the UI.

---

## 2) ArgoCD Project (optional but tidy)

`k8s/70-gitops/projects/atharva-project.yaml`

```
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: atharva
  namespace: argocd
spec:
  description: Atharva Dental Clinic - LLMOps Bootcamp
  sourceRepos:
    - https://github.com/xxxxxx/atharva-dental-assistant.git
  destinations:
    - namespace: atharva-ml
      server: https://kubernetes.default.svc
    - namespace: atharva-app
      server: https://kubernetes.default.svc
    - namespace: monitoring
      server: https://kubernetes.default.svc
    - namespace: keda
      server: https://kubernetes.default.svc
    - namespace: argocd
      server: https://kubernetes.default.svc
  clusterResourceWhitelist:
    - group: '*'
      kind: '*'
```

Ensure you replace `xxxxxx` with the actual Git Repository 


Apply:

```
kubectl apply -f k8s/70-gitops/projects/atharva-project.yaml
```

---

## 3) App-of-Apps (the “root” Application)

`k8s/70-gitops/app-of-apps.yaml`

```
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: atharva-root
  namespace: argocd
spec:
  project: atharva
  source:
    repoURL: https://github.com/xxxxxx/atharva-dental-assistant.git
    targetRevision: main
    path: k8s/70-gitops/apps
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```
Ensure you replace `xxxxxx` with the actual Git Repository 

Apply:

```
kubectl apply -f k8s/70-gitops/app-of-apps.yaml
```

---

## 4) Child Applications (one per stack)

> We’ll sequence with **sync-waves** so infra comes first, then serving, then autoscaling:

* Wave 0: **observability** (Prometheus already installed in Lab 0; this just adds SMs & dashboards)

* Wave 1: **data** (jobs), **train** (jobs), **model** (image-volume smoke)

* Wave 2: **serve** (retriever, vLLM, chat API)

* Wave 3: **autoscale** (HPA/VPA/KEDA)

Create files in `k8s/70-gitops/apps/`:

```
cd k8s/70-gitops/apps/
```

### `app-observability.yaml` (wave 0)

```
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: atharva-observability
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "0"
spec:
  project: atharva
  source:
    repoURL: https://github.com/xxxxxx/atharva-dental-assistant.git
    targetRevision: main
    path: k8s/50-observability
  destination:
    server: https://kubernetes.default.svc
    namespace: monitoring
  syncPolicy:
    automated: { prune: true, selfHeal: true }
```
Ensure you replace `xxxxxx` with the actual Git Repository 

### `app-serve.yaml` (wave 1)

Points at Lab 4 (vLLM RawDeployment + svc, chat API, cm).
Includes **ignoreDifferences** for KServe’s generated fields (if needed) and a health timeout.

```
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: atharva-serve
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "1"
spec:
  project: atharva
  source:
    repoURL: https://github.com/xxxxxx/atharva-dental-assistant.git
    targetRevision: main
    path: k8s/40-serve
  destination:
    server: https://kubernetes.default.svc
    namespace: atharva-ml
  syncPolicy:
    automated: { prune: true, selfHeal: true }
  ignoreDifferences:
    - group: serving.kserve.io
      kind: RawDeployment
      jsonPointers:
        - /spec/deployments/0/replicas    # controller may reconcile
  syncOptions:
    - Validate=true
    - PrunePropagationPolicy=background
  revisionHistoryLimit: 5
```
Ensure you replace `xxxxxx` with the actual Git Repository 

> If your Chat API manifests live in `atharva-app`, keep them in `k8s/40-serve/`—Argo will create resources in their native namespaces. (Your YAML already sets the namespace for each object.)

### `app-autoscale.yaml` (wave 2)

Points at Lab 6 (HPA/VPA/KEDA + loadgen).

```
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: atharva-autoscale
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "2"
spec:
  project: atharva
  source:
    repoURL: https://github.com/xxxxxx/atharva-dental-assistant.git
    targetRevision: main
    path: k8s/60-autoscale
  destination:
    server: https://kubernetes.default.svc
    namespace: atharva-app
  syncPolicy:
    automated: { prune: true, selfHeal: true }
```
Ensure you replace `xxxxxx` with the actual Git Repository 

In case if you want to bulk replace `xxxxxx` with your userid do this 
```
sed -i '' 's/xxxxxx/yourid/g' *.yaml
```
where `yourid` is what you are replacing the placeholders with 

Also create kustomization spec at `k8s/60-autoscale/kustomization.yaml` to apply selective manifests 

```
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - vpa-retriever.yaml
  - vpa-vllm.yaml
  - keda-scaledobject-chat.yaml
```

Commit & push these files.

```
# run this from atharva-dental-assistant path
git add k8s/*
git status
git commit -m "adding argocd app spec"
git push -u origin main
```
---

## 5) Register the Git repo (CLI alternative) 

If your repo is **private**, add credentials:

```
argocd repo add https://github.com/<you>/atharva-dental-assistant.git \
  --username <user> --password <token> --insecure-ignore-host-key
```

If **public**, no action needed.

---

## 6) Sync

From CLI:

```
argocd app sync atharva-root
argocd app list
```

Or click **Sync** in Argo UI for `atharva-root`.
Argo will create child Applications in wave order, then deploy resources from each path.

---

## 7) Model promotion (Git change → rollout)

The **one-line** you change to promote a model is the **ImageVolume reference** in `k8s/40-serve/rawdeployment-vllm.yaml`:

```
volumes:
- name: model
  image:
    reference: kind-registry:5001/atharva/smollm2-135m-merged:v2   # <— change tag v1 → v2
    pullPolicy: IfNotPresent
```

**Flow**

1. Build & push your new model image (Lab 3 scripts), e.g. `:v2`.

2. Edit the `reference:` tag to `:v2`, commit & push to `main`.

3. ArgoCD detects the diff, syncs, and the **vLLM** pod rolls to the new mounted model.

4. Validate with your Lab 4 smoke test.

5. **Rollback**: `git revert` that commit (or change back to `:v1`) → Argo rolls back.

⠀
> Bonus: add a **PreSync hook** job in `k8s/30-model/` to **cosign verify** the model image before serving. If verify fails, sync fails.

---

## 8) (Optional) App-of-Apps + Staging/Prod 

Create two roots: `atharva-root-staging` and `atharva-root-prod`, each pointing at a different branch (`staging`, `main`).
Promotion becomes a **PR/merge** from staging → main. The only change needed for a model rollout is still that **one tag**.

---

## 9) (Optional) Argo CD Image Updater

If you’d like automatic bumping when a new tag appears in the registry:

* Install **argocd-image-updater**.

* Annotate `app-serve.yaml` with tracking rules for `kind-registry:5001/atharva/smollm2-135m-merged`.

* It will open PRs or auto-commit tag bumps to your Git repo.

(Keep manual edits for the bootcamp exercises first.)

---

## 10) Quick commands cheat-sheet

```

# List applications
argocd app list

# Sync root
argocd app sync atharva-root

# See app tree
argocd app get atharva-root

# Rollback a child app to a past revision (if needed)
argocd app history atharva-serve
argocd app rollback atharva-serve <ID>
```

---

## Lab Summary 

This is what we accomplished in this lab

* Stood up **ArgoCD** and declared an **App-of-Apps** hierarchy.

* Put **every lab stack under GitOps** with clear sequencing.

* Practiced a **model promotion** by changing the **ImageVolume tag** in Git.

* Learned how to **rollback** using Git/Argo history.


