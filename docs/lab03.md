# Lab 3 - Packaging Model as OCI Image 

Awesome—moving into **Lab 3: Package the merged model as an OCI artifact and mount it via ImageVolumes**.
This lab gives you **production-like model delivery**: immutable, versioned, and ArgoCD/GitOps-friendly later.

---

# Lab 3 — Model as OCI Artifact + ImageVolume mount

## Goal

* Turn the merged model from **Lab 2** into an **OCI image** (no app code, just `/model` files).

* Push it to a **local registry** accessible by your KIND nodes.

* **Mount** it into a pod using **ImageVolumes** and verify the files.

> Works with the cluster you created in Lab 0 (ImageVolume feature-gate already enabled).

---

## What you’ll add in this lab

```
atharva-dental-assistant/
├─ training/
│  └─ Dockerfile.model-asset         # builds an image containing /model
├─ k8s/
│  └─ 30-model/
│     ├─ model-mount-check.yaml      # Pod that mounts the model via ImageVolume
│     └─ (optional) cosign-verify-job.yaml
└─ scripts/
   ├─ start_local_registry.sh        # runs registry at kind-registry:5001 and nets it to KIND
   ├─ build_model_image.sh           # builds + tags image from artifacts/train/<RUN_ID>/merged-model
   └─ model_mount_smoke.sh           # applies Pod, inspects /model
```

---

```
cd project/atharva-dental-assistant
mkdir k8s/30-model
```


## 1) Sign up to DockerHub Registry 

Sign up to https://hub.docker.com/ if you haven’t already.

---

## 2) Build a model “asset” image (only the weights)

We’ll COPY the **merged** model folder (created in Lab 2) into `/model` in an image and push it to `kind-registry:5001`.

### `training/Dockerfile.model-asset`

```
# Minimal base; just hosts files at /model
FROM alpine:3.20
RUN adduser -D -H -s /sbin/nologin model && mkdir -p /model && chown -R model /model
# Copy the merged Transformers folder produced by Lab 2
# (Contains config.json, .safetensors, tokenizer files, etc.)
COPY artifacts/train/REPLACE_RUN_ID/merged-model/ /model/
USER model
```

> We’ll patch `REPLACE_RUN_ID` at build time.

### `scripts/build_model_image.sh`

```
#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <RUN_ID> <TAG>   e.g. $0 20251001-113045 v1"
  exit 1
fi
RUN_ID="$1"
USER="$2"
TAG="$3"

IMG="${USER}/smollm2-135m-merged:${TAG}"

# Safety checks
[ -d "artifacts/train/${RUN_ID}/merged-model" ] || { echo "Merged model folder not found for RUN_ID=${RUN_ID}"; exit 1; }

# Create a temp Dockerfile with RUN_ID patched
TMP_DF=$(mktemp)
sed "s|REPLACE_RUN_ID|${RUN_ID}|g" training/Dockerfile.model-asset > "$TMP_DF"

echo "==> Building model asset image: ${IMG}"
docker build -f "$TMP_DF" -t "${IMG}" .

# Optional, enable if you want to push this image to DockerHub
#echo "==> Pushing to local registry ${IMG}"
#docker push "${IMG}"

echo "Done. Image: ${IMG}"
```

Run it:

```
# Example
bash scripts/build_model_image.sh <RUN_ID> <DOCKERHUB_USERNAME> v1
# e.g. bash scripts/build_model_image.sh 20251001-113045 initcron v1
```

You should now have an image `kind-registry:5001/atharva/smollm2-135m-merged:v1` in the local registry.

Load this image to `llmops-kind-worker` node 

```
kind load docker-image --name llmops-kind xxxxxx/smollm2-135m-merged:v2 --nodes llmops-kind-worker
```
where, replace `xxxxxx` with your DockerHub ID 

---

## 3) Test Mount the model via **ImageVolume**

This Pod mounts the image’s `/model` directory read-only at `/model` in the container. We then `ls -lah /model`.

### `k8s/30-model/model-mount-check.yaml`

```
apiVersion: v1
kind: Pod
metadata:
  name: model-mount-check
  namespace: atharva-ml
spec:
  restartPolicy: Never
  nodeName: llmops-kind-worker
  containers:
  - name: inspect
    image: debian:12-slim
    command: ["bash","-lc","ls -lah /model/model && head -n 50 /model/model/config.json || true && sleep 5"]
    volumeMounts:
    - name: model
      mountPath: /model
      readOnly: true
  volumes:
  - name: model
    image:
      reference: xxxxxx/smollm2-135m-merged:v1
      pullPolicy: IfNotPresent
```

where, replace `xxxxxx` with actual dockerhub user/org, update tag (e.g. v1) if necessary. 

> If you used a different tag or repo path, update `reference:` accordingly.

### `scripts/model_mount_smoke.sh`

```
#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/30-model/model-mount-check.yaml
echo "Waiting for Pod to complete..."
kubectl -n atharva-ml wait --for=condition=Ready pod/model-mount-check --timeout=120s || true
kubectl -n atharva-ml logs pod/model-mount-check
```

Run it:

```
bash scripts/model_mount_smoke.sh
```

Expected output (truncated):

```
/model
total 120M
-rw-r--r--    1 model    model     1.2K config.json
-rw-r--r--    1 model    model      512 tokenizer_config.json
-rw-r--r--    1 model    model       96 tokenizer.json
-rw-r--r--    1 model    model      120 model.safetensors
...
{
  "architectures": ["MistralForCausalLM"],
  "model_type": "mistral",
  ...
}
```

(Your `config.json` fields will reflect SmolLM2.)

------

## Lab Summary 

This is what we accomplished in this lab

* Stood up a **local OCI registry** that KIND nodes can reach (`kind-registry:5001`).

* Packaged your **merged Transformers model** into an **immutable image**.

* **Mounted** it with **ImageVolumes**—the exact pattern you’ll use for **vLLM** in KServe.

* (Optional) Practiced **image signing** to prepare for a secure rollout.



#courses/llmops/labs/v1
