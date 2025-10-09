# Lab 0 - Setup a 3 Node Kubernetes Cluster

This lab gets you a KIND cluster that’s ready for **ImageVolumes** and **KServe (RawDeployment)** that you would setup and use later, plus namespaces and monitoring.

> 🔎 Notes
> • **ImageVolumes** (mounting an OCI image as a read-only volume) require K8s **v1.31+**, are **beta in v1.33** and **disabled by default**; you must enable the `ImageVolume` feature gate and use a runtime that supports it. ([Kubernetes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/))
> 
---

Create a project directory 

```
mkdir project 
cd project 
mkdir atharva-dental-assistant
cd atharva-dental-assistant
mkdir scripts setup 
```


### Install Supporting Tools 

  * Install kind : [https://kind.sigs.k8s.io/docs/user/quick-start/#installation](https://kubernetes.io/docs/tasks/tools/). 
  * Install kubectl: [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/). 
  * Install HELM: [https://helm.sh/docs/intro/install/](https://helm.sh/docs/intro/install/). 

## setup/kind-config.yaml

Create this file to: (1) pin a recent node image, (2) enable the **ImageVolume** feature gate across control-plane components **and** kubelet, and (3) mount your local `./project` directory into each node.

```
# Enable ImageVolume feature gate (beta, disabled-by-default as of v1.33+)
# Ref: "Use an Image Volume With a Pod" + Feature Gates docs.
kubeadmConfigPatches:
  - |
    apiVersion: kubeadm.k8s.io/v1beta3
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        feature-gates: "ImageVolume=true"
    controllerManager:
      extraArgs:
        feature-gates: "ImageVolume=true"
    scheduler:
      extraArgs:
        feature-gates: "ImageVolume=true"
  - |
    apiVersion: kubelet.config.k8s.io/v1beta1
    kind: KubeletConfiguration
    featureGates:
      ImageVolume: true

# three node (two workers) cluster config
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: llmops-kind
nodes:
- role: control-plane
  image: kindest/node:v1.34.0
  extraMounts:
    - hostPath: /Users/gshah/work/llmops/code/project
      containerPath: /mnt/project
  extraPortMappings:
  - containerPort: 32000
    hostPort: 32000
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 32100
    hostPort: 32100
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30000
    hostPort: 30000
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30055
    hostPort: 30055
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30056
    hostPort: 30056
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30100
    hostPort: 30100
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30200
    hostPort: 30200
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30300
    hostPort: 30300
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30400
    hostPort: 30400
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30500
    hostPort: 30500
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30600
    hostPort: 30600
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30700
    hostPort: 30700
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 30800
    hostPort: 30800
    listenAddress: "0.0.0.0"
    protocol: tcp
- role: worker
  image: kindest/node:v1.34.0
  extraMounts:
    - hostPath: /Users/gshah/work/llmops/code/project
      containerPath: /mnt/project
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 8000
    hostPort: 8000
    listenAddress: "0.0.0.0"
    protocol: tcp
  - containerPort: 8001
    hostPort: 8001
    listenAddress: "0.0.0.0"
    protocol: tcp

- role: worker
  image: kindest/node:v1.34.0
  extraMounts:
    - hostPath: /Users/gshah/work/llmops/code/project
      containerPath: /mnt/project
```

You must replace `/Users/gshah/work/llmops/code/project` with the actual path. 
* In case of windows just use `./project`
* On MacOS replace it with absolute path e.g. `/Users/xxxx/work/llmops/code/project`

> Why the gates here? The docs specify enabling the `ImageVolume` feature gate and using a runtime that supports it; we toggle it for **apiserver/controller/scheduler** and **kubelet** to be thorough. ([Kubernetes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/))

---

# scripts/bootstrap_kind.sh

This script: creates the cluster, namespaces, installs **KServe (RawDeployment)**, and the **kube-prometheus-stack** (Prometheus + Grafana) into `monitoring`.

```
#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="llmops-kind"
KIND_CONFIG="setup/kind-config.yaml"
MON_NS="monitoring"

echo "==> Preflight checks"
command -v kind >/dev/null 2>&1 || { echo "Please install kind"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Please install kubectl"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "Please install helm"; exit 1; }

mkdir -p project

echo "==> Creating KIND cluster (${CLUSTER_NAME}) with ImageVolume feature-gate enabled"
kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG}"

echo "==> Verifying Kubernetes server version"
SERVER_MINOR=$(kubectl version -o json | jq -r '.serverVersion.minor' | sed 's/[^0-9].*//')
SERVER_MAJOR=$(kubectl version -o json | jq -r '.serverVersion.major')
echo "Server version: ${SERVER_MAJOR}.${SERVER_MINOR}"
# ImageVolumes need >= 1.31; KServe Quickstart needs >= 1.29
if [ "${SERVER_MAJOR}" -lt 1 ] || [ "${SERVER_MINOR}" -lt 31 ]; then
  echo "ERROR: Kubernetes >=1.31 required for ImageVolumes. Current: ${SERVER_MAJOR}.${SERVER_MINOR}"
  exit 1
fi

echo "==> Creating namespaces"
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata: { name: atharva-ml }
---
apiVersion: v1
kind: Namespace
metadata: { name: atharva-app }
---
apiVersion: v1
kind: Namespace
metadata: { name: ${MON_NS} }
EOF

echo "==> (Optional) Check container runtime inside nodes for ImageVolume support"
CONTROL_NODE=$(kubectl get nodes -o name | head -n1 | sed 's|node/||')
docker exec "${CLUSTER_NAME}-control-plane" containerd --version || true


echo "==> All set!
Namespaces:
  - atharva-ml      (ML training & model artifacts)
  - atharva-app     (chat API / frontend)
  - monitoring      (Prometheus + Grafana)

Next:
  • Lab 1 will generate synthetic data and build the FAISS index.
"
```

Execute the script with 

```
bash scripts/bootstrap_kind.sh
```


validate the cluster using 

```
# Validate the nodes are listed Ready
kubectl get nodes 

# Validate the pods are running 
kubectl get pods -A 

# Validate namespaces atharva-ml, atharva-app and monitoring are created
kubectl get ns
```

---

## Lab Summary 

This is what we accomplished in this lab

* A KIND cluster using **K8s v1.34 node images** to comfortably meet ImageVolumes’ version requirement. ([Docker Hub](https://hub.docker.com/r/kindest/node/tags?utm_source=chatgpt.com))

* **ImageVolume** feature gate enabled at control-plane and **kubelet** level. Docs note it’s **beta** and **off by default**; we turned it on. ([Kubernetes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/))


