# Lab 6 — Autoscaling vLLM + RAG + Chat API

## Goal

* **HPA**: scale Chat API on CPU (simple), and (optionally) on **Prometheus RPS** via Prometheus Adapter.

* **VPA**: get resource **recommendations** for vLLM & Retriever, then (optionally) apply.

* **KEDA**: event-driven **Prometheus** trigger (e.g., RPS per pod).

> Works with Labs 0–5 (Prometheus/Grafana already installed).

---

## What you’ll add

```
atharva-dental-assistant/
├─ k8s/
│  └─ 60-autoscale/
│     ├─ metrics-server-values.yaml
│     ├─ hpa-chat-api-cpu.yaml
│     ├─ adapter-values.yaml           # Prometheus Adapter (external/custom metrics)
│     ├─ hpa-chat-api-rps.yaml         # HPA using external metrics (RPS)
│     ├─ vpa-vllm.yaml
│     ├─ vpa-retriever.yaml
│     ├─ keda-values.yaml
│     ├─ keda-scaledobject-chat.yaml   # Prometheus trigger -> scale Chat API
│     └─ loadgen-job.yaml              # simple load generator
└─ scripts/
   └─ deploy_autoscaling.sh
```

Begin the la by creating the directory 

```
cd project/atharva-dental-assistant
mkdir k8s/60-autoscale/
```
---

## 0) Pre-reqs (install once)

### A) metrics-server (HPA on CPU needs this)

Kind often lacks it by default. you can check it by running the following 

```
kubectl top pod

kubectl top node
```
If you do not see the metrics but `error: Metrics API not available`, you need to install this component. 

```
# switch to home dir. alternately you pick a path
cd ~
git clone https://github.com/schoolofdevops/metrics-server.git
kubectl apply -k metrics-server/manifests/overlays/release
```

validate:

```
kubectl top pod

kubectl top node
```

[sample output]
```
➜  kubernetes kubectl top node
NAME                        CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)
llmops-kind-control-plane   285m         4%       1344Mi          13%
llmops-kind-worker          82m          1%       2721Mi          27%
llmops-kind-worker2         110m         1%       1377Mi          13%
➜  kubernetes kubectl top pods
NAME                                      CPU(cores)   MEMORY(bytes)
atharva-retriever-75b5996b54-w5vgc        4m           98Mi
atharva-vllm-predictor-64845bf646-dcdmf   14m          2117Mi
```

### B) Add Resource Spec to Chat API 


Step 2: Add Resource Spec to the Pod 

`File: k8s/40-serve/deploy-chat-api.yaml`
```
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
        resources:
          requests:
            cpu: "50m"
            memory: "64Mi"
          limits:
            cpu: "250m"
            memory: "256Mi"
```

```
kubectl apply -f k8s/40-serve/deploy-chat-api.yaml
```

validate 

```
kubectl describe pod -n atharva-app -l "app=atharva-chat-api"
```

Look for the container section where you should see the resource spec added. 

---

## 1) HPA on **CPU** (Chat API)

`k8s/60-autoscale/hpa-chat-api.yaml`

```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-chat-api
  namespace: atharva-app
spec:
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: ContainerResource
      containerResource:
        name: cpu
        container: api  # change this as per actual container name
        target:
          type: Utilization
          averageUtilization: 50
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: atharva-chat-api
  behavior:
    scaleDown:
      policies:
      - type: Pods
        value: 2
        periodSeconds: 120
      - type: Percent
        value: 25
        periodSeconds: 120
      stabilizationWindowSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 45
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 2
        periodSeconds: 15
      selectPolicy: Max
```

Apply:

```
kubectl apply -f k8s/60-autoscale/hpa-chat-api.yaml
kubectl -n atharva-app get hpa hpa-chat-api -w
```

[expected output]

```
NAME           REFERENCE                     TARGETS              MINPODS   MAXPODS   REPLICAS   AGE
hpa-chat-api   Deployment/atharva-chat-api   cpu: <unknown>/50%   2         6         2          2m54s
hpa-chat-api   Deployment/atharva-chat-api   cpu: 10%/50%         2         6         2          3m

```

Wait till you see the metrics appear and the minimum pods set to 2 as per HPA spec to proceed with load test next. 
------


## 3) VPA (recommendations for vLLM & Retriever)

Install the VPA components (once):

```
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler/ 

# deploy vpa 
./hack/vpa-up.sh
```


Recommend-only mode first:

`k8s/60-autoscale/vpa-vllm.yaml`

```
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-vllm
  namespace: atharva-ml
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: atharva-vllm-predictor
  updatePolicy:
    updateMode: "Off"     # "Auto" to apply automatically
```

`k8s/60-autoscale/vpa-retriever.yaml`

```
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-retriever
  namespace: atharva-ml
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: atharva-retriever
  updatePolicy:
    updateMode: "Off"
```

Apply:

```
kubectl apply -f k8s/60-autoscale/vpa-vllm.yaml
kubectl apply -f k8s/60-autoscale/vpa-retriever.yaml
# After some traffic:
kubectl -n atharva-ml describe vpa vpa-vllm | sed -n '1,200p'
kubectl -n atharva-ml describe vpa vpa-retriever | sed -n '1,200p'
```

---


## 5) Load generator (to trigger scaling)

A tiny **Job** that hammers the Chat API. Adjust `CONCURRENCY` & `REQUESTS`.

`k8s/60-autoscale/loadgen-job.yaml`

```
apiVersion: batch/v1
kind: Job
metadata:
  generateName: loadgen-job-
  namespace: atharva-app
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: hey
        image: schoolofdevops/hey:latest
        command: ["hey"]
        args:
          - "-z"
          - "4m"                                 # duration
          - "-q"
          - "50"                                 # target QPS
          - "-c"
          - "25"                                # concurrency
          - "-m"
          - "POST"
          - "-H"
          - "Content-Type: application/json"
          - "-d"
          - '{"question":"Do you accept UPI Payments?","k":4,"max_tokens":160}'
          - "http://atharva-chat-api.atharva-app.svc.cluster.local:8080/chat"
```

Run it:

```
kubectl create -f k8s/60-autoscale/loadgen-job.yaml
kubectl -n atharva-app get jobs 
kubectl -n atharva-app logs -f job/loadgen-job-xxxx
```

Watch scaling [in a new terminal]:

```
# Watch Horizontal Scling with HPA
kubectl -n atharva-app get hpa hpa-chat-api -w

# Check Resource Scaling Recommendations with VPA 
kubectl get vpa -w

# If using KEDA:
kubectl -n atharva-app get deploy/atharva-chat-api -w
```

Once the load test is complete you could delete HPA with 
```
kubectl delete -f k8s/60-autoscale/hpa-chat-api.yaml
```

---



### C) KEDA (for event-driven scale on Prometheus)


Install KEDA as 

```
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace
```

validate 
```
kubectl get all -n keda
```


---


## 4) KEDA ScaledObject (Prometheus trigger)

Scale **Chat API** when **RPS > 1 per replica** (same signal as HPA external metric).
Point KEDA to Prometheus from kube-prometheus-stack.

`k8s/60-autoscale/keda-scaledobject-chat.yaml`

```
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: chat-api-qps
  namespace: atharva-app
spec:
  scaleTargetRef:
    name: atharva-chat-api
  minReplicaCount: 1
  maxReplicaCount: 8
  cooldownPeriod: 60
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://kps-kube-prometheus-stack-prometheus.monitoring.svc:9090
        metricName: vllm_queue_length
        threshold: "5"
        query: sum(vllm:num_requests_waiting)
```

Apply:

```
kubectl apply -f k8s/60-autoscale/keda-scaledobject-chat.yaml
kubectl -n atharva-app get scaledobject
```

launch the load test and watch ScaledObject + HPA
```
kubectl create -f k8s/60-autoscale/loadgen-job.yaml
watch kubectl -n atharva-app get scaledobject,hpa
```

> You don’t need Prometheus Adapter for KEDA if you use the `prometheus` trigger (it queries Prom directly). We keep Adapter anyway for the HPA external-metric demo.

---

## 5) Load generator (to trigger scaling)

A tiny **Job** that hammers the Chat API. Adjust `CONCURRENCY` & `REQUESTS`.

`k8s/60-autoscale/loadgen-job.yaml`

```
apiVersion: batch/v1
kind: Job
metadata:
  generateName: loadgen-job-
  namespace: atharva-app
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: hey
        image: schoolofdevops/hey:latest
        command: ["hey"]
        args:
          - "-z"
          - "3m"                                 # duration
          - "-q"
          - "5"                                 # target QPS
          - "-c"
          - "5"                                # concurrency
          - "-m"
          - "POST"
          - "-H"
          - "Content-Type: application/json"
          - "-d"
          - '{"question":"Do you accept UPI Payments?","k":4,"max_tokens":160}'
          - "http://atharva-chat-api.atharva-app.svc.cluster.local:8080/chat"
```

Run it:

```
kubectl create -f k8s/60-autoscale/loadgen-job.yaml
kubectl get pods -n atharva-app
kubectl -n atharva-app logs -f xxxx
```

Watch scaling:

```
# If using HPA CPU:
kubectl -n atharva-app get hpa hpa-chat-api -w

# If using KEDA:
kubectl -n atharva-app get deploy/atharva-chat-api -w
```

---


## Tips / Gotchas

* **Pick one scaler** per target at a time to avoid fights. For demos:

  1. Start with **CPU HPA**,

  2. Switch to **RPS HPA** (delete CPU HPA),

  3. Try **KEDA**.

* HPA external metrics require **Prometheus Adapter** (we installed & mapped `chat_requests_per_second`).

* KEDA Prometheus trigger queries Prom directly; Adapter not required for that path.

* **VPA in “Off”** mode shows recommendations under `describe vpa`. Flip to `"Auto"` if you want automatic updates (be mindful of restarts).

---

## Lab Summary 

This is what we accomplished in this lab

* Horizontal scaling of **Chat API** on **CPU** or **RPS** (Prometheus-backed).

* Vertical **recommendations** for **vLLM** and **Retriever** to right-size resources.

* Event-driven scaling with **KEDA** using a Prometheus trigger.

* A simple **load generator** to visualize scaling behavior.


#courses/llmops/labs/v1
