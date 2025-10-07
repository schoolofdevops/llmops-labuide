# Lab 2: CPU LoRA fine-tuning (SmolLM2-135M)

Amazing—here’s **Lab 2: CPU LoRA fine-tuning (SmolLM2-135M)**, ready to drop into your repo and run on the KIND cluster from Lab 0.

This lab:

* fine-tunes **HuggingFaceTB/SmolLM2-135M-Instruct** on **CPU** with **PEFT LoRA**, using the synthetic chat JSONL you generated in Lab 1,

* saves **LoRA adapters** + checkpoints under `artifacts/train/<run-id>/`,

* **merges** the LoRA into base weights for serving,

* produces a **Transformers-style model folder**, and a **tarball** you’ll later wrap into an OCI image for ImageVolumes.

---

# 📁 Files added in this lab

```
atharva-dental-assistant/
├─ training/
│  ├─ Dockerfile
│  ├─ train_lora.py
│  ├─ merge_lora.py
│  └─ prompt_utils.py
├─ k8s/
│  └─ 20-train/
│     ├─ job-train-lora.yaml
│     └─ job-merge-model.yaml
└─ scripts/
   ├─ train_lora.sh
   └─ merge_model.sh
```

> Assumes your repo still lives under `./project/atharva-dental-assistant` on the host (mounted into nodes at `/mnt/project/atharva-dental-assistant` per Lab 0).

---
```
# make sure you are in the right project path 
project/atharva-dental-assistant

# create directories you would need as part of this lab
mkdir training k8s/20-train
```

## training/Dockerfile

CPU-only image with pinned libs. (No CUDA; uses PyTorch CPU wheel.)

File : `training/Dockerfile`

```
#training/Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch + Transformers/PEFT/Datasets/Accelerate
RUN pip install --no-cache-dir "torch==2.3.1" \
    "transformers==4.43.3" "peft==0.12.0" "datasets==2.20.0" \
    "accelerate==0.33.0" "tqdm==4.66.5" "bitsandbytes==0.43.1" \
    "sentencepiece==0.2.0" "numpy==1.26.4" "pydantic==2.8.2"

WORKDIR /workspace
COPY training/train_lora.py training/merge_lora.py training/prompt_utils.py /workspace/
```

> `bitsandbytes` is optional; it will no-op on CPU (kept for parity if you demonstrate QLoRA later).

---

## training/prompt_utils.py

Utilities to turn our `messages` (from JSONL) into tokenizable text.
We try `tokenizer.apply_chat_template` (preferred) and fallback to a simple template.

```
# training/prompt_utils.py
from typing import List, Dict

DEFAULT_SYSTEM = (
    "You are Atharva Dental Clinic assistant in Pune, India (INR). "
    "Always use INR (₹) as a currency for prices and cost ranges, "
    "Be concise, safety-minded, ask follow-ups if info is missing, "
    "Consider the context provided and derive answered based on that, "
    "and always include a final 'Source:' line citing file#section."
)

def to_chat(messages: List[Dict], default_system: str = DEFAULT_SYSTEM):
    """
    Ensure we have system→user→assistant message order for a single-turn sample.
    Our dataset already stores messages with roles. We enforce one assistant turn.
    """
    sys_seen = any(m["role"] == "system" for m in messages)
    msgs = []
    if not sys_seen:
        msgs.append({"role": "system", "content": default_system})
    msgs.extend(messages)
    # Basic guard: keep only first assistant answer for label masking
    out, assistant_added = [], False
    for m in msgs:
        if m["role"] == "assistant":
            if assistant_added:  # drop extra assistant turns for simplicity
                continue
            assistant_added = True
        out.append(m)
    return out

def simple_template(messages: List[Dict]) -> str:
    """
    Fallback formatting if tokenizer has no chat template.
    """
    lines = []
    for m in messages:
        role = m["role"]
        prefix = {"system":"[SYS]", "user":"[USER]", "assistant":"[ASSISTANT]"}.get(role, f"[{role.upper()}]")
        lines.append(f"{prefix}\n{m['content'].strip()}\n")
    # Ensure the string ends with assistant text (trainer expects labels on last turn)
    return "\n".join(lines).strip()
```

---

## training/train_lora.py

* Loads `datasets/training/train.jsonl` and `val.jsonl`

* Uses **chat template** if model has one; otherwise falls back

* **Masks labels** so loss is computed only on the assistant span

* Trains with **Trainer** on CPU with your hyperparams

* Saves adapter + tokenizer + minimal `run.json`

```
# training/train_lora.py  (fast demo edition)
import os, json, time, math, random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt_utils import to_chat, simple_template, DEFAULT_SYSTEM

BASE_DIR = Path("/mnt/project/atharva-dental-assistant")
DATA_DIR = BASE_DIR / "datasets" / "training"

# ------------------------------
# Demo-friendly defaults (override via env)
# ------------------------------
BASE_MODEL   = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
MAX_SEQ_LEN  = int(os.environ.get("MAX_SEQ_LEN", "256"))     # ↓ from 512
LORA_R       = int(os.environ.get("LORA_R", "4"))            # ↓ from 8
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "8"))        # ↓ from 16
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LR           = float(os.environ.get("LR", "2e-4"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.02"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))        # ↓ from 8
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "80"))        # ↓ from 400 (≈5–10 min)
# Optional dataset subsample for speed (0 = use all)
DEMO_MAX_TRAIN_SAMPLES = int(os.environ.get("DEMO_MAX_TRAIN_SAMPLES", "0"))
DEMO_MAX_VAL_SAMPLES   = int(os.environ.get("DEMO_MAX_VAL_SAMPLES", "0"))

OUTPUT_ROOT = BASE_DIR / "artifacts" / "train" / time.strftime("%Y%m%d-%H%M%S")

# Use all CPU cores for a faster demo
torch.set_num_threads(max(1, os.cpu_count()))

print(f"Base model: {BASE_MODEL}")
print(f"Output dir: {OUTPUT_ROOT}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None
)
model = prepare_model_for_kbit_training(model)  # safe on CPU

# Trim LoRA to attention projections only (fewer trainable params)
peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

def build_example(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    msgs = to_chat(messages, DEFAULT_SYSTEM)
    use_chat_template = hasattr(tokenizer, "apply_chat_template")
    text_prompt = (
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        if use_chat_template else simple_template(msgs)
    )
    # Find assistant content for label masking
    assistant_text = [m["content"] for m in msgs if m["role"]=="assistant"][-1]
    _ = assistant_text.strip()

    tok = tokenizer(text_prompt, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)

    prefix_msgs = [m for m in msgs if m["role"]!="assistant"]
    prefix_text = (
        tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
        if use_chat_template else simple_template(prefix_msgs) + "\n[ASSISTANT]\n"
    )
    prefix_tok = tokenizer(prefix_text, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)

    input_ids = tok["input_ids"]
    labels = input_ids.copy()
    mask_len = min(len(prefix_tok["input_ids"]), len(labels))
    labels[:mask_len] = [-100] * mask_len
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1]*len(input_ids)}

def load_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        yield json.loads(line)

# ------------------------------
# Load + optional subsample
# ------------------------------
train_records = list(load_jsonl(DATA_DIR/"train.jsonl"))
val_records   = list(load_jsonl(DATA_DIR/"val.jsonl"))

if DEMO_MAX_TRAIN_SAMPLES > 0 and len(train_records) > DEMO_MAX_TRAIN_SAMPLES:
    random.seed(42)
    train_records = random.sample(train_records, DEMO_MAX_TRAIN_SAMPLES)
if DEMO_MAX_VAL_SAMPLES > 0 and len(val_records) > DEMO_MAX_VAL_SAMPLES:
    random.seed(123)
    val_records = random.sample(val_records, DEMO_MAX_VAL_SAMPLES)

train_ds = [build_example(rec["messages"]) for rec in train_records]
val_ds   = [build_example(rec["messages"]) for rec in val_records]

@dataclass
class Collator:
    pad_token_id: int = tokenizer.pad_token_id
    def __call__(self, batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad    = [self.pad_token_id] * (maxlen - len(x["input_ids"]))
            maskpd = [0] * (maxlen - len(x["attention_mask"]))
            lblpd  = [-100] * (maxlen - len(x["labels"]))  # ← fixed missing ')'
            input_ids.append(x["input_ids"] + pad)
            labels.append(x["labels"] + lblpd)
            attn.append(x["attention_mask"] + maskpd)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Training args: no eval during train, single final save, fewer steps
# ------------------------------
args = TrainingArguments(
    output_dir=str(OUTPUT_ROOT),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    max_steps=MAX_STEPS,
    lr_scheduler_type="cosine",
    logging_steps=50,
    evaluation_strategy="no",         # ← no eval to save time
    save_steps=10_000_000,            # ← avoid mid-run checkpoints
    save_total_limit=1,               # ← keep only final
    bf16=False, fp16=False,
    dataloader_num_workers=0,
    report_to="none"
)

# Helpful run summary
N = len(train_ds)
steps_per_epoch = max(1, math.ceil(N / (BATCH_SIZE * GRAD_ACCUM)))
est_epochs = args.max_steps / steps_per_epoch
print(f"Train examples: {N}, steps/epoch: {steps_per_epoch}, "
      f"optimizer steps: {args.max_steps}, ~epochs: {est_epochs:.2f}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=None,    # no eval during training
    data_collator=Collator(),
)

trainer.train()

# Save adapter + tokenizer
model.save_pretrained(str(OUTPUT_ROOT / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_ROOT / "tokenizer"))

# Run manifest
(OUTPUT_ROOT / "run.json").write_text(json.dumps({
    "base_model": BASE_MODEL,
    "max_seq_len": MAX_SEQ_LEN,
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
    "lr": LR, "warmup_ratio": WARMUP_RATIO,
    "batch": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "max_steps": MAX_STEPS,
    "demo_max_train_samples": DEMO_MAX_TRAIN_SAMPLES,
    "demo_max_val_samples": DEMO_MAX_VAL_SAMPLES
}, indent=2), encoding="utf-8")

print(f"Training complete. Artifacts at {OUTPUT_ROOT}")
```

---

## training/merge_lora.py

Merges the adapter into the base model and writes a **Transformers** folder you can point vLLM at later. Also creates a `model.tgz` tarball for OCI packaging in the next lab.

```
# training/merge_lora.py
import os, json, shutil, tarfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path("/mnt/project/atharva-dental-assistant")
ART_ROOT = BASE_DIR / "artifacts" / "train"

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
RUN_ID = os.environ["RUN_ID"]  # e.g., 20250928-121314 (folder under artifacts/train)

run_dir = ART_ROOT / RUN_ID
adapter_dir = run_dir / "lora_adapter"
tok_dir = run_dir / "tokenizer"
out_dir = run_dir / "merged-model"

assert adapter_dir.exists(), f"Missing adapter at {adapter_dir}"

print(f"Loading base {BASE_MODEL} and merging adapter from {adapter_dir}")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map=None)
merged = PeftModel.from_pretrained(base, str(adapter_dir))
merged = merged.merge_and_unload()  # apply LoRA into base weights
tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else BASE_MODEL, use_fast=True)

out_dir.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)

# Create a tarball for OCI packaging in Lab 3
#tgz_path = run_dir / "model.tgz"
#with tarfile.open(tgz_path, "w:gz") as tar:
#    tar.add(out_dir, arcname="model")
#print(f"Merged model saved at {out_dir}, tarball at {tgz_path}")
```

---

## k8s/20-train/job-train-lora.yaml

Runs the training container mounting your repo path. Adjust `MAX_STEPS` to 300–600.

```
apiVersion: batch/v1
kind: Job
metadata:
  name: atharva-train-lora
  namespace: atharva-ml
spec:
  ttlSecondsAfterFinished: 7200   # auto-clean up pod 2h after completion
  template:
    spec:
      nodeName: llmops-kind-worker
      restartPolicy: Never
      containers:
      - name: train
        image: schoolofdevops/lora-build-python:3.11-slim
        imagePullPolicy: IfNotPresent
        command: ["bash","-lc"]
        args:
          - |
            python /mnt/project/atharva-dental-assistant/training/train_lora.py
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
          value: "80"
        - name: DEMO_MAX_TRAIN_SAMPLES
          value: "0"
        - name: DEMO_MAX_VAL_SAMPLES
          value: "0"
        - name: HF_HOME
          value: "/cache/hf"              # model/dataset cache across runs
        - name: HF_HUB_DISABLE_TELEMETRY
          value: "1"
        - name: TOKENIZERS_PARALLELISM
          value: "true"
        # Thread caps to avoid oversubscription; align to CPU limits below
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
      volumes:
      - name: host
        hostPath:
          path: /mnt/project
          type: Directory
      - name: hf-cache
        hostPath:
          path: /mnt/hf-cache               # create once on the node
          type: DirectoryOrCreate
```

> You can switch the container to the purpose-built image from `training/Dockerfile` later; the above installs deps inline to keep the lab minimal.

---

## k8s/20-train/job-merge-model.yaml

Merges the adapter and creates a `model.tgz`. Provide `RUN_ID` (the timestamped folder created by the training job).

```
apiVersion: batch/v1
kind: Job
metadata:
  name: atharva-merge-model
  namespace: atharva-ml
spec:
  template:
    spec:
      nodeName: llmops-kind-worker
      restartPolicy: Never
      containers:
      - name: merge
        image: schoolofdevops/lora-build-python:3.11-slim
        command: ["bash","-lc"]
        args:
          - |
            python /mnt/project/atharva-dental-assistant/training/merge_lora.py
        env:
        - name: BASE_MODEL
          value: "HuggingFaceTB/SmolLM2-135M-Instruct"
        - name: RUN_ID
          value: "REPLACE_WITH_RUN_ID"
        volumeMounts:
        - name: host
          mountPath: /mnt/project
      volumes:
      - name: host
        hostPath: { path: /mnt/project, type: Directory }
```

---

## scripts/train_lora.sh

```
#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/20-train/job-train-lora.yaml
kubectl -n atharva-ml wait --for=condition=complete job/atharva-train-lora --timeout=12h
kubectl -n atharva-ml logs job/atharva-train-lora
echo "Artifacts under artifacts/train/<run-id>/"
```

> After it finishes, list the timestamped folder:

> `ls -1 artifacts/train` → copy the folder name (e.g., `20251001-1130xx`).

## scripts/merge_model.sh

```
#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <RUN_ID>"
  exit 1
fi
RUN_ID="$1"

# Patch the manifest with the RUN_ID and apply
tmp=$(mktemp)
sed "s/REPLACE_WITH_RUN_ID/${RUN_ID}/g" k8s/20-train/job-merge-model.yaml > "$tmp"
kubectl apply -f "$tmp"
kubectl -n atharva-ml wait --for=condition=complete job/atharva-merge-model --timeout=60m
kubectl -n atharva-ml logs job/atharva-merge-model

echo "Merged model at artifacts/train/${RUN_ID}/merged-model"
#echo "Tarball at artifacts/train/${RUN_ID}/model.tgz"
```

---

## 🧪 Run the lab

```
# 0) (If not done) Ensure Lab 1’s data exists under datasets/training/{train,val}.jsonl

# 1) Load Base Image to a node 

docker image pull schoolofdevops/lora-build-python:3.11-slim

kind load docker-image --name llmops-kind schoolofdevops/lora-build-python:3.11-slim --nodes llmops-kind-worker

# 2) Start fine-tuning (CPU). This can take a while; steps are small on purpose.
bash scripts/train_lora.sh

# 3) Find the run-id (timestamp folder) created by the job:
ls -1 artifacts/train

# 4) Merge adapters into base & create tarball
bash scripts/merge_model.sh <RUN_ID>

# 5) Inspect outputs
tree artifacts/train/<RUN_ID> | sed -n '1,200p'
```

You should see:

* `lora_adapter/` (adapter weights),

* `tokenizer/`,

* `run.json`,

* `merged-model/` (Transformers folder with `config.json`, `model.safetensors`, tokenizer files),

* `model.tgz` (for OCI packaging in the next lab).

---

## Lab Summary 

This is what we accomplished in this lab

* Fine-tuned **SmolLM2-135M** on **Kubernetes (CPU)** with **LoRA** to learn Atharva’s style:

  * concise steps, safety tone, ask-back, and **always include `Source:`**.

* Produced reproducible artifacts under `artifacts/train/<run-id>/`.

* **Merged** adapters → a **single folder** that vLLM can serve.

* Created a **tarball** ready to be wrapped into an **OCI image** and mounted via **ImageVolumes** later.

#courses/llmops/labs/v1
