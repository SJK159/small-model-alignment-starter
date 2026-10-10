#!/usr/bin/env python
"""PA2: SFT + DPO Training Pipeline (Tasks 2-6)

Run on Kaggle with GPU T4 x2 accelerator.
Outputs are saved under /kaggle/working/outputs/ and preserved via Save Version.
"""

import subprocess, os, sys, time, json, traceback

WORK = "/kaggle/working"
REPO = f"{WORK}/small-model-alignment-starter"
OUTPUTS = f"{WORK}/outputs"
timings = {}


def run_task(name, cmd, env=None):
    """Run a task, print output, return success bool."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = subprocess.run(cmd, env=env, check=True,
                                capture_output=False, text=True)
        elapsed = time.time() - start
        timings[name] = elapsed
        print(f"\n[OK] {name} completed in {elapsed/60:.1f} min")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        timings[name] = -elapsed
        print(f"\n[FAIL] {name} failed after {elapsed/60:.1f} min")
        traceback.print_exc()
        return False


ENV = lambda: {**os.environ, "PYTHONPATH": REPO}

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1: Setup
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1: Environment Setup")
print("=" * 60)

subprocess.run(["nvidia-smi"], check=True)

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
               capture_output=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.43.0,<5", "datasets>=2.18,<5",
    "huggingface_hub", "accelerate", "tqdm", "pyyaml",
    "sentencepiece", "safetensors", "pytest",
    "peft>=0.11.0", "trl>=0.12.0",
], check=True)

if not os.path.exists(REPO):
    subprocess.run([
        "git", "clone", "https://github.com/SJK159/small-model-alignment-starter.git", REPO
    ], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO, "--no-deps"], check=True)

os.chdir(REPO)
sys.path.insert(0, REPO)

import torch, transformers, datasets
print(f"\nPyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Smoke tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_public_adapter_smoke.py"],
    capture_output=True, text=True, env=ENV()
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    print("WARNING: Smoke tests failed!")
else:
    print("Smoke tests passed!")

os.makedirs(OUTPUTS, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: Baseline Generation
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(f"{OUTPUTS}/baseline", exist_ok=True)

run_task("TASK 2a: Baseline AlpacaEval", [
    sys.executable, "scripts/generate_baseline.py",
    "--eval_set", "alpaca_eval",
    "--output_path", f"{OUTPUTS}/baseline/alpaca_eval_base.json",
    "--batch_size", "8",
], env=ENV())

run_task("TASK 2b: Baseline HH-RLHF", [
    sys.executable, "scripts/generate_baseline.py",
    "--eval_set", "hh_rlhf",
    "--output_path", f"{OUTPUTS}/baseline/hh_rlhf_base.json",
    "--batch_size", "8",
], env=ENV())

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: SFT Training
# ══════════════════════════════════════════════════════════════════════════════
sft_ok = run_task("TASK 3: SFT Training", [
    sys.executable, "scripts/train_sft.py",
    "--output_dir", f"{OUTPUTS}/sft",
    "--train_examples", "30000",
    "--val_examples", "1000",
    "--batch_size", "2",
    "--grad_accum", "16",
    "--lr", "2e-5",
    "--epochs", "1",
    "--eval_steps", "50",
    "--log_steps", "10",
    "--seed", "42",
], env=ENV())

if sft_ok:
    subprocess.run(["zip", "-qr", f"{WORK}/sft_outputs.zip", f"{OUTPUTS}/sft"])

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4a: DPO Training - HH-RLHF
# ══════════════════════════════════════════════════════════════════════════════
if sft_ok:
    dpo_hh_ok = run_task("TASK 4a: DPO HH-RLHF", [
        sys.executable, "scripts/train_dpo.py",
        "--sft_checkpoint", f"{OUTPUTS}/sft/best",
        "--dataset", "hh_rlhf",
        "--output_dir", f"{OUTPUTS}/dpo_hh_rlhf",
        "--train_pairs", "15000",
        "--val_pairs", "1000",
        "--batch_size", "1",
        "--grad_accum", "32",
        "--lr", "1e-6",
        "--beta", "0.1",
        "--epochs", "1",
        "--eval_steps", "100",
        "--log_steps", "10",
        "--seed", "42",
    ], env=ENV())
else:
    print("\nSkipping TASK 4a: SFT failed")
    dpo_hh_ok = False

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4b: DPO Training - UltraFeedback
# ══════════════════════════════════════════════════════════════════════════════
if sft_ok:
    dpo_uf_ok = run_task("TASK 4b: DPO UltraFeedback", [
        sys.executable, "scripts/train_dpo.py",
        "--sft_checkpoint", f"{OUTPUTS}/sft/best",
        "--dataset", "ultrafeedback",
        "--output_dir", f"{OUTPUTS}/dpo_ultrafeedback",
        "--train_pairs", "15000",
        "--val_pairs", "1000",
        "--batch_size", "1",
        "--grad_accum", "32",
        "--lr", "1e-6",
        "--beta", "0.1",
        "--epochs", "1",
        "--eval_steps", "100",
        "--log_steps", "10",
        "--seed", "42",
    ], env=ENV())
else:
    print("\nSkipping TASK 4b: SFT failed")
    dpo_uf_ok = False

# ══════════════════════════════════════════════════════════════════════════════
# TASK 5: LoRA DPO Persona Training
# ══════════════════════════════════════════════════════════════════════════════
lora_ok = run_task("TASK 5: LoRA DPO Persona", [
    sys.executable, "scripts/train_lora_dpo.py",
    "--output_dir", f"{OUTPUTS}/lora_dpo",
    "--epochs", "3",
    "--batch_size", "2",
    "--grad_accum", "4",
    "--lr", "5e-5",
    "--seed", "42",
], env=ENV())

# ══════════════════════════════════════════════════════════════════════════════
# TASK 6: Generate Evaluation Outputs
# ══════════════════════════════════════════════════════════════════════════════
if sft_ok and dpo_uf_ok:
    run_task("TASK 6a: Generate UltraFeedback eval", [
        sys.executable, "scripts/generate_outputs.py",
        "--sft_checkpoint", f"{OUTPUTS}/sft/best",
        "--dpo_checkpoint", f"{OUTPUTS}/dpo_ultrafeedback/best",
        "--eval_set", "alpaca_eval",
        "--output_dir", f"{OUTPUTS}/eval_ultrafeedback",
        "--batch_size", "8",
    ], env=ENV())
else:
    print("\nSkipping TASK 6a: prerequisites not met")

if sft_ok and dpo_hh_ok:
    run_task("TASK 6b: Generate HH-RLHF eval", [
        sys.executable, "scripts/generate_outputs.py",
        "--sft_checkpoint", f"{OUTPUTS}/sft/best",
        "--dpo_checkpoint", f"{OUTPUTS}/dpo_hh_rlhf/best",
        "--eval_set", "hh_rlhf",
        "--output_dir", f"{OUTPUTS}/eval_hh_rlhf",
        "--batch_size", "8",
    ], env=ENV())
else:
    print("\nSkipping TASK 6b: prerequisites not met")

# ══════════════════════════════════════════════════════════════════════════════
# Archive everything
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Archiving all outputs")
print("=" * 60)
subprocess.run(["zip", "-qr", f"{WORK}/all_outputs.zip", OUTPUTS])

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
for name, t in timings.items():
    status = "OK" if t > 0 else "FAIL"
    t_abs = abs(t)
    if t_abs > 3600:
        print(f"  [{status}] {name}: {t_abs/3600:.1f} hours")
    else:
        print(f"  [{status}] {name}: {t_abs/60:.1f} min")
print("=" * 60)
print("\nREMEMBER: Click 'Save Version' to preserve outputs!")
