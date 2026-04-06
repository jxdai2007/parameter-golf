"""
Parameter Golf - Autoresearch Preparation Script
=================================================
Validates GPU environment and downloads competition data.
Run once before starting the autoresearch loop.

Usage: python prepare.py
"""

import os
import sys
import subprocess
import shutil

def check_gpu():
    """Verify CUDA GPU is available."""
    print("=" * 60)
    print("STEP 1: Checking GPU...")
    print("=" * 60)
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Need an NVIDIA GPU.")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  GPU count: {gpu_count}")
        print(f"  VRAM: {vram_gb:.1f} GB")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
        # Quick allocation test
        t = torch.zeros(1, device='cuda')
        del t
        print("  GPU allocation test: PASSED")
        print()
        return gpu_count
    except Exception as e:
        print(f"ERROR: GPU check failed: {e}")
        sys.exit(1)

def check_repo_structure():
    """Verify we're in the parameter-golf repo with required files."""
    print("=" * 60)
    print("STEP 2: Checking repo structure...")
    print("=" * 60)
    required_files = {
        "train_gpt.py": "Training script (agent modifies this)",
        "eval.py": "Evaluation script (read-only)",
    }
    optional_files = {
        "data/tokenizer_specs.json": "Tokenizer configurations",
        "requirements.txt": "Pip dependencies",
    }
    all_good = True
    for f, desc in required_files.items():
        if os.path.exists(f):
            print(f"  FOUND: {f} — {desc}")
        else:
            print(f"  MISSING: {f} — {desc}")
            all_good = False
    for f, desc in optional_files.items():
        if os.path.exists(f):
            print(f"  FOUND: {f} — {desc}")
        else:
            print(f"  OPTIONAL: {f} — {desc} (not found, may be OK)")
    if not all_good:
        print("\nERROR: Missing required files. Are you in the parameter-golf repo?")
        print("Run: git clone https://github.com/YOUR_USERNAME/parameter-golf.git")
        sys.exit(1)
    print()

def download_data():
    """Download competition data if not present."""
    print("=" * 60)
    print("STEP 3: Checking/downloading data...")
    print("=" * 60)
    data_dir = "data/datasets"
    tokenizer_dir = "data/tokenizers"

    # Check if data already exists
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        n_files = len(os.listdir(data_dir))
        print(f"  Data directory exists with {n_files} files. Skipping download.")
    else:
        print("  Data not found. Running download script...")
        # Try the standard download script
        download_scripts = [
            "data/cached_challenge_fineweb.py",
            "data/download_hf_docs_and_tokenize.py",
        ]
        downloaded = False
        for script in download_scripts:
            if os.path.exists(script):
                print(f"  Running: python {script}")
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=False
                )
                if result.returncode == 0:
                    downloaded = True
                    break
                else:
                    print(f"  WARNING: {script} failed, trying next...")
        if not downloaded:
            print("  ERROR: Could not download data automatically.")
            print("  Please run the download script manually:")
            print("    python data/cached_challenge_fineweb.py")
            print("  or")
            print("    python data/download_hf_docs_and_tokenize.py")
            sys.exit(1)

    # Check tokenizers
    if os.path.exists(tokenizer_dir) and len(os.listdir(tokenizer_dir)) > 0:
        n_files = len(os.listdir(tokenizer_dir))
        print(f"  Tokenizer directory exists with {n_files} files.")
    else:
        print("  WARNING: Tokenizer directory missing or empty.")
        print("  The download script should have created it.")
    print()

def check_records():
    """List available record submissions for reference."""
    print("=" * 60)
    print("STEP 4: Checking record submissions...")
    print("=" * 60)
    records_dir = "records/track_10min_16mb"
    if os.path.exists(records_dir):
        records = sorted(os.listdir(records_dir))
        print(f"  Found {len(records)} record submissions:")
        for r in records[-5:]:  # Show last 5
            print(f"    - {r}")
        if len(records) > 5:
            print(f"    ... and {len(records) - 5} more")
    else:
        print("  No records directory found (OK for fresh clone)")
    print()

def init_results():
    """Create results.tsv if it doesn't exist."""
    print("=" * 60)
    print("STEP 5: Initializing results tracking...")
    print("=" * 60)
    results_file = "results.tsv"
    if os.path.exists(results_file):
        with open(results_file) as f:
            lines = f.readlines()
        print(f"  results.tsv exists with {len(lines) - 1} experiment(s)")
    else:
        with open(results_file, "w") as f:
            f.write("commit\tval_bpb\tartifact_mb\tmemory_gb\tstatus\tdescription\n")
        print("  Created results.tsv with header row")
    print()

def setup_git():
    """Ensure git is configured for autoresearch commits."""
    print("=" * 60)
    print("STEP 6: Checking git configuration...")
    print("=" * 60)
    # Check if git user is configured
    result = subprocess.run(
        ["git", "config", "user.name"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        subprocess.run(["git", "config", "user.name", "autoresearch-agent"])
        subprocess.run(["git", "config", "user.email", "autoresearch@local"])
        print("  Set git user to 'autoresearch-agent'")
    else:
        print(f"  Git user: {result.stdout.strip()}")

    # Show current branch
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True
    )
    print(f"  Current branch: {result.stdout.strip()}")
    print()

def test_training(gpu_count):
    """Run a quick smoke test of train_gpt.py (10 seconds)."""
    print("=" * 60)
    print("STEP 7: Quick training smoke test...")
    print("=" * 60)
    print("  Skipping full smoke test (takes 5+ min).")
    print("  The autoresearch agent will run the baseline as its first experiment.")
    print()

def print_summary(gpu_count):
    """Print final summary and instructions."""
    print("=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print()
    print("Your environment is ready for autoresearch on Parameter Golf.")
    print()
    print(f"  GPUs: {gpu_count}")
    print(f"  Mode: {'Single-GPU (dev)' if gpu_count == 1 else f'{gpu_count}-GPU (can do record submissions)'}")
    print()
    print("Next steps:")
    print("  1. Start Claude Code:  claude")
    print('  2. Tell it: "Hi have a look at program.md and let\'s kick off a new experiment! Let\'s do the setup first."')
    print()
    print("The agent will:")
    print("  - Read program.md for instructions")
    print("  - Study record submissions in records/")
    print("  - Run the baseline first")
    print("  - Then iterate autonomously")
    print()
    print("TIP: Run inside tmux so experiments continue if you disconnect:")
    print("  tmux")
    print("  claude")
    print("  # Detach: Ctrl+B then D")
    print("  # Reattach: tmux attach")
    print()

if __name__ == "__main__":
    print()
    print("Parameter Golf — Autoresearch Setup")
    print("====================================")
    print()
    gpu_count = check_gpu()
    check_repo_structure()
    download_data()
    check_records()
    init_results()
    setup_git()
    test_training(gpu_count)
    print_summary(gpu_count)
