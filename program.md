# parameter-golf autoresearch

This is an experiment to have the LLM do its own research on the OpenAI Parameter Golf competition.

**Goal**: Train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100 SXM GPUs. Score is **val_bpb** (bits per byte) — lower is better. Baseline is 1.2244.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is larger than standard autoresearch. Read these files for full context:
   * `README.md` — competition rules, constraints, submission format.
   * `train_gpt.py` — the file you modify. Model architecture, optimizer, training loop, quantization, compression. Everything.
   * `eval.py` — evaluation script. Computes val_bpb. Do not modify.
   * `data/tokenizer_specs.json` — tokenizer configurations. Read-only context.
   * `records/` — browse the most recent record submissions. Each subfolder has a README and a `train_gpt.py`. **Study these carefully** — they represent the current frontier.
4. **Read the competition tracker**: Fetch `https://github.com/openai/parameter-golf/issues/140` — this is the live commentary with technique analysis, idea lineage, and estimated gains. Parse out the current SOTA BPB, the technique stack, and any untried ideas listed.
5. **Verify data exists**: Check that `./data/datasets/` contains training shards and that `./data/tokenizers/` has the tokenizer model. If not, tell the human to run `python data/download_hf_docs_and_tokenize.py`.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good. Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a **single GPU** for fast iteration (1xH100). The training script trains for a **fixed time budget** controlled by the script's internal settings. For development runs, you may reduce training steps/time to iterate faster, but **always restore full settings before recording a result**.

You launch fast experiments via the orchestrator: `python run_autoresearch.py fast --desc "description" > run.log 2>&1`

This runs training for 180s on 12 fixed shards, skips GPTQ and validation eval, and uses the **final train_loss as a proxy metric**. Total time: ~3 min per experiment. Full calibration runs with eval are done periodically.

**What you CAN do:**

* Modify `train_gpt.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization scheme, compression method, tokenizer choice, depth, width, activation functions, attention variants, etc.

**What you CANNOT do:**

* Modify `eval.py`. It is read-only. It contains the ground truth BPB calculation.
* Modify anything in `data/` after initial setup.
* Install new packages beyond what's in `requirements.txt`.
* Make network calls during evaluation. The artifact must be self-contained.
* Exceed the 16,000,000 byte artifact size limit (code + compressed model).
* Exceed 10 minutes training time on 8xH100 SXM (for record submissions).

**The goal is simple: get the lowest val_bpb while keeping artifact ≤ 16MB.**

**Dual constraint check**: After every successful run, you MUST verify BOTH:
1. `val_bpb` — did it improve?
2. Artifact size — run `ls -la` on the output artifact and confirm ≤ 16,000,000 bytes.

If BPB improved but artifact exceeds 16MB, the run is a **discard**. If BPB improved and artifact is within budget, it's a **keep**.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.0005 val_bpb improvement that adds 50 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from a clean architectural change? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## Output format

After training completes, the script prints a summary. Extract the key metrics:

```
grep "val_bpb" run.log
```

After eval:
```
grep "bpb" eval.log
```

Also check artifact size:
```
ls -la *.pt *.pth *.bin 2>/dev/null || echo "check output artifact path"
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	artifact_mb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.1234) — use 0.000000 for crashes
3. artifact size in MB, round to .1f (e.g. 15.8) — use 0.0 for crashes
4. peak memory in GB, round to .1f — use 0.0 for crashes
5. status: `keep`, `discard`, `crash`, or `oversize`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	artifact_mb	memory_gb	status	description
a1b2c3d	1.2244	15.9	44.0	keep	baseline
b2c3d4e	1.1800	15.2	44.2	keep	int6 quantization + mlp 3x expansion
c3d4e5f	1.1750	16.4	44.0	oversize	added smeargate but artifact too large
d4e5f6g	0.0000	0.0	0.0	crash	OOM from 13-layer depth
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr5` or `autoresearch/apr5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Think about what to try next. Prioritize ideas by estimated train_loss improvement vs. complexity. Consult the **Research Directions** section below. Phase 1 ideas: architecture, optimizer, hyperparameters, quantization schemes, MLP width, number of layers, LR schedules — anything that only changes the trained model.
3. Tune `train_gpt.py` with an experimental idea by directly hacking the code.
4. git commit.
5. Run the fast experiment:
   ```bash
   python run_autoresearch.py fast --desc "short description" > run.log 2>&1
   ```
   This internally runs `train_gpt.py` with:
   - `MAX_WALLCLOCK_SECONDS=180` (~3 min training)
   - `SKIP_QUANT=1` (skip GPTQ/compression, exit after training)
   - `SKIP_EVAL=1` (skip validation eval — use final train_loss as proxy)
   - `EVAL_STRIDE=0` (no sliding window)
   - Fixed 12 shards with seed=42 for reproducibility
   - `TRAIN_LOG_EVERY=10` (frequent loss logging for early stopping)
   Total time: ~3 min per experiment.
6. Read out the results: `grep "fast_result\|train_loss\|early_stop\|RECOMMEND" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
8. The orchestrator logs results to `experiments.jsonl` automatically and prints a delta against the fast baseline.
9. If train_loss improved (negative delta), the experiment is a **success** — advance the branch, keep the commit.
10. If train_loss is equal or worse, **discard** — `git reset --hard` back to where you started.

## Research Directions

Prioritize these roughly in order. The agent should work through them systematically, not randomly.

### Tier 1 — Table Stakes (do these first if not already present)
- **Int6 quantization**: Store weights as 6-bit integers. Smaller artifact → fit more parameters.
- **MLP 3x expansion**: Widen feedforward layers to 3x model dimension.
- **FP16 tied embeddings**: Keep embedding/unembedding in full precision (sensitive to quantization).
- **Zstd-22 compression**: Replace zlib with zstd at level 22 for smaller artifacts.
- **Muon optimizer**: Replace Adam/AdamW with Muon for faster convergence.
- **Sliding window eval**: Overlapping context windows during evaluation (eval-only, free BPB).

### Tier 2 — Proven Techniques (from record submissions)
- **11-layer depth**: Increase from 9 to 11 layers. Most frontier submissions use 11L.
- **EMA (Exponential Moving Average)**: Running average of weights during training.
- **GPTQ post-training quantization**: Smarter weight rounding that compensates for errors.
- **QAT (Quantization-Aware Training)**: Simulate quantization during training so model learns robustness. Start at ~15% of training.
- **Warmdown schedule**: Gradually decrease LR at end of training (~3500 steps).
- **OrthoInit**: Initialize weights as orthogonal matrices. Critical for SmearGate to work.
- **SmearGate + BigramHash**: Mix bigram statistics into model. Requires OrthoInit.
- **XSA (Cross-head Shared Attention)**: Remove self-value bias from attention. Apply to last 3-4 layers (Partial XSA) or all layers. Near-universal in frontier submissions.

### Tier 3 — Frontier / Experimental
- **VRL (Value Residual Learning)**: Residual connections in value computation.
- **LeakyReLU²**: Squared LeakyReLU activation function.
- **Test-Time Training (TTT)**: Adapt model on test data during eval (backward-looking only, legal).
- **LaCT (Large Chunk TTT)**: Document-sized chunks for TTT with 70% GPU utilization.
- **Int5 quantization**: Push to 5-bit. Requires careful GPTQ to avoid quality loss.
- **Depth recurrence**: Share weights across layer pairs to fit more effective depth.
- **Selective pruning**: Remove least-important weights before quantization.
- **QKNorm**: L2-normalize Q and K before dot product with learned temperature.
- **SLOT (output-head TTT)**: Learnable delta vector at last hidden layer during eval.
- **Fused Triton kernels**: RMSNorm, linear+CE, residual+norm fusion for throughput.

### Tier 4 — Speculative / Novel
- **Prune-then-quantize ordering**: Apply pruning before quantization (opposite of current convention). Estimated 0.001-0.003 BPB free gain.
- **Spectral embedding init**: Initialize embeddings using spectral methods.
- **Adjacent K/V sharing**: Share K/V projections between adjacent layers. Saves ~0.5MB artifact.
- **Block-partitioned attention residuals**: Efficient AttnRes variant. <2% overhead.
- **Decoupled magnitude/angle attention**: Separate content (magnitude) from position (angle).

## Reading Other Submissions

When you feel stuck or want inspiration, read the `records/` directory:
1. `ls records/track_10min_16mb/` to see all record submissions.
2. Read the most recent (highest BPB improvement) README files.
3. Diff their `train_gpt.py` against yours to find techniques you haven't tried.
4. Do NOT blindly copy — understand why each change helps, then implement cleanly.

## Reading Research Papers

When a technique references an arXiv paper:
1. Note the arXiv ID from the technique description or Issue #140.
2. Think about the core mechanism — what does this change about the forward/backward pass?
3. Estimate: does this save parameters, improve convergence, or compress better?
4. Implement a minimal version first. If it shows life, optimize.

## Crash Handling

**Timeout**: Each experiment should take roughly the configured training time (+ startup/eval overhead). If a run exceeds 2x the expected time, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken (e.g. model too large for VRAM), just skip it, log "crash" as the status in the tsv, and move on.

**OOM specifically**: If you OOM, try reducing batch size first. If that doesn't work, reduce model size or depth. Don't waste more than 2 attempts on an OOM idea.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, re-read the records directory for new angles, try combining previous near-misses, try more radical architectural changes, or revisit discarded ideas with different hyperparameters. The loop runs until the human interrupts you, period.
