# CLAUDE.md — Parameter Golf Project

## Project Context
This is my fork of OpenAI's Parameter Golf competition (https://github.com/openai/parameter-golf).
Goal: Train the best language model that fits in 16MB, training in <10 min on 8xH100s.
Score: bits per byte (BPB) on FineWeb validation set. Lower = better. Baseline = 1.2244.

## My Background
I'm a CS freshman. I know Python well and I'm learning ML. I do NOT know most of the advanced techniques people are using in this competition. When I ask about a technique, explain it simply first, then get into implementation details. Don't assume I know what things like "quantization-aware training" or "exponential moving average" mean — explain them.

## How I Want to Work
1. **Research first, code second.** When I mention a technique or paper, help me understand what it does and why it matters before writing any code.
2. **One change at a time.** Never stack multiple untested changes. Make one modification, test it, log the result, then decide what's next.
3. **Track everything.** Log every experiment in `experiments.jsonl` with: id, description, BPB, artifact size, training time, what changed, and notes.
4. **Plain English explanations.** I'll ask "why" a lot. That's not me being difficult — I genuinely want to understand the ML concepts, not just copy-paste code.

## Key Commands
```bash
# Download training data (run once)
python data/download_hf_docs_and_tokenize.py

# Train the model
python train_gpt.py

# Evaluate (get BPB score)
python eval.py

# Check artifact size
ls -la final_model.*.ptz
```

## Current Experiment State
Check `experiments.jsonl` for the latest results before suggesting changes.

## Rules I Must Follow
- Artifact (code + compressed model) ≤ 16,000,000 bytes
- Training ≤ 10 min on 8xH100 SXM
- Eval ≤ 10 min
- No network calls during eval
- No validation data stored in artifact
- Record submissions need 3 seeds + statistical significance (p<0.01, beat SOTA by ≥0.005 nats)
- Only backward-looking TTT is legal (adapt on already-graded tokens only)

## Paper Research Workflow
When I ask about a technique or paper:
1. Search for the arXiv paper or competition PR
2. Read the abstract and method section
3. Tell me: what does it do (1 sentence), why might it help here (1-2 sentences), how hard to implement, expected BPB gain, what could go wrong
4. If I say "let's try it" — implement it on a git branch, run training, log results

## Competition Intel
- Issue #140 on the repo has live AI commentary tracking all techniques and their results
- Study merged PRs in records/track_10min_16mb/ to see what worked
- The Discord #parameter-golf-discussions channel has active discussion