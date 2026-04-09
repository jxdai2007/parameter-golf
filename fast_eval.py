#!/usr/bin/env python3
"""Standalone fast evaluator for Parameter Golf checkpoints.

Used primarily for Phase 2 (eval-only experiments on a frozen checkpoint).
Imports model classes and eval functions from train_gpt.py.

Usage:
    # Fixed-context eval on full val set
    python fast_eval.py --checkpoint final_model.pt

    # Fixed-context eval on 10% val subset (faster)
    python fast_eval.py --checkpoint final_model.pt --val-fraction 0.1

    # Sliding window eval (Phase 2, ~5 min)
    python fast_eval.py --checkpoint best_model.pt --sliding --stride 64
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import sentencepiece as spm
import torch

# Import from train_gpt.py — safe because main() is guarded by __name__ == "__main__"
import train_gpt as tg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast eval for Parameter Golf checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint (FP32 state dict)")
    parser.add_argument("--val-fraction", type=float, default=1.0,
                        help="Fraction of val data to use (e.g., 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for val subset selection")
    parser.add_argument("--sliding", action="store_true",
                        help="Use sliding window eval (slow, ~5 min)")
    parser.add_argument("--stride", type=int, default=64,
                        help="Sliding window stride (only used with --sliding)")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = tg.Hyperparameters()

    # Load tokenizer + build byte-counting LUTs
    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    if int(sp.vocab_size()) != hp.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={hp.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        tg.build_sentencepiece_luts(sp, hp.vocab_size, device)
    )

    # Load validation tokens, optionally subsample a contiguous chunk
    val_tokens = tg.load_validation_tokens(hp.val_files, hp.train_seq_len)
    total_tokens = val_tokens.numel()
    if args_cli.val_fraction < 1.0:
        chunk_tokens = int(total_tokens * args_cli.val_fraction)
        # Align to seq_len boundaries (need +1 for next-token prediction)
        chunk_tokens = (chunk_tokens // hp.train_seq_len) * hp.train_seq_len + 1
        max_start = total_tokens - chunk_tokens
        if max_start > 0:
            start = random.Random(args_cli.seed).randint(0, max_start)
        else:
            start = 0
        val_tokens = val_tokens[start : start + chunk_tokens]
        print(f"fast_eval: using {val_tokens.numel()} / {total_tokens} val tokens "
              f"({100 * val_tokens.numel() / total_tokens:.1f}%)")

    # Build model and load checkpoint
    model = tg.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim,
        xsa_last_n=hp.xsa_last_n,
        rope_dims=hp.rope_dims,
        ln_scale=hp.ln_scale,
        ve_dim=hp.ve_dim,
        ve_layers=hp.ve_layers,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(model)

    state = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state

    # Evaluate
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if args_cli.sliding:
        val_loss, val_bpb = tg.eval_val_sliding(
            hp, model, rank=0, world_size=1, device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            stride=args_cli.stride,
        )
    else:
        val_loss, val_bpb = tg.eval_val(
            hp, model, rank=0, world_size=1, device=device,
            grad_accum_steps=1, val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )

    torch.cuda.synchronize()
    eval_time = time.perf_counter() - t0

    mode = f"sliding(stride={args_cli.stride})" if args_cli.sliding else "fixed_context"
    print(f"fast_eval val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f} "
          f"mode:{mode} eval_time:{eval_time:.1f}s")


if __name__ == "__main__":
    main()
