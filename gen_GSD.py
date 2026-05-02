import argparse
import os
import sys
import json
import gc
import logging
import random
import time

PATH = '/SSD'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
sys.path.append("./lumina_mgpt/")
sys.path.append("./")

import numpy as np
import torch
from PIL import Image

from lumina_mgpt.inference_solver import FlexARInferenceSolver
from scheduler.jacobi_iteration_lumina_mgpt import renew_pipeline_sampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GSD batch inference over prompt JSON")
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--save_dir",    type=str, required=True)
    parser.add_argument("--json_key",    type=str, default="caption")
    parser.add_argument("--begin",       type=int, default=0)
    parser.add_argument("--end",         type=int, default=None)
    parser.add_argument("--model_path",  type=str, default="/home/vlgd/Models/Lumina-mGPT-7B-768-bnb-int4")
    parser.add_argument("--target_size", type=int, default=768)
    parser.add_argument("--device",      type=str, default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    SEED = 42
    set_seed(SEED)

    target_size = args.target_size
    target_size_h = target_size_w = target_size
    template_prefix = f"Generate an image of {target_size_w}x{target_size_h} according to the following prompt:\n"

    with open(args.prompt_path, "r") as f:
        all_prompts = json.load(f)

    all_prompts = all_prompts[args.begin : args.end]
    end = args.end if args.end is not None else args.begin + len(all_prompts)

    logger.info(f"Prompt file : {args.prompt_path}")
    logger.info(f"Segment     : [{args.begin}, {end})  ({len(all_prompts)} prompts)")
    logger.info(f"Save dir    : {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="int4",
        target_size=target_size,
        cache_dir=PATH,
        device=args.device,
    )

    max_num_new_tokens      = 16
    multi_token_init_scheme = "random"
    image_top_k             = 2000
    text_top_k              = 10
    guidance_scale          = 3.0
    prefix_token_sampler_scheme = "speculative_jacobi"

    inference_solver = renew_pipeline_sampler(
        inference_solver,
        jacobi_loop_interval_l = 3,
        jacobi_loop_interval_r = (target_size // 16) ** 2 + target_size // 16 - 10,
        max_num_new_tokens      = max_num_new_tokens,
        guidance_scale          = guidance_scale,
        seed                    = SEED,
        multi_token_init_scheme = multi_token_init_scheme,
        do_cfg                  = True,
        image_top_k             = image_top_k,
        text_top_k              = text_top_k,
        prefix_token_sampler_scheme = prefix_token_sampler_scheme,
    )

    # ── Generation loop ────────────────────────────────────────────────────────
    for local_i, item in enumerate(all_prompts):
        global_idx = args.begin + local_i
        out_path = os.path.join(args.save_dir, f"{global_idx:05d}.png")

        if os.path.exists(out_path):
            logger.info(f"[{global_idx:05d}] already exists, skipping.")
            continue

        prompt = item[args.json_key] if isinstance(item, dict) else item
        q1 = template_prefix + prompt

        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t1.record()

        generated = inference_solver.generate(
            images=[],
            qas=[[q1, None]],
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(
                cfg=guidance_scale, image_top_k=image_top_k
            ),
        )

        t2.record()
        torch.cuda.synchronize()
        elapsed = t1.elapsed_time(t2) / 1000

        _, new_image = generated[0], generated[1][0]
        result_image = inference_solver.create_image_grid([new_image], 1, 1)
        result_image.save(out_path)

        logger.info(f"[{global_idx:05d}/{end-1:05d}] saved {out_path}  ({elapsed:.1f}s)  prompt: {prompt[:60]}")

    del inference_solver
    gc.collect()
    logger.info("Done.")


if __name__ == "__main__":
    main()
