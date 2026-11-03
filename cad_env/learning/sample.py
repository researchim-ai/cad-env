"""Collect real policy samples, verify them, and mine SFT/preference examples."""

import argparse
import json
from pathlib import Path

from .factory import jsonl
from .rewards import CADReward, group_statistics


def main():
    parser = argparse.ArgumentParser(
        description="Sample a chosen local/HF code policy and execute each CAD candidate"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--freecad")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.limit < 1
        or args.num_generations < 2
        or args.max_new_tokens < 1
        or args.temperature <= 0
    ):
        parser.error("Invalid sampling budget or temperature")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    root, output = Path(args.dataset).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        json.loads(line)
        for line in (root / "exports" / f"{args.split}.grpo.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ][: args.limit]
    if not tasks:
        raise ValueError("The selected task split is empty")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    reward = CADReward(
        root / "episodes", output / "cad-rollouts", executable=args.freecad
    )
    scores, successful, preferences, groups = [], [], [], {}
    for task in tasks:
        encoded = tokenizer.apply_chat_template(
            task["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        candidates = []
        for i in range(args.num_generations):
            with torch.inference_mode():
                generated = model.generate(
                    encoded,
                    attention_mask=torch.ones_like(encoded),
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(
                generated[0, encoded.shape[-1] :], skip_special_tokens=True
            )
            score = reward.score(task["task_id"], text)
            row = {
                "task_id": task["task_id"],
                "candidate": i,
                "completion": text,
                **score,
            }
            candidates.append(row)
            scores.append(row)
            completion = [{"role": "assistant", "content": text}]
            if score["task_success"]:
                successful.append({**task, "completion": completion})
            # Flush every candidate so interrupted collection retains failed attempts too.
            with (output / "scores.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        groups[task["task_id"]] = group_statistics(
            [row["reward"] for row in candidates]
        )
        best, worst = (
            max(candidates, key=lambda r: r["reward"]),
            min(candidates, key=lambda r: r["reward"]),
        )
        if best["task_success"] and best["reward"] > worst["reward"]:
            preferences.append(
                {
                    **task,
                    "chosen": [{"role": "assistant", "content": best["completion"]}],
                    "rejected": [{"role": "assistant", "content": worst["completion"]}],
                }
            )
    # Keep validation/test generations for evaluation; never export them as training supervision.
    if args.split == "train":
        jsonl(output / "rejection.sft.jsonl", successful)
        jsonl(output / "mined.preference.jsonl", preferences)
    report = {
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "candidates": len(scores),
        "task_successes": sum(r["task_success"] for r in scores),
        "groups": groups,
        "warning": "Offline normalized advantages are diagnostics, not a GRPO optimizer.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
