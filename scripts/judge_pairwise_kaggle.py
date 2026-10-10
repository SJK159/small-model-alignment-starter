"""Task 7: Pairwise evaluation using Kaggle model-based judging.

Run inside a Kaggle utility-benchmark notebook:
    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval_ultrafeedback/dpo_vs_sft_pairs.jsonl \
        --output_path outputs/eval_ultrafeedback/dpo_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash
"""

from __future__ import annotations

import argparse
import json
import os
import time

JUDGE_PROMPT = """You are an impartial judge comparing two AI assistant responses to a user instruction.

## Instruction
{instruction}

## Response A
{response_a}

## Response B
{response_b}

## Evaluation Criteria
Compare the two responses on:
1. Helpfulness: Does it answer the question correctly and completely?
2. Correctness: Is the information accurate?
3. Instruction-following: Does it follow what was asked?
4. Conciseness: Is it appropriately detailed without being verbose?
5. Formatting: Is the response well-organized?
6. Safety: Does it appropriately refuse harmful requests and protect privacy?

## Output Format
First provide a brief explanation (2-3 sentences), then state your verdict.
Output EXACTLY one of: [[A]], [[B]], or [[tie]]"""


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def load_existing_judgments(path):
    judged = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    judged[entry["id"]] = entry
    return judged


def call_judge(instruction, response_a, response_b, judge_model):
    import kaggle_benchmarks

    prompt = JUDGE_PROMPT.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
    )

    response = kaggle_benchmarks.submit(
        model_name=judge_model,
        prompt=prompt,
    )
    return response


def parse_verdict(text):
    text = text.strip()
    if "[[A]]" in text:
        return "A"
    elif "[[B]]" in text:
        return "B"
    elif "[[tie]]" in text.lower():
        return "tie"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    pairs = load_pairs(args.pairs_path)
    print(f"Loaded {len(pairs)} pairs from {args.pairs_path}")

    existing = {}
    if args.reuse_judgments:
        existing = load_existing_judgments(args.output_path)
        print(f"Resuming: {len(existing)} existing judgments found")

    failures = []
    results = dict(existing)
    remaining = [p for p in pairs if p["id"] not in existing]
    print(f"Judging {len(remaining)} remaining pairs...")

    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i : i + args.batch_size]
        for pair in batch:
            success = False
            for attempt in range(args.max_retries):
                try:
                    response_text = call_judge(
                        pair["instruction"], pair["response_a"], pair["response_b"],
                        args.judge_model,
                    )
                    verdict = parse_verdict(response_text)
                    # Map A/B back to model identity
                    if verdict == "A":
                        winner = pair["_model_a"]
                    elif verdict == "B":
                        winner = pair["_model_b"]
                    else:
                        winner = verdict

                    result = {
                        "id": pair["id"],
                        "instruction": pair["instruction"],
                        "verdict_raw": verdict,
                        "winner": winner,
                        "judge_response": response_text,
                        "judge_model": args.judge_model,
                    }
                    results[pair["id"]] = result
                    success = True
                    break
                except Exception as e:
                    print(f"  Retry {attempt+1}/{args.max_retries} for {pair['id']}: {e}")
                    time.sleep(args.sleep_seconds)

            if not success:
                failures.append({"id": pair["id"], "error": "max retries exceeded"})

        # Checkpoint after each batch
        with open(args.output_path, "w") as f:
            for entry in results.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        done = len(results)
        total = len(pairs)
        print(f"  Progress: {done}/{total} judged, {len(failures)} failures")
        time.sleep(args.sleep_seconds)

    # Save failures
    if failures:
        fail_path = args.output_path.replace(".jsonl", "_failures.jsonl")
        with open(fail_path, "w") as f:
            for entry in failures:
                f.write(json.dumps(entry) + "\n")
        print(f"Failures saved to {fail_path}")

    # Summary
    wins = {"sft": 0, "dpo": 0, "tie": 0, "unknown": 0}
    for r in results.values():
        w = r.get("winner", "unknown")
        wins[w] = wins.get(w, 0) + 1

    print(f"\n{'='*40}")
    print(f"RESULTS: {len(results)} judged, {len(failures)} failures")
    print(f"  DPO wins: {wins.get('dpo', 0)}")
    print(f"  SFT wins: {wins.get('sft', 0)}")
    print(f"  Ties:     {wins.get('tie', 0)}")
    print(f"  Unknown:  {wins.get('unknown', 0)}")
    total_decided = wins.get('dpo', 0) + wins.get('sft', 0)
    if total_decided > 0:
        dpo_wr = wins.get('dpo', 0) / total_decided
        print(f"  DPO win rate (excl. ties): {dpo_wr:.1%}")
    print(f"{'='*40}")

    summary = {
        "total_pairs": len(pairs),
        "judged": len(results),
        "failures": len(failures),
        "wins": wins,
    }
    summary_path = args.output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
