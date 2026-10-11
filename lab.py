import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from scorer import score_output


TASK = "Classify each customer support message as urgent or normal."
ROOT = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return cases


def load_prompt_variants(path: Path) -> dict[str, str]:
    variants = {}
    for prompt_file in sorted(path.glob("*.txt")):
        variants[prompt_file.stem] = prompt_file.read_text(encoding="utf-8").strip()
    if not variants:
        raise ValueError(f"No prompt files found in {path}")
    return variants


def call_model(client: OpenAI, model: str, system_prompt: str, input_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def run_variant(
    client: OpenAI,
    model: str,
    variant_name: str,
    prompt: str,
    cases: list[dict],
    scoring_mode: str,
) -> list[dict]:
    rows = []
    for case in cases:
        output = call_model(client, model, prompt, case["input"])
        score = score_output(
            mode=scoring_mode,
            client=client,
            model=model,
            task=TASK,
            case=case,
            output=output,
        )
        rows.append(
            {
                "variant": variant_name,
                "case_id": case["id"],
                "input": case["input"],
                "expected": case["expected"],
                "output": output,
                "score": score["score"],
                "passed": score["passed"],
                "reason": score["reason"],
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_variant = {}
    for row in rows:
        stats = by_variant.setdefault(
            row["variant"],
            {"variant": row["variant"], "score": 0.0, "passed": 0, "total": 0},
        )
        stats["score"] += row["score"]
        stats["passed"] += 1 if row["passed"] else 0
        stats["total"] += 1

    summary = []
    for stats in by_variant.values():
        total = stats["total"]
        summary.append(
            {
                "variant": stats["variant"],
                "score": stats["score"] / total if total else 0.0,
                "passed": stats["passed"],
                "total": total,
            }
        )
    return sorted(summary, key=lambda item: item["score"], reverse=True)


def print_report(summary: list[dict]) -> None:
    print("Prompt variant     Score     Passed     Total")
    for item in summary:
        print(
            f"{item['variant']:<18} "
            f"{item['score']:.2f}      "
            f"{item['passed']:<10} "
            f"{item['total']}"
        )
    if summary:
        print()
        print(f"Winner: {summary[0]['variant']}")


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring", choices=["exact", "judge"], default="exact")
    parser.add_argument("--cases", default="data/testcases.jsonl")
    parser.add_argument("--prompts", default="prompts")
    parser.add_argument("--out", default="runs/latest-results.jsonl")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    cases = load_jsonl(ROOT / args.cases)
    variants = load_prompt_variants(ROOT / args.prompts)

    all_rows = []
    for variant_name, prompt in variants.items():
        all_rows.extend(
            run_variant(
                client=client,
                model=model,
                variant_name=variant_name,
                prompt=prompt,
                cases=cases,
                scoring_mode=args.scoring,
            )
        )

    write_results(ROOT / args.out, all_rows)
    print_report(summarize(all_rows))


if __name__ == "__main__":
    main()
