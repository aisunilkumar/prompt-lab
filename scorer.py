import json


def exact_match_score(output: str, expected: str) -> dict:
    cleaned = output.strip().lower()
    target = expected.strip().lower()
    passed = cleaned == target
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reason": "exact match" if passed else f"expected {target}, got {cleaned}",
    }


def build_judge_messages(task: str, input_text: str, expected: str, output: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict evaluator. Return only JSON with keys "
                "score, passed, and reason. score must be 1 or 0. "
                "passed must be true or false."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "input": input_text,
                    "expected": expected,
                    "model_output": output,
                    "grading_rule": (
                        "Pass if the model output means the same label as expected. "
                        "Fail if it chooses the wrong label or includes unclear extra text."
                    ),
                }
            ),
        },
    ]


def judge_score(client, model: str, task: str, case: dict, output: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=build_judge_messages(
            task=task,
            input_text=case["input"],
            expected=case["expected"],
            output=output,
        ),
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "passed": False,
            "reason": "judge returned invalid JSON",
        }

    score = float(data.get("score", 0))
    passed = bool(data.get("passed", score >= 1))
    reason = str(data.get("reason", "no reason provided"))
    return {"score": score, "passed": passed, "reason": reason}


def score_output(mode: str, client, model: str, task: str, case: dict, output: str) -> dict:
    if mode == "exact":
        return exact_match_score(output, case["expected"])
    if mode == "judge":
        return judge_score(client, model, task, case, output)
    raise ValueError(f"Unknown scoring mode: {mode}")
