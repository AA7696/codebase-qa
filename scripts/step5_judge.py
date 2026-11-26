import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
judge_model = os.environ.get("JUDGE_MODEL", os.environ["OPENAI_MODEL"])

answers = json.loads(Path("data/answers_final.json").read_text(encoding="utf-8"))

JUDGE_PROMPT = """You are grading an answer about the Python 'requests' library.

You get numbered code chunks, a question, the answer that was given, and a reference answer.

Grade two things separately.
1. faithful: is every factual claim in the answer supported by the numbered code chunks?
   "yes" = all claims are supported, "partly" = some claim is not supported,
   "no" = most claims are not supported.
   Ignore the reference answer for this one. Judge only against the chunks.
2. correct: does the answer agree with the reference answer on the points the question asks about?
   "yes" = agrees, "partly" = right but missing a key point or adding a wrong detail,
   "no" = contradicts it or misses the main point. Extra correct detail is fine.

Reply with JSON only, like:
{"faithful": "yes", "unsupported": ["claim text"], "correct": "partly", "reason": "one sentence"}
"""


def format_chunks(items):
    parts = []
    for number, c in enumerate(items, start=1):
        header = f"[{number}] {c['file']} lines {c['start']}-{c['end']} ({c['name']})"
        parts.append(header + "\n" + c["code"])
    return "\n\n".join(parts)


def judge(item):
    user = (
        f"Code chunks:\n\n{format_chunks(item['retrieved'])}\n\n"
        f"Question: {item['question']}\n\n"
        f"Answer to grade:\n{item['answer']}\n\n"
        f"Reference answer:\n{item['reference_answer']}"
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"faithful": "error", "unsupported": [], "correct": "error", "reason": text[:200]}


for item in answers:
    if item["refused"]:
        item["judge"] = None
        continue
    item["judge"] = judge(item)
    print(".", end="", flush=True)
print()

unanswerable = [a for a in answers if a["kind"] == "unanswerable"]
answerable = [a for a in answers if a["kind"] != "unanswerable"]
print()
print(f"Judge model: {judge_model}")
print(f"Unanswerable questions refused: {sum(a['refused'] for a in unanswerable)} of {len(unanswerable)}")
print(f"Answerable questions refused:   {sum(a['refused'] for a in answerable)} of {len(answerable)}")

for label, wanted in (("expected chunk FOUND", True), ("expected chunk NOT found", False)):
    group = [a for a in answerable if a["found_expected_chunk"] == wanted and a["judge"]]
    print()
    print(f"Answered, {label} ({len(group)} answers):")
    print("  faithful:", dict(Counter(a["judge"].get("faithful") for a in group)))
    print("  correct: ", dict(Counter(a["judge"].get("correct") for a in group)))

print()
print("Look at these (judge did not say yes and yes):")
for a in answers:
    j = a["judge"]
    if j and (j.get("faithful") != "yes" or j.get("correct") != "yes"):
        print(f"  row {a['id']} [{a['kind']}] chunk found: {a['found_expected_chunk']}")
        print(f"    {a['question']}")
        print(f"    faithful: {j.get('faithful')}   correct: {j.get('correct')}")
        print(f"    reason: {j.get('reason')}")
        for claim in j.get("unsupported", [])[:3]:
            print(f"    unsupported: {claim}")

Path("data/judge_results.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
print()
print("Saved to data/judge_results.json")