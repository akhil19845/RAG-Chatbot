import json
import time
from pathlib import Path
from deepeval import evaluate
from deepeval.models import GeminiModel
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from statistics import mean
from collections import defaultdict
import json
import os
from dotenv import load_dotenv
from app.config import (
    INPUT_EVAL_DATA_FILE,
    OUTPUT_EVAL_DATA_FILE
)

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

model = GeminiModel(
    model_name="gemini-2.5-flash-lite",
    api_key=api_key,
    temperature=0
)

custom_llm = model


def load_test_cases(ground_truth_path: str):
    p = Path(ground_truth_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    test_cases = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("chatbot_answer", "")
        truth = item.get("ground_truth", "")
        contexts = item.get("contexts", [])

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=truth,
                retrieval_context=contexts,
            )
        )
    return test_cases



test_cases = load_test_cases(INPUT_EVAL_DATA_FILE)
print(f"Loaded {len(test_cases)} test cases")

# ---- Define metrics ----
metrics = [
    # ContextualPrecisionMetric(model=custom_llm, include_reason=True),
    # ContextualRelevancyMetric(model=custom_llm, include_reason=True),
    # ContextualRecallMetric(model=custom_llm, include_reason=True)
    FaithfulnessMetric(model=custom_llm, include_reason=True)
    # AnswerRelevancyMetric(model=custom_llm, include_reason=True)
]


eval_results = []

for metric in metrics:
    metric_name = metric.__class__.__name__
    print(f"\nRunning {metric_name} ...")

    metric_scores = []

    for idx, case in enumerate(test_cases, 0):
        for attempt in range(2):  # attempt 0 = first try, 1 = retry
            try:
                res = metric.measure(case)
                metric_scores.append(res)
                print(f"  [{idx}] {metric_name} score={res:.3f}")
                eval_results.append({
                    "metric": metric_name,
                    "question": case.input,
                    "chatbot_answer": case.actual_output,
                    "ground_truth": case.expected_output,
                    "retrieval_context": case.retrieval_context,
                    "score": res
                })
                break
            except Exception as e:
                if attempt == 0:
                    print(f"Error in case #{idx} ({metric_name}), retrying once: {e}")
                    # time.sleep(2)  # brief pause before retry (optional)
                else:
                    print(f"Failed again in case #{idx} ({metric_name}): {e}")
            finally:
                time.sleep(60)

    # compute average for this metric
    if metric_scores:
        avg_score = mean([s for s in metric_scores if s is not None])
        print(f"Average {metric_name} score: {avg_score:.3f}")
    else:
        print(f"No valid scores for {metric_name}.")



out_path = Path(OUTPUT_EVAL_DATA_FILE)
with out_path.open("w", encoding="utf-8") as f:
    for r in eval_results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nResults written to {out_path.resolve()}")
print("Each line includes: metric, question, chatbot_answer, ground_truth, contexts, score")



metric_groups = defaultdict(list)
for r in eval_results:
    if r.get("score") is not None:
        metric_groups[r["metric"]].append(r["score"])
        
print("\nFinal average scores per metric:")
print("===================================")
for metric_name, scores in metric_groups.items():
    avg_score = mean(scores)
    print(f"{metric_name:<30} | Avg Score: {avg_score:.3f}")