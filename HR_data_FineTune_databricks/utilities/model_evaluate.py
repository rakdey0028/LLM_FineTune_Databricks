import  json
import os
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety
from pathlib import Path
from  utilities.inference import DatabricksInference



def load_eval_set(path: str) -> list:
    return json.loads(Path(path).read_text())


def make_predict_fn(endpoint_name: str, max_new_tokens: int = 150):
    inference=DatabricksInference(endpoint_name)

    @mlflow.trace
    def predict_fn(question: str) -> dict:
        answer = inference.inference_generate(question, max_new_tokens=max_new_tokens)
        return {"response": answer}

    return predict_fn


def evaluate_endpoint(
endpoint_name: str,
eval_set_path: str,
judge_model: str | None = None,
host: str | None = None,
max_new_tokens: int = 150,
):
    eval_items = load_eval_set(eval_set_path)
    data = [
        {
            "inputs": {"question": item["question"]},
            "expectations": {"expected_facts": item["expected_facts"]},
        }
        for item in eval_items
    ]

    predict_fn = make_predict_fn(endpoint_name, max_new_tokens=max_new_tokens)

    scorer_kwargs = {"model": judge_model} if judge_model else {}
    scorers = [
        Correctness(**scorer_kwargs),       # needs expectations -- factual accuracy vs. expected_facts
        RelevanceToQuery(**scorer_kwargs),  # no ground truth needed -- does the answer address the question
        Safety(**scorer_kwargs),            # no ground truth needed -- flags unsafe/inappropriate content
    ]

    
    os.environ['MLFLOW_GENAI_EVAL_MAX_WORKERS']="1"
    mlflow.set_experiment("/Shared/LLM/HRPolicy")
    with mlflow.start_run(run_name=f"llm_judge_{endpoint_name}_v1"):
        results = mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)

    return eval_items, results