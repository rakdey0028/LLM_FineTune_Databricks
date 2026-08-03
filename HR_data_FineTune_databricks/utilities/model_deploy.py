from transformers import pipeline
from mlflow.models import infer_signature
import transformers
import torch
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from .model_serving import create_serving_endpoint


mlflow.set_registry_uri("databricks-uc")

def model_logging_serving(path:str,registered_model_name:str,run_name:str,endpoint_name: str,
                  workload_size: str = "Small", scale_to_zero: bool = True):
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    tokenizer.chat_template = None

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    prompt = "What is the company's leave policy?"

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    signature = infer_signature(
        model_input=prompt,
        model_output=generated_text,
    )
    mlflow.set_experiment("/Shared/LLM/HRPolicy")
    with mlflow.start_run(run_name=run_name):

        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer,
            },
            task="text-generation",
            artifact_path="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=prompt,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"transformers=={transformers.__version__}",
                f"torch=={torch.__version__.split('+')[0]}",
                "accelerate==1.11.0",
                "bitsandbytes==0.47.0",
            ],
        )

    print(f"✓ Model logged and registered: {model_info.model_uri}")
    print(f"✓ Model name:{registered_model_name}")

    create_serving_endpoint(registered_model_name, endpoint_name, workload_size, scale_to_zero)

    