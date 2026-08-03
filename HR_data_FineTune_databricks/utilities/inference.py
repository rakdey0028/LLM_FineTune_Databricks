
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Databricks
from langchain_core.output_parsers import StrOutputParser
import mlflow
import requests

class DatabricksInference:

    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name

    mlflow.set_experiment("/Shared/LLM/HRPolicy")
    mlflow.langchain.autolog(log_traces=True)

    @mlflow.trace
    def inference_generate(self, instruction: str, max_new_tokens: int = 150):
        """
        Send raw instruction to the endpoint.
        The model was trained with Alpaca format, so we apply it here.
        """
        # Apply the same prompt format used during training
        formatted_prompt = f"""Below is an instruction from an employee. Write an accurate, policy-grounded response for the HR Policy Assistant.

### Instruction:
{instruction}

### Response:
"""

        llm = Databricks(
            endpoint_name=self.endpoint_name,
            extra_params={
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "repetition_penalty": 1.1,
            }
        )

        # Send the formatted prompt directly
        response = llm.invoke(formatted_prompt)

        # Clean up the response
        if "### Response:" in response:
            response = response.split("### Response:")[-1]
        
        return response.strip()
    
    @mlflow.trace
    def inference_generate_direct_api(self, instruction: str, max_new_tokens: int = 150):
        """
        Alternative method using MLflow deployment client.
        Handles authentication automatically in notebook context.
        """
        # Apply the same prompt format used during training
        formatted_prompt = f"""Below is an instruction from an employee. Write an accurate, policy-grounded response for the HR Policy Assistant.

### Instruction:
{instruction}

### Response:
"""
        
        # Use MLflow deployment client - handles notebook auth automatically
        from mlflow.deployments import get_deploy_client
        
        client = get_deploy_client("databricks")
        
        # Call the endpoint using MLflow client
        response = client.predict(
            endpoint=self.endpoint_name,
            inputs={
                "inputs": [formatted_prompt],
                "params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "do_sample": False,
                    "repetition_penalty": 1.1
                }
            }
        )
        
        # Extract generated text
        if isinstance(response, dict):
            if "predictions" in response:
                generated_text = response["predictions"][0]
            elif "choices" in response:
                generated_text = response["choices"][0]["text"]
            else:
                generated_text = str(response)
        else:
            generated_text = str(response)
        
        # Clean up the response
        if "### Response:" in generated_text:
            generated_text = generated_text.split("### Response:")[-1]
        
        return generated_text.strip()