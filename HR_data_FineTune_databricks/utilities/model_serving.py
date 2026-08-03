from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput
import mlflow

def create_serving_endpoint(registered_model_name:str, endpoint_name:str, workload_size:str, scale_to_zero:bool=True, workload_type:str='CPU'):
    # Create serving endpoint
    w = WorkspaceClient()
    # Get the latest version using Unity Catalog registry
    from mlflow import MlflowClient
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{registered_model_name}'")
    latest_version = max([int(mv.version) for mv in model_versions])

    existing = [e.name for e in w.serving_endpoints.list()]
    served_entity = ServedEntityInput(
        entity_name=registered_model_name,
        entity_version=str(latest_version),
        workload_size=workload_size,
        scale_to_zero_enabled=scale_to_zero
    )

    if endpoint_name in existing:
        print(f"Updating existing endpoint {endpoint_name} to version {latest_version} ...")
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=[served_entity])
    else:
        print(f"Creating endpoint {endpoint_name} (version {latest_version}) ...")
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
        )