# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Project Setup - Run this ONCE per cluster/session
# MAGIC %md
# MAGIC # Project Setup
# MAGIC
# MAGIC **How to use in other notebooks:**
# MAGIC
# MAGIC At the top of ANY notebook in your project, just add:
# MAGIC ```python
# MAGIC %run ./00_setup
# MAGIC ```
# MAGIC
# MAGIC Then you can import:
# MAGIC ```python
# MAGIC from utilities import model_deploy
# MAGIC from utilities.model_serving import create_serving_endpoint
# MAGIC ```
# MAGIC
# MAGIC **Works from any folder** - the path adjusts automatically!

# COMMAND ----------

# DBTITLE 1,Install project package
import sys
import os

# Dynamically find and add project root to sys.path
def find_project_root(start_path=None):
    if start_path is None:
        start_path = os.getcwd()
    
    current = start_path
    while current != '/':
        if os.path.exists(os.path.join(current, 'setup.py')):
            return current
        current = os.path.dirname(current)
    
    raise FileNotFoundError("Could not find setup.py in any parent directory")

project_root = find_project_root()

# Add project root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✓ Added {project_root} to sys.path")
else:
    print(f"✓ {project_root} already in sys.path")

print("✓ Setup complete! You can now import: from utilities import model_deploy")

# COMMAND ----------

