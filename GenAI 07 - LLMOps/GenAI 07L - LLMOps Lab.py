# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LLMOps Lab
# MAGIC
# MAGIC This notebook demonstrates the end-to-end workflow for question answering using Hugging Face's Transformers library, integrated with MLflow Tracking for experiment tracking and the MLflow Model Registry for managing model versions.
# MAGIC
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you should be able to:
# MAGIC 1. Create a question-answering model using a pre-trained Hugging Face model.
# MAGIC 1. Utilize MLflow Tracking to log parameters, metrics, and artifacts during the model development process.
# MAGIC 1. Understand the role of the MLflow Model Registry in managing different versions of the model.
# MAGIC 1. Deploy a model for batch inference using MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12, 13.3.x-gpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell to setup the classroom environment:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Overview
# MAGIC
# MAGIC In this demo we will be using the `SQuAD` dataset hosted on HuggingFace. This is a reading comprehension dataset which consists of questions and answers based on the provided context. Let's load and inspect the structure of the SQuAD dataset.
# MAGIC
# MAGIC **📌 Note:** If you get an error due to dataset's *splits sizes verification*, you can turn off the **`verification_mode`** by uncommenting the `verification_mode` line.

# COMMAND ----------

from datasets import load_dataset

# Note: We specify cache_dir to use pre-cached data.
dataset = load_dataset(
    "squad", version="1.2.0", 
    cache_dir=DA.paths.datasets,
    verification_mode="no_checks"
)

dataset_sample = dataset["train"].select(range(10))
display(dataset_sample.to_pandas())

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Develop an LLM pipeline
# MAGIC
# MAGIC In this section, you will develop an LLM pipeline to answer questions based on the provided context. 
# MAGIC
# MAGIC Our LLMOps goals during development are (a) to track what we do carefully for later auditing and reproducibility and (b) to package models or pipelines in a format which will make future deployment easier.  Step-by-step, we will:
# MAGIC * Load data.
# MAGIC * Build an LLM pipeline.
# MAGIC * Test applying the pipeline to data, and log queries and results to MLflow Tracking.
# MAGIC * Log the pipeline to the MLflow Tracking server as an MLflow Model.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Create a HuggingFace Q&A Pipeline

# COMMAND ----------

from transformers import pipeline
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

# COMMAND ----------

# TODO

Construct a question-answering pipeline
qa_pipeline = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Track LLM Development with MLflow
# MAGIC
# MAGIC After creating the Q&A pipeline, you will need to track the pipeline with MLFlow's Tracking component. You will need to log pipeline details such as model parameters, metrics and input/output examples.
# MAGIC

# COMMAND ----------

import mlflow

inference_config = {"min_length": 20, "max_length": 40, "truncation": True, "do_sample": True}
params = {"model_name": "distilbert-base-cased-distilled-squad"}
input_example = [{
    "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the 'golden anniversary' with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as 'Super Bowl L'), so that the logo could prominently feature the Arabic numerals 50.", 
    "question": "Which NFL team represented the AFC at Super Bowl 50?"}]

output_example = qa_pipeline(context=input_example[0]["context"], question=input_example[0]["question"])

# COMMAND ----------

# Tell MLflow Tracking to use this explicit experiment path,
# which is located on the left hand sidebar under Machine Learning -> Experiments 
mlflow.set_experiment(f"/Users/{DA.username}/LLM-07L-LLMOps-Lab")

# COMMAND ----------

# TODO

Infer signiture for the model using input and output examples 
signature = <FILL_IN>

Start the run
<FILL_IN> (run_name = "LLMOps_Lab_QA_Pipeline"):

log custom paramters
<FILL_IN>

log transformers model
model_info = <FILL_IN>


# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model
# MAGIC
# MAGIC Next, you will need to register the tracked model to Model Registry. 

# COMMAND ----------

# unique model name scoped to each user
model_name = f"LLMOps_QA_Lab_{DA.unique_name(sep='_')}"

# COMMAND ----------

# TODO
Register the model
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Model State Management and Batch Inference
# MAGIC
# MAGIC In this section, you will manage the registered model's state. First, you will search for the registered model and inspect the current state. Then, you will move the model to `Staging` and use it for batch inference. In the last step, you will promote the model to `Production` stage and serve it with **Model Serving**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search and Inspect Registered Model
# MAGIC
# MAGIC Search for the model by name and show model details.

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# COMMAND ----------

# TODO
search the model with model_name
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote the Model Stage
# MAGIC Promote the model to `Staging` stage.

# COMMAND ----------

# TODO
promote the first version to the staging
model_version = 1
client.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the Model for Batch Inference
# MAGIC
# MAGIC Load the registered model from `staging` and use it for batch inference.
# MAGIC
# MAGIC Let's use dataset's validation set's first 100 rows. 

# COMMAND ----------

val_dataset = dataset["validation"].select_columns(["context", "question"]).select(range(100)).to_pandas()
display(val_dataset)

# COMMAND ----------

# TODO

load model from staging 
staging_model = <FILL_IN>

use dataset for inference
predictions = <FILL_IN>

# COMMAND ----------

# convert the prediction list to pd dataframe to display in nice tabular format
import pandas as pd
display(pd.DataFrame(predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serve with Model Serving
# MAGIC
# MAGIC As last step of LLMOps workflow, you will promote the model to `Production` stage and serve it with **Model Serving**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote the Model to Production
# MAGIC
# MAGIC Before serving the model, promote it to `Production` stage and serve it from that stage. 

# COMMAND ----------

# TODO
promote the model from staging to production
model_version = 1
client.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serve the Model in Real-time with Model Serving
# MAGIC
# MAGIC In this step, you will need to use the UI to find the model in Model Registery and serve with Model Serving in real-time.
# MAGIC
# MAGIC Follow these steps;
# MAGIC
# MAGIC - Go to **Models**, select the model you have built.
# MAGIC
# MAGIC - Click the **Use model for inference** button on the top right.
# MAGIC
# MAGIC - Select the **Real-time** tab.
# MAGIC
# MAGIC - Select the **model version** and provide an **endpoint name**. Let's enter *"LLMOps-Lab-QA-Pipeline"* as endpoint name.
# MAGIC
# MAGIC - Select the compute size for your endpoint, and specify if your endpoint should **scale to zero** when not in use.
# MAGIC
# MAGIC - Click **Create endpoint**. 
# MAGIC
# MAGIC
# MAGIC **View Model Serving Endpoint:**
# MAGIC
# MAGIC - Go to **"Serving"** page.
# MAGIC
# MAGIC - Select the endpoint we just created. The Serving endpoints page appears with Serving endpoint state shown as Not Ready. After a few minutes, Serving endpoint state changes to Ready.
# MAGIC
# MAGIC - Test the model by clicking **Query endpoint** button on top right. Select **Browser** tab and use provided example to test the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitor the Model
# MAGIC
# MAGIC After querying the endpoint, view the **Metrics** and **Logs** tabs located at the bottom of the **Serving** page. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Summary
# MAGIC
# MAGIC In this lab, we've offered a practical LLMOps demonstration. Initially, you crafted an LLM pipeline for question answering. Subsequently, you registered this LLM model in the Model Registry. The subsequent stage involved promoting the model to the staging environment and utilizing it for inference.
# MAGIC
# MAGIC Moving to the second section of the lab, you advanced the registered model to the production stage. Here, you served it in real-time using Databricks Model serving. This lab provides a comprehensive example, illustrating the end-to-end process of LLM development and deployment.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>