# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Generative AI Engineering with Databricks
# MAGIC
# MAGIC This course is aimed at data scientists, machine learning engineers, and other data practitioners looking to build Generative AI Engineering applications with the latest and most popular frameworks. In this course, you will build common Generative AI applications using Hugging Face, develop retrieval-augmented generation (RAG) applications, create multi-stage reasoning pipelines using LangChain, fine-tune LLMs for specific tasks, assess and address societal considerations of using LLMs, and learn how to deploy your models at scale leveraging LLMOps best practices.
# MAGIC
# MAGIC By the end of this course, you will have built an end-to-end Generative AI workflow that is ready for production!
# MAGIC
# MAGIC ## Course agenda
# MAGIC
# MAGIC | Time | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|-------------|
# MAGIC | 55m    | **[Introduction]($./GenAI 00 - Introduction/GenAI 00a - Install Datasets)**    | Generative AI and LLMs</br>Practical NLP Primer</br>Databricks and LLMs</br>[Demo: Install Datasets]($./GenAI 00 - Introduction/GenAI 00a - Install Datasets)|
# MAGIC | 85m    | **[Common Applications with LLMs]($./GenAI 01 - Common Applications with LLMs/GenAI 01 - Common Applications)** | Common Applications Overview </br> [Common Applications Demo]($./GenAI 01 - Common Applications with LLMs/GenAI 01 - Common Applications) </br> [Common Applications Lab]($./GenAI 01 - Common Applications with LLMs/GenAI 01L - Common Applications Lab) | 
# MAGIC | 10m  | **Break**                                            ||
# MAGIC | 90m  | **[Retrieval-Augmented Generation (RAG)]($./GenAI 02 - Retrieval-Augmented Generation [RAG]/GenAI 02 - RAG with FAISS and Chroma)** | Retrieval-augmented Generation Overview </br> [Retrieval-augmented Generation Demo]($./GenAI 02 - Retrieval-Augmented Generation [RAG]/GenAI 02 - RAG with FAISS and Chroma) </br> [Retrieval-augmented Generation Lab]($./GenAI 02 - Retrieval-Augmented Generation [RAG]/GenAI 02L - RAG Lab) </br> [RAG with Pinecone [OPTIONAL]]($./GenAI 02 - Retrieval-Augmented Generation [RAG]/GenAI 02a - RAG with Pinecone [OPTIONAL]) </br> [RAG with Weaviate [OPTIONAL]]($./GenAI 02 - Retrieval-Augmented Generation [RAG]/GenAI 02b - RAG with Weaviate [OPTIONAL])| 
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 35m    | **[Multi-stage Reasoning with LLM Chains]($./GenAI 03 - Multi-stage Reasoning with LLM Chains/GenAI 03 - Building LLM Chains)**    | Multi-stage Reasoning Overview </br> [Multi-stage Reasoning Demo]($./GenAI 03 - Multi-stage Reasoning with LLM Chains/GenAI 03 - Building LLM Chains) </br> [Multi-stage Reasoning Lab]($./GenAI 03 - Multi-stage Reasoning with LLM Chains/GenAI 03L - Building LLM Chains Lab) |
# MAGIC | 10m | **Break**                                               ||
# MAGIC | 90m  | **[Fine-tuning LLMs]($./GenAI 04 - Fine-tuning LLMs/GenAI 04 - Fine-tuning LLMs)**       | Fine-tuning LLMs Overview </br> [Fine-tuning LLMs Demo]($./GenAI 04 - Fine-tuning LLMs/GenAI 04 - Fine-tuning LLMs) </br> [Fine-tuning LLMs Lab]($./GenAI 04 - Fine-tuning LLMs/GenAI 04L - Fine-tuning LLMs Lab) |
# MAGIC | 75m  | **[Evaluating LLMs]($./GenAI 05 - Evaluating LLMs/GenAI 05 - Evaluating LLMs)**      | Evaluating LLMs Overview </br> [Evaluating LLMs Demo]($./GenAI 05 - Evaluating LLMs/GenAI 05 - Evaluating LLMs) </br> [Evaluating LLMs Lab]($./GenAI 05 - Evaluating LLMs/GenAI 05L - Evaluating LLMs Lab) |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 75m |**[LLMs and Society]($./GenAI 06 - LLMs and Society/GenAI 06 - LLMs and Society)** |  Society and LLMs Overview </br> [Society and LLMs Demo]($./GenAI 06 - LLMs and Society/GenAI 06 - LLMs and Society) </br> [Society and LLMs Lab]($./GenAI 06 - LLMs and Society/GenAI 06L - LLMs and Society Lab) |
# MAGIC | 10m  | **Break**                                               ||
# MAGIC | 70m  | **[LLMOps]($./GenAI 07 - LLMOps/GenAI 07 - LLMOps)**  | LLMOps Overview </br>[LLMOps Demo]($./GenAI 07 - LLMOps/GenAI 07 - LLMOps) </br>[LLMOps Lab]($./GenAI 07 - LLMOps/GenAI 07L - LLMOps Lab) |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12, 13.3.x-gpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>