# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluating LLMs Lab
# MAGIC
# MAGIC This lab aims to assess the translation performance of two LLMs. Translation tasks are commonly evaluated using BLEU scores. Comparing these scores for each model helps us comprehend their translation capabilities. In real-world scenarios, model evaluation involves examining multiple metrics, but for the sake of simplicity in this lab, we'll focus on a single metric.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this lab, you will be able to:
# MAGIC
# MAGIC * Utilize an LLM for translation tasks.
# MAGIC * Evaluate translation quality by calculating BLEU scores, which measure the accuracy and fluency of machine translations.
# MAGIC * Discuss BLEU scores and interpret their significance concerning model performance.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12, 13.3.x-gpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cells to setup the classroom environment.

# COMMAND ----------

# MAGIC %pip install sacrebleu
# MAGIC %pip install sacremoses

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Dataset Preparation 
# MAGIC
# MAGIC We will use a subset of the `cnn_dailymail` dataset from See et al., 2017, downloadable from the [Hugging Face `datasets` hub](https://huggingface.co/datasets/cnn_dailymail).
# MAGIC
# MAGIC This dataset provides news article paired with summaries (in the "highlights" column).  Let's load the data and take a look at some examples.
# MAGIC
# MAGIC **📌 Note:** If you get an error due to dataset's *splits sizes verification*, you can turn off the **`verification_mode`** by uncommenting the `verification_mode` line.

# COMMAND ----------

from transformers import pipeline
from datasets import load_dataset

# We specify cache_dir to use pre-cached data.
full_dataset = load_dataset(
    "cnn_dailymail", "3.0.0",
    split='train',
    cache_dir=DA.paths.datasets,
    verification_mode="no_checks"
)  

# Use a small sample of the data during this lab, for speed.
sample_size = 10
sample = (
    full_dataset
    .filter(lambda r: "CNN" in r["article"][:25])
    .shuffle(seed=42)
    .select(range(sample_size))
)
sample

# COMMAND ----------

display(sample.to_pandas())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Translation with LLMs
# MAGIC
# MAGIC We will leverage two different LLMs to translate the *highlights* field of the dataset. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Translation with T5-Small Model
# MAGIC
# MAGIC First, translate the text using **T5-Small**. You should build an **English-to-French** translation pipeline and translate the **highlights** field.

# COMMAND ----------

# TODO
translator_pipeline = <FILL_IN>
t5_translated_text = <FILL_IN>
print(t5_translated_text)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Translation with Helsinki-NLP Model
# MAGIC
# MAGIC As the second model, use **Helsinki-NLP** for translation. You should build an **English-to-French** translation pipeline and translate the **highlights** field.  

# COMMAND ----------

# TODO
translator_pipeline_hnlp = <FILL_IN>
hnlp_translated_text = <FILL_IN>
print(hnlp_translated_text)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## BLEU Score for Translation Evaluation
# MAGIC
# MAGIC To assess the quality of our translations, we will calculate the BLEU score which provides a numerical estimation of translation quality.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Reference Translations and Candidate Translations
# MAGIC
# MAGIC As reference translation, we are going to use translations that we manually generated using GPT-3.5. Normally, we would use a human-translated text for reference. 

# COMMAND ----------

import sacrebleu
import statistics

reference_translations = [
    "La Papouasie-Nouvelle-Guinée se trouve sur le soi-disant Cercle de Feu. C'est sur un arc de lignes de faille qui est sujet à des tremblements de terre fréquents. Aucune alerte au tsunami n'a été émise.",
    "L'Australie s'effondre à 88 tout en ouvrant le jour du deuxième test contre le Pakistan à Leeds. Le Pakistan, cherchant à égaliser la série de deux matchs, a atteint 148-3 lorsque la mauvaise lumière a interrompu le jeu. Le capitaine australien Ricky Ponting a étonnamment choisi de frapper en premier par temps couvert. Son équipe n'a pas réussi à atteindre les trois chiffres dans un test pour la première fois depuis 1984.",
    "Jared Loughner refuse la demande du gouvernement pour un échantillon d'écriture. Les autorités le veulent pour le comparer avec des notes trouvées chez lui après la fusillade. Loughner est confronté à 49 chefs d'accusation liés à une fusillade de masse devant un marché de Tucson.",
    "La victime de la fusillade, McKayla Hicks, est allée à l'audience pour l'accusé de meurtre James Holmes. Elle a dit qu'elle pouvait ressentir 'toute la colère que tout le monde avait pour' Holmes. L'incident l'a changée, a déclaré Hicks. Une balle s'est logée dans sa mâchoire - les médecins ont dit qu'il était plus sûr de la laisser là.",
    "Oscar Pistorius deviendra le premier athlète doublement amputé aux Jeux olympiques de Londres. Le jeune homme de 25 ans a été sélectionné dans les 400 mètres individuels et le relais 4x400 mètres. Pistorius s'est fait amputer les deux jambes lorsqu'il avait 11 mois. Il a remporté une médaille d'argent lors des Championnats du monde de l'année dernière dans le relais 4 x 400 mètres.",
    "NOUVEAU : L'avocat de Perry qualifie les inculpations d'« abus politique du système judiciaire ». L'inculpation par un grand jury du comté au Texas découle d'un effort pour évincer un procureur local. Perry aurait menacé de veto le financement d'un programme géré par le procureur de district à Austin. L'inculpation pourrait avoir des implications politiques.",
    "Le procureur dit au juge : Assez de preuves pour poursuivre l'enquête sur le président. Un autre procureur, avant sa mort, a allégué que le président avait caché la prétendue implication de l'Iran dans l'attentat à la bombe. La présidente Cristina Fernández de Kirchner et d'autres responsables nient toute dissimulation.",
    "NOUVEAU : Le président de l'UEFA, Michel Platini, exhorte les fans à se comporter lors des matches décisifs de samedi. L'UEFA affirme qu'il y a eu des chants racistes de la part des supporters croates lors d'un match contre l'Italie. La question du racisme menace de ternir le tournoi de football Euro 2012. Une commission disciplinaire examinera le cas de la Croatie mardi.",
    "Un nouveau groupe de haut niveau pour discuter de la coopération économique se réunira à l'automne. Obama dit que les liens entre les États-Unis et le Mexique vont au-delà de la sécurité et de l'immigration. Le président du Mexique déclare que son administration est engagée dans la lutte contre le crime organisé. Le président américain se rendra au Costa Rica vendredi pour rencontrer les dirigeants de l'Amérique centrale.",
    "Quatre détenus s'échappent de la prison de St. Tammany Parish, en Louisiane. Trois ont été retrouvés dans une zone près de la prison au nord de La Nouvelle-Orléans, selon un responsable. Un homme inculpé de meurtre est toujours en fuite, selon un responsable. Des adjoints ratissent les quartiers à la recherche du fugitif."
]

hnlp_translations = [t["translation_text"] for t in hnlp_translated_text]
t5small_translations = [t["translation_text"] for t in t5_translated_text]

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Calculate BLEU Scores for Each Model
# MAGIC
# MAGIC Let's calculate the BLEU score for each model compared to reference translation.

# COMMAND ----------

# Computing BLEU score using sacrebleu for Helsinki-NLP
bleu_scores_hnlp = []
for refs, cands in zip(reference_translations, hnlp_translations):
    bleu = sacrebleu.raw_corpus_bleu(cands, [refs])
    bleu_scores_hnlp.append(bleu.score)

# COMMAND ----------

# Computing BLEU score using sacrebleu for T5-Small
bleu_scores_t5small = []
for refs, cands in zip(reference_translations, t5small_translations):
    bleu = sacrebleu.raw_corpus_bleu(cands, [refs])
    bleu_scores_t5small.append(bleu.score)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### BLEU Score Interpretation
# MAGIC
# MAGIC Having calculated the BLEU score for each model per row in the previous step, it's now time to compute the mean of these scores for each model and interpret the results!
# MAGIC
# MAGIC What are your thoughts on mean BLEU scores? Which model exhibits better performance?
# MAGIC

# COMMAND ----------

# TODO
print("Mean BLEU scores for H-NLP:", round(statistics.mean(bleu_scores_hnlp), 1))
print("Mean BLEU scores for T5-Sm:", round(statistics.mean(bleu_scores_t5small), 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Throughout this lab, we employed two LLMs for a translation task and evaluated their performance based on reference translations. Initially, we loaded a news dataset containing a 'highlights' field. Subsequently, we utilized two LLMs to translate this field. In the latter part of the lab, we computed BLEU scores for each model and engaged in a discussion regarding the obtained results.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>