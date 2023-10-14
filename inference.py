
# pip install tensorflow[and-cuda]

# checkpoints = {

#     # t5-small : Paulius
#     "t5-s": "t5-small",
#     "t5-s-xsum": "pki/t5-small-finetuned_xsum",   
    
#     # t5-large : Luka
#     "t5-large": "t5-large",
#     "t5-large-xsum": "sysresearch101/t5-large-finetuned-xsum",

#     # t5-large xsum-cnn Skipping for now
#     # "t5-large-xsum-cnn" : "sysresearch101/t5-large-finetuned-xsum-cnn",

#     # google/pegasus-xsum" :  Erencan
#     "pegasus-xsum": "google/pegasus-xsum",
# }

import argparse
import torch
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from model import get_model_tok
from dataset import process_dataset
import numpy as np
import pandas as pd
import os
from datetime import datetime
import numpy as np
import nltk
import torch

from metric.summac.summac.model_summac import SummaCZS, SummaCConv
# from summac.model_summac import SummaCZS, SummaCConv
from metric.factsumm.factsumm.factsumm import FactSumm
import os
from transformers import pipeline
from bleurt.score import BleurtScorer
# import wandb
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# TODO: Change to the BLEURT-20 model instead!
# /home/scur0666/dl4nlp-text-summarization/metric/bleurt/BLEURT-20
BLEURT_MODEL = "metric/bleurt/BLEURT-20-D3"
if os.path.exists(BLEURT_MODEL):
    checkpoint = BLEURT_MODEL
elif os.path.exists("metric/bleurt/BLEURT-20"):
    checkpoint = "metric/bleurt/BLEURT-20/" 
else:
    print("BLEURT MODEL NOT FOUND")
    checkpoint = None

factsumm = FactSumm()
bleurt_scorer = BleurtScorer(checkpoint=checkpoint)
summac_scorer = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence",
               nli_labels="e", device=device, start_file="default", agg="mean")


def compute_metrics_pipeline2(articles, summaries):
    qags_scores = []
    rouge_scores = []
    fact_scores = []
    bleurt_scores = []
    summac_scores = []
    ensemble_scores = []

    # try:
    #     bleurt_scores = bleurt_scorer.score(references=articles, candidates=summaries)
    # except Exception as ex:
    #     # TODO: bad idea if only one fails out of all of them?
    #     print(f"BLEURT ERROR! {ex}")
    #     bleurt_scores = [0] * len(articles)
    #
    # try:
    #     summac_scores = summac_scorer.score(articles, summaries)
    # except Exception as ex:
    #     print(f"SUMMAC ERROR! {ex}")
    #     summac_scores = [0] * len(articles)

    for i in range(len(articles)):
        article, summary = articles[i], summaries[i]


        print("eval", i)
        print("article", article)
        print("summary", summary)
        try:
            qags = factsumm.extract_qas(article, summary, verbose=True, device=device)
        except TypeError:
            # TODO: BIAS! Take care of this?
            qags = 0
        qags_scores.append(qags)

        try:
            rouge = factsumm.calculate_rouge(article, summary)[-1] # Take ROUGE-L
        except Exception as ex:
            print(f"ROUGE ERROR! {ex}")
            rouge = 0
        rouge_scores.append(rouge)

        # Doesn't work, idk
        # BERT = factsumm.calculate_bert_score(article, summary)

        try:
            facts = factsumm.extract_facts(article, summary, verbose=True, device=device)[-1]
        except Exception as ex:
            print(f"FACTS ERROR! {ex}")
            facts = 0
        fact_scores.append(facts)

        references = [article]
        candidates = [summary]
        try:
            bleurt_score = bleurt_scorer.score(references=references,
                                               candidates=candidates)[0]
        except Exception as ex:
            print(f"BLEURT ERROR! {ex}")
            bleurt_score = 0
        bleurt_scores.append(bleurt_score)

        try:
            summac_score = summac_scorer.score(references, candidates)['scores'][0]
        except Exception as ex:
            print(f"SUMMAC ERROR! {ex}")
            summac_score = 0
        summac_scores.append(summac_score)
        
        ensemble = np.mean([qags, rouge, facts, bleurt_score, summac_score])
        ensemble_scores.append(ensemble)

    # Calculate the average ensemble score over the batch
    # batch_ensemble_score = np.mean(ensemble_scores)

    return {
        "qags": qags_scores,
        "rouge": rouge_scores,
        "triples": fact_scores,
        # "BERT": BERT,
        "bleurt": bleurt_scores,
        "summac": summac_scores,
        "ensemble": ensemble_scores,
    }



article = ["""The injured pedestrian - a young man - is thought to have been walking with a group of people from a graduation ceremony at the Caird Hall. The incident took place on High Street at about 18:00. The man's injuries are believed not to be life-threatening. The driver of the taxi is thought to be uninjured."""]

# hallucinated_Summary = ["""During a Caird Hall graduation ceremony, a pedestrian accident on High Street at 18:00 left a young man and a taxi driver in critical condition."""]

# Not_Hallucinated_Summary = ["""A pedestrian has been struck by a taxi in Dundee after it mounted the pavement."""]

# Not_Hallucinated_Summary_Len = ["""A young man, part of a group returning from a Caird Hall graduation ceremony, was injured on High Street around 18:00. His injuries are not life-threatening, and the taxi driver is uninjured."""]

# golden_summary = ["""A pedestrian has been struck by a taxi in Caird Hall after it mounted the pavement."""]

test_summary = ["""A pedestrian has been struck by a taxi in High Street after it mounted the pavement."""]

print(test_summary)
print("article", len(article[0]))
# print("hallucinated_Summary", len(hallucinated_Summary))
# print("Not_Hallucinated_Summary", len(Not_Hallucinated_Summary))
# print("Not_Hallucinated_Summary_Len", len(Not_Hallucinated_Summary_Len))
print("golden_summary", len(test_summary[0]))

# custom_metric_hall = compute_metrics_pipeline2(articles=article, 
#                                                  summaries=hallucinated_Summary)
# custom_metric_not_hall = compute_metrics_pipeline2(articles=article, 
#                                                  summaries=Not_Hallucinated_Summary)
# custom_metric_not_hall_len = compute_metrics_pipeline2(articles=article, 
#                                                  summaries=Not_Hallucinated_Summary_Len)

custom_metric_test = compute_metrics_pipeline2(articles=article, 
                                                 summaries=test_summary)

print("custom_metric_test",
      custom_metric_test)
