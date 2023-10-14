"""Script that evaluates one summarization model with hallucination metrics."""
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from factsumm import FactSumm
from metric.metrics import compute_metrics, compute_metrics_pipeline
import numpy as np

MODEL_NAME = "t5-small"
summarizer = pipeline("summarization", model=MODEL_NAME)

BASE_PROMPT = "Summarize the following text: "
dataset = load_dataset("xsum", split="test[:1000]")

if '/' in MODEL_NAME:
    MODEL_NAME = MODEL_NAME.replace('/', '-')

import torch
# Check if GPU is available
print("GPU available: ", torch.cuda.is_available())
# print device names
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Initialize scorers
factsumm = FactSumm()

def calculate_rouge(article, summary):
    try:
        rouge = factsumm.calculate_rouge(article, summary)
        return rouge
    except Exception as ex:
        print(f"ROUGE ERROR! {ex}")
        return (0, 0, 0)

def remove_newlines(text):
    return text.replace("\n", " ")

def compute_summarization_metrics(articles, summaries):
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    
    for article, summary in zip(articles, summaries):
        rouge_scores = calculate_rouge(article, summary)
        rouge1_scores.append(rouge_scores[0])
        rouge2_scores.append(rouge_scores[1])
        rougel_scores.append(rouge_scores[2])

    return {
        "rouge1": rouge1_scores,
        "rouge2": rouge2_scores,
        "rougel": rougel_scores
    }

# Create an empty DataFrame
columns = ["xsum_id", "document", "gold_summary", "pred_summary", "rouge1", "rouge2", "rougel", "qags", "rouge", "triples", "bleurt", "summac", "ensemble"]
df = pd.DataFrame(columns=columns)

for idx, d in enumerate(dataset):
    article = d["document"]
    prompt = f"{BASE_PROMPT}{article}"
    
    pred_summary = summarizer(prompt)[0]['summary_text']
    gold_summary = d["summary"]

    print(f"Gold:\n{gold_summary}")
    print(f"Summary:\n{pred_summary}")

    metric_res = compute_metrics_pipeline([gold_summary], [pred_summary])

    xsum_id = d['id']
    metric_res_rouge = calculate_rouge([gold_summary], [pred_summary])
   
    rouge1 = (metric_res_rouge[0])
    rouge2 = (metric_res_rouge[1])
    rougel = (metric_res_rouge[2])

    # print(f"{d['id']},{metric_res['qags'][0]},{metric_res['rouge'][0]},{metric_res['triples'][0]},{metric_res['bleurt'][0]},{metric_res['summac'][0]},{metric_res['ensemble'][0]}\n")

    # Append a new row to the DataFrame
    new_row = pd.DataFrame([[xsum_id, remove_newlines(article), remove_newlines(gold_summary), remove_newlines(pred_summary), rouge1, rouge2, rougel, metric_res['qags'][0], metric_res['rouge'][0], metric_res['triples'][0], metric_res['bleurt'][0], metric_res['summac'][0], metric_res['ensemble'][0]]], columns=columns)
    df = pd.concat([df, new_row], ignore_index=True)

    print()

    # Write the DataFrame to a CSV file, in case job crashes
    df.to_csv(f"summ-metrics-{MODEL_NAME}.csv", index=False)
