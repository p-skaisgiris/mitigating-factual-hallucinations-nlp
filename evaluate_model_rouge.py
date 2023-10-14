"""
Calculates CLASSIC ROUGE scores for a model on the XSum dataset.
Score between two summaries, the gold summary and the predicted summary.
"""
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from factsumm import FactSumm

# parse arguments
import argparse

parser = argparse.ArgumentParser()

# argum for model_name
parser.add_argument("--model_name", type=str, 
                    required=True,
                    default="facebook/bart-large-xsum")

cfg = parser.parse_args()

MODEL_NAME = cfg.model_name
# MODEL_NAME = "t5-small"
summarizer = pipeline("summarization", model=MODEL_NAME)

if '/' in MODEL_NAME:
    MODEL_NAME = MODEL_NAME.replace('/', '-')

print("MODEL_NAME: ", MODEL_NAME)

BASE_PROMPT = "Summarize the following text: "
dataset = load_dataset("xsum", split="test[:1000]")

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
columns = ["xsum_id", "document", "gold_summary", "pred_summary", "rouge1", "rouge2", "rougel"]
df = pd.DataFrame(columns=columns)

for idx, d in enumerate(dataset):
    article = d["document"]
    prompt = f"{BASE_PROMPT}{article}"
    
    pred_summary = summarizer(prompt)[0]['summary_text']
    gold_summary = d["summary"]

    print(f"Gold:\n{gold_summary}")
    print(f"Pred Summary:\n{pred_summary}")

    metric_res = compute_summarization_metrics([gold_summary], [pred_summary])

    xsum_id = d['id']
    rouge1 = metric_res['rouge1'][0]
    rouge2 = metric_res['rouge2'][0]
    rougel = metric_res['rougel'][0]

    # Append a new row to the DataFrame using concat
    new_row = pd.DataFrame([[xsum_id, remove_newlines(article), remove_newlines(gold_summary), remove_newlines(pred_summary), rouge1, rouge2, rougel]], columns=columns)
    df = pd.concat([df, new_row], ignore_index=True)

    print()

    # Write the DataFrame to a CSV file
    df.to_csv(f"summ-metrics-{MODEL_NAME}_rouge.csv", index=False)
