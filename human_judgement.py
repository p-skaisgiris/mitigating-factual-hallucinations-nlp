from datasets import load_dataset
import pandas as pd

# ----------------------------------------------------
import pandas as pd

path = "metric/factuality_annotations_xsum_summaries.csv"

df = pd.read_csv(path)
# drop system columns
summ = df["summary"].copy()

# Drop system columns
df = df.drop(columns=["system"])

# Convert "is_factual" to lowercase and replace values with 1 for "yes" and 0 for "no"
df["is_factual"] = df["is_factual"].str.lower().replace({"yes": 1, "no": 0})

# Group by 'bbcid' and calculate the mean of 'is_factual' while keeping the first summary
df = df.groupby("bbcid", as_index=False).agg({'is_factual': 'mean', 'summary': 'first'})

# Rename 'is_factual' to 'human_judgement'
df = df.rename(columns={"is_factual": "human_judgement"})
df 
print(df.head())
print(df.shape)
# # ----------------------------------------------------
# # # ----------------------------------------------------
# # probably only in test
# # dataset = load_dataset("xsum",  split="train+test")
dataset = load_dataset("xsum",  split="test")

# print(dataset[0])

# keeping track of the ids, if you know the d[id] of the dataset, you can get the document
record = {}

# Loop over the dataset to populate the 'record' dictionary
for i, d in enumerate(dataset):
    bbcid = d["id"]
    record[str(bbcid)] = {"document": d["document"], "dataset_index": i}


# ## Loop over the DataFrame to match 'bbcid' with the dataset
for idx, row in df.iterrows():
    bbcid = str(int(row["bbcid"]))
    if bbcid in record:
        # Retrieve the dataset index and document from the 'record' dictionary
        dataset_index = record[bbcid]["dataset_index"]
        document = record[bbcid]["document"]
        # Store the document and dataset index in the DataFrame
        df.loc[idx, 'document'] = document
        # df.loc[idx, 'summary'] = df["summary"][idx]
        df.loc[idx, 'dataset_index'] = int(dataset_index)

print(df.columns)
print(df.head())

df.to_csv("test.csv", index=False)

import numpy as np

# loop over df, 
metrics = ["qags", "rouge", "triples", "bleurt", "summac", "ensemble"]
# create these column names in df
df["qags"] = None
df["rouge"] = None
df["triples"] = None
df["bleurt"] = None
df["summac"] = None
df["ensemble"] = None

from datetime import datetime

ID = datetime.now().strftime("%Y_%m_%d_%H_%M")

print(ID)
print(df.head())
print(df.shape)

"""Script that evaluates one summarization model with hallucination metrics."""
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from factsumm import FactSumm
from metric.metrics import compute_metrics, compute_metrics_pipeline
import numpy as np

# MODEL_NAME = "t5-small"
# summarizer = pipeline("summarization", model=MODEL_NAME)

BASE_PROMPT = "Summarize the following text: "
dataset = load_dataset("xsum", split="test[:1000]")

# if '/' in MODEL_NAME:
#     MODEL_NAME = MODEL_NAME.replace('/', '-')

# import torch
# # Check if GPU is available
# print("GPU available: ", torch.cuda.is_available())
# # print device names
# print(torch.cuda.get_device_name(torch.cuda.current_device()))

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

# Create an empty DataFrame
columns = ["qags", "rouge", "triples", "bleurt", "summac", "ensemble"]

for idx, row in df.iterrows():

    article = row["document"]
    summary = row['summary']

    metric_res = compute_metrics_pipeline([article], [summary])

    print(metric_res)
    # loop over metric_res dict and store to df row
    for key, value in metric_res.items():
        if key in columns:
            df.loc[idx, key] = value[0]
    
    # save df to csv after each iteration
    df.to_csv(f"human_judge_summ_metrics_{ID}.csv", index=False)

