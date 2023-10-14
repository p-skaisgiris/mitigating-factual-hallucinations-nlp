"""Script that evaluates one summarization model with hallucination metrics."""

from metric.metrics import compute_metrics_pipeline

from transformers import pipeline
from datasets import load_dataset
from metric.metrics import compute_metrics, compute_metrics_pipeline

# Important! For logging results. Make sure you don't have / in the name
# MODEL_NAME = "t5-small-finetuned"
# MODEL_NAME = "googl/e/pegasus-xsum"
# MODEL_NAME = 'sysresearch101/t5-large-finetuned-xsum'

# MODEL_NAME = "facebook/bart-large-xsum"
MODEL_NAME = "facebook/bart-large"

# parse arguments
import argparse

parser = argparse.ArgumentParser()

# argum for model_name
parser.add_argument("--model_name", type=str, default="facebook/bart-large-xsum")

cfg = parser.parse_args()

MODEL_NAME = cfg.model_name

print("MODEL_NAME", MODEL_NAME)
# CHANGE THE ACTUAL MODEL HERE
summarizer = pipeline("summarization", model=MODEL_NAME)


BASE_PROMPT = "Summarize the following text: "
dataset = load_dataset("xsum", split="test[:1000]")

# with open(f"eval-results-{MODEL_NAME}.csv", "w") as f:
with open(f"eval-results-{MODEL_NAME}.csv", "w") as f:
    f.write("xsum_id,qags,rouge,triples,bleurt,summac,ensemble\n")
    for idx, d in enumerate(dataset):
        # Early stopping for testing
        # if idx == 20:
        #     break

        article = d["document"]
        # Add the actual article to the prompt
        prompt = f"{BASE_PROMPT}{article}"
        # Summarize the article
        pred_summary = summarizer(prompt)[0]['summary_text']

        print(f"Prompt:\n{prompt}")
        print(f"Summary:\n{pred_summary}")

        metric_res = compute_metrics_pipeline([article], [pred_summary])

        print(f"{d['id']},{metric_res['qags'][0]},{metric_res['rouge'][0]},{metric_res['triples'][0]},{metric_res['bleurt'][0]},{metric_res['summac'][0]},{metric_res['ensemble'][0]}\n")
        # Write metrics to csv file
        f.write(f"{d['id']},{metric_res['qags'][0]},{metric_res['rouge'][0]},{metric_res['triples'][0]},{metric_res['bleurt'][0]},{metric_res['summac'][0]},{metric_res['ensemble'][0]}\n")
        print()

        f.flush()