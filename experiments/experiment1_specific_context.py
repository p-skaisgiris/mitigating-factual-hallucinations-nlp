"""Script that runs the increasing context specificity experiment on a selected model.

The model is tested on the test split of the XSum dataset. Each prompt increases in specificity
attempting to force the model to refer only to the provided source text and not to its internal
"knowledge". It is an experiment to reduce factual hallucinations.
"""

from metric.metrics import compute_metrics_pipeline

from transformers import pipeline
from datasets import load_dataset

# Important! For logging results
MODEL_NAME = "t5-small"
# TODO: Change the summarizer model to our own!
summarizer = pipeline("summarization", model="t5-small")

prompt1 = "Summarize the following text: "
prompt2 = "Using the exact wording of the text, summarize the following text: "
prompt3 = "Summarize the following text by including direct quotes where they are essential to convey the author's message accurately: "
prompts = [prompt1, prompt2, prompt3]
dataset = load_dataset("xsum", split="test")

with open(f"exp1-results-{MODEL_NAME}.csv", "w") as f:
    f.write("xsum_id,prompt_id,qags,rouge,triples,bleurt,summac,ensemble\n")
    for idx, d in enumerate(dataset):
        # Early stopping for testing
        # if idx == 3:
        #     break
        article = d["document"]
        for p_idx, prompt in enumerate(prompts):
            # Add the actual article to the prompt
            prompt += article
            # Summarize the article
            pred_summary = summarizer(prompt)[0]['summary_text']

            print(f"Prompt:\n{prompt}")
            print(f"Summary:\n{pred_summary}")

            metric_res = compute_metrics_pipeline([article], [pred_summary])

            print(f"{d['id']},{p_idx},{metric_res['qags'][0]},{metric_res['rouge'][0]},{metric_res['triples'][0]},{metric_res['bleurt'][0]},{metric_res['summac'][0]},{metric_res['ensemble'][0]}\n")
            # Write metrics to csv file
            f.write(f"{d['id']},{p_idx},{metric_res['qags'][0]},{metric_res['rouge'][0]},{metric_res['triples'][0]},{metric_res['bleurt'][0]},{metric_res['summac'][0]},{metric_res['ensemble'][0]}\n")
            print()