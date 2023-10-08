"""Script that runs the Chain of Verification (CoVe) experiment for a selected model.

The model is tested on the test split of the XSum dataset. The CoVe pipeline looks 
as follows:
1. Ask LLM $M_S$ to summarize a text, receive response $resp$
2. Generate questions from $resp$ using a question-generating model $M_{QG}$
3. Ask LLM $M_S$ to answer the generated questions
4. Create a new prompt that is comprised of the generated questions by $M_{QG}$, the answers by $M_S$, and the original prompt to summarize a text
5. Receive a verified response $resp_v$
"""

import torch

from metric.metrics import compute_metrics_pipeline
import metric.metrics
import sys

from transformers import pipeline
from datasets import load_dataset

# Important! For logging results
MODEL_NAME = "t5-small"
# TODO: Change to our trained models!!
summarizer = pipeline("summarization", model="t5-small")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

dataset = load_dataset("xsum", split="test")
BASE_PROMPT = "Summarize the following text: "


def chain_of_verification(baseline_summary: str, prompt: str, summ_model):
    # Generate questions based on baseline response
    questions = metric.metrics.factsumm.extract_questions(baseline_summary, verbose=False, device=device)

    # Make a modified prompt based on the answers to generated questions    
    # and their answers
    verified_prompt = ""
    for question in questions:
        # TODO: is this correct? Should we have fine-tuned our summarization model on question-answering?
        resp_q = summ_model(question)[0]['summary_text']
        verified_prompt += f"{question}\n{resp_q}\n"
    verified_prompt += f"\n{prompt}"

    # Get verified response
    resp_v = summ_model(verified_prompt)[0]['summary_text']
    return verified_prompt, resp_v


with open(f"exp2-results-{MODEL_NAME}.csv", "w") as f:
    f.write("xsum_id,qags,rouge,triples,bleurt,summac,ensemble\n")
    for idx, d in enumerate(dataset):
        # Early stopping for testing
        # if idx == 3:
        #     break
        article = d["document"]

        # Add the actual article to the prompt
        prompt = f"{BASE_PROMPT}{article}"
        # Summarize the article
        pred_summary = summarizer(prompt)[0]["summary_text"]

        verified_prompt, verified_pred_summary = chain_of_verification(
            pred_summary, prompt, summarizer
        )

        print(f"Prompt:\n{verified_prompt}")
        print(f"Summary:\n{verified_pred_summary}")

        metric_res = compute_metrics_pipeline([article], [verified_pred_summary])
        print(
            f"{d['id']},"
            f"{metric_res['qags'][0]},"
            f"{metric_res['rouge'][0]},"
            f"{metric_res['triples'][0]},"
            f"{metric_res['bleurt'][0]},"
            f"{metric_res['summac'][0]},"
            f"{metric_res['ensemble'][0]}\n"
        )
        # Write metrics to csv file
        f.write(
            f"{d['id']},"
            f"{metric_res['qags'][0]},"
            f"{metric_res['rouge'][0]},"
            f"{metric_res['triples'][0]},"
            f"{metric_res['bleurt'][0]},"
            f"{metric_res['summac'][0]},"
            f"{metric_res['ensemble'][0]}\n"
        )
        print()
