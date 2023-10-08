import numpy as np
import nltk
import torch

from summac.model_summac import SummaCZS, SummaCConv
from factsumm import FactSumm
import os
from transformers import pipeline
from bleurt.score import BleurtScorer
import wandb
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# TODO: Change to the BLEURT-20 model instead!
BLEURT_MODEL = "metric/bleurt/BLEURT-20-D3"
if os.path.exists(BLEURT_MODEL):
    checkpoint = BLEURT_MODEL
elif os.path.exists("metric/bleurt/bleurt/BLEURT-20/"):
    checkpoint = "metric/bleurt/bleurt/BLEURT-20/" 
else:
    print("BLEURT MODEL NOT FOUND")
    checkpoint = None

factsumm = FactSumm()
bleurt_scorer = BleurtScorer(checkpoint=checkpoint)
summac_scorer = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence",
               nli_labels="e", device=device, start_file="default", agg="mean")


def compute_metrics_pipeline(articles, summaries):
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
        try:
            qags = factsumm.extract_qas(article, summary, device=device)
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
            facts = factsumm.extract_facts(article, summary, device=device)[-1]
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


def compute_metrics(
    tokenizer,
    document_texts_batch,
    summary_texts_batch,
    summary_texts_pred_batch,
    device,
):
    num_samples = len(document_texts_batch)
    ensemble_scores = []

    factsumm = FactSumm()

    for i in range(num_samples):
        article = document_texts_batch[i]
        summary_texts = summary_texts_batch[i]  # List of vocab IDs
        summary_pred_ids = summary_texts_pred_batch[i]  # List of vocab IDs

        # PREDICTED
        # Decode the summary_ids and summary_pred_ids to text summaries
        summary_pred_text = tokenizer.decode(summary_pred_ids, skip_special_tokens=True)

        article, summary_texts = str(article), str(summary_texts)
        print(f"article {type(article)}\n", article)
        print(f"summary_texts {type(summary_texts)}\n", summary_texts)
        print("summary", summary_pred_text)
        summary = summary_pred_text

        print_QA = False

        print("device", device)

        QA_based = factsumm.extract_qas(article, summary)
        ROUGE = factsumm.calculate_rouge(article, summary)
        facts = factsumm.extract_facts(article, summary)
        # BERT = factsumm.calculate_bert_score(article, summary) #  device="cuda")
        BERT = 1

        # Load the BLEURT scorer model
        checkpoint = "../bleurt/BLEURT-20"
        bleurt_scorer = pipeline(
            task="table-question-generation", model=checkpoint, device=device
        )
        # Define your reference and candidate sentences
        references = [summary_texts]
        candidates = [summary_pred_text]
        # Compute BLEURT scores for the candidates
        BLEURT = bleurt_scorer(references=references, candidates=candidates)

        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=device, start_file="default", agg="mean")
        SUMMAC = model_conv.score([article], [summary], device=device)

        ensemble = np.mean([QA_based, ROUGE, BERT, BLEURT, SUMMAC])

        ensemble_scores.append(ensemble)

    # Calculate the average ensemble score over the batch
    batch_ensemble_score = np.mean(ensemble_scores)

    return {
        "QA_based": QA_based,
        "ROUGE": ROUGE,
        "Triples": facts,
        "BERT": BERT,
        "BLEURT": BLEURT,
        "SUMMAC": SUMMAC,
        "ensemble": batch_ensemble_score,  # Return the average ensemble score
    }
