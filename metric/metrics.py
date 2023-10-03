"Eval Metric"

# import evaluate
# rouge = evaluate.load("rouge")
# metric = load("rouge")

import numpy as np
import nltk

from .summac.summac.model_summac import SummaCZS, SummaCConv
from factsumm import FactSumm

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)

#     return {k: round(v, 4) for k, v in result.items()}




# def computre_rouge(eval_pred, tokenizer):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     # Rouge expects a newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
#     # Note that other metrics may not have a `use_aggregator` parameter
#     # and thus will return a list, computing a metric for each sentence.
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
#     # Extract a few results
#     result = {key: value * 100 for key, value in result.items()}
    
#     # Add mean generated length
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)
    
#     return {k: round(v, 4) for k, v in result.items()}


def compute_metrics(tokenizer, document_texts_batch, summary_texts_batch, summary_texts_pred_batch, device):

    num_samples = len(document_texts_batch)
    ensemble_scores = []

    factsumm = FactSumm()

    print("-------- Start evaluation --------")
    print("num_samples", num_samples)

    batch_scores = {
        "QA_based": [],
        "ROUGE": [],
        "BERT": [],
        "BLEURT": [],
        "SUMMAC": [],
        "ensemble": []
    }
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

        QA_based = factsumm.extract_qas(article, summary, verbose=print_QA) # device="cuda")
        ROUGE = factsumm.calculate_rouge(article, summary) # device="cuda")
        
        # BERT = factsumm.calculate_bert_score(article, summary) #  device="cuda")
        BERT = 1

        # calculate Bert scores using 
        import torch

        # remove cache
        torch.cuda.empty_cache()

        from evaluate import load
        bertscore = load("bertscore")  
      # predictions = ["hello there", "general kenobi"]
        # references = ["hello there", "general kenobi"]
        predictions = [article]
        references = [summary]

        results = bertscore.compute(predictions=predictions, references=references, lang="en")
        print(results, "takign only f1 score")
        BERT = results["f1"]


        from transformers import pipeline
        # Load the BLEURT scorer model
        
        # check in model for BLEURT-20-D3 or BLEURT-20
        import os
        if os.path.exists("metric/bleurt/bleurt/BLEURT-20-D3/"):
            checkpoint = "metric/bleurt/bleurt/BLEURT-20-D3/"
        else:
            checkpoint = "metric/bleurt/bleurt/BLEURT-20/"
            
        bleurt_scorer = pipeline(task="table-question-generation", model=checkpoint, device=device)
        # Define your reference and candidate sentences
        references = [article]
        candidates = [summary_pred_text]
        # Compute BLEURT scores for the candidates
        BLEURT = bleurt_scorer(references=references, candidates=candidates)


        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=device, start_file="default", agg="mean")
        SUMMAC = model_conv.score([article], [summary_pred_text],  device=device)

        # update scores
        batch_scores["QA_based"].append(QA_based)
        batch_scores["ROUGE"].append(ROUGE)
        batch_scores["BERT"].append(BERT)
        batch_scores["BLEURT"].append(BLEURT)
        batch_scores["SUMMAC"].append(SUMMAC)
        batch_scores["ensemble"].append(np.mean([QA_based, ROUGE, BERT, BLEURT, SUMMAC]))
        
        print("batch_scores", batch_scores)
    # Calculate the average ensemble score over the batch
    batch_ensemble_score = np.mean(ensemble_scores)

    return {
        "QA_based": QA_based,
        "ROUGE": ROUGE,
        "BERT": BERT,
        "BLEURT": BLEURT,
        "SUMMAC": SUMMAC,
        "ensemble": batch_ensemble_score  # Return the average ensemble score
    }