"Eval Metric"

# import evaluate
# rouge = evaluate.load("rouge")
# metric = load("rouge")
import torch
import numpy as np
import nltk
import wandb
from .summac.summac.model_summac import SummaCZS, SummaCConv
from factsumm import FactSumm

def compute_metrics(tokenizer, document_texts_batch, summary_texts_batch, 
                    summary_texts_pred_batch, device, NumberOfEvals, print_QA = False):

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
    
    # overwrites the number of evals if the batch size is smaller
    if NumberOfEvals < num_samples:
        num_samples = NumberOfEvals  
    
    for i in range(0, num_samples):
        article = document_texts_batch[i]
        summary_texts = summary_texts_batch[i]  # List of vocab IDs
        summary_pred_ids = summary_texts_pred_batch[i]  # List of vocab IDs
        
        # PREDICTED
        # Decode the summary_ids and summary_pred_ids to text summaries
        # print("summary_pred_ids", summary_pred_ids)
        summary_pred_text = tokenizer.decode(summary_pred_ids, skip_special_tokens=True)
        # print("summary_pred_text", summary_pred_text)
        summary_pred_text = summary_pred_text if type(summary_pred_text) == list else summary_pred_text
        
        article, summary_texts = str(article), str(summary_texts)
        # print()
        summary = summary_pred_text

        
        # print("device", device)
        torch.cuda.empty_cache()

        try:
            QA_BASED = factsumm.extract_qas(article, summary, verbose=print_QA)
        except:
            QA_BASED = None
        print("QA_BASED", QA_BASED)
        torch.cuda.empty_cache()

        R1, R2, R_L = factsumm.calculate_rouge(article, summary)
        ROUGE = R_L
        print("ROUGE", ROUGE)
        torch.cuda.empty_cache()

        BERT, BLEURT = 0, 0
        # BERT = factsumm.calculate_bert_score(article, summary)
        print("BERT", BERT)
        # torch.cuda.empty_cache()
        # F1, Precsion, Recall
        # from bleurt import score
        # import os
        # # check in model for BLEURT-20-D3 or BLEURT-20    
        # if os.path.exists("metric/bleurt/bleurt/BLEURT-20-D3/"):
        #     checkpoint = "metric/bleurt/bleurt/BLEURT-20-D3/"
        # else:
        #     checkpoint = "metric/bleurt/bleurt/BLEURT-20/"
        # scorer = score.BleurtScorer(checkpoint)
        # BLEURT = scorer.score(references=[article], candidates=[summary])
        # print("BLEURT", BLEURT)
        # torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=device, start_file="default", agg="mean")
        SUMMAC = model_conv.score([article], [summary])['scores'][0]
        print("SUMMAC", SUMMAC)
        torch.cuda.empty_cache()

        # update scores
        batch_scores["QA_based"].append(QA_BASED)
        batch_scores["ROUGE"].append(ROUGE)
        batch_scores["BERT"].append(BERT)
        batch_scores["BLEURT"].append(BLEURT)
        batch_scores["SUMMAC"].append(SUMMAC)
        print(QA_BASED, ROUGE, BERT, BLEURT, SUMMAC)
        combine = [QA_BASED, ROUGE, BERT, BLEURT, SUMMAC]
        filtered_list = list(filter(lambda x: x is not None, combine))
        batch_scores["ensemble"].append(np.mean(filtered_list))

        wandb.log({
            "QA_based": QA_BASED,
            "ROUGE": ROUGE,
            "BERT": BERT,
            "BLEURT": BLEURT,
            "SUMMAC": SUMMAC,
            "ensemble": np.mean(filtered_list)  # Ensemble score
        })
        
    filtered_list_QA = list(filter(lambda x: x is not None, batch_scores["QA_based"]))

    return {
        "QA_based": np.mean(filtered_list_QA),
        "ROUGE": np.mean(batch_scores["ROUGE"]),
        "BERT": np.mean(batch_scores["BERT"]),
        "BLEURT": np.mean(batch_scores["BLEURT"]),
        "SUMMAC": np.mean(batch_scores["SUMMAC"]),
        "ensemble": np.mean(batch_scores["ensemble"]),  # Return the average ensemble score
    }
