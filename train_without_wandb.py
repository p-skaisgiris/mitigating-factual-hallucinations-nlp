import argparse
import torch
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from model import get_model_tok
from dataset import process_dataset
from metric.metrics import compute_metrics_pipeline, compute_summarization_metrics
import numpy as np
import pandas as pd
import os
from datetime import datetime

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument("--data_split", type=str, default="train[:]+validation[:]")
    # parser.add_argument("--data_split", type=str, default="train[:3%]+validation[:3%]")
    
    # Model Parameters
    parser.add_argument("--model", default="t5-small", help='choose a model architecture')
    
    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--wandb-project", type=str, default="model-finetuning")
    parser.add_argument("--wandb-name", type=str, default="dl4nlp_group7")
    # parser.add_argument("--wandb-entity", type=str, default="-")
    parser.add_argument("--wandb-mode", type=str, default="disabled")  # disabled
    

    cfg = parser.parse_args()
     
    run_name = f"{cfg.model}_{cfg.epochs}_{cfg.learning_rate}_{cfg.batch_size}"

    # Set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    dataset = load_dataset("xsum", split=cfg.data_split)
    model, tokenizer = get_model_tok(cfg.model)
    train_dataset, eval_dataset, data_collator = process_dataset(dataset, tokenizer, model)
    
    # Create a model ID based on the current timestamp
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M")
    model_id = f"models/a_training/{cfg.model}_{ID}/"

    if not os.path.exists(model_id):
        os.makedirs(model_id)
    
    print("model_id id", model_id)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_id,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        weight_decay=0.01,
        save_total_limit=10,
        num_train_epochs=cfg.epochs,
        predict_with_generate=True,
        evaluation_strategy="no",
        do_eval=False,
    )


    # Initialize custom metric variables
    custom_metrics = {
        "xsum_id": [],
        "document": [],
        "gold_summary": [],
        "pred_summary": [],
        "classic_rouge1": [],
        "classic_rouge2": [],
        "classic_rougel": [],
        "qags": [],
        "rouge": [],
        "triples": [],
        "bleurt": [],
        "summac": [],
        "ensemble": [],
    }

    training_loss_list = []  # List to store training loss values

    train_loss, eval_loss = 0.0, 0.0
    # Training loop
    for epoch in range(1000):
        shuffled_dataset = train_dataset.shuffle(seed=epoch)
        subset_size = 2000  # Subset size for training
        train_subset = shuffled_dataset.select([i for i in range(subset_size)])

        # Initialize the trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        metrics = trainer.train()
        # print("metrics", metrics)
        train_loss = metrics.training_loss 

        # take subset of eval_dataset
        eval_dataset = eval_dataset.shuffle(seed=epoch)
        eval_size = 15
        eval_subset = eval_dataset.select([i for i in range(eval_size)])
        

        eval_results = trainer.predict(eval_subset)
        # print("eval_results", eval_results)
        eval_loss = eval_results.metrics["test_loss"]
        document_texts = [example['document'] for example in eval_subset]
        summary_texts = [example['summary'] for example in eval_subset]
        summary_texts_pred = eval_results.predictions

        custom_metric = compute_metrics_pipeline(articles=document_texts, 
                                                 summaries=summary_texts_pred,
                                                 tokenizer=tokenizer)

        custom_metrics["xsum_id"].append(eval_dataset["id"])
        custom_metrics["document"].append(document_texts)
        custom_metrics["gold_summary"].append(summary_texts)
        custom_metrics["pred_summary"].append(summary_texts_pred)
        custom_metrics["qags"].append(custom_metric['qags'])
        custom_metrics["rouge"].append(custom_metric['rouge'])
        custom_metrics["triples"].append(custom_metric['triples'])
        custom_metrics["bleurt"].append(custom_metric['bleurt'])
        custom_metrics["summac"].append(custom_metric['summac'])
        custom_metrics["ensemble"].append(custom_metric['ensemble'])


        classic_rouge = compute_summarization_metrics(summary_texts, summary_texts_pred,
                                                      tokenizer)
        custom_metrics["classic_rouge1"].append(classic_rouge['classic_rouge1'])
        custom_metrics["classic_rouge2"].append(classic_rouge['classic_rouge2'])
        custom_metrics["classic_rougel"].append(classic_rouge['classic_rougel'])

        # trainer.save_model(f"{model_id}/checkpoints/epoch_{epoch}/")

        print(f"Epoch {epoch + 1} - Custom Metric ensemble: {custom_metric['ensemble']}")

        # Save training loss to a local CSV file

        # Append the training loss
        training_loss_list.append((train_loss, eval_loss))

        training_loss_df = pd.DataFrame(training_loss_list, columns=["training_loss", "val_loss"])
        training_loss_df.to_csv(f"{model_id}/training_loss.csv", index=False)

        # Save custom metrics to a local CSV file
        custom_metrics_df = pd.DataFrame(custom_metrics)
        custom_metrics_df.to_csv(f"{model_id}/custom_metrics.csv", index=False)

    print("Done training and saved training loss and custom metrics.")
    trainer.save_model(f"{model_id}/checkpoints/epoch_{epoch}_last/")
