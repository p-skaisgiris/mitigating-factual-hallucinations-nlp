import argparse

# import pytorch_lightning as pl
import torch
import torch.utils.data as data
import wandb
# from pytorch_lightning.loggers import WandbLogger

from dataset import process_dataset
from model import get_model_tok
# from utils import config_from_args
import numpy as np
from datasets import load_dataset
from evaluate import load
import evaluate

import os
from metric.metrics import compute_metrics
import nltk
nltk.download('punkt')


# rouge = evaluate.load("rouge")

import numpy as np

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Data Parameters

    # TODO: Model Parameters
    parser.add_argument("--model", choices=['t5-small','gpt-tiny', 'gpt-Neo', 'gpt2', 'PEGASUS', 'phi-1.5'], 
                        default="t5-small", help='choose a model architecture')
    
    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # from transformers import TrainingArguments, Trainer


    # WandB Parameters
    parser.add_argument("--wandb-project", type=str, default="model-finetuning")
    # parser.add_argument("--wandb-name", type=str, default="dl4nlp_group7")
    # parser.add_argument("--wandb-entity", type=str, default="-")
    parser.add_argument("--wandb-mode", type=str, default="online")  # disabled
    
    parser.add_argument("--server", type=str, default="Snellius")
    
    cfg = parser.parse_args()
     

    run_name = f"{cfg.model}_{cfg.epochs}_{cfg.learning_rate}_{cfg.batch_size}"

    wandb.init(
        name=run_name,
        project=cfg.wandb_project,
        # entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        tags=["baseline"],
        config={
            "learning_rate": cfg.learning_rate,
            "architecture": cfg.model,
            "dataset": "xsum",
            "epochs": cfg.epochs,
            }
    )    
    
    callbacks = []


    # Function for setting the seed
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_seed(cfg.seed)


    # from sklearn.model_selection import train_test_split


    # dataset = XSUM()
    # dataset = load_dataset("xsum",  split="train[:1%]")
    dataset = load_dataset("xsum",  split="validation[:1%]")

    model, tokenizer = get_model_tok(cfg.model)

    train_dataset, eval_dataset, data_collator = process_dataset(dataset, tokenizer, model)
    # tokenized_dataset, data_collator = process_dataset(dataset, tokenizer, model)

    # using datetime module YYYY_MM_DD_HH_MM
    from datetime import datetime
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M")

    if not os.path.exists(f"models/"):
        os.makedirs(f"models/")
    model_id = f"models/{cfg.model}_{ID}/"
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_id}/",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,  # TODO Understand Param.
        num_train_epochs=cfg.epochs,
        predict_with_generate=True,
        # fp16=True, only if cuda
        # push_to_hub=True,
        report_to="wandb",
    )

    # torch.cuda.set_device(1)
    print(torch.cuda.current_device())
    
    # print available devices
    print("Available devices:")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # metric = load("rouge")
    # def compute_metrics(eval_pred):
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use the same data collator for both train and eval
        # compute_metrics=lambda pred: compute_metrics(pred, tokenizer, eval_dataset),
        # compute_metrics=compute_metrics,  # Pass the compute_metrics function directly

        # logger="wandb",
        # device=cfg.device,
    )

    # print("trainer.model.device: ", trainer.device)

    # print("---------Pre-loading and init. fractsumm metric---------")
    # from factsumm import FactSumm
    # factsumm = FactSumm()

    print("-------- Start training --------")
    for epoch in range(training_args.num_train_epochs):

        print(f"-------- Epoch {epoch + 1} --------")
        trainer.train()

        print("-------- Start evaluation --------")        
        # Perform evaluation during training
        eval_results = trainer.predict(eval_dataset) # .evaluate() --> predict + compute metrics

        # Extract relevant information for custom metric computation
        document_texts = [example['document'] for example in eval_dataset]
        summary_texts = [example['summary'] for example in eval_dataset]
        summary_texts_pred = eval_results.predictions

        # Compute the custom metric using your function
        custom_metric = compute_metrics(tokenizer, document_texts, summary_texts, summary_texts_pred, cfg.device)

        print(f"Epoch {epoch + 1} - Custom Metric ensemble: {custom_metric['ensemble']}")

        # Log the custom metric to WandB
        for key, value in custom_metric.items():
            wandb.log({key: value})

    wandb.finish()



    print("-------- Done training --------")

    trainer.save_model(f"models/{cfg.model}/")
    # save training argument
    
    import pickle
    # Save the args object to a file using pickle
    with open(f"models/{cfg.model}/args.pkl", "wb") as args_file:
        pickle.dump(cfg, args_file)
        
    print("-------- Done saving model & params. --------")

    # trainer.evaluate()

    