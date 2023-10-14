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
from metric.metrics import compute_metrics, compute_metrics_pipeline
import nltk
nltk.download('punkt')

# rouge = evaluate.load("rouge")

import numpy as np

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Data Parameters
    # parser.add_argument("--data_split", type=str, default="train[:]+validation[:]")  # train[:50]+validation[:1%]
    parser.add_argument("--data_split", type=str, default="train[:]+validation[:]")  # train[:50]+validation[:1%]

    # TODO: Model Parameters
    parser.add_argument("--model", 
                        # choices=['google/t5-efficient-tiny, t5-small','gpt-tiny', 'gpt-Neo', 'gpt2', 'PEGASUS', 'phi-1.5'], 
                        default="t5-small", help='choose a model architecture')
    
    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
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
    parser.add_argument("--wandb-name", type=str, default="dl4nlp_group7")
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
            
            # device name
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    set_seed(cfg.seed)


    dataset = load_dataset("xsum",  split=cfg.data_split)

    model, tokenizer = get_model_tok(cfg.model)
    # model, tokenizer = get_model_tok('google/t5-efficient-tiny') #DON't use, will return empty summary
    # model, tokenizer = get_model_tok('t5-small')
    
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
        learning_rate=2e-5,
        per_device_train_batch_size=1   ,
        # per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,  # TODO max saved checkpoints
        num_train_epochs=cfg.epochs,
        predict_with_generate=True,
        evaluation_strategy="no", 
        # evaluation_strategy="epoch",
        do_eval=False,
        # fp16=True, only if cuda
        # push_to_hub=True,
        # report_to=["wandb"],
    )

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     # eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,  # Use the same data collator for both train and eval
    #     # device=cfg.device,
    # )

    # print("trainer.model.device: ", trainer.device)

    # print("---------Pre-loading and init. fractsumm metric---------")
    # from factsumm import FactSumm
    # factsumm = FactSumm()

    # remove memory torch
    torch.cuda.empty_cache()

    # number of eval per batch per epoch
    NumberOfEvals = 15
    
    print("-------- Start training --------")
    for epoch in range(training_args.num_train_epochs):

        print(f"-------- Epoch {epoch + 1} --------")

        # Shuffle the entire training dataset to ensure randomness
        shuffled_dataset = train_dataset.shuffle(seed=epoch)  # Use 'epoch' as the seed for reproducibility
        
        # Determine the size of the subset (e.g., 10% of the data)
        subset_size = int(0.01 * len(dataset))

        subset_size = int(200)
        print(f"subset_size: {subset_size}")
        
        # Create a subset of the shuffled training data
        train_subset = shuffled_dataset.select([i for i in range(subset_size)])
        
        # Define the trainer with the current training subset
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Perform training on the current subset
        # trainer.train()
        torch.cuda.empty_cache()

        print("-------- Start evaluation --------")        
         # Perform evaluation only for the first X examples
        eval_dataset = eval_dataset.shuffle()
                
        eval_results = trainer.predict(eval_dataset) # .evaluate() --> predict + compute metrics

        torch.cuda.empty_cache()
        # Extract relevant information for custom metric computation
        document_texts = [example['document'] for example in eval_dataset]
        summary_texts = [example['summary'] for example in eval_dataset]
        summary_texts_pred = eval_results.predictions

        # Compute the custom metric using your function
        custom_metric = compute_metrics_pipeline(articles=document_texts, 
                                                 summaries=summary_texts_pred)

   
        # custom_metric = compute_metrics(tokenizer, document_texts, summary_texts, 
        #                                 summary_texts_pred, cfg.device, NumberOfEvals)
        
        print(custom_metric)
        print(f"Epoch {epoch + 1} - Custom Metric ensemble: {custom_metric['ensemble']}")

        wandb.log(custom_metric)

        # Log the custom metric to WandB
        for key, value in custom_metric.items():
            key = f"eval_{key}_on_epoch"
            print(key, value)
            wandb.log({key: value})
        torch.cuda.empty_cache()

        # save model checkpoint with epoch number

        trainer.save_model(f"models/{cfg.model}_{ID}/checkpoints/epoch_{epoch}/")
        # Log the eval loss to WandB
        
        wandb.log({"eval_loss": eval_results.metrics['test_loss']})

        # wandb.log({"eval_loss": eval_results.metrics["eval_loss"]}) DOES NOT WORK
 
        
    wandb.finish()

    print("-------- Done training --------")

    # save training argument
    
    import pickle
    # Save the args object to a file using pickle

    with open(f"models/{cfg.model}_{ID}/args.pkl", "wb") as args_file:
        pickle.dump(cfg, args_file)
        
    print("-------- Done saving model & params. --------")

    # trainer.evaluate()

    