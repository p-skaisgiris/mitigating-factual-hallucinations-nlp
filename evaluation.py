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
import nltk
nltk.download('punkt')

# rouge = evaluate.load("rouge")

import numpy as np

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO: Data Parameters
    parser.add_argument("--data_split", type=str, default="train[:1]+validation[:1]")  # train[:50]+validation[:1%]

    # TODO: Model Parameters
    parser.add_argument("--model", 
                        # choices=['google/t5-efficient-tiny, t5-small','gpt-tiny', 'gpt-Neo', 'gpt2', 'PEGASUS', 'phi-1.5'], 
                        default="t5-small", help='choose a model architecture')
    
    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
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
     
    run_name = f"{cfg.model}_evaluation_with_or_without_XSUM"

    wandb.init(
        name=run_name,
        project=cfg.wandb_project,
        # entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        tags=["evaluation"],
        config={
            "architecture": cfg.model,
            "dataset": "xsum",
            }
    )    
    
    callbacks = []

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"    

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
    
    train_dataset, eval_dataset, data_collator = process_dataset(dataset, tokenizer, model)

    eval_dataset = eval_dataset.shuffle()

    # using datetime module YYYY_MM_DD_HH_MM
    from datetime import datetime
    ID = datetime.now().strftime("%Y_%m_%d_%H_%M")

    if not os.path.exists(f"models/"):
        os.makedirs(f"models/")
    model_id = f"models/baseline/{cfg.model}_{ID}/"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_id}/",
        learning_rate=2e-5,
        per_device_train_batch_size=1   ,
        # per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,  # TODO max saved checkpoints
        num_train_epochs=1,
        predict_with_generate=True,
        evaluation_strategy="no", 
        # evaluation_strategy="epoch",
        do_eval=False,
        # fp16=True, # only if cuda
        # push_to_hub=True,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use the same data collator for both train and eval
        # device=cfg.device,
    )

    torch.cuda.empty_cache()

    # number of eval per batch per epoch
    NumberOfEvals = 3
    
    print("-------- Start Eval --------")

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
    from metric.metrics import compute_metrics

    custom_metric = compute_metrics(tokenizer, document_texts, summary_texts, 
                                    summary_texts_pred, cfg.device, NumberOfEvals)
    
    print(custom_metric)
    print(f"Epoch- Custom Metric ensemble: {custom_metric['ensemble']}")

    # Log the custom metric to WandB
    for key, value in custom_metric.items():
        key = f"eval_{key}_on_epoch"
        wandb.log({key: value})
    torch.cuda.empty_cache()

    # save model checkpoint with epoch number
    
    wandb.log({"eval_loss": eval_results.metrics['test_loss']})
    
    wandb.finish()

    print("-------- Done Eval --------")

    # save training argument
  
    # trainer.evaluate()

    