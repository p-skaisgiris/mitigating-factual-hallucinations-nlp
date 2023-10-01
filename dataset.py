
from transformers import DataCollatorForSeq2Seq

prefix = "summarize: "
# prefix = "summarize, and only  "




def process_dataset(dataset, tokenizer, checkpoint):

    xsum_dataset = dataset.train_test_split(test_size=0.2)

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["document"]]
        MAX_LEN = 42
        model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = xsum_dataset["train"]
    eval_dataset = xsum_dataset["test"]
    print("Size of train set", len(train_dataset))
    print("Size of eval set", len(eval_dataset))

    # tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # Ensure the datasets are in the correct format
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    return train_dataset, eval_dataset, data_collator

