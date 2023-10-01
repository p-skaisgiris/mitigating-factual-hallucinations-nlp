from transformers import AutoTokenizer

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer



from transformers import AutoTokenizer

checkpoint = "ComCom/gpt2-small"

# checkpoint = "google/pegasus-xsum"
# checkpoint = "lucadiliello/bart-small"
# checkpoint = "prajjwal1/bert-tiny"
checkpoint = "t5-small"

def get_model_tok(model_name):

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return model, tokenizer