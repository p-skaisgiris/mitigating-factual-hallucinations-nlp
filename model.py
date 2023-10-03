from transformers import AutoTokenizer

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer



from transformers import AutoTokenizer

checkpoint = "ComCom/gpt2-small"

# checkpoint = "google/pegasus-xsum"
# checkpoint = "lucadiliello/bart-small"
# checkpoint = "prajjwal1/bert-tiny"
checkpoint = "t5-small"

def get_model_tok(model_name):

    print("model", model_name)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    return model, tokenizer