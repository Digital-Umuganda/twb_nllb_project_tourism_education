import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import wandb
from datasets.combine import concatenate_datasets
from datasets import load_dataset, load_metric
import numpy as np

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
KIN_LANG_CODE = "kin_Latn"
EN_LANG_CODE = "eng_Latn"

en_kin_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=EN_LANG_CODE, tgt_lang=KIN_LANG_CODE)
kin_en_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=KIN_LANG_CODE, tgt_lang=EN_LANG_CODE)
wandb.init(entity='')

edu_train = pd.read_csv("education_train_data.tsv",sep="\t")
tourism_train = pd.read_csv("tourism_train_data.tsv",sep="\t")
du_train = pd.read_csv("du_train.tsv",sep="\t")
edu_val = pd.read_csv("education_val_data.tsv",sep="\t")
tourism_val = pd.read_csv("tourism_val_data.tsv",sep="\t") 
du_val = pd.read_csv("du_val.tsv",sep="\t")
edu_test = pd.read_csv("education_test_data.tsv",sep="\t")
tourism_test = pd.read_csv("tourism_test_data.tsv",sep="\t")
du_test = pd.read_csv("du_test.tsv",sep="\t")


train_data_en = list(edu_train['source'])  + list(tourism_train['source']) + list(du_train['source'])
train_data_kin = list(edu_train['phrase'])  + list(tourism_train['phrase']) + list(du_train['phrase'])
val_data_en = list(edu_val['source']) + list(tourism_val['source']) + list(du_val['source'])
val_data_kin = list(edu_val['phrase']) + list(tourism_val['phrase']) + list(du_val['phrase'])
test_data_en = list(edu_test['source']) + list(tourism_test['source']) + list(du_test['source'])
test_data_kin = list(edu_test['phrase']) + list(tourism_test['phrase']) + list(du_test['phrase'])

from datasets import Dataset, DatasetDict, load_dataset


max_source_length = max_target_length = 128
padding = "max_length"
truncation = True



en_kin_src_key = "sentence_" + EN_LANG_CODE
en_kin_tgt_key = "sentence_" + KIN_LANG_CODE
kin_en_src_key = "sentence_" + KIN_LANG_CODE
kin_en_tgt_key = "sentence_" + EN_LANG_CODE



train_data_en_kin = [{en_kin_src_key: str(src), en_kin_tgt_key: str(tgt)} for src,tgt in zip(train_data_en,train_data_kin)]
val_data_en_kin = [{en_kin_src_key: str(src), en_kin_tgt_key: str(tgt) } for src,tgt in zip(val_data_en,val_data_kin)]
test_data_en_kin = [{en_kin_src_key: str(src), en_kin_tgt_key: str(tgt)} for src,tgt in zip(test_data_en,test_data_kin)]

train_data_kin_en = [{kin_en_src_key: str(tgt), kin_en_tgt_key: str(src)} for src,tgt in zip(train_data_en,train_data_kin)]
val_data_kin_en = [{kin_en_src_key: str(tgt), kin_en_tgt_key: str(src)} for src,tgt in zip(val_data_en,val_data_kin)]
test_data_kin_en = [{kin_en_src_key: str(tgt), kin_en_tgt_key: str(src)} for src,tgt in zip(test_data_en,test_data_kin)]



train_dataset_en_kin = Dataset.from_list(train_data_en_kin)
val_dataset_en_kin = Dataset.from_list(val_data_en_kin)
test_dataset_en_kin = Dataset.from_list(test_data_en_kin)


train_dataset_kin_en = Dataset.from_list(train_data_kin_en)
val_dataset_kin_en = Dataset.from_list(val_data_kin_en)
test_dataset_kin_en = Dataset.from_list(test_data_kin_en)

def en_kin_tokenize_fn(examples):

    inputs = examples[en_kin_src_key]
    targets = examples[en_kin_tgt_key]

    model_inputs = en_kin_tokenizer(inputs, max_length = max_source_length, padding  = padding, truncation = truncation)
    with en_kin_tokenizer.as_target_tokenizer():
        labels = en_kin_tokenizer(targets, max_length = max_target_length, padding = padding, truncation = truncation)
    labels["input_ids"] = [[(i if i != en_kin_tokenizer.pad_token_id else -100) for i in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

en_kin_tokenized_train_data = train_dataset_en_kin.map(en_kin_tokenize_fn, batched=True)
en_kin_tokenized_val_data = val_dataset_en_kin.map(en_kin_tokenize_fn, batched=True)

def kin_en_tokenize_fn(examples):

    inputs = examples[kin_en_src_key]
    targets = examples[kin_en_tgt_key]

    model_inputs = kin_en_tokenizer(inputs, max_length = max_source_length, padding  = padding, truncation = truncation)
    with kin_en_tokenizer.as_target_tokenizer():
        labels = kin_en_tokenizer(targets, max_length = max_target_length, padding = padding, truncation = truncation)
    labels["input_ids"] = [[(i if i != kin_en_tokenizer.pad_token_id else -100) for i in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

kin_en_tokenized_train_data = train_dataset_kin_en.map(kin_en_tokenize_fn, batched=True)
kin_en_tokenized_val_data = val_dataset_kin_en.map(kin_en_tokenize_fn, batched=True)

tokenized_train_data = concatenate_datasets([en_kin_tokenized_train_data,kin_en_tokenized_train_data])
tokenized_val_data = concatenate_datasets([en_kin_tokenized_val_data,kin_en_tokenized_val_data])

metric = load_metric("sacrebleu")

def metrics_calc(data):
    preds, true_labels = data
    decoded_preds = en_kin_tokenizer.batch_decode(preds, skip_special_tokens = True)
    true_labels = np.where(true_labels != -100, true_labels, en_kin_tokenizer.pad_token_id)
    decoded_labels = en_kin_tokenizer.batch_decode(true_labels, skip_special_tokens = True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions = decoded_preds, references = decoded_labels)
    spm_result = metric.compute(predictions = decoded_preds, references = decoded_labels,tokenize='spm')
    chrf_metric = load_metric("chrf")
    chrf_result = chrf_metric.compute(predictions=decoded_preds,references=decoded_labels,word_order=2)
    ter_metric = load_metric("ter")
    ter_result = ter_metric.compute(predictions = decoded_preds,references = decoded_labels)
    result = {"bleu":result["score"],"spbleu":spm_result['score'],'ter':ter_result['score'],'chrf++':chrf_result['score']}
    prediction_lens = [np.count_nonzero(pred != en_kin_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


from transformers import  Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainingArguments, Trainer, logging

# NB: We work with small batch sizes and checkpointing due to the limitations of
# this free instance of Colab. In practice you'd want to use settings closer to
# what we use in the paper.
training_args = Seq2SeqTrainingArguments(
    output_dir="tmp",
    num_train_epochs=1,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
#    gradient_accumulation_steps=4,
#    eval_accumulation_steps=4,
  #  gradient_checkpointing=True,
    save_strategy="steps",
    do_train=True,
    do_eval=True,
    do_predict=True,
    predict_with_generate=True,
#    fp16=True,
    save_steps=1000,
    save_total_limit = 2,
#    fp16_full_eval=True,
    evaluation_strategy="steps"
)

trainer = Seq2SeqTrainer(
    model=model,
    compute_metrics=metrics_calc,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,
)
trainer.train()

print("################################ EN to Kin test########################")
print("----------------------------------------------------------------------\n")
tokenized_test_data_en_kin = test_dataset_en_kin.map(en_kin_tokenize_fn, batched=True)

print(trainer.predict(tokenized_test_data_en_kin))
print("\n\n###############################Kin to En test#######################")
print("------------------------------------------------------------------------\n")

tokenized_test_data_kin_en = test_dataset_kin_en.map(kin_en_tokenize_fn, batched=True)

print(trainer.predict(tokenized_test_data_kin_en))


