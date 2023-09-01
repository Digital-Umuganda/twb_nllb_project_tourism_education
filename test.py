import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import wandb
from datasets.combine import concatenate_datasets
from datasets import load_dataset, load_metric
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p',"--test_data_path")
parser.add_argument("-m","--model_path",default="facebook/nllb-200-distilled-1.3B")

args = parser.parse_args()

test_data_path = args.test_data_path
MODEL_PATH = args.model_path



KIN_LANG_CODE = "kin_Latn"
EN_LANG_CODE = "eng_Latn"

LANG_CODES = ["kin_Latn","eng_Latn"]

#MODEL_PATH = "DigitalUmuganda/finetuned-nllb-1.3B"

print("....Using model: ",MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

test_data = pd.read_csv(test_data_path,sep="\t")


data = {"kin_Latn":list(test_data['phrase']),"eng_Latn":list(test_data['source'])}

metric = load_metric("sacrebleu")
chrf_metric = load_metric("chrf")
ter_metric = load_metric("ter")

for i in range(len(LANG_CODES)):

    print("-----------------------------\n")
    print("direction: ",LANG_CODES[0]," --> ",LANG_CODES[1])
    true_translations = list(data[LANG_CODES[1]])
    true_translations = [[element] for element in true_translations]
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang=LANG_CODES[0], tgt_lang=LANG_CODES[1])
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0,src_lang=LANG_CODES[0], tgt_lang=LANG_CODES[1])
    inputs = [str(element) for element in list(data[LANG_CODES[0]])]
    predicted_translations = translator(inputs)
    predicted_translations = [list(element.values())[0] for element in predicted_translations]
    blue = metric.compute(predictions = predicted_translations, references = true_translations)['score']
    spbleu = metric.compute(predictions = predicted_translations, references = true_translations,tokenize='spm')['score']
    chrf = chrf_metric.compute(predictions=predicted_translations,references=true_translations,word_order=2)['score']
    ter = ter_metric.compute(predictions = predicted_translations,references = true_translations)['score']
    print("\t** BLEU: ",blue)
    print("\t** spBLEU: ",spbleu)
    print("\t** chrf++: ",chrf)
    print("\t** TER: ",ter)
    LANG_CODES.reverse()
