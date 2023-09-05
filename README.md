# twb_nllb_finetuning

---
## Datasets:
- [mbazaNLP/NMT_Tourism_parallel_data_en_kin](https://huggingface.co/datasets/mbazaNLP/NMT_Tourism_parallel_data_en_kin)
- [mbazaNLP/NMT_Education_parallel_data_en_kin](https://huggingface.co/datasets/mbazaNLP/NMT_Education_parallel_data_en_kin)
- [mbazaNLP/Kinyarwanda_English_parallel_dataset](https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset)
#### Languages:
- en
- rw
#### library_name: 
- transformers
---
## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This is a Machine Translation model, finetuned from [NLLB](https://huggingface.co/facebook/nllb-200-distilled-1.3B)-200's distilled 1.3B model, it is meant to be used in machine translation for education-related data.



- **Finetuning code repository:** the code used to finetune this model can be found [here](https://github.com/Digital-Umuganda/twb_nllb_finetuning)


<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->


## How to Get Started with the Model

Use the code below to get started with the model.


### Training Procedure 

The model was finetuned on three datasets; a [general](https://huggingface.co/datasets/mbazaNLP/Kinyarwanda_English_parallel_dataset) purpose dataset, a [tourism](https://huggingface.co/datasets/mbazaNLP/NMT_Tourism_parallel_data_en_kin), and an [education](https://huggingface.co/datasets/mbazaNLP/NMT_Education_parallel_data_en_kin) dataset.

The model was finetuned in two phases.

#### Phase one:
- General purpose dataset
- Education dataset
- Tourism dataset

#### Phase two:
- Education dataset or Tourism dataset (Depending on the downstream task)

Other than the dataset changes between phase one, and phase two finetuning; no other hyperparameters were modified. In both cases, the model was trained on an A100 40GB GPU for two epochs.


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->


#### Testing Data

<!-- This should link to a Data Card if possible. -->


#### Metrics

Model performance was measured using BLEU, spBLEU, TER, and chrF++ metrics.

### Results
#### Tourism metrics:

|Lang. Direction| BLEU |  spBLEU  | chrf++ |TER   |
|:----|:----:|:----:|:----:|----:|
| Eng -> Kin     | 28.37   | 40.62 | 56.48 |   59.71   |
| Kin -> Eng     | 42.54   |  44.84  |   61.54 |  43.87     |


#### Education metrics:

|Lang. Direction| BLEU |  spBLEU  | chrf++ |TER   |
|:----|:----:|:----:|:----:|----:|
| Eng -> Kin     | 45.96   | 59.20 | 68.79 |   41.61    |
| Kin -> Eng     | 43.98   |  44.94  |   63.05 |  41.41     |

<!-- [More Information Needed] -->




