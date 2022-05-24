# Transformers_Question-Answering_distillBERT_SQUAD_Finetuned
Transformers Question and Answering Model based on distill BERT model fine tuned on SQUAD question answering dataset deployed using Gradio
*** 
## Download SQUAD Dataset
```Python
from datasets import load_dataset

dataset = load_dataset("squad")
```
## Download Dstill BERT
```Python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
```
or

Download Files from [here](https://huggingface.co/distilbert-base-uncased/tree/main) 

## To FineTune Model Run
``` python
Model_Tuned_SQUAD_QnA.ipynb
```

## Run Final Model
```python
Python Question_Answering.py
```
