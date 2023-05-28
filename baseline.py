import os
import numpy as np
import pandas as pd
import argparse
import wandb
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp

from tqdm import tqdm

# Kobert
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

# transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# skleran
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# custom libraries
from utils import get_timezone
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# typing
from numpy.typing import ArrayLike
from typing import List

'''
    Define Dataset, Model
'''
class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)

def calc_confusion_matrix(preds: ArrayLike, labels: ArrayLike) -> None:
    '''
        label과 pred의 logits를 받아 confusion matrix를 계산해주는 함수 입니다.
        classes : '정치', '경제', '사회', '생활문화', '세계', 'IT과학', '스포츠'

        args:
            preds(ArrayLike) : model's output (예측 데이터)
            labels(ArrayLike) : target data (정답 데이터)
        return: None
    '''
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    cmn = cmn.astype("int")

    # set pyplot
    fig = plt.figure(figsize=(22, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # set non-normalized confusion matrix
    cm_plot = sns.heatmap(cm, cmap="Blues", fmt="d", annot=True, ax=ax1)
    cm_plot.set_xlabel("pred")
    cm_plot.set_ylabel("true")
    cm_plot.set_title("confusion matrix")

    # set normalized confusion matrix
    cmn_plot = sns.heatmap(cmn, cmap="Blues", fmt="d", annot=True, ax=ax2)
    cmn_plot.set_xlabel("pred")
    cmn_plot.set_ylabel("true")
    cmn_plot.set_title("confusion matrix normalize")
    
    wandb.log({"confusion_matrix": wandb.Image(fig)})

'''
    Define Metric
'''
def compute_metrics(eval_pred):
    f1 = evaluate.load('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    calc_confusion_matrix(predictions, labels)
    wandb.log({"eval_accuracy": accuracy_score(predictions, labels)})
    
    return f1.compute(predictions=predictions, references=labels, average='macro')

def set_seed(seed:int) -> None:
    '''
        seed값을 고정 시키기 위한 함수 입니다.

        args : seed(int) : seed 값
        returns : None
    '''
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    '''
        Set Hyperparameters
    '''
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    OUTPUT_DIR = os.path.join(BASE_DIR, '../output')
    SEED = 456

    set_seed(SEED)  # or any other number

    ## Setting parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5
    
    '''
        Load Tokenizer and Model
    '''
    model_name = 'monologg/kobert'
    tokenizer = KoBertTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
    
    '''
        Define Dataset
    '''
    data = pd.read_csv(os.path.join(DATA_DIR, args.file))
    dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)
    
    data_train = BERTDataset(dataset_train, tokenizer)
    data_valid = BERTDataset(dataset_valid, tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    '''
        Train
    '''
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate= 2e-05,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
        
    '''
        Evaluate Model
    '''
    dataset_eval = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    model.eval()
    preds = []
    for idx, sample in dataset_eval.iterrows():
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    
    '''
        Save output file
    '''
    dataset_eval['target'] = preds
    
    # save output file with checking folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_eval.to_csv(os.path.join(OUTPUT_DIR, 'output.csv'), index=False)
    
if __name__ == "__main__":
    # arguments로 train_data의 파일을 입력했으면 해당 파일을 사용하도록 설정하고 아니면 train.csv를 사용하도록 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="train.csv")
    args = parser.parse_args()
    
    running_name = f"{args.file[:-4]}-{get_timezone()}"

    # Wandb 설정
    wandb.init(project="DataCentric", name = running_name)

    main(args)

    # Wandb 종료
    wandb.finish()