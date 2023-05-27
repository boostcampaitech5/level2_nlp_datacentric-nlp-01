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
    def __init__(self, dataset, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        texts = dataset['input_text'].tolist()
        targets = dataset['target'].tolist()

        self.sentences = [transform(i) for i in texts]
        self.labels = [np.int32(i) for i in targets]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)
    
def calc_accuracy(X,Y):
    _, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]       # accuracy
    return train_acc
    
    
def calc_f1_score(preds : ArrayLike, labels : ArrayLike) -> float:
    '''
        label과 pred의 logits를 받아 f1 score를 계산해주는 함수 입니다.

        args:
            preds(ArrayLike) : model's output (예측 데이터)
            labels(ArrayLike) : target data (정답 데이터)
        return: float : f1 score
    '''
    return f1_score(labels, preds, average='weighted')
    
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

def set_seed(seed:int) -> None:
    '''
        seed값을 고정 시키기 위한 함수 입니다.

        args : seed(int) : seed 값
        returns : None
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  
    random.seed(seed)  
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
    SEED = 42

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
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    '''
        Define Dataset
    '''
    data = pd.read_csv(os.path.join(DATA_DIR, args.file))
    dataset_train, dataset_eval = train_test_split(data, train_size=0.7, random_state=SEED)
    
    data_train = BERTDataset(dataset_train, tok, max_len, True, False)
    data_eval = BERTDataset(dataset_eval, tok, max_len, True, False)
    
    train_dataloader = DataLoader(data_train, batch_size=batch_size)
    eval_dataloader = DataLoader(data_eval, batch_size=batch_size)
    
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(DEVICE)
    
    '''
        Define Optimizer and Scheduler
    '''
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
    '''
        Train
    '''
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(DEVICE)
            segment_ids = segment_ids.long().to(DEVICE)
            valid_length= valid_length
            label = label.long().to(DEVICE)
            
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            
            train_acc += calc_accuracy(out, label)
            
            # logging train information to Wandb
            if batch_id % 5 == 0:
                wandb.log({"train_loss" : loss.item(), 'train/epoch': batch_id+1})     # 
                wandb.log({"train_acc": train_acc / (batch_id+1), 'train/epoch': batch_id+1})      # 
            
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        
        model.eval()
        outs : List = []
        labels : List = []

        # Evalutation
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                token_ids = token_ids.long().to(DEVICE)
                segment_ids = segment_ids.long().to(DEVICE)
                valid_length= valid_length
                label = label.long().to(DEVICE)
                out = model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)
                
                ### convert prob to logits for confusion matrix
                _, max_indices = torch.max(out, 1)
                outs.extend(max_indices.detach().cpu().numpy())
                labels.extend(label.detach().cpu().numpy())

            outs = np.array(outs)
            labels = np.array(labels)
            calc_confusion_matrix(outs, labels)

            print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
            
            # logging eval information to Wandb
            wandb.log({"eval_f1": calc_f1_score(outs, labels), 'epoch': e+1})     #
            wandb.log({"eval_acc": test_acc / (batch_id+1), 'epoch': e+1})        # 
        
    '''
        Test
    '''
    dataset_eval = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    dataset_eval['target'] = [0]*len(dataset_eval)
    data_eval = BERTDataset(dataset_eval, tok, max_len, True, False)
    eval_dataloader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)
    
    preds = []
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, _) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        token_ids = token_ids.long().to(DEVICE)
        segment_ids = segment_ids.long().to(DEVICE)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        _, max_indices = torch.max(out, 1)
        preds.extend(list(max_indices))

    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    preds = [int(p) for p in preds]
    
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