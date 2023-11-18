import os, sys, time
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from torchmetrics import Accuracy
import torch
from torch import nn
from copy import deepcopy
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
from bert_architecture import BERT_Arch
import argparse
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Subset, SequentialSampler
from sklearn.model_selection import StratifiedShuffleSplit
def printlog(info):
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")


class StepRunner:
    def __init__(self, net, loss_fn, stage="train", metrics_dict=None, optimizer=None):
        self.net, self.loss_fn, self.metrics_dict, self.stage = (
            net,
            loss_fn,
            metrics_dict,
            stage,
        )
        self.optimizer = optimizer

    def step(self, features, mask, real_seq_len, labels):
        # loss
        preds = self.net(features, mask, real_seq_len)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(preds, labels).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics

    def train_step(self, features, mask, real_seq_len, labels):
        self.net.train()  # 训练模式, dropout层发生作用
        return self.step(features, mask, real_seq_len, labels)

    @torch.no_grad()
    def eval_step(self, features, mask, real_seq_len, labels):
        self.net.eval()  # 预测模式, dropout层不发生作用
        return self.step(features, mask, real_seq_len, labels)

    def __call__(self, features, mask, real_seq_len, labels):
        if self.stage == "train":
            return self.train_step(features, mask, real_seq_len, labels)
        else:
            return self.eval_step(features, mask, real_seq_len, labels)


class EpochRunner:
    def __init__(self, steprunner, device):
        self.steprunner = steprunner
        self.device = device
        self.stage = steprunner.stage

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        for i, batch in loop:
            batch = [r.to(device) for r in batch]
            loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {
                    self.stage + "_" + name: metric_fn.compute().item()
                    for name, metric_fn in self.steprunner.metrics_dict.items()
                }
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(
    net,
    optimizer,
    loss_fn,
    metrics_dict,
    train_data,
    device,
    val_data=None,
    epochs=10,
    ckpt_path="./BERT/checkpoint.pt",
    patience=5,
    monitor="val_loss",
    mode="min",
    
):
    history = {}

    for epoch in range(1, epochs + 1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(
            net=net,
            stage="train",
            loss_fn=loss_fn,
            metrics_dict=deepcopy(metrics_dict),
            optimizer=optimizer,
        )
        train_epoch_runner = EpochRunner(train_step_runner, device)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(
                net=net,
                stage="val",
                loss_fn=loss_fn,
                metrics_dict=deepcopy(metrics_dict),
            )
            val_epoch_runner = EpochRunner(val_step_runner, device)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = (
            np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        )
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print(
                "<<<<<< reach best {0} : {1} >>>>>>".format(
                    monitor, arr_scores[best_score_idx]
                )
            )
        if len(arr_scores) - best_score_idx > patience:
            print(
                "<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                    monitor, patience
                )
            )
            break
    net.load_state_dict(torch.load(ckpt_path))

    return pd.DataFrame(history)

# text data
def get_data_loader(batch_size, data):
    
    no_padding_data_seq_len = []
    mask_ids = []
    labels = []
    ori_flow_seq_len = data.shape[1] + 2 # [CLS] + seq len + [SEP]
    for flow in data:
        flow = flow[~(np.all(flow == 0, axis = 1))]
        seq_len = flow.shape[0]
        no_padding_data_seq_len.append(seq_len)
        
        # [CLS]  ... [SEP] [PAD] in total the len is 18
        mask_id = np.concatenate((np.ones(seq_len + 2), np.zeros(ori_flow_seq_len - seq_len - 2)))
        mask_ids.append(mask_id)
        if flow[0, -1] == 1:
            labels.append([0, 1])
        else:
            labels.append([1, 0])
            
    # Convert Integer Sequences to Tensors

    # for train set
    train_seq = torch.tensor(data[:, :, :2]).float()
    train_mask = torch.tensor(mask_ids)
    train_no_padding_data_seq_len = torch.tensor(no_padding_data_seq_len)
    train_y = torch.tensor(labels).float()

    # Create DataLoaders
    # wrap tensors
    dataset = TensorDataset(train_seq, train_mask, train_no_padding_data_seq_len, train_y)
    
    # split dataset 
    # Define the proportions for each split (e.g., 70% training, 20% validation, 10% test)
    train_prop = 0.7
    val_prop = 0.2
    test_prop = 0.1

    # Create a StratifiedShuffleSplit instance
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_prop, random_state=42)

    # Split the dataset into training and test sets
    train_val_idx, test_idx = next(sss.split(train_seq, train_y))
    
    print(len(test_idx))
    # Further split the train_val set into training and validation sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_prop / (train_prop + val_prop), random_state=42)
    train_idx, val_idx = next(sss.split(train_seq[train_val_idx], train_y[train_val_idx]))
    print(len(train_idx))
    # Create the final datasets using the indices
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)


    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_dataset)
    # dataLoader for train set
    train_dataloader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=batch_size)
    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_dataset)
    # dataLoader for train set
    val_dataloader = DataLoader(val_dataset,
                                sampler=val_sampler,
                                batch_size=batch_size)
    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_dataset)
    # dataLoader for train set
    test_dataloader = DataLoader(test_dataset,
                                sampler=test_sampler,
                                batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


# 使用方法如下：from torchmetrics import Accuracy
if __name__=="__main__":
    torch.cuda.is_available()
    torch.cuda.manual_seed_all(3407)
    torch.manual_seed(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', default='32', type=int)
    parser.add_argument('-lr', '--learning_rate', default='2e-4', type=float)
    parser.add_argument('-dc', '--decay', default='0.00001', type=float)
    parser.add_argument('-ep', '--epochs', default='50', type=int)
    args = parser.parse_args()
    #define a batch size, learning rate and weight decay
    batch_size = args.batch_size
    lr = args.learning_rate
    decay = args.decay
    # number of training epochs
    epochs = args.epochs
    
    # load data
    
    # load model
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # specify GPU
    device = torch.device("cuda")
    model_name = 'bert-base-uncased'
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    data = np.load("./dataset/ddos_sampled_dataset.npy")
    train_dataloader, val_dataloader, test_dataloader = get_data_loader(batch_size, data)
    # import BERT-base pretrained model
    config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    bert = BertForMaskedLM.from_pretrained(model_name, config=config)
    model = BERT_Arch(bert, 2, tokenizer, device)
    model = model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=decay)
    metrics_dict = {"acc": Accuracy("binary").to(device)}

    for param in bert.parameters():
        param.requires_grad = False
    
    
    
    
    dfhistory = train_model(
        model,
        optimizer,
        loss_fn,
        metrics_dict,
        device=device,
        train_data=train_dataloader,
        val_data=val_dataloader,
        epochs=10,
        patience=5,
        monitor="val_acc",
        mode="max",
    )
