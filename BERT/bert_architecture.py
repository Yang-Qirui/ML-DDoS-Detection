import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
import os


class BERT_Arch(nn.Module):
    def __init__(self, bert, input_feature_size, tokenizer, device):
        super(BERT_Arch, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.input_feature_size = input_feature_size
        self.fc0 = nn.Linear(input_feature_size, 768)
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)
        # softmax activation function
        self.softmax = nn.Softmax(dim=1)
        

    def embedding_layer(self, features, real_seq_lens, seq_len):
        new_features = []
        special_token_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "[PAD]"])
        embeddings = self.bert.get_input_embeddings()
        special_embeds = embeddings(torch.tensor(special_token_ids).to(self.device))
        cls_embed = special_embeds[0].to(self.device)
        sep_embed = special_embeds[1].to(self.device)
        pad_embed = special_embeds[2].to(self.device)
        
        for i, real_seq_len in enumerate(real_seq_lens):
            if (seq_len == real_seq_len):
                new_feature = torch.concat([cls_embed.view(1, -1), features[i, :real_seq_len, :], sep_embed.view(1, -1)])
            else: 
                pad_embeds = torch.stack([pad_embed for i in range(seq_len - real_seq_len)])
                new_feature = torch.concat([cls_embed.view(1, -1), features[i, :real_seq_len, :], sep_embed.view(1, -1), pad_embeds])
            new_features.append(new_feature)
        
        return torch.stack(new_features)
    # define the forward pass
    # features: (batch_size, seq_len, feature_size)
    def forward(self, features, mask_ids, real_seq_len):
        # pass the inputs to the model
        seq_len = features.shape[1]
        inputs_embeds = self.fc0(features)
        inputs_embeds = self.embedding_layer(inputs_embeds, real_seq_len, seq_len)
        # inputs_embeds = inputs_embeds.view(batch_size, seq_len, -1)
        # inputs_embeds = torch.cat([self.cls_embed, inputs_embeds, self.sep_embed])
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=mask_ids)
        cls_hs = outputs.hidden_states[-1][:, 0, :]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x