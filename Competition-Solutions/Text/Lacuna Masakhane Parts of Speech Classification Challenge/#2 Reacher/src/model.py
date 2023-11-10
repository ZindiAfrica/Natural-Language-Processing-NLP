import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,

)



def odd_layer_freeze(module):
    for i in range(1,24,2):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False

class ResidualLSTM(nn.Module):

    def __init__(self, d_model,rnn):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        if rnn=='GRU':
            self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM=nn.LSTM(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)



class TransformerModel(nn.Module):
    def __init__(self,PRETRAINED_MODEL, rnn='LSTM', num_classes=17):
        super(TransformerModel, self).__init__()
        
        config_model = AutoConfig.from_pretrained(PRETRAINED_MODEL)
        self.backbone=AutoModel.from_pretrained(
                           PRETRAINED_MODEL,config=config_model)

        self.hidden_dim = 1024
        self.lstm= ResidualLSTM(self.hidden_dim,rnn)
        self.bilstm = nn.LSTM(config_model.hidden_size, (config_model.hidden_size) // 2, dropout=config_model.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        self.dropout = nn.Dropout(config_model.hidden_dropout_prob)
        self.classification_head=nn.Linear(self.hidden_dim,num_classes)
        ## Freeze
        odd_layer_freeze(self.backbone)
        

    def forward(self,x,attention_mask):
        x=self.backbone(input_ids=x,attention_mask=attention_mask,return_dict=False)[0]
        x=self.lstm(x.permute(1,0,2)).permute(1,0,2)
        x=self.classification_head(x)
        return x[:,:,:]