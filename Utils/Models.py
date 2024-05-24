#@title Importación de librerías
from transformers import BertModel, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch

#Dispositivo sobre el que se corre el modelo de ML
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentAnalysis(nn.Module):
    def __init__(self, bert_model, dropout_rate, num_classes, neurons_capa_1, neurons_capa_2, eval_bert):
        super(SentimentAnalysis, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model)
        if not eval_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, neurons_capa_1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(neurons_capa_1, neurons_capa_2)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        self.fc_out = nn.Linear(neurons_capa_2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        temp = self.bert_model(input_ids, attention_mask)
        CLS = temp[1]
        out = self.dropout(CLS)

        out2 = self.dropout2(self.fc(out))
        out2 = F.leaky_relu(out2)
        
        out3 = self.dropout3(self.fc2(out2))
        out3 = F.leaky_relu(out3)

        return self.fc_out(out3)
    
class SentimentAnalysisFromMaskedModel(nn.Module):
    def __init__(self, bert_model, dropout_rate, num_classes, neurons_capa_1, neurons_capa_2, eval_bert):
        super(SentimentAnalysisFromMaskedModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model)
        if not eval_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, neurons_capa_1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(neurons_capa_1, neurons_capa_2)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        self.fc_out = nn.Linear(neurons_capa_2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        temp = self.bert_model(input_ids, attention_mask, output_hidden_states = True).hidden_states
        CLS = temp[-1][:,0,:]
        out = self.dropout(CLS)

        out2 = self.dropout2(self.fc(out))
        out2 = F.leaky_relu(out2)
        
        out3 = self.dropout3(self.fc2(out2))
        out3 = F.leaky_relu(out3)

        return self.fc_out(out3)

class SentimentAnalysisPretrainedBert(nn.Module):
    def __init__(self, bert_model, dropout_rate, num_classes, neurons_capa_1, neurons_capa_2, eval_bert):
        super(SentimentAnalysisPretrainedBert, self).__init__()
        self.bert_model = torch.load(
            bert_model,
            map_location=torch.device(DEVICE)
        )
        if not eval_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.bert_model.config.hidden_size, neurons_capa_1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(neurons_capa_1, neurons_capa_2)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        self.fc_out = nn.Linear(neurons_capa_2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):

        temp = self.bert_model(input_ids, attention_mask, output_hidden_states = True).hidden_states
        CLS = temp[-1][:,0,:]
        out = self.dropout(CLS)

        out2 = self.dropout2(self.fc(out))
        out2 = F.leaky_relu(out2)
        
        out3 = self.dropout3(self.fc2(out2))
        out3 = F.leaky_relu(out3)

        return self.fc_out(out3)