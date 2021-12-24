import torch
import torch.nn as nn


import torch.nn as nn

from .transformer import TransformerEncoderBlock, get_padding_mask
from .blocks import BERTInputBlock
from .tasks import MaskedLanguageModelTask, NextSentencePredictionTask




class NeuralNet(nn.Module):
    def __init__(self, input_size, voc_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        self.bert = BERTModel(voc_size)
        self.BERTLanguageModel = BERTLanguageModel(self.bert, voc_size)

        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = x
        #out, _ = self.BERTLanguageModel(x)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        print(out.shape)
        out = torch.reshape(out, (-1,12))
        # no activation and no softmax at the end
        return out


class FineTuneModel(nn.Module):

    def __init__(self, pretrained_model, hidden_size, num_classes):
        super(FineTuneModel, self).__init__()

        self.pretrained_model = pretrained_model

        new_classification_layer = nn.Linear(hidden_size, num_classes)
        self.pretrained_model.classification_layer = new_classification_layer

    def forward(self, inputs):
        sequence, segment = inputs
        token_predictions, classification_outputs = self.pretrained_model((sequence, segment))
        return classification_outputs


class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, max_seq=512, n_layers=12, n_heads=12, dropout=0.1, pad_idx=0,
                 pos_pad=True):
        super().__init__()
        self._info = {'vocab_size': vocab_size,
                      'hidden_dim': hidden_dim,
                      'ff_hidden_dim': hidden_dim * 4,
                      'max_seq': max_seq,
                      'n_layers': n_layers,
                      'n_atttention_heads': n_heads,
                      'dropout': dropout,
                      'padding_idx': pad_idx}

        self.input = BERTInputBlock(vocab_size, hidden_dim, max_seq, dropout, pad_idx, pos_pad)
        self.encoder = TransformerEncoderBlock(n_layers, hidden_dim, hidden_dim * 4, n_heads, 'gelu', dropout)

    def forward(self, x_input, x_segment=None):
        if x_segment is not None:
            assert x_input.shape == x_segment.shape, "input and segment must have same dimension"

        x_mask = get_padding_mask(x_input)
        x_emb = self.input(x_input.type(torch.LongTensor), x_segment)
        output = self.encoder(x_emb, x_mask)

        return output

    def get_info(self, key=None):
        if key:
            return self._info.get(key)
        return self._info


class BERTMaskedLanguageModel(nn.Module):
    def __init__(self, bert: BERTModel, vocab_size, hidden_dim = 768):
        super().__init__()
        self.bert = bert
        self.mlm = MaskedLanguageModelTask(vocab_size=vocab_size, hidden_dim=hidden_dim)

    def forward(self, x, x_segment=None):
        x = self.bert(x, x_segment)
        return self.mlm(x)


class BERTLanguageModel(nn.Module):
    def __init__(self, bert: BERTModel, vocab_size, hidden_dim = 768):
        super().__init__()
        self.bert = bert
        self.mlm = MaskedLanguageModelTask(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.nsp = NextSentencePredictionTask(hidden_dim)

    def forward(self, x, x_segment=None):
        x = self.bert(x, x_segment)
        return self.mlm(x), self.nsp(x),
