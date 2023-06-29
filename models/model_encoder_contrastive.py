#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
<<<<<<< Updated upstream
from TextEncoder import BertEncoder
from BERT_Config import MODELS
=======
from TextEncoder import BertEncoder
from BERT_Config import MODELS
from contrastive_loss import NTXent
>>>>>>> Stashed changes


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class ASE(nn.Module):

    def __init__(self, config):
        super(ASE, self).__init__()

        self.l2 = config.training.l2_norm
        self.in_dim_audio = config.in_dim_audio
        self.joint_embed = config.joint_embed


        self.audio_linear = nn.Sequential(
            nn.Linear(self.in_dim_audio, self.joint_embed * 2),
            nn.ReLU(),
            nn.Linear(self.joint_embed * 2, self.joint_embed)
        )

        self.text_enc = BertEncoder(config)
        bert_type = config.bert_encoder.type
        self.text_linear = nn.Sequential(
            nn.Linear(MODELS[bert_type][2], self.joint_embed * 2),
            nn.ReLU(),
            nn.Linear(self.joint_embed * 2, self.joint_embed)
        )

    def encode_text(self, captions):
        return self.text_enc(captions)

    def forward(self, audio_encoded, captions, mels_id):
        contrastive_loss = NTXent()
        
        caption_encoded = self.encode_text(captions) # bz x 768 x 1

        caption_embed = self.text_linear(caption_encoded)
                
        # audio_encoded: bz x 256 x 1
        audio_embed = self.audio_linear(audio_encoded) 

        if self.l2:
            # apply l2-norm on the embeddings
            audio_embed = l2norm(audio_embed)
            caption_embed = l2norm(caption_embed)

        return contrastive_loss(audio_embed, caption_embed, mels_id), audio_embed, caption_embed
