#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.TextEncoder import BertEncoder
from models.BERT_Config import MODELS


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
        joint_embed = config.joint_embed


        self.audio_linear = nn.Sequential(
            nn.Linear(1024, joint_embed * 2),
            nn.ReLU(),
            nn.Linear(joint_embed * 2, joint_embed)
        )

        self.text_enc = BertEncoder(config)
        bert_type = config.bert_encoder.type
        self.text_linear = nn.Sequential(
            nn.Linear(MODELS[bert_type][2], joint_embed * 2),
            nn.ReLU(),
            nn.Linear(joint_embed * 2, joint_embed)
        )

    def encode_text(self, captions):
        return self.text_enc(captions)

    def forward(self, audio_encoded, captions):
        caption_encoded = self.encode_text(captions) # bz x 768 x 1

        caption_embed = self.text_linear(caption_encoded)
                
        # audio_encoded: bz x 1024 x 1
        audio_embed = self.audio_linear(audio_encoded) 

        if self.l2:
            # apply l2-norm on the embeddings
            audio_embed = l2norm(audio_embed)
            caption_embed = l2norm(caption_embed)

        return audio_embed, caption_embed
