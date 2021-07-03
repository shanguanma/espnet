#!/usr/bin/env python3


from  typing import Tuple
from typing import Optional

from typeguard import check_argument_types

import torch
import torch.nn as nn
import logging
'''
reference: 'CHANNEL-ATTENTION DENSE U-NET FOR MULTICHANNEL SPEECH ENHANCEMENT'
https://arxiv.org/pdf/2001.11542.pdf
'''
class ChannelSelfAttention1(nn.Module):
    def __init__(
        self,
        feats_dim: int = 80,
        dim: int = 40,
        residual: bool = True,
    ):  
        check_argument_types()
        super().__init__()
        self.residual = residual
        self.q = nn.Conv2d(feats_dim, dim, 1, 1)
        self.k = nn.Conv2d(feats_dim, dim, 1, 1)
        self.v = nn.Conv2d(feats_dim, feats_dim, 1, 1)
        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input = torch.stack((x1, x2), dim=3).permute(0,2,1,3) # B x F x T x 2
        q = self.q(input) # B x d x T x 2
        k = self.k(input) # B x d x T x 2
        v = self.v(input) # B x F x T x 2
        #b,d,T,_ = q.size() 
        #q = q.view(b,T,-1,d)       
        #k = k.view(b,T,d, -1)
        
        q = q.permute(0, 2, 3, 1 ) # B x T x 2 x d
        k = k.permute(0, 2, 1, 3 ) # B x T x d x 2
        qk = torch.matmul(q, k) # B x T x 2 x 2
        score = torch.softmax(qk, dim=-1)
        logging.info(f"score shape is {score.shape}")
        v = v.permute(0,2,1,3)
        logging.info(f"v shape is {v.shape}")
        output = torch.matmul(v, score) # B x T x F x 2
        #v = v.permute(0,2,1,3)
        #output = output.permute(1,2)
        output1 = output.unbind(dim=-1)[0] # B x T x F
        output2 = output.unbind(dim=-1)[1] # B x T x F
        if self.residual: 
            output1 = x1 + output1
            output2 = x2 + output2
        else:
            output1 = output1
            output2 = output2
        return output1, output2
            

        









