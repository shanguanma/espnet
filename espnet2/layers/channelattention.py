#!/usr/bin/env python3


from  typing import Tuple
from typing import Optional

from typeguard import check_argument_types
import logging
import torch
import torch.nn as nn

'''
reference: 'CHANNEL-ATTENTION DENSE U-NET FOR MULTICHANNEL SPEECH ENHANCEMENT'
https://arxiv.org/pdf/2001.11542.pdf
'''
class ChannelSelfAttention(nn.Module):
    def __init__(
        self,
        num_channels: int = 2,
        residual: bool = True,
    ):  
        check_argument_types()
        super().__init__()
        self.q = nn.Conv2d(num_channels, num_channels, 1, 1)
        self.k = nn.Conv2d(num_channels, num_channels, 1, 1)
        self.v = nn.Conv2d(num_channels, num_channels, 1, 1)
        self.residual = residual        
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input = torch.stack((x1, x2), dim=1) # B x 2 x T x F
        logging.info(f"input shape is {input.shape}")
        q = self.q(input).permute(0,2,3,1) # B x T x F x 2 
        logging.info(f"q shape is {q.shape}")
        k = self.k(input).permute(0,2,1,3) # B x T x 2 x F
        v = self.v(input).permute(0,2,3,1) # B x T x F x 2

        qk = torch.matmul(q, k) # B x T x F x F
        score = torch.softmax(qk, dim=-1)
        output = torch.matmul(score, v) # B x T x F x 2
        output1 = output.unbind(dim=-1)[0] # B x T x F
        output2 = output.unbind(dim=-1)[1] # B x T x F
        if self.residual: 
            output1 = x1 + output1
            output2 = x2 + output2
        else:
            output1 = output1
            output2 = output2
        return output1, output2
            

        









