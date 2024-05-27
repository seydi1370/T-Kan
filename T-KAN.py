# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:42:42 2024

@author: Seyd Teymoor Seydi,

Email: seydi.7021@gmail.com

 This is inspired by Kolmogorov-Arnold Networks but using trigonometric polynomials instead of splines coefficients
"""

import torch
import torch.nn as nn


class TrigKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(TrigKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree
        self.trig_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1, 2))
        nn.init.normal_(self.trig_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # View and repeat input degree + 1 times
        x = x.view((-1, self.input_dim, 1)).expand(-1, -1, self.degree + 1)
        # shape = (batch_size, input_dim, self.degree + 1)

        # Compute sine and cosine terms
        x_sin = torch.sin(x * self.arange)
        x_cos = torch.cos(x * self.arange)

        # Concatenate sine and cosine terms
        x_trig = torch.stack((x_sin, x_cos), dim=-1)
        # shape = (batch_size, input_dim, self.degree + 1, 2)

        # Compute the trigonometric interpolation
        y = torch.einsum("bidk,iodk->bo", x_trig, self.trig_coeffs)
        # shape = (batch_size, out_dim)

        y = y.view(-1, self.out_dim)
        return y

class T_Kan(nn.Module):
    def __init__(self):
        super(T_Kan, self).__init__()
        self.trigkan1 = TrigKANLayer(784, 32, 3)
        self.ln1 = nn.LayerNorm(32)
        self.trigkan2 = TrigKANLayer(32, 16,3)
        self.ln2 = nn.LayerNorm(16)
        self.trigkan3 = TrigKANLayer(16, 10, 3)

    def forward(self, x):
        x = self.trigkan1(x)
        x = self.ln1(x)
        x = self.trigkan2(x)
        x = self.ln2(x)
        x = self.trigkan3(x)
        return x
