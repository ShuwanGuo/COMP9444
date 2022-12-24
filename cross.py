"""
   cross.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.in_to_hid = torch.nn.Linear(2, hid)
        self.hid_to_hid = torch.nn.Linear(hid, hid)
        self.hid_to_out = torch.nn.Linear(hid, 1)

    def forward(self, input):
        hid1_sum = self.in_to_hid(input)
        self.hid1 = torch.tanh(hid1_sum)
        hid2_sum = self.hid_to_hid(self.hid1)
        self.hid2 = torch.tanh(hid2_sum)
        output_sum = self.hid_to_out(self.hid2)
        output = torch.sigmoid(output_sum)
        return output 

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.in_to_hid = torch.nn.Linear(2, hid)
        self.hid_to_hid_1 = torch.nn.Linear(hid, hid)
        self.hid_to_hid_2 = torch.nn.Linear(hid, hid)
        self.hid_to_out = torch.nn.Linear(hid, 1)
 
    def forward(self, input):
        hid1_sum = self.in_to_hid(input)
        self.hid1 = torch.tanh(hid1_sum)
        hid2_sum = self.hid_to_hid_1(self.hid1)
        self.hid2 = torch.tanh(hid2_sum)
        hid3_sum = self.hid_to_hid_2(self.hid2)
        self.hid3 = torch.tanh(hid3_sum)
        output_sum = self.hid_to_out(self.hid3)
        output = torch.sigmoid(output_sum)
        return output 

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.in_to_hid = torch.nn.Linear(2, num_hid)

        self.in_to_hid2 = torch.nn.Linear(2, num_hid, bias=False)
        self.hid1_to_hid2 = torch.nn.Linear(num_hid, num_hid)

        self.in_to_out = torch.nn.Linear(2, 1, bias=False)
        self.hid1_to_out = torch.nn.Linear(num_hid, 1, bias=False)
        self.hid2_to_out = torch.nn.Linear(num_hid, 1)

    def forward(self, input):
        hid1_sum = self.in_to_hid(input)
        self.hid1 = torch.tanh(hid1_sum)
        hid2_sum_1 = self.in_to_hid2(input)
        hid2_sum_2 = self.hid1_to_hid2(self.hid1)
        hid2_sum = hid2_sum_1+hid2_sum_2
        self.hid2 = torch.tanh(hid2_sum)

        out_sum_1 = self.in_to_out(input)
        out_sum_2 = self.hid1_to_out(self.hid1)
        out_sum_3 = self.hid2_to_out(self.hid2)
        out_sum = out_sum_1+out_sum_2+out_sum_3
        output = torch.sigmoid(out_sum)
        return output 
