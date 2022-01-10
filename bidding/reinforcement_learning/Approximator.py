import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def load_data(train_dir, batch_n, b_sample_size, b_bound, dim):
    NB = []
    Dnb = []
    with open(train_dir, 'r') as fin:
        for n in batch_n:
            line = fin.readline()
            line = line[:len(line) - 1].split("\t")
            line = line[1:]
            b_list = [i for i in range(b_bound, len(line))]
            np.random.shuffle(b_list)
            if b_sample_size > 0:
                b_list = b_list[:b_sample_size]
                for b in b_list:
                    nb = [n, b]
                    dnb = float(line[b])
                    NB.append(nb)   
                    Dnb.append([dnb])
    NB = np.array(NB)
    Dnb = np.array(Dnb)
    return NB, Dnb#x_vecs, labels 

class ApproxNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ApproxNet, self).__init__()
        self.layer1 = nn.linear(n_input, n_hidden)
        self.layer2 = nn.linear(n_input, n_hidden)
        self.predict = nn.linear(n_input, n_output)

    def forward(self, input):
        out = self.layer1(input)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out
    
    def loss(self, x, y):
        l = nn.MSELoss()
        return l(x, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='approx framework')
    parser.add_argument("--input", help="input data")
    args = parser.parse_args()
    datafile = args.input
    dataset = pd.read_csv(datafile)
    model = ApproxNet(4, 4, 4)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for input, target in dataset:
        optimizer.zero_grad()
        output = model.forward(input)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()    
        