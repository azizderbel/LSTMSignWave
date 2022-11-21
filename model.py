import torch
import torch.nn as nn

class Lstmp(torch.nn.Module):
    def __init__(self,n_hidden=50) -> None:
        super(Lstmp,self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(input_size=1,hidden_size=n_hidden)
        self.lstm2 = nn.LSTMCell(input_size=n_hidden,hidden_size=n_hidden)
        self.linear = nn.Linear(n_hidden,1) # one value is predicted at a time

    def forward(self,x,future=0):
        outputs = []
        # x shape [97,999]
        n_samples = x.shape[0] # 97
        h01 = torch.zeros(n_samples,self.n_hidden,dtype=torch.float32)
        cs1 = torch.zeros(n_samples,self .n_hidden,dtype=torch.float32)
        h02 = torch.zeros(n_samples,self.n_hidden,dtype=torch.float32)
        cs2 = torch.zeros(n_samples,self .n_hidden,dtype=torch.float32)
        
        
        # predict the next value for each sample
        for input in x.split(1,dim=1): # get one simple
            # input shape [97,1]
            h01,cs1 = self.lstm1(input,(h01,cs1))
            h02,cs2 = self.lstm2(h01,(h02,cs2))
            # h02.shape [97,50]
            outputs.append(self.linear(h02))
            # output length = 999
        # predict the x future value for each sample
        if future > 0:
            for i in range(future):
                h01,cs1 = self.lstm1(outputs[-1],(h01,cs1))
                h02,cs2 = self.lstm2(h01,(h02,cs2))
                outputs.append(self.linear(h02))
        outputs = torch.cat(outputs,dim=1)
        return outputs

        



    


