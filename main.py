import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from model import Lstmp
import graphs as gr

N = 100  # number of sign wave
L = 1000 # the number of values for each sign waves
T = 20  # width of waves.

x = np.empty((N,L),np.float32) # randomly picked

x[:] = np.array(range(L)) + np.random.randint(-4*T,4*T,N).reshape(N,1)


y = np.sin(x/1.0/T).astype(np.float32)

# y 100,1000
train_input = torch.from_numpy(y[3:,:-1]) # 97,999

train_label = torch.from_numpy(y[3:,1:]) # 97,999
test_input = torch.from_numpy(y[:3,:-1]) # 3,999
test_label = torch.from_numpy(y[:3,1:]) # 3,999

model = Lstmp()
loss = torch.nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(),lr=0.8,max_iter=20)
n_steps = 10

for i in range(n_steps):
    print("Step",i)
    def closure(): # do 20 iteration at each step
        optimizer.zero_grad() # input shape [97,999]
        out=model(train_input) # out shape [97,999]
        l=loss(out,train_label)
        print('loss',l.item())
        l.backward()
        return l

    optimizer.step(closure)

    with torch.no_grad():
        future_values = 1000
        n_real_values = train_input.shape[1]
        colors = ['red','blue','green']
        out = model(test_input,future=future_values)
        l=loss(out[:,:-future_values],test_label) # exclude future value
        print('test loss',l.item())
        y = out.detach().numpy()

        plt.figure(figsize=(12,6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        for j in range(y.shape[0]): # plot 3 test waves
            gr.plot(i,y[j],colors[j])
        
        plt.savefig("predict%d.pdf"%i)
        plt.close()

         









