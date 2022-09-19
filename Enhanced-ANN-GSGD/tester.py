import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy

n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 100, 0.2

data_x = torch.randn(batch_size, n_input)
data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()

print(data_x.size())
print(data_y.size())

model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.Sigmoid())

model2 = copy.deepcopy(model)
print(model)
print('model 2')
print(model2)

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)

losses = []
losses2 = []
for epoch in range(5000):
    pred_y = model(data_x)
    loss = loss_function(pred_y, data_y)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()


for epoch in range(5000):
    pred_y = model2(data_x)
    loss = loss_function(pred_y, data_y)
    losses2.append(loss.item())

    model2.zero_grad()
    loss.backward()

    optimizer2.step()

plt.plot(losses2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()