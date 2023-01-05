import torch

x = torch.tensor([1., 2., 3., 4., 5., 6.]).reshape(6, 1)
y = torch.tensor([3., 5., 7., 9., 11., 13.]).reshape(6, 1)

w = torch.tensor([-0.1], requires_grad=True)
b = torch.tensor([0.], requires_grad=True)

epochs = 1000
lr = 0.01

for _ in range(epochs):
    y_hat = x@w + b
    loss = 0.5 * ((y_hat - y)**2).mean()
    loss.sum().backward()

    dw = w.grad
    db = b.grad
    with torch.no_grad():
        w -= lr * dw
        b -= lr * db

print(f"w = {w}\nb = {b}")
