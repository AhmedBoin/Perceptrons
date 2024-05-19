# Perceptrons
Tensor deployment with autograd in pure rust.

short story long
an example of python code

```python
import torch

x = torch.tensor([1., 2., 3., 4., 5., 6.]).reshape(6, 1)
y = torch.tensor([3., 5., 7., 9., 11., 13.]).reshape(6, 1)

w = torch.tensor([-0.1], requires_grad=True)
b = torch.tensor([0.], requires_grad=True)

epochs = 1000
lr = 0.01

for _ in range(epochs):
    # forward propagation
    y_hat = x@w + b

    # loss calculation
    loss = 0.5 * ((y_hat - y)**2).mean()
    
    # back propagation
    w.grad.zero_()
    b.grad.zero_()
    loss.sum().backward()

    # optimization
    dw = w.grad
    db = b.grad
    with torch.no_grad():
        w = w - lr * dw
        b = b - lr * db


print(f"w = {w}\nb = {b}")
```


equivalent rust code
```rust
use perceptron::prelude::*;

fn main() {
    let x = tensor![1., 2., 3., 4., 5., 6.].reshape((6, 1));
    let y = tensor![3., 5., 7., 9., 11., 13.].reshape((6, 1));

    let mut w = rand_tensor![1, 1].requires_grad(true);
    let mut b = rand_tensor![1].requires_grad(true);

    let epochs = 1000;
    let lr = 0.01;

    for _ in 0..epochs {
        // forward propagation
        let y_hat = x.dot(&w) + &b;

        // loss calculation
        let loss = 0.5 * (&y_hat - &y).pow(2.0).mean();

        // back propagation
        loss.backward();

        // optimization
        no_grad!({
            w = &w - lr * w.grad().unwrap();
            b = &b - lr * b.grad().unwrap();
        });

        w.zero_grad();
        b.zero_grad();
    }
    println!("w = {w}\n\nb = {b}");
}
```
This code is working in Rust the same as Python. There are many additional features that can be tested.

In conclusion, please use Display instead of Debug if you can. You can see that there is no difference between Python and Rust.

This project may not be completed due to the lack of interest from GitHub's Rust and ML developers. Feel free to leave a comment if you are interested.

