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
    y_hat = x@w + b
    loss = 0.5 * ((y_hat - y)**2).mean()
    
    w.grad.zero_()
    b.grad.zero_()
    loss.sum().backward()

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
    let x = tensor![1., 2., 3., 4., 5., 6.].reshape((3, 2));
    let y = tensor![3., 7., 11.].reshape((3, 1));

    let mut w = rand_tensor![2, 1].requires_grad(true);
    let mut b = rand_tensor![1].requires_grad(true);

    let epochs = 1000;
    let lr = 0.02;

    for _ in 0..epochs {
        // forward propagation
        let y_hat = x.dot(&w) + &b;

        // loss calculation
        let loss = 0.5 * (&y_hat - &y).pow(2.0).mean();

        // back propagation
        loss.backward();

        // // optimization
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
this code is working in rust better than python, there is many additional fature can be test.

at last, please use display instad of debug if you could, you can see there is no difference between python and rust.

maybe this project not be completed due to no interest of github rust and ML developers, leave a comment if you are interest

