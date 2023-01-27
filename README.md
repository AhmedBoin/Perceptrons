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
    loss.backward()

    # optimization
    dw = w.grad
    db = b.grad
    with torch.no_grad():
        w -= lr * dw
        b -= lr * db

print(f"w = {w}\nb = {b}")
```


equivalent rust code
```rust
use perceptron::prelude::*;

fn main() {
    let x = tensor!(array![1., 2., 3., 4., 5., 6.]).reshape((6, 1));
    let y = tensor!(array![3., 5., 7., 9., 11., 13.]).reshape((6, 1));

    let w = tensor!(rand_array![1, 1]).requires_grad(true);
    let b = tensor!(array![[0.0]]).requires_grad(true);

    let epochs = 1000;
    let lr = 0.01;

    for _ in 0..epochs {
        // forward propagation
        let y_hat = x.dot(&w) + &b;

        // loss calculation
        let loss = 0.5 * (&y_hat - &y).pow(2.0).mean();

        // back propagation
        loss.backward();

        // // optimization
        let dw = w.grad().unwrap() * lr;
        let db = b.grad().unwrap() * lr;
        w.optimize_with_dyn_array(dw);
        b.optimize_with_dyn_array(db);

        w.zero_grad();
        b.zero_grad();
    }
    println!("w = {w}\nb = {b}");
}
```
this code is working in rust better than python, there is many additional fature can be test.

at last, please use display instad of debug if you could, you can see there is no difference between python and rust.

maybe this project not be completed due to no interest of github rust and ML developers, leave a comment if you are interest

