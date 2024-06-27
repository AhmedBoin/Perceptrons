pub mod prelude;

pub mod functional;
pub mod nn;
pub mod optimizers;
pub mod tensors;
pub mod utils;

#[allow(unused_imports)]
use perceptrons_macros::mask_tensor;

#[allow(unused_imports)]
use crate::prelude::*;

#[test]
fn test1() {
    let x = tensor!(array![1., 2., 3., 4., 5., 6.]).reshape((6, 1));
    let y = tensor!(array![3., 5., 7., 9., 11., 13.]).reshape((6, 1));
    println!("{}", mask_tensor!(x != y) * &x);

    let w = tensor!(rand_array![1, 1]).requires_grad(true);
    let b = tensor!(rand_array![1]).requires_grad(true);

    let epochs = 1000;
    let lr = 0.1;

    for _ in 0..epochs {
        // forward propagation
        let y_hat = x.dot(&w) + &b;

        // loss calculation
        let loss = 0.5 * (&y_hat - &y).pow(2.0).mean();

        // back propagation
        loss.backward();

        // // optimization
        let dw = w.grad_tensor().unwrap() * lr;
        let db = b.grad_tensor().unwrap() * lr;
        w.optimize(dw);
        b.optimize(db);

        w.zero_grad();
        b.zero_grad();
    }
    println!("w = {w}\n\nb = {b}");
}

#[test]
fn test2() {
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
            w = &w - lr * w.grad_tensor().unwrap();
            b = &b - lr * b.grad_tensor().unwrap();
        });

        w.zero_grad();
        b.zero_grad();
    }
    println!("w = {w}\n\nb = {b}");
}
