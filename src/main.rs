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
    }
    println!("w = {w}\nb = {b}");
}
