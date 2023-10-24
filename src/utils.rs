use crate::prelude::*;

#[macro_export]
macro_rules! rand_array {
    ($($x:expr),*) => {
        {
            Array::random(($($x,)*), Uniform::new(-0.01, 0.01))
        }
    };
}

#[macro_export]
macro_rules! tensor {
    ($x:expr) => {
        Tensor::new($x)
    };
    ($($x:expr),*) => {
        Tensor::new(array![$($x,)*])
    };
}

#[macro_export]
macro_rules! rand_tensor {
    ($($x:expr),*) => {
        Tensor::new(Array::random(($($x,)*), Uniform::new(-0.01, 0.01)))
    };
}

pub static NO_GRAD: Mutex<bool> = Mutex::new(false);

#[macro_export]
macro_rules! no_grad {
    ($($block:block)*) => {{
        *NO_GRAD.lock().unwrap() = true;
        $($block;)*
        *NO_GRAD.lock().unwrap() = false;
    }};
}

pub type DynArray = ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;
