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
}

pub type DynArray = ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
