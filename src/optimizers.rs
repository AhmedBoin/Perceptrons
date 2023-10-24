#[allow(unused)]
use crate::prelude::*;

pub trait TensorOptimize {
    fn optimize(&self, grad: Tensor);
}

pub trait ModelOptimize {
    fn optimize(&self);
}
