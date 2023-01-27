use crate::prelude::*;

#[derive(Debug, Clone)]
pub struct TensorBase {
    data: DynArray,
    requires_grade: bool,
    depends_on: Option<GradFn>,
    grad: Option<DynArray>,
}

#[rustfmt::skip]
impl TensorBase {
    pub fn backward(&mut self, back: DynArray) {
        if self.requires_grade {
            self.grad = match self.grad.clone() {
                Some(data) => Some(data + back.clone()),
                None => Some(back.clone()), 
            }
        }

        self.depends_on = match self.depends_on.clone() {
            Some(mut data) => {
                data.backward(back);
                None
            },
            None => None,
        }
    }
}

#[derive(Clone)]
pub struct Tensor(Arc<Mutex<TensorBase>>);

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prev = self.0.lock().unwrap();
        let data = prev.data.clone();
        let req = prev.requires_grade.clone();
        match prev.depends_on.as_ref() {
            Some(dep) => write!(f, "Tensor({:?}, grad_fn=<{:?}>)", data, dep),
            None => {
                if req {
                    write!(f, "Tensor({:?}, requires_grad={:?})", data, req)
                } else {
                    write!(f, "Tensor({:?})", data)
                }
            }
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let prev = self.0.lock().unwrap();
        let data = prev.data.clone();
        let req = prev.requires_grade.clone();
        match prev.depends_on.as_ref() {
            Some(dep) => write!(f, "Tensor({}, grad_fn=<{}>)", data, dep),
            None => {
                if req {
                    write!(f, "Tensor({}, requires_grad={})", data, req)
                } else {
                    write!(f, "Tensor({})", data)
                }
            }
        }
    }
}

pub trait ToDynArray {
    fn into_dyn(self) -> DynArray;
}

macro_rules! to_dyn_array {
    ($($ty:ty),+) => {$(
        impl ToDynArray for $ty {
            fn into_dyn(self) -> DynArray {
                self.into_dyn()
            }
        }
    )*};
}

to_dyn_array!(
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 0]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 4]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 5]>>,
    ArrayBase<OwnedRepr<f64>, Dim<[usize; 6]>>
);

impl ToDynArray for Vec<f64> {
    fn into_dyn(self) -> DynArray {
        Array::from_vec(self).into_dyn()
    }
}

impl ToDynArray for &[f64] {
    fn into_dyn(self) -> DynArray {
        Array::from_vec(self.iter().map(|&f| f).collect::<Vec<f64>>()).into_dyn()
    }
}

impl Tensor {
    pub fn new(data: impl ToDynArray) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: data.into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn from_dyn(data: DynArray) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: data,
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn from_op(data: DynArray, depends_on: GradFn) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: data,
            requires_grade: false,
            depends_on: Some(depends_on),
            grad: None,
        })))
    }

    pub fn eye(n: usize) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: Array::<f64, _>::eye(n).into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: Array::<f64, _>::ones(shape).into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: Array::<f64, _>::zeros(shape).into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn ones_like(data: &Tensor) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: Array::<f64, _>::ones(data.0.lock().unwrap().data.shape()),
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn zeros_like(data: &Tensor) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: Array::<f64, _>::zeros(data.0.lock().unwrap().data.shape()),
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn requires_grad(self, boolean: bool) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            requires_grade: boolean,
            ..self.0.lock().unwrap().to_owned()
        })))
    }

    pub fn set_grad(&self, boolean: bool) {
        self.0.lock().unwrap().requires_grade = boolean;
    }

    pub fn detach(&self) -> Self {
        Self(Arc::new(Mutex::new(TensorBase {
            data: self.0.lock().unwrap().to_owned().data,
            requires_grade: false,
            depends_on: None,
            grad: None,
        })))
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.lock().unwrap().data.shape().into()
    }

    pub fn grad(&self) -> Option<DynArray> {
        self.0.lock().unwrap().grad.clone()
    }

    pub fn grad_tensor(&self) -> Option<Tensor> {
        if self.0.lock().unwrap().grad.is_some() {
            Some(Self(Arc::new(Mutex::new(TensorBase {
                data: self.0.lock().unwrap().grad.clone().unwrap(),
                requires_grade: false,
                depends_on: None,
                grad: None,
            }))))
        } else {
            None
        }
    }

    pub fn zero_grad(&self) {
        self.0.lock().unwrap().grad = None;
    }

    pub fn backward(&self) {
        assert!(
            self.0.lock().unwrap().depends_on.is_some(),
            "tensor doesn't have a grad_fn"
        );

        let shape: Vec<usize> = self.0.lock().unwrap().data.shape().into();
        self.0.lock().unwrap().backward(Array::ones(shape));
    }

    pub fn backward_with_array(&self, array: impl ToDynArray) {
        assert!(
            self.0.lock().unwrap().depends_on.is_some(),
            "tensor doesn't have a grad_fn"
        );

        self.0.lock().unwrap().backward(array.into_dyn());
    }

    pub fn backward_with_tensor(&self, tensor: Tensor) {
        assert!(
            self.0.lock().unwrap().depends_on.is_some(),
            "tensor doesn't have a grad_fn"
        );

        self.0
            .lock()
            .unwrap()
            .backward(tensor.0.lock().unwrap().to_owned().data);
    }

    pub fn optimize_with_dyn_array(&self, grad: DynArray) {
        let s = self.0.lock().unwrap().to_owned().data;
        self.0.lock().unwrap().data = s - grad;
    }

    pub fn optimize_with_array(&self, grad: impl ToDynArray) {
        let s = self.0.lock().unwrap().to_owned().data;
        self.0.lock().unwrap().data = s - grad.into_dyn();
    }

    pub fn optimize_with_tensor(&self, grad: Tensor) {
        let s = self.0.lock().unwrap().to_owned().data;
        let grad = grad.0.lock().unwrap().to_owned().data;
        self.0.lock().unwrap().data = s - grad;
    }

    #[rustfmt::skip]
    pub fn reshape(&self, shape: impl ShapeArg) -> Self {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let d = self.0.lock().unwrap().data.clone();
        let data = match d.ndim() {
            0 => d.into_dimensionality::<Ix0>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            1 => d.into_dimensionality::<Ix1>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            2 => d.into_dimensionality::<Ix2>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            3 => d.into_dimensionality::<Ix3>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            4 => d.into_dimensionality::<Ix4>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            5 => d.into_dimensionality::<Ix5>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            6 => d.into_dimensionality::<Ix6>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            _ => panic!(),
        };
        Self(Arc::new(Mutex::new(TensorBase {
            data: data,
            requires_grade: false,
            depends_on: if req || dep {
                Some(GradFn::Reshape(self.0.clone()))
            } else {
                None
            },
            grad: None,
        })))
    }

    pub fn sum(&self) -> Self {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        Self(Arc::new(Mutex::new(TensorBase {
            data: arr0(self.0.lock().unwrap().data.sum()).into_dyn(),
            requires_grade: false,
            depends_on: if req || dep {
                Some(GradFn::Sum(self.0.clone()))
            } else {
                None
            },
            grad: None,
        })))
    }

    pub fn mean(&self) -> Self {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        Self(Arc::new(Mutex::new(TensorBase {
            data: arr0(self.0.lock().unwrap().data.mean().unwrap() as f64).into_dyn(),
            requires_grade: false,
            depends_on: if req || dep {
                Some(GradFn::Mean(self.0.clone()))
            } else {
                None
            },
            grad: None,
        })))
    }

    pub fn pow(&self, num: f64) -> Self {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        Self(Arc::new(Mutex::new(TensorBase {
            data: self.0.lock().unwrap().data.mapv(|f| f.powf(num)),
            requires_grade: false,
            depends_on: if req || dep {
                Some(GradFn::Pow(num, self.0.clone()))
            } else {
                None
            },
            grad: None,
        })))
    }

    pub fn dot(&self, rhs: &Tensor) -> Self {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        Self(Arc::new(Mutex::new(TensorBase {
            data: (self
                .0
                .lock()
                .unwrap()
                .data
                .dot(rhs.0.lock().unwrap().data.clone()))
            .into_dyn(),
            requires_grade: false,
            depends_on: if req || dep {
                Some(GradFn::Dot(self.0.clone(), rhs.0.clone()))
            } else {
                None
            },
            grad: None,
        })))
    }
}

#[rustfmt::skip]
pub trait Dot {
    fn dot(&self, rhs: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>) -> ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
    fn dot_rev(&self, rhs: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>, place: bool) -> ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>;
}

#[rustfmt::skip]
impl Dot for ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> {
    fn dot(&self, rhs: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>) -> ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> {
        let shape0 = self.shape();
        let shape1 = rhs.shape();
        let lhs = self.clone();
        match (self.ndim(), rhs.ndim()) {
            (1, 1) => arr0(lhs.into_dimensionality::<Ix1>().unwrap().dot(&rhs.into_dimensionality::<Ix1>().unwrap())).into_dyn(),
            (2, 1) => (lhs.into_dimensionality::<Ix2>().unwrap().dot(&rhs.into_dimensionality::<Ix1>().unwrap()).insert_axis(Axis(1))).into_dyn(),
            (1, 2) => (lhs.into_dimensionality::<Ix1>().unwrap().insert_axis(Axis(0)).dot(&rhs.into_dimensionality::<Ix2>().unwrap())).into_dyn(),
            (2, 2) => (lhs.into_dimensionality::<Ix2>().unwrap().dot(&rhs.into_dimensionality::<Ix2>().unwrap())).into_dyn(),
            (3, 2) => dot3_2(lhs, rhs).into_dyn(),
            (_, _) => panic!("lhs dim: {shape0:?} can't be multiplied by rhs dim: {shape1:?}"),
        }
    }
    
    fn dot_rev(&self, rhs: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>, place: bool) -> ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>> {
        let shape0 = self.shape();
        let shape1 = rhs.shape();
        let lhs = self.clone();
        match (self.ndim(), rhs.ndim(), place) {
            (1, 1, _) => arr0(lhs.into_dimensionality::<Ix1>().unwrap().dot(&rhs.into_dimensionality::<Ix1>().unwrap())).into_dyn(),
            (2, 1, true) => (lhs.into_dimensionality::<Ix2>().unwrap().t().dot(&rhs.into_dimensionality::<Ix1>().unwrap())).into_dyn(),
            (2, 1, false) => (lhs.into_dimensionality::<Ix2>().unwrap().dot(&rhs.into_dimensionality::<Ix1>().unwrap().insert_axis(Axis(0)).t())).into_dyn(),
            (1, 2, true) => (lhs.into_dimensionality::<Ix1>().unwrap().insert_axis(Axis(0)).dot(&rhs.into_dimensionality::<Ix2>().unwrap())).into_dyn(),
            (1, 2, false) => (lhs.into_dimensionality::<Ix1>().unwrap().insert_axis(Axis(0)).dot(&rhs.into_dimensionality::<Ix2>().unwrap())).into_dyn(),
            (2, 2, true) => (lhs.into_dimensionality::<Ix2>().unwrap().t().dot(&rhs.into_dimensionality::<Ix2>().unwrap())).into_dyn(),
            (2, 2, false) => (lhs.into_dimensionality::<Ix2>().unwrap().dot(&rhs.into_dimensionality::<Ix2>().unwrap().t())).into_dyn(),
            (3, 2, _) => dot3_2(lhs, rhs.into_dimensionality::<Ix2>().unwrap().t().into_dyn().to_owned()).into_dyn(),
            (_, _, _) => panic!("lhs dim: {shape0:?} can't be multiplied by rhs dim: {shape1:?}"),
        }
    }
}

#[rustfmt::skip]
fn dot3_2(lhs: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>, rhs: ArrayBase<OwnedRepr<f64>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>> {
    let lhs = lhs.into_dimensionality::<Ix3>().unwrap();
    let rhs = rhs.into_dimensionality::<Ix2>().unwrap();
    let mut ans = vec![];
    for lhs in lhs.axis_iter(Axis(0)) {
        ans.push(lhs.dot(&rhs));
    }
    stack(Axis(0), ans.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice(),)
    .unwrap()
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data + rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data + rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data + rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data + rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::AddN(rhs, self.0.clone());
        let result = self.0.lock().unwrap().to_owned().data + rhs;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::AddN(rhs, self.0.clone());
        let result = self.0.lock().unwrap().to_owned().data + rhs;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add<&Tensor> for f64 {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let req = rhs.0.lock().unwrap().requires_grade;
        let dep = rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::AddN(self, rhs.0.clone());
        let result = self + rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Add<Tensor> for f64 {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        let req = rhs.0.lock().unwrap().requires_grade;
        let dep = rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::AddN(self, rhs.0.clone());
        let result = self + rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data - rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data - rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data - rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data - rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::SubN(1.0, self.0.clone());
        let result = self.0.lock().unwrap().to_owned().data - rhs;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::SubN(1.0, self.0.clone());
        let result = self.0.lock().unwrap().to_owned().data - rhs;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub<&Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let req = rhs.0.lock().unwrap().requires_grade;
        let dep = rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::SubN(-1.0, rhs.0.clone());
        let result = self - rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Sub<Tensor> for f64 {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let req = rhs.0.lock().unwrap().requires_grade;
        let dep = rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::SubN(-1.0, rhs.0.clone());
        let result = self - rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data * rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data * rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data * rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade || rhs.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some()
            || rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = self.0.lock().unwrap().to_owned().data * rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::MulN(rhs, self.0.clone());
        let result = self.0.lock().unwrap().to_owned().data * rhs;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let req = self.0.lock().unwrap().requires_grade;
        let dep = self.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::MulN(rhs, self.0.clone());
        let result = self.0.lock().unwrap().to_owned().data * rhs;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let req = rhs.0.lock().unwrap().requires_grade;
        let dep = rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::MulN(self, rhs.0.clone());
        let result = self * rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let req = rhs.0.lock().unwrap().requires_grade;
        let dep = rhs.0.lock().unwrap().depends_on.is_some();
        let grad = GradFn::MulN(self, rhs.0.clone());
        let result = self * rhs.0.lock().unwrap().to_owned().data;
        if req || dep {
            Tensor::from_op(result, grad)
        } else {
            Tensor::from_dyn(result)
        }
    }
}

#[derive(Debug, Clone)]
pub enum GradFn {
    Sum(Arc<Mutex<TensorBase>>),
    Mean(Arc<Mutex<TensorBase>>),
    Add(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    AddN(f64, Arc<Mutex<TensorBase>>),
    Sub(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    SubN(f64, Arc<Mutex<TensorBase>>),
    Dot(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    Mul(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    MulN(f64, Arc<Mutex<TensorBase>>),
    Pow(f64, Arc<Mutex<TensorBase>>),
    Reshape(Arc<Mutex<TensorBase>>),
}

#[allow(unused)]
impl GradFn {
    fn backward(&mut self, back: DynArray) {
        use GradFn::*;
        match self {
            Sum(tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                let shape: Vec<usize> = tensor.lock().unwrap().data.shape().into();
                if req || dep {
                    tensor.lock().unwrap().backward(Array::ones(shape) * back);
                }
            }
            Mean(tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                let shape: Vec<usize> = tensor.lock().unwrap().data.shape().into();
                let num: f64 = tensor.lock().unwrap().data.len() as f64;
                if req || dep {
                    tensor
                        .lock()
                        .unwrap()
                        .backward((1. / num) * Array::ones(shape) * back);
                }
            }
            Add(tensor0, tensor1) => {
                let shape0 = tensor0.lock().unwrap().data.shape().to_vec();
                let shape1 = tensor1.lock().unwrap().data.shape().to_vec();
                let req0 = tensor0.lock().unwrap().requires_grade;
                let req1 = tensor1.lock().unwrap().requires_grade;
                let dep0 = tensor0.lock().unwrap().depends_on.is_some();
                let dep1 = tensor1.lock().unwrap().depends_on.is_some();
                if shape0.iter().sum::<usize>() == shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(back.clone());
                    }
                    if req1 || dep1 {
                        tensor1.lock().unwrap().backward(back);
                    }
                } else if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(back.clone());
                    }
                    if req1 || dep1 {
                        tensor1
                            .lock()
                            .unwrap()
                            .backward(back.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0)));
                    }
                } else if shape0.iter().sum::<usize>() < shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(
                            back.clone()
                                .mean_axis(Axis(0))
                                .unwrap()
                                .insert_axis(Axis(0)),
                        );
                    }
                    if req1 || dep1 {
                        tensor1.lock().unwrap().backward(back);
                    }
                }
            }
            AddN(_, tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                if req || dep {
                    tensor.lock().unwrap().backward(back.clone());
                }
            }
            Sub(tensor0, tensor1) => {
                let shape0 = tensor0.lock().unwrap().data.shape().to_vec();
                let shape1 = tensor1.lock().unwrap().data.shape().to_vec();
                let req0 = tensor0.lock().unwrap().requires_grade;
                let req1 = tensor1.lock().unwrap().requires_grade;
                let dep0 = tensor0.lock().unwrap().depends_on.is_some();
                let dep1 = tensor1.lock().unwrap().depends_on.is_some();
                if shape0.iter().sum::<usize>() == shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(back.clone());
                    }
                    if req1 || dep1 {
                        tensor1.lock().unwrap().backward(-back);
                    }
                } else if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(back.clone());
                    }
                    if req1 || dep1 {
                        tensor1
                            .lock()
                            .unwrap()
                            .backward(-back.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0)));
                    }
                } else if shape0.iter().sum::<usize>() < shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(
                            back.clone()
                                .mean_axis(Axis(0))
                                .unwrap()
                                .insert_axis(Axis(0)),
                        );
                    }
                    if req1 || dep1 {
                        tensor1.lock().unwrap().backward(-back);
                    }
                }
            }
            SubN(number, tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                if req || dep {
                    if *number > 0.0 {
                        tensor.lock().unwrap().backward(back);
                    } else {
                        tensor.lock().unwrap().backward(-back);
                    }
                }
            }
            Dot(tensor0, tensor1) => {
                let req0 = tensor0.lock().unwrap().requires_grade;
                let req1 = tensor1.lock().unwrap().requires_grade;
                let dep0 = tensor0.lock().unwrap().depends_on.is_some();
                let dep1 = tensor1.lock().unwrap().depends_on.is_some();
                if req0 || dep0 {
                    tensor0.lock().unwrap().backward(
                        back.clone()
                            .dot_rev(tensor1.lock().unwrap().to_owned().data, false),
                    );
                }
                if req1 || dep1 {
                    tensor1.lock().unwrap().backward(
                        tensor0
                            .lock()
                            .unwrap()
                            .to_owned()
                            .data
                            .dot_rev(back.clone(), true),
                    );
                }
            }
            Mul(tensor0, tensor1) => {
                let shape0 = tensor0.lock().unwrap().data.shape().to_vec();
                let shape1 = tensor1.lock().unwrap().data.shape().to_vec();
                let req0 = tensor0.lock().unwrap().requires_grade;
                let req1 = tensor1.lock().unwrap().requires_grade;
                let dep0 = tensor0.lock().unwrap().depends_on.is_some();
                let dep1 = tensor1.lock().unwrap().depends_on.is_some();
                if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0
                            .lock()
                            .unwrap()
                            .backward(back.clone() * tensor1.lock().unwrap().to_owned().data);
                    }
                    if req1 || dep1 {
                        tensor1
                            .lock()
                            .unwrap()
                            .backward(back * tensor0.lock().unwrap().to_owned().data);
                    }
                } else if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0
                            .lock()
                            .unwrap()
                            .backward(back.clone() * tensor1.lock().unwrap().to_owned().data);
                    }
                    if req1 || dep1 {
                        tensor1.lock().unwrap().backward(
                            back.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0))
                                * tensor0.lock().unwrap().to_owned().data,
                        );
                    }
                } else if shape0.iter().sum::<usize>() < shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        tensor0.lock().unwrap().backward(
                            back.clone()
                                .mean_axis(Axis(0))
                                .unwrap()
                                .insert_axis(Axis(0))
                                * tensor1.lock().unwrap().to_owned().data,
                        );
                    }
                    if req1 || dep1 {
                        tensor1
                            .lock()
                            .unwrap()
                            .backward(back * tensor0.lock().unwrap().to_owned().data);
                    }
                }
            }
            MulN(number, tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                if req || dep {
                    tensor.lock().unwrap().backward(back.clone() * *number);
                }
            }
            Pow(number, tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                let data = tensor.lock().unwrap().data.clone();
                if req || dep {
                    tensor
                        .lock()
                        .unwrap()
                        .backward(data.mapv(|f| f.powf(*number - 1.)) * *number * back);
                }
            }
            Reshape(tensor) => {
                let req = tensor.lock().unwrap().requires_grade;
                let dep = tensor.lock().unwrap().depends_on.is_some();
                let shape: Vec<usize> = tensor.lock().unwrap().data.shape().into();
                if req || dep {
                    tensor
                        .lock()
                        .unwrap()
                        .backward(back.to_shape(shape).unwrap().to_owned());
                }
            }
        }
    }
}

impl fmt::Display for GradFn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use GradFn::*;
        match self {
            Sum(_) => write!(f, "SumBackward"),
            Mean(_) => write!(f, "MeanBackward"),
            Add(_, _) => write!(f, "AddBackward"),
            AddN(_, _) => write!(f, "AddBackward"),
            Sub(_, _) => write!(f, "SubBackward"),
            SubN(_, _) => write!(f, "SubBackward"),
            Dot(_, _) => write!(f, "DotBackward"),
            Mul(_, _) => write!(f, "MulBackward"),
            MulN(_, _) => write!(f, "MulBackward"),
            Pow(_, _) => write!(f, "PowBackward"),
            Reshape(_) => write!(f, "ReshapeBackward"),
        }
    }
}
