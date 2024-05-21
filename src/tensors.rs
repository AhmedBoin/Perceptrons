use crate::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBase {
    data: DynArray,
    requires_grade: bool,
    #[serde(skip)]
    depends_on: Option<GradFn>,
    grad: Option<DynArray>,
}

#[rustfmt::skip]
impl TensorBase {
    pub fn backward(&mut self, back: DynArray) {
        if self.requires_grade {
            self.grad = Some(match self.grad.take() {
                Some(data) => data + &back,
                None => back.clone(),
            });
        }

        if let Some(mut data) = self.depends_on.take() {
            data.backward(back);
        }
    }

    pub fn backward_strict(&mut self, back: DynArray) {
        if self.depends_on.is_none() {
            panic!("tensor doesn't have a grad_fn");
        }
        self.backward(back);
    }

    fn wrap(self) -> Tensor {
        Tensor(Arc::new(Mutex::new(self)))
    }
}

#[derive(Clone)]
pub struct Tensor(Arc<Mutex<TensorBase>>);

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.lock().unwrap().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Tensor, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tensor_base = TensorBase::deserialize(deserializer)?;
        Ok(Tensor(Arc::new(Mutex::new(tensor_base))))
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prev = self.0.lock().unwrap();
        let req: bool = prev.requires_grade;
        write!(f, "Tensor({:?}", prev.data)?;
        match prev.depends_on.as_ref() {
            Some(dep) => {
                write!(f, ", grad_fn=<{:?}>)", dep)?;
            }
            None => {
                if req {
                    write!(f, ", requires_grad=true")?;
                }
            }
        }
        write!(f, ")")
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let prev = self.0.lock().unwrap();
        let req: bool = prev.requires_grade;
        write!(f, "Tensor({}", prev.data)?;
        match prev.depends_on.as_ref() {
            Some(dep) => {
                write!(f, ", grad_fn=<{}>)", dep)?;
            }
            None => {
                if req {
                    write!(f, ", requires_grad=true")?;
                }
            }
        }
        write!(f, ")")
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        // Compare the underlying TensorBase for equality
        let self_lock = self.0.lock().unwrap().data.clone();
        let other_lock = other.0.lock().unwrap().data.clone();
        self_lock == other_lock
    }
}

impl Eq for Tensor {}

pub trait ToDynArray {
    fn into_dyn(self) -> DynArray;
}

impl<D: ndarray::Dimension> ToDynArray for ArrayBase<OwnedRepr<f32>, D> {
    fn into_dyn(self) -> DynArray {
        ArrayBase::into_dyn(self)
    }
}

impl ToDynArray for Vec<f32> {
    fn into_dyn(self) -> DynArray {
        Array::from_vec(self).into_dyn()
    }
}

impl ToDynArray for &[f32] {
    fn into_dyn(self) -> DynArray {
        Array::from_iter(self.iter().copied()).into_dyn()
    }
}

impl ToDynArray for Tensor {
    fn into_dyn(self) -> DynArray {
        self.0.lock().unwrap().data.clone()
    }
}

impl ToDynArray for f32 {
    fn into_dyn(self) -> DynArray {
        array![self].into_dyn()
    }
}

#[rustfmt::skip]
impl Tensor {
    pub fn new<T: ToDynArray>(data: T) -> Self {
        TensorBase {
            data: data.into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn from_dyn(data: DynArray) -> Self {
        TensorBase {
            data: data,
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn from_op<T: ToDynArray>(data: T, depends_on: GradFn, requires_grade: bool) -> Self {
        TensorBase {
            data: data.into_dyn(),
            requires_grade,
            depends_on: if *NO_GRAD.lock().unwrap() {
                None
            } else {
                Some(depends_on)
            },
            grad: None,
        }.wrap()
    }

    pub fn eye(n: usize) -> Self {
        TensorBase {
            data: Array::<f32, _>::eye(n).into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn ones(shape: &[usize]) -> Self {
        TensorBase {
            data: Array::<f32, _>::ones(shape).into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn zeros(shape: &[usize]) -> Self {
        TensorBase {
            data: Array::<f32, _>::zeros(shape).into_dyn(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn ones_like(data: &Tensor) -> Self {
        TensorBase {
            data: Array::<f32, _>::ones(data.0.lock().unwrap().data.shape()),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn zeros_like(data: &Tensor) -> Self {
        TensorBase {
            data: Array::<f32, _>::zeros(data.0.lock().unwrap().data.shape()),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    fn get_mut_then<F: FnOnce(&mut TensorBase)>(mut self, f: F) -> Self {
        match Arc::get_mut(&mut self.0) {
            Some(x) => {
                f(x.get_mut().unwrap());
                self
            }
            None => {
                let mut x = self.0.lock().unwrap().to_owned();
                f(&mut x);
                x.wrap()
            }
        }
    }

    pub fn requires_grad(self, boolean: bool) -> Self {
        self.get_mut_then(|lock| { lock.requires_grade = boolean; })
    }

    pub fn set_grad(&self, boolean: bool) {
        self.0.lock().unwrap().requires_grade = boolean;
    }

    pub fn detach(&self) -> Self {
        TensorBase {
            data: self.0.lock().unwrap().data.to_owned(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.lock().unwrap().data.shape().into()
    }

    pub fn grad(&self) -> Option<DynArray> {
        self.0.lock().unwrap().grad.clone()
    }

    pub fn grad_tensor(&self) -> Option<Tensor> {
        self.0.lock().unwrap().grad.as_ref().map(|grad| TensorBase {
            data: grad.clone(),
            requires_grade: false,
            depends_on: None,
            grad: None,
        }.wrap())
    }

    pub fn zero_grad(&self) {
        self.0.lock().unwrap().grad = None;
    }

    pub fn backward(&self) {
        let mut lock = self.0.lock().unwrap();
        let shape: Vec<usize> = lock.data.shape().into();
        lock.backward_strict(Array::ones(shape));
    }

    pub fn backward_with_array(&self, array: impl ToDynArray) {
        let mut lock = self.0.lock().unwrap();
        lock.backward_strict(array.into_dyn());
    }

    pub fn backward_with_tensor(&self, tensor: Tensor) {
        let mut lock = self.0.lock().unwrap();
        let tensor_lock = tensor.0.lock().unwrap();
        lock.backward_strict(tensor_lock.data.to_owned());
    }

    pub fn optimize_with_dyn_array(&self, grad: DynArray) {
        self.0.lock().unwrap().data -= &grad;
    }

    pub fn optimize_with_array(&self, grad: impl ToDynArray) {
        self.0.lock().unwrap().data -= &grad.into_dyn();
    }

    pub fn optimize_with_tensor(&self, grad: Tensor) {
        self.0.lock().unwrap().data -= &grad.0.lock().unwrap().data;
    }

    #[rustfmt::skip]
    pub fn reshape(&self, shape: impl ShapeArg) -> Self {
        let lock = self.0.lock().unwrap();
        let req = lock.requires_grade;
        let dep = lock.depends_on.is_some();
        let d = lock.data.clone();
        let data = match d.ndim() {
            0 => d.into_dimensionality::<Ix0>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            1 => d.into_dimensionality::<Ix1>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            2 => d.into_dimensionality::<Ix2>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            3 => d.into_dimensionality::<Ix3>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            4 => d.into_dimensionality::<Ix4>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            5 => d.into_dimensionality::<Ix5>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            6 => d.into_dimensionality::<Ix6>().unwrap().to_shape(shape).unwrap().to_owned().into_dyn(),
            _ => panic!("There is no dimensionality greater than 6"),
        };
        let depends_on = if *NO_GRAD.lock().unwrap() {
            None
        } else if req || dep {
            Some(GradFn::Reshape(self.0.clone()))
        } else {
            None
        };
        TensorBase {
            data,
            requires_grade: false,
            depends_on,
            grad: None,
        }.wrap()
    }

    pub fn sum(&self) -> Self {
        let lock = self.0.lock().unwrap();
        let req = lock.requires_grade;
        let dep = lock.depends_on.is_some();
        let depends_on = if *NO_GRAD.lock().unwrap() {
            None
        } else if req || dep {
            Some(GradFn::Sum(self.0.clone()))
        } else {
            None
        };
        TensorBase {
            data: arr0(lock.data.sum()).into_dyn(),
            requires_grade: false,
            depends_on,
            grad: None,
        }.wrap()
    }

    pub fn mean(&self) -> Self {
        let lock = self.0.lock().unwrap();
        let req = lock.requires_grade;
        let dep = lock.depends_on.is_some();
        let depends_on = if *NO_GRAD.lock().unwrap() {
            None
        } else if req || dep {
            Some(GradFn::Mean(self.0.clone()))
        } else {
            None
        };
        TensorBase {
            data: arr0(lock.data.mean().unwrap() as f32).into_dyn(),
            requires_grade: false,
            depends_on,
            grad: None,
        }.wrap()
    }

    pub fn pow(&self, num: f32) -> Self {
        let lock = self.0.lock().unwrap();
        let req = lock.requires_grade;
        let dep = lock.depends_on.is_some();
        let depends_on = if *NO_GRAD.lock().unwrap() {
            None
        } else if req || dep {
            Some(GradFn::Pow(num, self.0.clone()))
        } else {
            None
        };
        TensorBase {
            data: lock.data.mapv(|f| f.powf(num)),
            requires_grade: false,
            depends_on,
            grad: None,
        }.wrap()
    }

    #[rustfmt::skip]
    pub fn dot(&self, rhs: &Tensor) -> Self {
        let lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lock.requires_grade || rhs_lock.requires_grade;
        let dep = lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let depends_on = if *NO_GRAD.lock().unwrap() {
            None
        } else if req || dep {
            Some(GradFn::Dot(self.0.clone(), rhs.0.clone()))
        } else {
            None
        };
        TensorBase {
            data: lock
                .data
                .dot(rhs_lock.data.clone())
            .into_dyn(),
            requires_grade: false,
            depends_on,
            grad: None,
        }.wrap()
    }
}

impl TensorOptimize for Tensor {
    fn optimize(&self, grad: Tensor) {
        self.0.lock().unwrap().data = self.clone().into_dyn() - grad.into_dyn();
    }
}

#[rustfmt::skip]
pub trait Dot {
    fn dot(&self, rhs: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;
    fn dot_rev(&self, rhs: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, place: bool) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;
}

#[rustfmt::skip]
impl Dot for ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    fn dot(&self, rhs: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
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

    fn dot_rev(&self, rhs: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, place: bool) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
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
fn dot3_2(lhs: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, rhs: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>> {
    let lhs = lhs.into_dimensionality::<Ix3>().unwrap();
    let rhs = rhs.into_dimensionality::<Ix2>().unwrap();
    let mut ans = vec![];
    for lhs in lhs.axis_iter(Axis(0)) {
        ans.push(lhs.dot(&rhs));
    }
    stack(Axis(0), ans.iter().map(|a| a.view()).collect::<Vec<_>>().as_slice(),)
    .unwrap()
}

#[derive(Debug)]
pub enum Grad {
    Grad(Tensor),
    NoGrad,
}

impl Grad {
    pub fn unwrap(self) -> Tensor {
        match self {
            Grad::Grad(tensor) => tensor,
            Grad::NoGrad => panic!("There is no grad on Unwrap Grad"),
        }
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() + &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() + &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() + &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Add(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() + &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let req = lhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some();
        let grad = GradFn::AddN(rhs, self.0.clone());
        let result = lhs_lock.data.to_owned() + rhs;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let req = lhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some();
        let grad = GradFn::AddN(rhs, self.0.clone());
        let result = lhs_lock.data.to_owned() + rhs;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add<f32> for Grad {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        if let Grad::Grad(ref slf) = self {
            let lock = slf.0.lock().unwrap();
            let req = lock.requires_grade;
            let dep = lock.depends_on.is_some();
            let grad = GradFn::AddN(rhs, slf.0.clone());
            let result = lock.data.to_owned() + rhs;
            if req || dep {
                Tensor::from_op(result, grad, req)
            } else {
                Tensor::new(result)
            }
        } else {
            Tensor::new(rhs)
        }
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let rhs_lock = rhs.0.lock().unwrap();
        let req = rhs_lock.requires_grade;
        let dep = rhs_lock.depends_on.is_some();
        let grad = GradFn::AddN(self, rhs.0.clone());
        let result = self + &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Add<Tensor> for f32 {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        let rhs_lock = rhs.0.lock().unwrap();
        let req = rhs_lock.requires_grade;
        let dep = rhs_lock.depends_on.is_some();
        let grad = GradFn::AddN(self, rhs.0.clone());
        let result = self + &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() - &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() - &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() - &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Sub(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() - &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let req = lhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some();
        let grad = GradFn::SubN(1.0, self.0.clone());
        let result = lhs_lock.data.to_owned() - rhs;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let req = lhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some();
        let grad = GradFn::SubN(1.0, self.0.clone());
        let result = lhs_lock.data.to_owned() - rhs;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let rhs_lock = rhs.0.lock().unwrap();
        let req = rhs_lock.requires_grade;
        let dep = rhs_lock.depends_on.is_some();
        let grad = GradFn::SubN(-1.0, rhs.0.clone());
        let result = self - &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Sub<Tensor> for f32 {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let rhs_lock = rhs.0.lock().unwrap();
        let req = rhs_lock.requires_grade;
        let dep = rhs_lock.depends_on.is_some();
        let grad = GradFn::SubN(-1.0, rhs.0.clone());
        let result = self - &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul<f32> for Grad {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        if let Grad::Grad(ref slf) = self {
            let lock = slf.0.lock().unwrap();
            let req = lock.requires_grade;
            let dep = lock.depends_on.is_some();
            let grad = GradFn::MulN(rhs, slf.0.clone());
            let result = lock.data.to_owned() * rhs;
            if req || dep {
                Tensor::from_op(result, grad, req)
            } else {
                Tensor::new(result)
            }
        } else {
            Tensor::new(rhs)
        }
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() * &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() * &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() * &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let rhs_lock = rhs.0.lock().unwrap();
        let req = lhs_lock.requires_grade || rhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some()
            || rhs_lock.depends_on.is_some();
        let grad = GradFn::Mul(self.0.clone(), rhs.0.clone());
        let result = lhs_lock.data.to_owned() * &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let req = lhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some();
        let grad = GradFn::MulN(rhs, self.0.clone());
        let result = lhs_lock.data.to_owned() * rhs;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        let lhs_lock = self.0.lock().unwrap();
        let req = lhs_lock.requires_grade;
        let dep = lhs_lock.depends_on.is_some();
        let grad = GradFn::MulN(rhs, self.0.clone());
        let result = lhs_lock.data.to_owned() * rhs;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let rhs_lock = rhs.0.lock().unwrap();
        let req = rhs_lock.requires_grade;
        let dep = rhs_lock.depends_on.is_some();
        let grad = GradFn::MulN(self, rhs.0.clone());
        let result = self * &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let rhs_lock = rhs.0.lock().unwrap();
        let req = rhs_lock.requires_grade;
        let dep = rhs_lock.depends_on.is_some();
        let grad = GradFn::MulN(self, rhs.0.clone());
        let result = self * &rhs_lock.data;
        if req || dep {
            Tensor::from_op(result, grad, req)
        } else {
            Tensor::new(result)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradFn {
    #[serde(skip)]
    Sum(Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Mean(Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Add(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    AddN(f32, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Sub(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    SubN(f32, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Dot(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Mul(Arc<Mutex<TensorBase>>, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    MulN(f32, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Pow(f32, Arc<Mutex<TensorBase>>),
    #[serde(skip)]
    Reshape(Arc<Mutex<TensorBase>>),
}

#[allow(unused)]
impl GradFn {
    fn backward(&mut self, back: DynArray) {
        use GradFn::*;
        match self {
            Sum(tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                let shape: Vec<usize> = lock.data.shape().into();
                if req || dep {
                    lock.backward(Array::ones(shape) * back);
                }
            }
            Mean(tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                let shape: Vec<usize> = lock.data.shape().into();
                let num: f32 = lock.data.len() as f32;
                if req || dep {
                    lock.backward((1. / num) * Array::ones(shape) * back);
                }
            }
            Add(tensor0, tensor1) => {
                let mut lock0 = tensor0.lock().unwrap();
                let mut lock1 = tensor1.lock().unwrap();
                let shape0 = lock0.data.shape().to_vec();
                let shape1 = lock1.data.shape().to_vec();
                let req0 = lock0.requires_grade;
                let req1 = lock1.requires_grade;
                let dep0 = lock0.depends_on.is_some();
                let dep1 = lock1.depends_on.is_some();
                if shape0.iter().sum::<usize>() == shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(back.clone());
                    }
                    if req1 || dep1 {
                        lock1.backward(back);
                    }
                } else if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(back.clone());
                    }
                    if req1 || dep1 {
                        lock1.backward(back.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0)));
                    }
                } else if shape0.iter().sum::<usize>() < shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(
                            back.clone()
                                .mean_axis(Axis(0))
                                .unwrap()
                                .insert_axis(Axis(0)),
                        );
                    }
                    if req1 || dep1 {
                        lock1.backward(back);
                    }
                }
            }
            AddN(_, tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                if req || dep {
                    lock.backward(back.clone());
                }
            }
            Sub(tensor0, tensor1) => {
                let mut lock0 = tensor0.lock().unwrap();
                let mut lock1 = tensor1.lock().unwrap();
                let shape0 = lock0.data.shape().to_vec();
                let shape1 = lock1.data.shape().to_vec();
                let req0 = lock0.requires_grade;
                let req1 = lock1.requires_grade;
                let dep0 = lock0.depends_on.is_some();
                let dep1 = lock1.depends_on.is_some();
                if shape0.iter().sum::<usize>() == shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(back.clone());
                    }
                    if req1 || dep1 {
                        lock1.backward(-back);
                    }
                } else if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(back.clone());
                    }
                    if req1 || dep1 {
                        lock1.backward(-back.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0)));
                    }
                } else if shape0.iter().sum::<usize>() < shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(
                            back.clone()
                                .mean_axis(Axis(0))
                                .unwrap()
                                .insert_axis(Axis(0)),
                        );
                    }
                    if req1 || dep1 {
                        lock1.backward(-back);
                    }
                }
            }
            SubN(number, tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                if req || dep {
                    lock.backward(if *number > 0.0 {
                        back
                    } else {
                        -back
                    });
                }
            }
            Dot(tensor0, tensor1) => {
                let mut lock0 = tensor0.lock().unwrap();
                let mut lock1 = tensor1.lock().unwrap();
                let req0 = lock0.requires_grade;
                let req1 = lock1.requires_grade;
                let dep0 = lock0.depends_on.is_some();
                let dep1 = lock1.depends_on.is_some();
                if req0 || dep0 {
                    lock0.backward(
                        back.clone()
                            .dot_rev(lock1.data.to_owned(), false),
                    );
                }
                if req1 || dep1 {
                    lock1.backward(
                        lock0.data
                            .to_owned()
                            .dot_rev(back.clone(), true),
                    );
                }
            }
            Mul(tensor0, tensor1) => {
                let mut lock0 = tensor0.lock().unwrap();
                let mut lock1 = tensor1.lock().unwrap();
                let shape0 = lock0.data.shape().to_vec();
                let shape1 = lock1.data.shape().to_vec();
                let req0 = lock0.requires_grade;
                let req1 = lock1.requires_grade;
                let dep0 = lock0.depends_on.is_some();
                let dep1 = lock1.depends_on.is_some();
                if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(back.clone() * lock1.data.to_owned());
                    }
                    if req1 || dep1 {
                        lock1.backward(back * lock0.data.to_owned());
                    }
                } else if shape0.iter().sum::<usize>() > shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(back.clone() * lock1.data.to_owned());
                    }
                    if req1 || dep1 {
                        lock1.backward(
                            back.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0))
                                * lock0.data.to_owned(),
                        );
                    }
                } else if shape0.iter().sum::<usize>() < shape1.iter().sum::<usize>() {
                    if req0 || dep0 {
                        lock0.backward(
                            back.clone()
                                .mean_axis(Axis(0))
                                .unwrap()
                                .insert_axis(Axis(0))
                                * lock1.data.to_owned(),
                        );
                    }
                    if req1 || dep1 {
                        lock1.backward(back * lock0.data.to_owned());
                    }
                }
            }
            MulN(number, tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                if req || dep {
                    lock.backward(back.clone() * *number);
                }
            }
            Pow(number, tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                let data = lock.data.clone();
                if req || dep {
                    lock.backward(data.mapv(|f| f.powf(*number - 1.)) * *number * back);
                }
            }
            Reshape(tensor) => {
                let mut lock = tensor.lock().unwrap();
                let req = lock.requires_grade;
                let dep = lock.depends_on.is_some();
                let shape: Vec<usize> = lock.data.shape().into();
                if req || dep {
                    lock.backward(back.to_shape(shape).unwrap().to_owned());
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
