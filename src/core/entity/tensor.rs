use super::TransformerError;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TransformerError> {
        let size = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| TransformerError::InvalidTensor("shape overflow".into()))?;
        if size != data.len() {
            return Err(TransformerError::InvalidTensor(format!(
                "shape size {size} does not match data length {}",
                data.len()
            )));
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(TransformerError::InvalidTensor("non-finite value".into()));
        }
        Ok(Self { data, shape })
    }
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: vec![0.0; shape.iter().product()],
            shape: shape.to_vec(),
        }
    }
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1]
    }
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TransformerError> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 || self.cols() != rhs.rows() {
            return Err(TransformerError::InvalidTensor(
                "matmul dimension mismatch".into(),
            ));
        }
        let (m, k, n) = (self.rows(), self.cols(), rhs.cols());
        let mut out = Self::zeros(&[m, n]);
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    out.data[i * n + j] += a * rhs.data[p * n + j];
                }
            }
        }
        Ok(out)
    }
    pub fn transpose(&self) -> Result<Self, TransformerError> {
        if self.shape.len() != 2 {
            return Err(TransformerError::InvalidTensor(
                "transpose expects rank 2".into(),
            ));
        }
        let mut out = Self::zeros(&[self.cols(), self.rows()]);
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                out.data[j * self.rows() + i] = self.data[i * self.cols() + j];
            }
        }
        Ok(out)
    }
}
