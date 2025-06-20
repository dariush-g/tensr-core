use crate::error::Result;
use crate::error::TensorError;
use crate::tensor::Tensor;

impl<T: Clone> Tensor<T> {
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.shape.len() {
            return Err(TensorError::ShapeMismatch);
        }

        if !is_valid_perm(dims) {
            return Err(TensorError::InvalidPermutation);
        }

        let new_shape = dims.iter().map(|&i| self.shape[i]).collect();
        let new_strides = dims.iter().map(|&i| self.strides[i]).collect();

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: *self.get_offset(),
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let mut dims: Vec<usize> = (0..self.shape.len()).collect();
        dims.swap(dim1, dim2);
        self.permute(&dims)
    }
}

fn is_valid_perm(dims: &[usize]) -> bool {
    let mut seen = vec![false; dims.len()];

    for &d in dims {
        if d >= dims.len() || seen[d] {
            return false;
        }
        seen[d] = true;
    }
    true
}
