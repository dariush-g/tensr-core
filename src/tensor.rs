use crate::error::{Result, TensorError};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl<T: Clone> Tensor<T> {
    pub fn get_offset(&self) -> &usize {
        &self.offset
    }

    pub fn set_offset(&mut self, new: usize) -> Result<()> {
        if self.data.len() < new {
            return Err(TensorError::IndexOutOfBounds);
        }

        self.offset = new;
        Ok(())
    }

    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    pub fn set_data(&mut self, new: Vec<T>) -> Result<()> {
        if self.data.len() != new.len() {
            return Err(TensorError::DimensionalMismatch);
        }

        self.data = new;
        Ok(())
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn set_shape(&mut self, new: Vec<usize>) -> Result<()> {
        if self.shape.len() != new.len() {
            return Err(TensorError::ShapeMismatch);
        }

        self.shape = new;
        Ok(())
    }

    pub fn get_strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn set_strides(&mut self, new: Vec<usize>) -> Result<()> {
        if self.strides.len() != new.len() {
            return Err(TensorError::DimensionalMismatch);
        }

        self.strides = new;
        Ok(())
    }

    pub fn get_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds);
        }

        Ok(indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum())
    }

    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        if let Ok(index) = self.get_index(indices) {
            if let Some(val) = self.data.get(index) {
                return Ok(val);
            }
        }
        Err(TensorError::IndexOutOfBounds)
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T> {
        if let Ok(index) = self.get_index(indices) {
            if let Some(val) = self.data.get_mut(index) {
                return Ok(val);
            }
        }
        Err(TensorError::IndexOutOfBounds)
    }

    pub fn set(&mut self, indices: &[usize], new: T) -> Result<()> {
        if let Ok(value) = self.get_mut(indices) {
            *value = new;
            return Ok(());
        }
        Err(TensorError::IndexOutOfBounds)
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<Self> {
        if self.data.len() != new_shape.iter().product::<usize>() {
            println!("Total size must remain the same");
            return Err(TensorError::ShapeMismatch);
        }

        self.strides = compute_strides(&new_shape);
        Ok(self.clone())
    }

    pub fn new(shape: Vec<usize>, fill: T) -> Self {
        let total = shape.iter().product();
        let data = vec![fill; total];
        let strides = compute_strides(&shape);

        Self {
            data,
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn from_data(shape: Vec<usize>, data: Vec<T>) -> Result<Self> {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape does not match data length"
        );
        if shape.iter().product::<usize>() != data.len() {
            return Err(TensorError::ShapeMismatch);
        }

        let strides = compute_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
            offset: 0,
        })
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<T: Copy> Tensor<T> {
    pub fn assert_same_shape(&self, other: &Self) {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
    }
}
