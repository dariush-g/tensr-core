use crate::error::{Result, TensorError};
use crate::tensor::Tensor;
use std::ops::Range;

pub struct TensorView<'data, T> {
    data: &'data [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

pub struct TensorViewMut<'data, T> {
    data: &'data mut [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
    //offset: usize,
}

impl<'data, T> TensorViewMut<'data, T> {
    pub fn slice_mut(
        &'data mut self,
        axis: usize,
        range: Range<usize>,
    ) -> Result<TensorViewMut<'data, T>> {
        if axis > self.shape.len() {
            return Err(TensorError::IndexOutOfBounds);
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = range.end - range.start;

        let new_offset = self.strides[axis] * range.start;

        let start_i = new_offset;
        let length = new_shape.iter().product::<usize>();
        let new_data = &mut self.data[start_i..start_i + length];

        Ok(TensorViewMut {
            data: new_data,
            shape: new_shape,
            strides: self.strides.clone(),
            //offset: 0
        })
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let mut flat_idx = 0;
        for (i, &idx_i) in indices.iter().enumerate() {
            if idx_i >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds);
            }
            flat_idx += self.strides[i] * idx_i;
        }
        Ok(self
            .data
            .get_mut(flat_idx)
            .ok_or(Err(TensorError::IndexOutOfBounds)?)?)
    }

    pub fn data(&'data self) -> &'data [T] {
        self.data
    }

    pub fn data_mut(&'data mut self) -> &'data mut [T] {
        self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl<T: Clone> Tensor<T> {
    pub fn view<'a>(&'a self) -> TensorView<'a, T> {
        TensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: *self.get_offset(),
        }
    }

    pub fn view_mut<'a>(&'a mut self) -> TensorView<'a, T> {
        TensorView {
            data: &mut self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl<'data, T> TensorView<'data, T> {
    
    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::IndexOutOfBounds);
        }
        let mut flat_idx = 0;
        for (i, &idx_i) in indices.iter().enumerate() {
            if idx_i >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds);
            }
            flat_idx += self.strides[i] * idx_i;
        }
        Ok(self
            .data
            .get(flat_idx)
            .ok_or(Err(TensorError::IndexOutOfBounds)?)?)
    }

    pub fn get_data(&self) -> &'data [T] {
        self.data
    }

    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn get_strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn get_offset(&self) -> usize {
        self.offset
    }

    pub fn reshape(&self, new_shape: Vec<usize>, new_strides: Vec<usize>) -> TensorView<'data, T> {
        TensorView {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn slice(&self, axis: usize, range: Range<usize>) -> Result<Self> {
        if axis > self.shape.len() {
            return Err(crate::error::TensorError::IndexOutOfBounds);
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = range.end - range.start;

        let new_offset = self.offset + self.strides[axis] * range.start;

        Ok(TensorView {
            data: self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }
}
