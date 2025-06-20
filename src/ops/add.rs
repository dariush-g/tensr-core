use std::ops::Add;

use crate::tensor::Tensor;

impl<T: Add<Output = T> + Clone + Copy> Add for Tensor<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.elementwise_add(&rhs)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Add<Output = T>,
{
    pub fn elementwise_add(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape, rhs.shape, "Shape mismatch for addition");
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}
