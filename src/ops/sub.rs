use std::ops::Sub;

use crate::tensor::Tensor;

impl<T: Sub<Output = T> + Clone + Copy> Sub for Tensor<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.elementwise_sub(&rhs)
    }
}
impl<T> Tensor<T>
where
    T: Copy + Sub<Output = T>,
{
    pub fn elementwise_sub(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}
