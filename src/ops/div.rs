use std::ops::Div;

use crate::tensor::Tensor;

impl<T: Div<Output = T> + Clone + Copy> Div for Tensor<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.elementwise_div(&rhs)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Div<Output = T>,
{
    pub fn elementwise_div(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}
