use std::ops::Mul;

use crate::{
    error::{Result, TensorError},
    tensor::Tensor,
};

impl<T> Tensor<T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(TensorError::ShapeMismatch);
        }

        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = rhs.shape[0];
        let n = rhs.shape[1];

        if k1 != k2 {
            return Err(TensorError::DimensionalMismatch);
        }

        let mut result = Tensor::new(vec![m, n], T::default());

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();

                for k in 0..k1 {
                    let a = *self.get(&[i, k])?;
                    let b = *rhs.get(&[k, j])?;
                    sum = sum + (a * b);
                }
                //let mut x = result.get_mut(&[i, j])?;
                //x = &mut sum;
                let _ = result.set(&[i, j], sum);
            }
        }

        Ok(result)
    }
}
impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T>,
{
    pub fn elementwise_mul(&self, rhs: &Self) -> Self {
        self.assert_same_shape(rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl<T: Mul<Output = T> + Clone + Copy> Mul for Tensor<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.elementwise_mul(&rhs)
    }
}
