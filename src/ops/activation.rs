use crate::tensor::Tensor;

impl Tensor<f32> {
    pub fn relu(&self) -> Self {
        let data = self.data.iter().map(|x| x.max(0.0)).collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: *self.get_offset(),
        }
    }
    pub fn sigmoid(&self) -> Self {
        let data = self.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: *self.get_offset(),
        }
    }
    pub fn tanh(&self) -> Self {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: *self.get_offset(),
        }
    }
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(axis < self.shape.len(), "Invalid axis");
        unimplemented!("softmax activation")
    }
}
