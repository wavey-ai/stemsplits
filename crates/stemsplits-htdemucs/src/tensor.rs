//! A plain row-major tensor, enough for a faithful port.
//!
//! Deliberately not a general ndarray: the shapes here are fixed by the
//! model, and explicit shapes keep the port readable against the reference.

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape {shape:?} does not match {} values",
            data.len()
        );
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let count = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; count],
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    /// Row-major flat index. The last axis is contiguous.
    pub fn flat(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.shape.len());
        let mut offset = 0;
        for (axis, &position) in index.iter().enumerate() {
            debug_assert!(position < self.shape[axis]);
            offset = offset * self.shape[axis] + position;
        }
        offset
    }

    pub fn at(&self, index: &[usize]) -> f32 {
        self.data[self.flat(index)]
    }

    pub fn set(&mut self, index: &[usize], value: f32) {
        let flat = self.flat(index);
        self.data[flat] = value;
    }
}
