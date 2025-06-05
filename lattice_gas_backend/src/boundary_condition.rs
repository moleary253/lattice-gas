use numpy::ndarray::ArrayView2;
use pyo3::prelude::*;

// TODO(Myles): Add documentation
pub trait BoundaryCondition<T>
where
    T: Clone,
{
    fn get(&self, state: &ArrayView2<T>, pos: [usize; 2]) -> T;
    fn adjacent(&self, state: &ArrayView2<T>, pos: [usize; 2]) -> Vec<T>;
    fn adjacent_indicies(&self, state: &ArrayView2<T>, pos: [usize; 2]) -> Vec<[usize; 2]>;
}

mod periodic_boundary;
pub use periodic_boundary::*;

pub fn extract(boundary: &Bound<'_, PyAny>) -> PyResult<Box<dyn BoundaryCondition<u32>>> {
    if let Ok(boundary) = boundary.extract::<Periodic>() {
        return Ok(Box::new(boundary));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} is not a boundary condition.",
        boundary
    )))
}
