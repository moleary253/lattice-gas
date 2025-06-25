use numpy::ndarray::ArrayView2;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;

use std::any::Any;

// TODO(Myles): Add documentation
#[typetag::serde(tag = "type")]
pub trait BoundaryCondition: Any {
    fn get(&self, state: &ArrayView2<u32>, pos: [usize; 2]) -> u32;
    fn adjacent(&self, state: &ArrayView2<u32>, pos: [usize; 2]) -> Vec<u32>;
    fn adjacent_indicies(&self, state: &ArrayView2<u32>, pos: [usize; 2]) -> Vec<[usize; 2]>;
}

mod periodic_boundary;
pub use periodic_boundary::*;

impl<'py> FromPyObject<'py> for Box<dyn BoundaryCondition> {
    fn extract_bound(boundary: &Bound<'py, PyAny>) -> PyResult<Box<dyn BoundaryCondition>> {
        if let Ok(boundary) = boundary.extract::<Periodic>() {
            return Ok(Box::new(boundary));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{} is not a boundary condition.",
            boundary
        )))
    }
}

impl<'py> IntoPyObject<'py> for Box<dyn BoundaryCondition> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let boundary_any = self as Box<dyn Any>;
        let boundary_any = match boundary_any.downcast::<Periodic>() {
            Ok(boundary) => {
                return Ok(boundary.into_bound_py_any(py)?);
            }
            Err(boundary_any) => boundary_any,
        };
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{:?} is not in the explicit list for boundary conditions.",
            boundary_any
        )))
    }
}
