use crate::boundary_condition::BoundaryCondition;
use crate::markov_chain::MarkovChain;
use crate::reaction::*;
use ndarray::Array1;

use numpy::{
    ndarray::{Array2, ArrayView2},
    prelude::*,
    PyArray1, PyArray2,
};
use pyo3::{conversion::IntoPyObjectExt, prelude::*};

use serde::{Deserialize, Serialize};
use std::any::Any;

#[typetag::serde(tag = "type")]
pub trait Analyzer: Any {
    fn init(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        chain: &Box<dyn MarkovChain>,
        previous_analyzers: &Vec<Box<dyn Analyzer>>,
    );

    fn update(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        chain: &Box<dyn MarkovChain>,
        reaction: BasicReaction<u32>,
        dt: f64,
        previous_analyzers: &Vec<Box<dyn Analyzer>>,
    );
}

mod droplets;
pub use droplets::*;

mod commitance_probability;
pub use commitance_probability::*;

mod largest_droplet_over_time;
pub use largest_droplet_over_time::*;

mod cnt_rates;
pub use cnt_rates::*;

impl<'py> FromPyObject<'py> for Box<dyn Analyzer> {
    fn extract_bound(py_chain: &Bound<'_, PyAny>) -> PyResult<Box<dyn Analyzer>> {
        if let Ok(chain) = py_chain.extract::<Droplets>() {
            return Ok(Box::new(chain));
        }
        if let Ok(chain) = py_chain.extract::<LargestDropletSizeAnalyzer>() {
            return Ok(Box::new(chain));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{} is not on the list of `Analyzer`s.",
            py_chain
        )))
    }
}

impl<'py> IntoPyObject<'py> for Box<dyn Analyzer> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let chain_any = self as Box<dyn Any>;
        let chain_any = match chain_any.downcast::<Droplets>() {
            Ok(chain) => {
                return Ok(chain.into_bound_py_any(py)?);
            }
            Err(chain_any) => chain_any,
        };
        let chain_any = match chain_any.downcast::<LargestDropletSizeAnalyzer>() {
            Ok(chain) => {
                return Ok(chain.into_bound_py_any(py)?);
            }
            Err(chain_any) => chain_any,
        };
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{:?} is not in the explicit list for `Analyzer`s.",
            chain_any
        )))
    }
}
