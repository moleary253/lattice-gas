use crate::boundary_condition::BoundaryCondition;
use crate::reaction::BasicReaction;

use numpy::ndarray::{Array2, ArrayView2};
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;

use std::any::Any;

// TODO(Myles): Add documentation
#[typetag::serde(tag = "type")]
pub trait MarkovChain: Any {
    fn num_possible_reactions(&self, state: &Array2<u32>) -> usize;
    fn rate(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> f64;
    fn on_reaction(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
        dt: f64,
    );
    fn initialize(&mut self, state: &ArrayView2<u32>, boundary: &Box<dyn BoundaryCondition>);
    fn indicies_affecting_reaction(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]>;
    fn reaction(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> BasicReaction<u32>;
}

mod homogenous_chain;
pub use homogenous_chain::*;
mod ising_chain;
pub use ising_chain::*;
mod homogenous_nvt_chain;
pub use homogenous_nvt_chain::*;
mod cnt_ladder_chain;
pub use cnt_ladder_chain::*;

impl<'py> FromPyObject<'py> for Box<dyn MarkovChain> {
    fn extract_bound(py_chain: &Bound<'_, PyAny>) -> PyResult<Box<dyn MarkovChain>> {
        if let Ok(chain) = py_chain.extract::<HomogenousChain>() {
            return Ok(Box::new(chain));
        }
        if let Ok(chain) = py_chain.extract::<IsingChain>() {
            return Ok(Box::new(chain));
        }
        if let Ok(chain) = py_chain.extract::<HomogenousNVTChain>() {
            return Ok(Box::new(chain));
        }
        if let Ok(chain) = py_chain.extract::<CNTLadderChain>() {
            return Ok(Box::new(chain));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{} is not a MarkovChain.",
            py_chain
        )))
    }
}

impl<'py> IntoPyObject<'py> for Box<dyn MarkovChain> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let chain_any = self as Box<dyn Any>;
        let chain_any = match chain_any.downcast::<HomogenousChain>() {
            Ok(chain) => {
                return Ok(chain.into_bound_py_any(py)?);
            }
            Err(chain_any) => chain_any,
        };
        let chain_any = match chain_any.downcast::<IsingChain>() {
            Ok(chain) => {
                return Ok(chain.into_bound_py_any(py)?);
            }
            Err(chain_any) => chain_any,
        };
        let chain_any = match chain_any.downcast::<HomogenousNVTChain>() {
            Ok(chain) => {
                return Ok(chain.into_bound_py_any(py)?);
            }
            Err(chain_any) => chain_any,
        };
        let chain_any = match chain_any.downcast::<CNTLadderChain>() {
            Ok(chain) => {
                return Ok(chain.into_bound_py_any(py)?);
            }
            Err(chain_any) => chain_any,
        };
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{:?} is not in the explicit list for markov chains.",
            chain_any
        )))
    }
}
