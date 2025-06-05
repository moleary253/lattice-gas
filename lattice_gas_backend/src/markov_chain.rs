use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use numpy::ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;

// TODO(Myles): Add documentation
pub trait MarkovChain<T, R: Reaction<T>>
where
    T: Clone,
{
    fn num_possible_reactions(&self, state: &Array2<T>) -> usize;
    fn rate(
        &self,
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        reaction_id: usize,
    ) -> f64;
    fn on_reaction(
        &mut self,
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        reaction_id: usize,
        dt: f64,
    );
    fn initialize(&mut self, state: &ArrayView2<T>, boundary: &Box<dyn BoundaryCondition<T>>);
    fn indicies_affecting_reaction(
        &mut self,
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]>;
    fn reaction(
        &self,
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        reaction_id: usize,
    ) -> R;
}

mod homogenous_chain;
pub use homogenous_chain::*;
mod ising_chain;
pub use ising_chain::*;
mod homogenous_nvt_chain;
pub use homogenous_nvt_chain::*;
mod cnt_ladder_chain;
pub use cnt_ladder_chain::*;

pub fn extract(
    py_chain: &Bound<'_, PyAny>,
) -> PyResult<Box<dyn MarkovChain<u32, crate::reaction::BasicReaction<u32>>>> {
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
