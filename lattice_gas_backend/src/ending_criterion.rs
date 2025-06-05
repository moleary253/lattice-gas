use crate::boundary_condition::BoundaryCondition;
use crate::markov_chain::MarkovChain;
use crate::reaction::*;
use numpy::ndarray::ArrayView2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

pub trait EndingCriterion<T, R: Reaction<T>>
where
    T: Clone,
{
    fn should_end(&self) -> bool;
    fn initialize(
        &mut self,
        system: &ArrayView2<T>,
        chain: &Box<dyn MarkovChain<T, R>>,
        boundary: &Box<dyn BoundaryCondition<T>>,
    );
    fn update(
        &mut self,
        system: &ArrayView2<T>,
        chain: &Box<dyn MarkovChain<T, R>>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        delta_t: f64,
        reaction: R,
    );
}

mod reaction_count;
pub use reaction_count::*;

mod largest_droplet_size;
pub use largest_droplet_size::*;

mod target_state;
pub use target_state::*;

mod particle_count;
pub use particle_count::*;

pub fn extract(
    py_ending_criterion: &Bound<'_, PyAny>,
) -> PyResult<Box<dyn EndingCriterion<u32, crate::reaction::BasicReaction<u32>> + Send + 'static>> {
    if let Ok(ending_criterion) = py_ending_criterion.extract::<ReactionCount>() {
        return Ok(Box::new(ending_criterion));
    }
    if let Ok(ending_criterion) = py_ending_criterion.extract::<LargestDropletSize>() {
        return Ok(Box::new(ending_criterion));
    }
    if let Ok(ending_criterion) = py_ending_criterion.extract::<TargetState>() {
        return Ok(Box::new(ending_criterion));
    }
    if let Ok(ending_criterion) = py_ending_criterion.extract::<ParticleCount>() {
        return Ok(Box::new(ending_criterion));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} is not an EndingCriterion.",
        py_ending_criterion
    )))
}
