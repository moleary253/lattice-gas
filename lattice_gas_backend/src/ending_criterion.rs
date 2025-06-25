use crate::boundary_condition::BoundaryCondition;
use crate::markov_chain::MarkovChain;
use crate::reaction::*;

use numpy::ndarray::ArrayView2;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;

use serde::{Deserialize, Serialize};

use std::any::Any;

#[typetag::serde(tag = "type")]
pub trait EndingCriterion: Any {
    fn should_end(&self) -> bool;
    fn initialize(
        &mut self,
        system: &ArrayView2<u32>,
        chain: &Box<dyn MarkovChain>,
        boundary: &Box<dyn BoundaryCondition>,
    );
    fn update(
        &mut self,
        system: &ArrayView2<u32>,
        chain: &Box<dyn MarkovChain>,
        boundary: &Box<dyn BoundaryCondition>,
        delta_t: f64,
        reaction: BasicReaction<u32>,
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

impl<'py> FromPyObject<'py> for Box<dyn EndingCriterion> {
    fn extract_bound(
        py_ending_criterion: &Bound<'py, PyAny>,
    ) -> PyResult<Box<dyn EndingCriterion>> {
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
}

impl<'py> IntoPyObject<'py> for Box<dyn EndingCriterion> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let criterion_any = self as Box<dyn Any>;
        let criterion_any = match criterion_any.downcast::<ReactionCount>() {
            Ok(criterion) => {
                return Ok(criterion.into_bound_py_any(py)?);
            }
            Err(criterion_any) => criterion_any,
        };
        let criterion_any = match criterion_any.downcast::<LargestDropletSize>() {
            Ok(criterion) => {
                return Ok(criterion.into_bound_py_any(py)?);
            }
            Err(criterion_any) => criterion_any,
        };
        let criterion_any = match criterion_any.downcast::<TargetState>() {
            Ok(criterion) => {
                return Ok(criterion.into_bound_py_any(py)?);
            }
            Err(criterion_any) => criterion_any,
        };
        let criterion_any = match criterion_any.downcast::<ParticleCount>() {
            Ok(criterion) => {
                return Ok(criterion.into_bound_py_any(py)?);
            }
            Err(criterion_any) => criterion_any,
        };
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{:?} is not in the explicit list for ending criteria.",
            criterion_any
        )))
    }
}
