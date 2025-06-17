use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};
use serde::{Deserialize, Serialize};

pub trait Reaction<T> {
    fn apply(&self, state: &mut Array2<T>);
    fn indicies_updated(&self) -> Vec<[usize; 2]>;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, IntoPyObject, FromPyObject)]
pub enum BasicReaction<T>
where
    T: Clone,
{
    Diffusion {
        #[pyo3(item)]
        from: [usize; 2],
        #[pyo3(item)]
        to: [usize; 2],
    },
    PointChange {
        #[pyo3(item)]
        from: T,
        #[pyo3(item)]
        to: T,
        #[pyo3(item)]
        position: [usize; 2],
    },
}

impl<T> BasicReaction<T>
where
    T: Clone,
{
    pub fn point_change(from: T, to: T, position: [usize; 2]) -> Self {
        BasicReaction::PointChange { from, to, position }
    }

    pub fn diffusion(to: [usize; 2], from: [usize; 2]) -> Self {
        BasicReaction::Diffusion { from, to }
    }
}

impl<T: Clone> Reaction<T> for BasicReaction<T> {
    fn apply(&self, state: &mut Array2<T>) {
        match self.clone() {
            BasicReaction::PointChange {
                from: _,
                to,
                position,
            } => state[position] = to,
            BasicReaction::Diffusion { from, to } => {
                (state[from], state[to]) = (state[to].clone(), state[from].clone());
            }
        }
    }

    fn indicies_updated(&self) -> Vec<[usize; 2]> {
        match self.clone() {
            BasicReaction::PointChange {
                from: _,
                to: _,
                position,
            } => vec![position],
            BasicReaction::Diffusion { from, to } => {
                vec![from, to]
            }
        }
    }
}

pub fn extract(py_reaction: &Bound<'_, PyAny>) -> PyResult<Box<dyn Reaction<u32>>> {
    if let Ok(reaction) = py_reaction.extract::<BasicReaction<u32>>() {
        return Ok(Box::new(reaction));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} is not a reaction.",
        py_reaction
    )))
}

pub fn extract_times(reactions: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
    let mut delta_times: Vec<f64> = Vec::with_capacity(reactions.len());
    for reaction in reactions.iter() {
        delta_times.push(
            reaction
                .downcast::<PyDict>()?
                .get_item("dt")?
                .unwrap()
                .downcast::<PyFloat>()?
                .extract()?,
        );
    }
    Ok(delta_times)
}
