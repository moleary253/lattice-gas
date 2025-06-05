use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

pub trait Reaction<T> {
    fn apply(&self, state: &mut Array2<T>);
    fn indicies_updated(&self) -> Vec<[usize; 2]>;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BasicReaction<T>
where
    T: Clone,
{
    Diffusion {
        from: [usize; 2],
        to: [usize; 2],
    },
    PointChange {
        from: T,
        to: T,
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

impl FromPyObject<'_> for BasicReaction<u32> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = ob.downcast::<PyDict>() {
            match dict
                .get_item("type")?
                .map(|value| value.extract::<String>())
                .transpose()?
                .as_ref()
                .map(|value| value.as_str())
            {
                Some("Diffusion") => {
                    return Ok(Self::diffusion(
                        dict.get_item("from")?
                            .expect("Diffusion reaction should have 'from' key")
                            .extract()?,
                        dict.get_item("to")?
                            .expect("Diffusion reaction should have 'to' key")
                            .extract()?,
                    ));
                }
                Some("PointChange") => {
                    return Ok(Self::point_change(
                        dict.get_item("from")?
                            .expect("PointChange reaction should have 'from' key.")
                            .extract()?,
                        dict.get_item("to")?
                            .expect("PointChange reaction should have 'to' key.")
                            .extract()?,
                        dict.get_item("position")?
                            .expect("PointChange reaction should have 'position' key.")
                            .extract()?,
                    ));
                }
                _ => {}
            }
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "{} is not a BasicReaction.",
            ob
        )))
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
