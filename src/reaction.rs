use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub trait Reaction<T> {
    fn apply(&self, state: &mut Array2<T>);
    fn indicies_updated(&self, state: &Array2<T>) -> Vec<[usize; 2]>;
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

    fn indicies_updated(&self, _state: &Array2<T>) -> Vec<[usize; 2]> {
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
