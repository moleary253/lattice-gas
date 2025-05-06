use crate::boundary_condition::BoundaryCondition;
use crate::markov_chain::MarkovChain;
use crate::reaction::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub trait EndingCriterion<T, R: Reaction<T>>
where
    T: Clone,
{
    fn should_end(&self) -> bool;
    fn initialize(
        &mut self,
        system: &Array2<T>,
        chain: &impl MarkovChain<T, R>,
        boundary: &impl BoundaryCondition<T>,
    );
    fn update(
        &mut self,
        system: &Array2<T>,
        chain: &impl MarkovChain<T, R>,
        boundary: &impl BoundaryCondition<T>,
        delta_t: f64,
        reaction: R,
    );
}

mod reaction_count;
pub use reaction_count::*;

mod largest_droplet_size;
pub use largest_droplet_size::*;
