use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use ndarray::Array2;

// TODO(Myles): Add documentation
pub trait MarkovChain<T, R: Reaction<T>>
where
    T: Clone,
{
    fn num_possible_reactions(&self, state: &Array2<T>) -> usize;
    fn rate(
        &self,
        state: &Array2<T>,
        boundary: &impl BoundaryCondition<T>,
        reaction_id: usize,
    ) -> f64;
    fn on_reaction(
        &mut self,
        state: &Array2<T>,
        boundary: &impl BoundaryCondition<T>,
        reaction_id: usize,
        dt: f64,
    );
    fn initialize(&mut self, state: &Array2<T>, boundary: &impl BoundaryCondition<T>);
    fn indicies_affecting_reaction(
        &mut self,
        state: &Array2<T>,
        boundary: &impl BoundaryCondition<T>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]>;
    fn reaction(&self, state: &Array2<T>, reaction_id: usize) -> R;
}

mod homogenous_chain;
pub use homogenous_chain::*;
