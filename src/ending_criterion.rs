use crate::markov_chain::MarkovChain;
use crate::reaction::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub trait EndingCriterion<T, R: Reaction<T>>
where
    T: Clone,
{
    fn should_end(&self) -> bool;
    fn initialize(&mut self, system: &Array2<T>, chain: &impl MarkovChain<T, R>);
    fn update(
        &mut self,
        system: &Array2<T>,
        chain: &impl MarkovChain<T, R>,
        delta_t: f64,
        reaction: R,
    );
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct ReactionCount {
    pub threshold: usize,
    count: usize,
}

impl ReactionCount {
    pub fn new(threshold: usize) -> Self {
        ReactionCount {
            threshold,
            count: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

impl<T: Clone, R: Reaction<T>> EndingCriterion<T, R> for ReactionCount {
    fn should_end(&self) -> bool {
        self.count >= self.threshold
    }

    fn initialize(&mut self, _system: &Array2<T>, _chain: &impl MarkovChain<T, R>) {
        self.count = 0;
    }

    fn update(
        &mut self,
        _system: &Array2<T>,
        _chain: &impl MarkovChain<T, R>,
        _delta_t: f64,
        _reaction: R,
    ) {
        self.count += 1;
    }
}

// pub struct ConcentrationEndingCriterion {
//     pub threshold:
// }
