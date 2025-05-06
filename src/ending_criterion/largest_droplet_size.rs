use super::*;
use crate::analysis::Droplets;
use std::marker::PhantomData;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LargestDropletSize<T: Clone, F: Fn(&T) -> bool> {
    pub threshold: usize,
    pub droplets: Option<Droplets>,
    pub is_droplet: F,
    phantom: PhantomData<T>,
}

impl<T: Clone, F: Fn(&T) -> bool> LargestDropletSize<T, F> {
    pub fn new(threshold: usize, is_droplet: F) -> Self {
        LargestDropletSize {
            threshold,
            droplets: None,
            is_droplet,
            phantom: PhantomData,
        }
    }
}

impl<T: Clone, R: Reaction<T>, F: Fn(&T) -> bool> EndingCriterion<T, R>
    for LargestDropletSize<T, F>
{
    fn should_end(&self) -> bool {
        self.droplets
            .as_ref()
            .expect("Should be initialized")
            .droplets
            .iter()
            .fold(0, |max, droplet| max.max(droplet.len()))
            >= self.threshold
    }

    fn initialize(
        &mut self,
        state: &Array2<T>,
        _chain: &impl MarkovChain<T, R>,
        boundary: &impl BoundaryCondition<T>,
    ) {
        self.droplets = Some(Droplets::new(state, boundary, &self.is_droplet));
    }

    fn update(
        &mut self,
        state: &Array2<T>,
        _chain: &impl MarkovChain<T, R>,
        boundary: &impl BoundaryCondition<T>,
        _delta_t: f64,
        reaction: R,
    ) {
        self.droplets
            .as_mut()
            .expect("Should be initialized")
            .update(state, boundary, &self.is_droplet, &reaction);
    }
}
