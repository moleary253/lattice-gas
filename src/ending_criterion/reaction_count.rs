use super::*;

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

    fn initialize(
        &mut self,
        _system: &Array2<T>,
        _chain: &impl MarkovChain<T, R>,
        _boundary: &impl BoundaryCondition<T>,
    ) {
        self.count = 0;
    }

    fn update(
        &mut self,
        _system: &Array2<T>,
        _chain: &impl MarkovChain<T, R>,
        _boundary: &impl BoundaryCondition<T>,
        _delta_t: f64,
        _reaction: R,
    ) {
        self.count += 1;
    }
}
