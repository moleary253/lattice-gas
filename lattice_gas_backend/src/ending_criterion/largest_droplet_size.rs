use super::*;
use crate::analysis::Droplets;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct LargestDropletSize {
    pub threshold: usize,
    pub droplets: Option<Droplets>,
    pub counts_as_droplet: Vec<u32>,
}

#[pymethods]
impl LargestDropletSize {
    #[new]
    pub fn new(threshold: usize, counts_as_droplet: Vec<u32>) -> Self {
        LargestDropletSize {
            threshold,
            droplets: None,
            counts_as_droplet,
        }
    }
}

#[typetag::serde]
impl EndingCriterion for LargestDropletSize {
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
        state: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        boundary: &Box<dyn BoundaryCondition>,
    ) {
        self.droplets = Some(Droplets::new(
            &state.view(),
            boundary,
            &self.counts_as_droplet,
        ));
    }

    fn update(
        &mut self,
        state: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        boundary: &Box<dyn BoundaryCondition>,
        _delta_t: f64,
        reaction: BasicReaction<u32>,
    ) {
        let reaction: Box<dyn Reaction<u32>> = Box::new(reaction);
        self.droplets
            .as_mut()
            .expect("Should be initialized")
            .update(&state.view(), boundary, &self.counts_as_droplet, &reaction);
    }
}
