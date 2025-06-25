use super::*;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
enum ComparisonMode {
    Above,
    Equal,
    Below,
}

impl ComparisonMode {
    fn compare<T: PartialOrd>(&self, left: T, right: T) -> bool {
        match self {
            ComparisonMode::Above => left <= right,
            ComparisonMode::Below => right <= left,
            ComparisonMode::Equal => left == right,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct ParticleCount {
    pub threshold: usize,
    count: usize,
    counts_as_particle: Vec<u32>,
    mode: ComparisonMode,
}

#[pymethods]
impl ParticleCount {
    #[new]
    pub fn new(threshold: usize, counts_as_particle: Vec<u32>, mode: &str) -> Self {
        ParticleCount {
            threshold,
            count: 0,
            counts_as_particle,
            mode: match mode {
                "above" => ComparisonMode::Above,
                "below" => ComparisonMode::Below,
                _ => ComparisonMode::Equal,
            },
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

#[typetag::serde]
impl EndingCriterion for ParticleCount {
    fn should_end(&self) -> bool {
        self.mode.compare(self.threshold, self.count)
    }

    fn initialize(
        &mut self,
        system: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        _boundary: &Box<dyn BoundaryCondition>,
    ) {
        self.count = 0;
        for site in system {
            if self.counts_as_particle.contains(site) {
                self.count += 1;
            }
        }
    }

    fn update(
        &mut self,
        _system: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        _boundary: &Box<dyn BoundaryCondition>,
        _delta_t: f64,
        reaction: BasicReaction<u32>,
    ) {
        match reaction {
            BasicReaction::Diffusion { from: _, to: _ } => {}
            BasicReaction::PointChange {
                from,
                to,
                position: _,
            } => {
                if self.counts_as_particle.contains(&to) {
                    self.count += 1;
                }
                if self.counts_as_particle.contains(&from) {
                    self.count -= 1;
                }
            }
        }
    }
}
