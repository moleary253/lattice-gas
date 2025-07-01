use super::*;
use crate::reaction::BasicReaction;
use serde::{Deserialize, Serialize};

/// Describes an NVT system with bonding and inert particles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct HomogenousNVTChain {
    #[pyo3(get)]
    pub beta: f64,
    #[pyo3(get)]
    pub bond_energy: f64,
    #[pyo3(get)]
    pub inert_to_bonding_rate: f64,
    #[pyo3(get)]
    pub bonding_to_inert_rate: f64,
    #[pyo3(get)]
    pub diffusion_constant: f64,
    sum_num_adjacent: Option<Vec<usize>>,
}

#[pymethods]
impl HomogenousNVTChain {
    #[new]
    pub fn new(
        beta: f64,
        bond_energy: f64,
        inert_to_bonding_rate: f64,
        bonding_to_inert_rate: f64,
        diffusion_constant: f64,
    ) -> Self {
        HomogenousNVTChain {
            beta,
            bond_energy,
            inert_to_bonding_rate,
            bonding_to_inert_rate,
            diffusion_constant,
            sum_num_adjacent: None,
        }
    }
}

impl HomogenousNVTChain {
    pub fn site_energy(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        position: [usize; 2],
    ) -> f64 {
        let mut energy = 0.;
        for adjacent_state in boundary.adjacent(&state, position) {
            if adjacent_state == 2 {
                energy += self.bond_energy;
            }
        }
        energy
    }

    pub fn is_possible(&self, state: &ArrayView2<u32>, reaction: BasicReaction<u32>) -> bool {
        match reaction {
            BasicReaction::PointChange {
                from,
                to: _to,
                position,
            } => state[position] == from,
            BasicReaction::Diffusion { from, to } => state[from] != state[to],
        }
    }
}

#[typetag::serde]
impl MarkovChain for HomogenousNVTChain {
    fn initialize(&mut self, state: &ArrayView2<u32>, boundary: &Box<dyn BoundaryCondition>) {
        let mut sum_num_adjacent = Vec::with_capacity(state.len());
        for i in 0..state.shape()[0] {
            for j in 0..state.shape()[1] {
                sum_num_adjacent.push(
                    sum_num_adjacent.last().unwrap_or(&0)
                        + boundary
                            .adjacent_indicies(&state, [i, j])
                            .into_iter()
                            .filter(|pos| pos[0] >= i && pos[1] >= j)
                            .count(),
                );
            }
        }
        self.sum_num_adjacent = Some(sum_num_adjacent);
    }

    fn on_reaction(
        &mut self,
        _state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition>,
        _reaction_id: usize,
        _dt: f64,
    ) {
    }

    fn num_possible_reactions(&self, state: &Array2<u32>) -> usize {
        state.len() * 2
            + self
                .sum_num_adjacent
                .as_ref()
                .expect("is initialized")
                .last()
                .unwrap()
    }

    fn reaction(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> BasicReaction<u32> {
        if reaction_id < state.len() * 2 {
            let (from, to) = if reaction_id % 2 == 1 {
                (INERT, BONDING)
            } else {
                (BONDING, INERT)
            };
            let position_index = reaction_id / 2;
            let position = [
                position_index / state.shape()[1],
                position_index % state.shape()[1],
            ];
            return BasicReaction::PointChange { from, to, position };
        }

        let position_id = reaction_id - state.len() * 2;

        let position_index = self
            .sum_num_adjacent
            .as_ref()
            .expect("is_initialized")
            .partition_point(|partial_sum| *partial_sum <= position_id);

        let from = [
            position_index / state.shape()[1],
            position_index % state.shape()[1],
        ];
        let adjacent_sites = boundary
            .adjacent_indicies(&state, from)
            .into_iter()
            .filter(|pos| pos[0] >= from[0] && pos[1] >= from[1])
            .collect::<Vec<[usize; 2]>>();

        let direction_id = reaction_id
            - state.len() * 2
            - self
                .sum_num_adjacent
                .as_ref()
                .expect("is_initialized")
                .get(position_index - 1)
                .unwrap_or(&0);
        let to = adjacent_sites[direction_id];
        return BasicReaction::Diffusion { from, to };
    }

    fn rate(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> f64 {
        use BasicReaction as BR;
        let reaction = self.reaction(state, boundary, reaction_id);
        if !self.is_possible(state, reaction) {
            return 0.0;
        }
        match reaction {
            BR::PointChange {
                from: INERT,
                to: BONDING,
                position: _,
            } => self.inert_to_bonding_rate,
            BR::PointChange {
                from: BONDING,
                to: INERT,
                position,
            } => {
                self.bonding_to_inert_rate
                    * (self.beta * self.site_energy(state, boundary, position)).exp()
            }
            BR::Diffusion { from, to } => {
                let delta_energy = if state[from] == BONDING {
                    self.site_energy(state, boundary, from)
                } else if state[to] == BONDING {
                    self.site_energy(state, boundary, to)
                } else {
                    0.0
                };
                self.diffusion_constant * (self.beta * delta_energy).exp()
            }
            _ => panic!("Unexpected reaction {:?}", reaction),
        }
    }

    fn indicies_affecting_reaction(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]> {
        let reaction = self.reaction(state, boundary, reaction_id);
        match reaction {
            BasicReaction::PointChange {
                from: _,
                to: _,
                position,
            } => {
                let mut reactions = boundary.adjacent_indicies(&state, position);
                reactions.push(position);
                reactions
            }
            BasicReaction::Diffusion { from, to } => {
                let mut reactions = boundary.adjacent_indicies(&state, from);
                reactions.extend(boundary.adjacent_indicies(&state, to));
                reactions
            }
        }
    }
}
