use super::*;
use crate::reaction::BasicReaction;
use serde::{Deserialize, Serialize};

/// Describes a homogenous bonding to inert reaction as described in Yeongik's PRL in 2023.
/// Sets $D$ to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct IsingChain {
    pub beta: f64,
    pub bond_energy: f64,
    pub magnetic_field: f64,
}

#[pymethods]
impl IsingChain {
    #[new]
    pub fn new(beta: f64, bond_energy: f64, magnetic_field: f64) -> Self {
        IsingChain {
            beta,
            bond_energy,
            magnetic_field,
        }
    }
}

impl IsingChain {
    pub fn site_energy(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        position: [usize; 2],
    ) -> f64 {
        let mut energy = 0.;
        for adjacent_state in boundary.adjacent(&state, position) {
            if adjacent_state == BONDING {
                energy += self.bond_energy;
            } else {
                energy -= self.bond_energy;
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
            _ => false,
        }
    }
}

#[typetag::serde]
impl MarkovChain for IsingChain {
    fn initialize(&mut self, _state: &ArrayView2<u32>, _boundary: &Box<dyn BoundaryCondition>) {}

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
    }

    fn reaction(
        &self,
        state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> BasicReaction<u32> {
        let possible_states = [EMPTY, BONDING];
        let position_index = reaction_id / 2;
        let position = [
            position_index / state.shape()[1],
            position_index % state.shape()[1],
        ];
        let from = possible_states[reaction_id % 2];
        let to = possible_states[(reaction_id + 1) % 2];
        BasicReaction::PointChange { from, to, position }
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
                from: EMPTY,
                to: BONDING,
                position,
            } => (self.beta
                * (-self.site_energy(state, boundary, position) + 4.0 * self.bond_energy)
                + self.magnetic_field * self.beta)
                .exp(),
            BR::PointChange {
                from: BONDING,
                to: EMPTY,
                position,
            } => (self.beta
                * (self.site_energy(state, boundary, position) + 4.0 * self.bond_energy)
                - self.magnetic_field * self.beta)
                .exp(),
            _ => panic!("Unexpected reaction {:?}", reaction),
        }
    }

    fn indicies_affecting_reaction(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]> {
        let position = if let BasicReaction::PointChange {
            from: _,
            to: _,
            position,
        } = self.reaction(state, boundary, reaction_id)
        {
            position
        } else {
            panic!(
                "Unexpected reaction {:?}",
                self.reaction(state, boundary, reaction_id)
            )
        };
        let mut reactions = boundary.adjacent_indicies(&state, position);
        reactions.push(position);
        reactions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction::BasicReaction as BR;
    use ndarray::arr2;

    #[test]
    fn detailed_balance() {
        let state = arr2(&[
            [0, 2, 0, 0, 0],
            [2, 0, 2, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [2, 2, 2, 2, 0],
        ]);

        let chain = IsingChain::new(1.0, -5.0, 7.0);

        let boundary = crate::boundary_condition::Periodic;

        let boundary_box = Box::new(boundary) as Box<dyn BoundaryCondition>;

        let reactions = vec![
            (
                BR::point_change(0, 2, [0, 4]),
                BR::point_change(2, 0, [1, 0]),
            ),
            (
                BR::point_change(0, 2, [1, 3]),
                BR::point_change(2, 0, [2, 1]),
            ),
            (
                BR::point_change(0, 2, [3, 0]),
                BR::point_change(2, 0, [3, 3]),
            ),
            (
                BR::point_change(0, 2, [0, 0]),
                BR::point_change(2, 0, [3, 1]),
            ),
            (
                BR::point_change(0, 2, [1, 1]),
                BR::point_change(2, 0, [4, 1]),
            ),
        ];
        let energy_differences = vec![
            2.0 * 7.0 - 8.0 * 5.0,
            2.0 * 7.0 - 4.0 * 5.0,
            2.0 * 7.0 - 0.0 * 5.0,
            2.0 * 7.0 + 4.0 * 5.0,
            2.0 * 7.0 + 8.0 * 5.0,
        ];

        for (&expected_energy, &(forward_reaction, backward_reaction)) in
            energy_differences.iter().zip(reactions.iter())
        {
            let BR::PointChange {
                from: _,
                to: _,
                position: forward_position,
            } = forward_reaction
            else {
                panic!()
            };
            let forward_id = 2 * (forward_position[0] * state.shape()[1] + forward_position[1]);
            assert_eq!(
                forward_reaction,
                chain.reaction(&state.view(), &boundary_box, forward_id)
            );

            let BR::PointChange {
                from: _,
                to: _,
                position: backward_position,
            } = backward_reaction
            else {
                panic!()
            };
            let backward_id =
                2 * (backward_position[0] * state.shape()[1] + backward_position[1]) + 1;

            assert_eq!(
                backward_reaction,
                chain.reaction(&state.view(), &boundary_box, backward_id)
            );

            let energy = (chain.rate(&state.view(), &boundary_box, forward_id)
                / chain.rate(&state.view(), &boundary_box, backward_id))
            .ln();
            println!("{energy} vs {expected_energy}");
            assert!((expected_energy - energy).abs() < 0.00001);
        }
    }
}
