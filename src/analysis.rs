use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use ndarray::Array2;

mod droplets;
pub use droplets::*;

/// Finds the largest size droplet at each time in the simulation.
///
/// Arguments:
///  - initial_state: The initial conditions of the simulation.
///  - boundary: The boundary conditions of the simulation.
///  - reactions: The reactions that took place during the simulation.
///  - is_droplet: Should return true for particles that can be part of a
///    droplet, and false if they can't be part of a droplet.
///
/// Returns sizes. Each entry of sizes is a tuple, with the first element being
/// the time, and the second beind the size of the largest droplet at that
/// time.
pub fn largest_droplet_size_over_time<T: Clone, R: Reaction<T>>(
    initial_state: &Array2<T>,
    boundary: &impl BoundaryCondition<T>,
    reactions: &Vec<(f64, R)>,
    is_droplet: impl Fn(&T) -> bool,
) -> Vec<(f64, usize)> {
    let mut sizes = Vec::new();
    let mut state = initial_state.clone();
    let mut droplets = Droplets::new(&state, boundary, &is_droplet);
    sizes.push((
        0.0,
        droplets
            .droplets
            .iter()
            .fold(0, |max, droplet| max.max(droplet.len())),
    ));
    let mut t = 0.0;
    for (dt, reaction) in reactions {
        reaction.apply(&mut state);
        t += dt;
        droplets.update(&state, boundary, &is_droplet, reaction);
        let size = droplets
            .droplets
            .iter()
            .fold(0, |max, droplet| max.max(droplet.len()));
        if size != sizes.last().unwrap().1 {
            sizes.push((t, size));
        }
    }
    sizes
}

#[cfg(test)]
pub mod tests {
    use crate::boundary_condition;

    use super::*;
    use ndarray::arr2;

    #[test]
    fn largest_droplet_size_over_time_works() {
        use crate::reaction::BasicReaction as BR;
        let initial_state = arr2(&[
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let reactions = vec![
            (1.0, BR::point_change(0, 1, [5, 1])),
            (1.0, BR::point_change(1, 0, [5, 2])),
            (1.0, BR::point_change(1, 0, [5, 1])),
            (1.0, BR::point_change(0, 1, [0, 2])),
            (1.0, BR::point_change(0, 1, [2, 0])),
            (1.0, BR::point_change(1, 0, [3, 4])),
            (1.0, BR::point_change(0, 1, [0, 1])),
            (1.0, BR::point_change(1, 0, [1, 1])),
        ];
        let sizes = largest_droplet_size_over_time(&initial_state, &boundary, &reactions, |site| {
            *site == 1
        });

        let expected_sizes = vec![
            (0.0, 6),
            (4.0, 7),
            (5.0, 13),
            (6.0, 10),
            (7.0, 11),
            (8.0, 10),
        ];
        assert_eq!(sizes.len(), expected_sizes.len());
        for (size, expected) in sizes.into_iter().zip(expected_sizes) {
            assert_eq!(size, expected);
        }
    }
}
