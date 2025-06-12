use super::*;
use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use numpy::ndarray::{Array2, ArrayView2};

/// Finds the largest size droplet at each time in the simulation.
///
/// Arguments:
///  - initial_state: The initial conditions of the simulation.
///  - boundary: The boundary conditions of the simulation.
///  - reactions: The reactions that took place during the simulation.
///  - is_droplet: A list of the states which should be tracked as particles
///    that can form a droplet.
///
/// Returns sizes, where each entry is the largest size at that sim step.
#[pyfunction]
#[pyo3(name = "largest_droplet_size_over_time")]
pub fn py_largest_droplet_size_over_time(
    initial_state: &Bound<'_, PyArray2<u32>>,
    boundary: &Bound<'_, PyAny>,
    reactions: &Bound<'_, PyList>,
    counts_as_droplet: Vec<u32>,
) -> PyResult<Vec<usize>> {
    let mut state = initial_state.to_owned_array();
    let boundary = crate::boundary_condition::extract(boundary)?;
    let (mut droplets, mut sizes) = initialize(&state.view(), &boundary, &counts_as_droplet);
    for reaction in reactions {
        let reaction = crate::reaction::extract(&reaction)?;
        advance_one_step(
            &mut state,
            &boundary,
            &mut droplets,
            &mut sizes,
            &counts_as_droplet,
            &reaction,
        );
    }
    Ok(sizes)
}

/// Finds the largest size droplet at each time in the simulation.
///
/// Arguments:
///  - state: The initial conditions of the simulation.
///  - boundary: The boundary conditions of the simulation.
///  - reactions: The reactions that took place during the simulation.
///  - is_droplet: A Vec of the states which should be tracked as particles
///    that can form a droplet.
///
/// Returns sizes, where each entry is the largest size at that sim step.
pub fn largest_droplet_size_over_time<T: Clone + PartialEq>(
    initial_state: &ArrayView2<T>,
    boundary: &Box<dyn BoundaryCondition<T>>,
    reactions: &Vec<Box<dyn Reaction<T>>>,
    counts_as_droplet: Vec<T>,
) -> Vec<usize> {
    let mut state = initial_state.to_owned();
    let (mut droplets, mut sizes) = initialize(&state.view(), &boundary, &counts_as_droplet);
    for reaction in reactions {
        advance_one_step(
            &mut state,
            &boundary,
            &mut droplets,
            &mut sizes,
            &counts_as_droplet,
            &reaction,
        );
    }
    sizes
}

fn initialize<T: Clone + PartialEq>(
    state: &ArrayView2<T>,
    boundary: &Box<dyn BoundaryCondition<T>>,
    counts_as_droplet: &Vec<T>,
) -> (Droplets, Vec<usize>) {
    let mut sizes = Vec::new();
    let droplets = Droplets::new(&state.view(), &boundary, &counts_as_droplet);
    sizes.push(
        droplets
            .droplets
            .iter()
            .fold(0, |max, droplet| max.max(droplet.len())),
    );
    (droplets, sizes)
}

fn advance_one_step<T: Clone + PartialEq>(
    state: &mut Array2<T>,
    boundary: &Box<dyn BoundaryCondition<T>>,
    droplets: &mut Droplets,
    sizes: &mut Vec<usize>,
    counts_as_droplet: &Vec<T>,
    reaction: &Box<dyn Reaction<T>>,
) {
    reaction.apply(state);
    droplets.update(&state.view(), &boundary, &counts_as_droplet, reaction);
    let size = droplets
        .droplets
        .iter()
        .fold(0, |max, droplet| max.max(droplet.len()));
    sizes.push(size);
}

#[cfg(test)]
mod tests {
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
            BR::point_change(0_u32, 1, [5, 1]),
            BR::point_change(1, 0, [5, 2]),
            BR::point_change(1, 0, [5, 1]),
            BR::point_change(0, 1, [0, 2]),
            BR::point_change(0, 1, [2, 0]),
            BR::point_change(1, 0, [3, 4]),
            BR::point_change(0, 1, [0, 1]),
            BR::point_change(1, 0, [1, 1]),
        ];
        let sizes = largest_droplet_size_over_time(
            &initial_state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &reactions
                .iter()
                .map(|reaction| Box::new(*reaction) as Box<dyn Reaction<u32>>)
                .collect(),
            vec![1],
        );

        let expected_sizes = vec![6, 6, 6, 6, 7, 13, 10, 11, 10];
        assert_eq!(expected_sizes, sizes);
    }
}
