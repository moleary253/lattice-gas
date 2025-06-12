use super::*;
use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub fn commitance_counts<T: Clone + PartialEq>(
    mut state: Array2<T>,
    boundary: &Box<dyn BoundaryCondition<T>>,
    counts_as_droplet: &Vec<T>,
    reactions: &Vec<Box<dyn Reaction<T>>>,
    bottom_absorbing_size: usize,
    top_absorbing_size: usize,
) -> (Array1<usize>, Array1<usize>) {
    let mut droplets = Droplets::new(&state.view(), &boundary, &counts_as_droplet);
    let mut sizes = Vec::with_capacity(droplets.droplets.len());
    let mut tracked_sizes = HashMap::new();
    for (i, droplet) in droplets.droplets.iter().enumerate() {
        let id = i + 1;
        sizes.push(droplet.len());
        if droplet.len() > bottom_absorbing_size && droplet.len() < top_absorbing_size {
            tracked_sizes.insert(id, vec![droplet.len()]);
        }
    }

    let mut times_seen = Array1::zeros(top_absorbing_size - 1 - bottom_absorbing_size);
    let mut times_succeeded = Array1::zeros(top_absorbing_size - 1 - bottom_absorbing_size);

    for reaction in reactions {
        reaction.apply(&mut state);
        let droplet_reactions =
            droplets.update(&state.view(), &boundary, &counts_as_droplet, &reaction);

        update_from_reactions(
            &mut tracked_sizes,
            &mut times_seen,
            bottom_absorbing_size,
            droplet_reactions,
        );

        update_from_sizes(
            &mut tracked_sizes,
            &mut times_seen,
            &mut times_succeeded,
            droplets.droplets.iter().map(|droplet| droplet.len()),
            bottom_absorbing_size,
            top_absorbing_size,
        );
    }

    (times_succeeded, times_seen)
}

/// Updates tracking from merge, id change, evaporation, or separation events.
///
/// Does not update tracking based on current size information. This function is
/// meant to switch the ids of the tracked droplets only, or record a droplet's
/// history as seen in the case that `bottom_absorbing_size == 0` and the
/// droplet evaporated.
fn update_from_reactions(
    tracked_sizes: &mut HashMap<usize, Vec<usize>>,
    times_seen: &mut Array1<usize>,
    bottom_absorbing_size: usize,
    droplet_reactions: Vec<DropletReaction>,
) {
    use droplets::DropletReaction as DR;

    for droplet_reaction in droplet_reactions {
        match droplet_reaction {
            DR::Merged { id, old_id } => {
                if let Some(old_sizes) = tracked_sizes.remove(&old_id) {
                    tracked_sizes
                        .entry(id)
                        .or_insert(Vec::new())
                        .extend(old_sizes);
                }
            }
            DR::Separated {
                new_id,
                old_id,
                new_size: _,
            } => {
                if tracked_sizes.contains_key(&old_id) {
                    tracked_sizes.insert(new_id, tracked_sizes[&old_id].clone());
                }
            }
            DR::ChangedId { old_id, new_id } => {
                if let Some(old_sizes) = tracked_sizes.remove(&old_id) {
                    tracked_sizes.insert(new_id, old_sizes);
                }
            }
            DR::Evaporated { id } => {
                if let Some(old_sizes) = tracked_sizes.remove(&id) {
                    for size in old_sizes {
                        times_seen[size - bottom_absorbing_size - 1] += 1;
                    }
                }
            }
            _ => {}
        }
    }
}

fn update_from_sizes(
    tracked_sizes: &mut HashMap<usize, Vec<usize>>,
    times_seen: &mut Array1<usize>,
    times_succeeded: &mut Array1<usize>,
    sizes: impl Iterator<Item = usize>,
    bottom_absorbing_size: usize,
    top_absorbing_size: usize,
) {
    for (i, size) in sizes.enumerate() {
        let id = i + 1;
        let tracked = tracked_sizes.contains_key(&id);
        if !tracked && size > bottom_absorbing_size {
            tracked_sizes.insert(id, vec![size]);
            continue;
        }
        if !tracked {
            continue;
        }
        if size > top_absorbing_size {
            for size in tracked_sizes.remove(&id).unwrap() {
                times_seen[size - bottom_absorbing_size - 1] += 1;
                times_succeeded[size - bottom_absorbing_size - 1] += 1;
            }
            continue;
        }
        if size < bottom_absorbing_size {
            for size in tracked_sizes.remove(&id).unwrap() {
                times_seen[size - bottom_absorbing_size - 1] += 1;
            }
            continue;
        }
        tracked_sizes.get_mut(&id).unwrap().push(size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use droplets::DropletReaction as DR;

    #[test]
    fn update_from_reactions_works() {
        let mut tracked_sizes = HashMap::new();
        tracked_sizes.insert(1, vec![1]);
        tracked_sizes.insert(2, vec![2]);
        tracked_sizes.insert(3, vec![3]);
        tracked_sizes.insert(4, vec![4]);
        tracked_sizes.insert(5, vec![5]);

        let droplet_reactions = vec![
            DR::Evaporated { id: 1 },
            DR::Merged { id: 2, old_id: 3 },
            DR::ChangedId {
                old_id: 5,
                new_id: 3,
            },
            DR::Separated {
                old_id: 3,
                new_id: 5,
                new_size: 2,
            },
            DR::Merged { id: 4, old_id: 6 },
            DR::Merged { id: 7, old_id: 5 },
        ];

        let bottom_absorbing_size = 0;
        let mut times_seen = Array1::zeros(9);

        update_from_reactions(
            &mut tracked_sizes,
            &mut times_seen,
            bottom_absorbing_size,
            droplet_reactions,
        );

        let mut expected_tracked = HashMap::new();
        expected_tracked.insert(2, vec![2, 3]);
        expected_tracked.insert(3, vec![5]);
        expected_tracked.insert(4, vec![4]);
        expected_tracked.insert(7, vec![5]);

        assert_eq!(tracked_sizes, expected_tracked);

        let mut expected_times_seen = Array1::zeros(9);
        expected_times_seen[0] = 1;

        assert_eq!(times_seen, expected_times_seen);
    }
}
