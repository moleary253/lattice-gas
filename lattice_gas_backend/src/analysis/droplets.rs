use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use ndarray::Array2;
use numpy::ndarray::ArrayView2;
use numpy::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Used to record a list of changes coarse-grained at the droplet level
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DropletReaction {
    /// A new droplet was formed with `id`.
    Formed { id: usize },
    /// A droplet evaporated with `id`.
    Evaporated { id: usize },
    /// Droplets `id` and `old_id` merged, and now have `id`.
    Merged { id: usize, old_id: usize },
    /// Droplet `old_id` separated into `old_id` and `new_id`. The new droplet has size `new_size`.
    Separated {
        old_id: usize,
        new_id: usize,
        new_size: usize,
    },
    /// Droplet `id` changed size by `change`.
    ChangedSize { id: usize, change: i32 },
    /// Droplet `old_id` changed id to `new_id`.
    ChangedId { old_id: usize, new_id: usize },
}

/// Keeps track of all the droplets currently in a simulation
///
/// Attributes:
///  - droplets is a Vec of Vecs. Each Vec represents one droplet, and contains
///    all the indicies of particles in that droplet
///  - labeled is an Array2 of usizes. Each position will be 0 if that index is
///    not part of a droplet, or a number if it is part of a droplet. The
///    number corresponds to the index in droplets (plus one) of the droplet
///    the particle is part of.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct Droplets {
    pub droplets: Vec<Vec<[usize; 2]>>,
    pub labeled: Array2<usize>,
}

#[pymethods]
impl Droplets {
    #[new]
    pub unsafe fn py_init(
        state: &Bound<'_, PyArray2<u32>>,
        boundary: &Bound<'_, PyAny>,
        counts_as_droplet: Vec<u32>,
    ) -> PyResult<Self> {
        Ok(Droplets::new(
            &state.as_array(),
            &crate::boundary_condition::extract(boundary)?,
            &counts_as_droplet,
        ))
    }

    #[pyo3(name = "update")]
    pub unsafe fn py_update(
        &mut self,
        state: &Bound<'_, PyArray2<u32>>,
        boundary: &Bound<'_, PyAny>,
        counts_as_droplet: Vec<u32>,
        reaction: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.update(
            &state.as_array(),
            &crate::boundary_condition::extract(boundary)?,
            &counts_as_droplet,
            &crate::reaction::extract(reaction)?,
        );
        Ok(())
    }

    pub fn droplets(&self) -> Vec<Vec<[usize; 2]>> {
        self.droplets.clone()
    }

    pub fn labeled<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
        self.labeled.clone().to_pyarray(py)
    }
}

impl Droplets {
    /// Creates a new Droplets
    pub fn new<T: Clone + PartialEq>(
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        counts_as_droplet: &Vec<T>,
    ) -> Self {
        let mut droplets = Vec::new();
        let mut labeled = Array2::default(state.raw_dim());

        for ((x, y), site) in state.indexed_iter() {
            let pos = [x, y];
            if !counts_as_droplet.contains(&site) || labeled[pos] != 0 {
                continue;
            }
            let mut droplet = Vec::new();
            let mut to_check = vec![pos];
            let mut next = Vec::new();
            while to_check.len() > 0 {
                for idx in to_check {
                    if !counts_as_droplet.contains(&state[idx]) || labeled[idx] != 0 {
                        continue;
                    }
                    droplet.push(idx);
                    labeled[idx] = droplets.len() + 1;
                    for adjacent in boundary.adjacent_indicies(state, idx) {
                        next.push(adjacent);
                    }
                }
                (to_check, next) = (next, Vec::new());
            }
            droplets.push(droplet);
        }

        Droplets { droplets, labeled }
    }

    /// Updates droplet data after a reaction occurs
    pub fn update<T: Clone + PartialEq>(
        &mut self,
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
        counts_as_droplet: &Vec<T>,
        reaction: &Box<dyn Reaction<T>>,
    ) -> Vec<DropletReaction> {
        let mut changes = Vec::new();
        let mut indicies_updated = reaction.indicies_updated();
        indicies_updated.sort_by(|pos1, pos2| self.labeled[*pos2].cmp(&self.labeled[*pos1]));
        for index in indicies_updated {
            if counts_as_droplet.contains(&state[index]) {
                changes.extend(self.update_added(index, &state, boundary));
            } else {
                changes.extend(self.update_removed(index, &state, boundary));
            }
        }
        changes
    }

    fn update_added<T: Clone>(
        &mut self,
        index: [usize; 2],
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
    ) -> Vec<DropletReaction> {
        use DropletReaction as DR;
        if self.labeled[index] != 0 {
            return Vec::new();
        }
        let mut adjacent_droplets = Vec::with_capacity(4);
        for adjacent in boundary.adjacent_indicies(&state, index.clone()) {
            if self.labeled[adjacent] == 0 || adjacent_droplets.contains(&self.labeled[adjacent]) {
                continue;
            }
            adjacent_droplets.push(self.labeled[adjacent]);
        }
        if adjacent_droplets.len() == 0 {
            self.droplets.push(vec![index]);
            self.labeled[index] = self.droplets.len();
            return vec![DR::Formed {
                id: self.droplets.len(),
            }];
        }
        if adjacent_droplets.len() == 1 {
            let droplet = adjacent_droplets[0];
            self.labeled[index] = droplet;
            self.droplets[droplet - 1].push(index);
            return vec![DR::ChangedSize {
                id: droplet,
                change: 1,
            }];
        }
        // NOTE(Myles): Need to merge droplets here
        adjacent_droplets.sort();
        adjacent_droplets.reverse();
        let smallest_droplet = *adjacent_droplets.iter().min().unwrap();
        self.labeled[index] = smallest_droplet;
        self.droplets[smallest_droplet - 1].push(index);

        let mut changes = vec![DR::ChangedSize {
            id: smallest_droplet,
            change: 1,
        }];
        for droplet in adjacent_droplets {
            if droplet == smallest_droplet {
                continue;
            }
            for index in self.droplets.swap_remove(droplet - 1) {
                self.labeled[index] = smallest_droplet;
                self.droplets[smallest_droplet - 1].push(index);
            }
            changes.push(DR::Merged {
                id: smallest_droplet,
                old_id: droplet,
            });
            if self.droplets.len() == droplet - 1 {
                continue;
            }
            changes.push(DR::ChangedId {
                old_id: self.droplets.len() + 1,
                new_id: droplet,
            });
            for index in self.droplets[droplet - 1].iter() {
                self.labeled[*index] = droplet;
            }
        }
        changes
    }

    fn update_removed<T: Clone>(
        &mut self,
        index: [usize; 2],
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
    ) -> Vec<DropletReaction> {
        use DropletReaction as DR;
        if self.labeled[index] == 0 {
            return Vec::new();
        }
        let mut adjacent_indicies = Vec::with_capacity(4);
        for adjacent in boundary.adjacent_indicies(&state, index.clone()) {
            if self.labeled[adjacent] == 0 || adjacent_indicies.contains(&adjacent) {
                continue;
            }
            adjacent_indicies.push(adjacent);
        }
        let old_droplet = self.labeled[index];
        let old_index_in_droplet = self.droplets[old_droplet - 1]
            .iter()
            .position(|idx| *idx == index)
            .unwrap();
        self.labeled[index] = 0;
        self.droplets[old_droplet - 1].swap_remove(old_index_in_droplet);
        if adjacent_indicies.len() == 1 {
            return vec![DR::ChangedSize {
                id: old_droplet,
                change: -1,
            }];
        }
        if adjacent_indicies.len() == 0 {
            self.droplets.swap_remove(old_droplet - 1);
            let mut changes = vec![DR::Evaporated { id: old_droplet }];
            if self.droplets.len() == old_droplet - 1 {
                return changes;
            }
            changes.push(DR::ChangedId {
                old_id: self.droplets.len() + 1,
                new_id: old_droplet,
            });
            for index in self.droplets[old_droplet - 1].iter() {
                self.labeled[*index] = old_droplet;
            }
            return changes;
        }
        // NOTE(Myles): Might need to split into multiple droplets here
        let mut checked = Vec::new();
        for _ in 0..adjacent_indicies.len() {
            checked.push(false);
        }
        let mut first_run = true;
        let mut changes = vec![DR::ChangedSize {
            id: old_droplet,
            change: -1,
        }];
        for (i, &pos) in adjacent_indicies.iter().enumerate() {
            if checked[i] {
                continue;
            }
            checked[i] = true;
            let mut droplet = Vec::new();
            let label = self.droplets.len() + 1;
            let mut to_check = vec![pos];
            let mut next = Vec::new();
            while to_check.len() > 0 {
                for idx in to_check {
                    if self.labeled[idx] != old_droplet {
                        continue;
                    }
                    droplet.push(idx);
                    self.labeled[idx] = label;

                    for adjacent in boundary.adjacent_indicies(state, idx) {
                        next.push(adjacent);
                    }

                    if let Some(j) = adjacent_indicies.iter().position(|adj| *adj == idx) {
                        checked[j] = true;
                    }

                    if first_run && checked.iter().all(|x| *x) {
                        next.clear();
                        break;
                    }
                }
                (to_check, next) = (next, Vec::new());
            }
            if !first_run {
                self.droplets.push(droplet);
                changes.push(DR::Separated {
                    old_id: old_droplet,
                    new_id: self.droplets.len(),
                    new_size: self.droplets.last().unwrap().len(),
                });
                continue;
            }
            first_run = false;
            if checked.iter().all(|x| *x) {
                for index in droplet.iter() {
                    self.labeled[*index] = old_droplet;
                }
                break;
            }
            self.droplets[old_droplet - 1] = droplet;
            for index in self.droplets[old_droplet - 1].iter() {
                self.labeled[*index] = old_droplet;
            }
        }
        changes
    }
}

#[cfg(test)]
pub mod tests {
    use crate::boundary_condition;

    use super::*;
    use ndarray::arr2;

    #[test]
    fn initialization() {
        let state: Array2<u32> = arr2(&[
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let droplets = Droplets::new(
            &state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &vec![1],
        );

        let expected_droplets = vec![
            vec![[0, 0], [1, 0], [1, 1], [2, 1], [0, 3], [0, 4]],
            vec![[2, 3], [2, 4], [3, 4], [4, 4], [4, 3]],
            vec![[5, 2]],
        ];

        let expected_labeled = arr2(&[
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 2, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 2, 2],
            [0, 0, 3, 0, 0],
        ]);

        assert_eq!(droplets.droplets.len(), expected_droplets.len());
        for (droplet, expected) in droplets.droplets.iter().zip(expected_droplets) {
            assert_eq!(droplet.len(), expected.len());
            for index in droplet {
                assert!(expected.contains(index));
            }
        }

        assert_eq!(expected_labeled, droplets.labeled);
    }

    #[test]
    fn registers_new_droplets() {
        use crate::reaction::BasicReaction as BR;
        let mut state: Array2<u32> = arr2(&[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let reactions = vec![
            BR::point_change(0, 1, [1, 1]),
            BR::point_change(0, 1, [3, 3]),
            BR::point_change(0, 1, [1, 2]),
            BR::point_change(0, 1, [2, 2]),
            BR::point_change(0, 1, [2, 3]),
        ];
        let mut droplets = Droplets::new(
            &state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &vec![1],
        );

        let expected_droplets = vec![
            vec![vec![[1, 1]]],
            vec![vec![[1, 1]], vec![[3, 3]]],
            vec![vec![[1, 1], [1, 2]], vec![[3, 3]]],
            vec![vec![[1, 1], [1, 2], [2, 2]], vec![[3, 3]]],
            vec![vec![[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]],
        ];
        for (reaction, expected) in reactions.iter().zip(expected_droplets) {
            reaction.apply(&mut state);
            droplets.update(
                &state.view(),
                &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
                &vec![1],
                &(Box::new(*reaction) as Box<dyn Reaction<u32>>),
            );

            assert_eq!(droplets.droplets.len(), expected.len());
            for (droplet, expected_droplet) in droplets.droplets.iter().zip(expected) {
                assert_eq!(droplet.len(), expected_droplet.len());
                for idx in droplet {
                    println!("{idx:?}");
                    assert!(expected_droplet.contains(&idx));
                }
            }
        }
    }

    #[test]
    fn can_merge_multiple_droplets() {
        use crate::reaction::BasicReaction as BR;
        let mut state = arr2(&[
            [0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let reactions = vec![
            BR::point_change(0, 1, [1, 1]),
            BR::point_change(0, 1, [1, 4]),
        ];
        let mut droplets = Droplets::new(
            &state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &vec![1],
        );

        let expected_droplets = vec![
            vec![
                vec![[0, 1], [1, 0], [1, 1], [2, 1]],
                vec![[0, 4]],
                vec![[2, 4]],
                vec![[1, 3]],
            ],
            vec![vec![
                [0, 1],
                [1, 0],
                [1, 1],
                [2, 1],
                [0, 4],
                [1, 3],
                [2, 4],
                [1, 4],
            ]],
        ];
        for (reaction, expected) in reactions.iter().zip(expected_droplets) {
            reaction.apply(&mut state);
            droplets.update(
                &state.view(),
                &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
                &vec![1],
                &(Box::new(*reaction) as Box<dyn Reaction<u32>>),
            );

            assert_eq!(droplets.droplets.len(), expected.len());
            for (droplet, expected_droplet) in droplets.droplets.iter().zip(expected) {
                assert_eq!(droplet.len(), expected_droplet.len());
                for idx in droplet {
                    assert!(expected_droplet.contains(&idx));
                }
            }
        }
    }

    #[test]
    fn droplet_reactions_work() {
        use crate::reaction::BasicReaction as BR;
        use DropletReaction as DR;
        let mut state = arr2(&[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let reactions = vec![
            BR::point_change(0, 1, [1, 1]),
            BR::point_change(0, 1, [1, 2]),
            BR::point_change(1, 0, [1, 1]),
            BR::point_change(1, 0, [3, 1]),
            BR::point_change(0, 1, [2, 3]),
            BR::point_change(1, 0, [2, 3]),
            BR::point_change(0, 1, [3, 4]),
            BR::point_change(1, 0, [3, 4]),
            BR::point_change(0, 1, [2, 5]),
            BR::point_change(1, 0, [2, 5]),
        ];
        let mut droplets = Droplets::new(
            &state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &vec![1],
        );

        let expected_drs = vec![
            vec![DR::Formed { id: 7 }],
            vec![DR::ChangedSize { id: 7, change: 1 }],
            vec![DR::ChangedSize { id: 7, change: -1 }],
            vec![
                DR::Evaporated { id: 4 },
                DR::ChangedId {
                    old_id: 7,
                    new_id: 4,
                },
            ],
            vec![
                DR::ChangedSize { id: 2, change: 1 },
                DR::Merged { id: 2, old_id: 5 },
                DR::ChangedId {
                    old_id: 6,
                    new_id: 5,
                },
            ],
            vec![
                DR::ChangedSize { id: 2, change: -1 },
                DR::Separated {
                    old_id: 2,
                    new_id: 6,
                    new_size: 1,
                },
            ],
            vec![
                DR::ChangedSize { id: 2, change: 1 },
                DR::Merged { id: 2, old_id: 6 },
                DR::Merged { id: 2, old_id: 5 },
            ],
            vec![
                DR::ChangedSize { id: 2, change: -1 },
                DR::Separated {
                    old_id: 2,
                    new_id: 5,
                    new_size: 3,
                },
                DR::Separated {
                    old_id: 2,
                    new_id: 6,
                    new_size: 1,
                },
            ],
            vec![
                DR::ChangedSize { id: 1, change: 1 },
                DR::Merged { id: 1, old_id: 5 },
                DR::ChangedId {
                    old_id: 6,
                    new_id: 5,
                },
                DR::Merged { id: 1, old_id: 3 },
                DR::ChangedId {
                    old_id: 5,
                    new_id: 3,
                },
                DR::Merged { id: 1, old_id: 2 },
                DR::ChangedId {
                    old_id: 4,
                    new_id: 2,
                },
            ],
            vec![
                DR::ChangedSize { id: 1, change: -1 },
                DR::Separated {
                    old_id: 1,
                    new_id: 4,
                    new_size: 3,
                },
                DR::Separated {
                    old_id: 1,
                    new_id: 5,
                    new_size: 2,
                },
                DR::Separated {
                    old_id: 1,
                    new_id: 6,
                    new_size: 1,
                },
            ],
        ];
        for (reaction, expected) in reactions.iter().zip(expected_drs) {
            reaction.apply(&mut state);
            let drs = droplets.update(
                &state.view(),
                &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
                &vec![1],
                &(Box::new(*reaction) as Box<dyn Reaction<u32>>),
            );

            assert_eq!(drs, expected);
        }
    }

    #[test]
    fn droplet_reactions_work_with_diffusion() {
        use crate::reaction::BasicReaction as BR;
        use DropletReaction as DR;
        let mut state = arr2(&[
            [0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let reactions = vec![BR::diffusion([1, 3], [1, 2]), BR::diffusion([3, 2], [2, 2])];
        let mut droplets = Droplets::new(
            &state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &vec![1],
        );

        let expected_drs = vec![
            vec![
                DR::ChangedSize { id: 2, change: -1 },
                DR::Separated {
                    old_id: 2,
                    new_id: 4,
                    new_size: 1,
                },
                DR::ChangedSize { id: 1, change: 1 },
                DR::Merged { id: 1, old_id: 3 },
                DR::ChangedId {
                    old_id: 4,
                    new_id: 3,
                },
            ],
            vec![
                DR::Evaporated { id: 2 },
                DR::ChangedId {
                    old_id: 3,
                    new_id: 2,
                },
                DR::Formed { id: 3 },
            ],
        ];
        for (reaction, expected) in reactions.iter().zip(expected_drs) {
            reaction.apply(&mut state);
            let drs = droplets.update(
                &state.view(),
                &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
                &vec![1],
                &(Box::new(*reaction) as Box<dyn Reaction<u32>>),
            );

            assert_eq!(drs, expected);
        }
    }

    #[test]
    fn droplet_reactions_work_with_central_evaporation() {
        use crate::reaction::BasicReaction as BR;
        use DropletReaction as DR;
        let mut state = arr2(&[
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let reactions = vec![BR::point_change(1, 0, [2, 2])];
        let mut droplets = Droplets::new(
            &state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
            &vec![1],
        );

        let expected_drs = vec![vec![DR::ChangedSize { id: 1, change: -1 }]];
        for (reaction, expected) in reactions.iter().zip(expected_drs) {
            reaction.apply(&mut state);
            let drs = droplets.update(
                &state.view(),
                &(Box::new(boundary) as Box<dyn BoundaryCondition<u32>>),
                &vec![1],
                &(Box::new(*reaction) as Box<dyn Reaction<u32>>),
            );

            assert_eq!(drs, expected);
        }
    }
}
