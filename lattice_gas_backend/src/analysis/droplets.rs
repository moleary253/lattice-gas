use crate::boundary_condition::BoundaryCondition;
use crate::reaction::Reaction;
use ndarray::Array2;
use numpy::ndarray::ArrayView2;
use numpy::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

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
        unsafe {
            Ok(Droplets::new(
                &state.as_array(),
                &crate::boundary_condition::extract(boundary)?,
                &counts_as_droplet,
            ))
        }
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
                    if counts_as_droplet.contains(&state[idx]) || labeled[idx] != 0 {
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
    ) {
        for index in reaction.indicies_updated() {
            if counts_as_droplet.contains(&state[index]) {
                self.update_added(index, &state, boundary);
            } else {
                self.update_removed(index, &state, boundary);
            }
        }
    }

    fn update_added<T: Clone>(
        &mut self,
        index: [usize; 2],
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
    ) {
        if self.labeled[index] != 0 {
            return;
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
            return;
        }
        if adjacent_droplets.len() == 1 {
            let droplet = adjacent_droplets[0];
            self.labeled[index] = droplet;
            self.droplets[droplet - 1].push(index);
            return;
        }
        // NOTE(Myles): Need to merge droplets here
        adjacent_droplets.sort();
        adjacent_droplets.reverse();
        let smallest_droplet = *adjacent_droplets.iter().min().unwrap();
        self.labeled[index] = smallest_droplet;
        self.droplets[smallest_droplet - 1].push(index);
        for droplet in adjacent_droplets {
            if droplet == smallest_droplet {
                continue;
            }
            for index in self.droplets.swap_remove(droplet - 1) {
                self.labeled[index] = smallest_droplet;
                self.droplets[smallest_droplet - 1].push(index);
            }
            if self.droplets.len() == droplet - 1 {
                continue;
            }
            for index in self.droplets[droplet - 1].iter() {
                self.labeled[*index] = droplet;
            }
        }
    }

    fn update_removed<T: Clone>(
        &mut self,
        index: [usize; 2],
        state: &ArrayView2<T>,
        boundary: &Box<dyn BoundaryCondition<T>>,
    ) {
        if self.labeled[index] == 0 {
            return;
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
            return;
        }
        if adjacent_indicies.len() == 0 {
            self.droplets.swap_remove(old_droplet - 1);
            if self.droplets.len() == old_droplet - 1 {
                return;
            }
            for index in self.droplets[old_droplet - 1].iter() {
                self.labeled[*index] = old_droplet;
            }
            return;
        }
        // NOTE(Myles): Might need to split into multiple droplets here
        let mut checked = Vec::new();
        for _ in 0..adjacent_indicies.len() {
            checked.push(false);
        }
        let mut first_run = true;
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
    }
}

// #[cfg(test)]
// pub mod tests {
//     use crate::boundary_condition;

//     use super::*;
//     use ndarray::arr2;

//     #[test]
//     fn initialization() {
//         let state = arr2(&[
//             [1, 0, 0, 1, 1],
//             [1, 1, 0, 0, 0],
//             [0, 1, 0, 1, 1],
//             [0, 0, 0, 0, 1],
//             [0, 0, 0, 1, 1],
//             [0, 0, 1, 0, 0],
//         ]);
//         let boundary = boundary_condition::Periodic;
//         let droplets = Droplets::new(&state, &boundary, &|site| *site == 1);

//         let expected_droplets = vec![
//             vec![[0, 0], [1, 0], [1, 1], [2, 1], [0, 3], [0, 4]],
//             vec![[2, 3], [2, 4], [3, 4], [4, 4], [4, 3]],
//             vec![[5, 2]],
//         ];

//         let expected_labeled = arr2(&[
//             [1, 0, 0, 1, 1],
//             [1, 1, 0, 0, 0],
//             [0, 1, 0, 2, 2],
//             [0, 0, 0, 0, 2],
//             [0, 0, 0, 2, 2],
//             [0, 0, 3, 0, 0],
//         ]);

//         assert_eq!(droplets.droplets.len(), expected_droplets.len());
//         for (droplet, expected) in droplets.droplets.iter().zip(expected_droplets) {
//             assert_eq!(droplet.len(), expected.len());
//             for index in droplet {
//                 assert!(expected.contains(index));
//             }
//         }

//         assert_eq!(expected_labeled, droplets.labeled);
//     }

//     #[test]
//     fn registers_new_droplets() {
//         use crate::reaction::BasicReaction as BR;
//         let mut state = arr2(&[
//             [0, 0, 0, 0, 0],
//             [0, 0, 0, 0, 0],
//             [0, 0, 0, 0, 0],
//             [0, 0, 0, 0, 0],
//             [0, 0, 0, 0, 0],
//         ]);
//         let boundary = boundary_condition::Periodic;
//         let reactions = vec![
//             BR::point_change(0, 1, [1, 1]),
//             BR::point_change(0, 1, [3, 3]),
//             BR::point_change(0, 1, [1, 2]),
//             BR::point_change(0, 1, [2, 2]),
//             BR::point_change(0, 1, [2, 3]),
//         ];
//         fn is_one(x: &usize) -> bool {
//             *x == 1
//         }
//         let mut droplets = Droplets::new(&state, &boundary, &is_one);

//         let expected_droplets = vec![
//             vec![vec![[1, 1]]],
//             vec![vec![[1, 1]], vec![[3, 3]]],
//             vec![vec![[1, 1], [1, 2]], vec![[3, 3]]],
//             vec![vec![[1, 1], [1, 2], [2, 2]], vec![[3, 3]]],
//             vec![vec![[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]]],
//         ];
//         for (reaction, expected) in reactions.iter().zip(expected_droplets) {
//             reaction.apply(&mut state);
//             droplets.update(&state, &boundary, &is_one, reaction);

//             assert_eq!(droplets.droplets.len(), expected.len());
//             for (droplet, expected_droplet) in droplets.droplets.iter().zip(expected) {
//                 assert_eq!(droplet.len(), expected_droplet.len());
//                 for idx in droplet {
//                     println!("{idx:?}");
//                     assert!(expected_droplet.contains(&idx));
//                 }
//             }
//         }
//     }

//     #[test]
//     fn can_merge_multiple_droplets() {
//         use crate::reaction::BasicReaction as BR;
//         let mut state = arr2(&[
//             [0, 1, 0, 0, 1],
//             [1, 0, 0, 1, 0],
//             [0, 1, 0, 0, 1],
//             [0, 0, 0, 0, 0],
//             [0, 0, 0, 0, 0],
//         ]);
//         let boundary = boundary_condition::Periodic;
//         let reactions = vec![
//             BR::point_change(0, 1, [1, 1]),
//             BR::point_change(0, 1, [1, 4]),
//         ];
//         fn is_one(x: &usize) -> bool {
//             *x == 1
//         }
//         let mut droplets = Droplets::new(&state, &boundary, &is_one);

//         let expected_droplets = vec![
//             vec![
//                 vec![[0, 1], [1, 0], [1, 1], [2, 1]],
//                 vec![[0, 4]],
//                 vec![[2, 4]],
//                 vec![[1, 3]],
//             ],
//             vec![vec![
//                 [0, 1],
//                 [1, 0],
//                 [1, 1],
//                 [2, 1],
//                 [0, 4],
//                 [1, 3],
//                 [2, 4],
//                 [1, 4],
//             ]],
//         ];
//         for (reaction, expected) in reactions.iter().zip(expected_droplets) {
//             reaction.apply(&mut state);
//             droplets.update(&state, &boundary, &is_one, reaction);

//             assert_eq!(droplets.droplets.len(), expected.len());
//             for (droplet, expected_droplet) in droplets.droplets.iter().zip(expected) {
//                 assert_eq!(droplet.len(), expected_droplet.len());
//                 for idx in droplet {
//                     assert!(expected_droplet.contains(&idx));
//                 }
//             }
//         }
//     }
// }
