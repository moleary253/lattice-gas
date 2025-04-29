use super::*;
use serde::{Deserialize, Serialize};

// TODO(Myles): Add documentation
#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub struct Periodic;

impl Periodic {
    pub fn new() -> Self {
        Periodic
    }
}

impl<T: Clone> BoundaryCondition<T> for Periodic {
    fn get(&self, state: &Array2<T>, pos: [usize; 2]) -> T {
        state[[
            (pos[0]).rem_euclid(state.shape()[0]) as usize,
            (pos[1]).rem_euclid(state.shape()[1]) as usize,
        ]]
        .clone()
    }

    fn adjacent_indicies(&self, state: &Array2<T>, pos: [usize; 2]) -> Vec<[usize; 2]> {
        let [x, y] = pos;
        let (x, y) = (x as i32, y as i32);
        return [(-1, 0), (1, 0), (0, 1), (0, -1)]
            .into_iter()
            .map(move |(dx, dy)| {
                [
                    (dx + x).rem_euclid(state.shape()[0] as i32) as usize,
                    (dy + y).rem_euclid(state.shape()[1] as i32) as usize,
                ]
            })
            .collect();
    }

    fn adjacent(&self, state: &Array2<T>, pos: [usize; 2]) -> Vec<T> {
        let [x, y] = pos;
        let (x, y) = (x as i32, y as i32);
        return [(-1, 0), (1, 0), (0, 1), (0, -1)]
            .into_iter()
            .map(move |(dx, dy)| {
                state[[
                    (dx + x).rem_euclid(state.shape()[0] as i32) as usize,
                    (dy + y).rem_euclid(state.shape()[1] as i32) as usize,
                ]]
                .clone()
            })
            .collect();
    }
}
