use ndarray::Array2;

// TODO(Myles): Add documentation
pub trait BoundaryCondition<T>
where
    T: Clone,
{
    fn get(&self, state: &Array2<T>, pos: [usize; 2]) -> T;
    fn adjacent(&self, state: &Array2<T>, pos: [usize; 2]) -> Vec<T>;
    fn adjacent_indicies(&self, state: &Array2<T>, pos: [usize; 2]) -> Vec<[usize; 2]>;
}

mod periodic_boundary;
pub use periodic_boundary::*;
