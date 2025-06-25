use std::ops::Add;

/// BinarySumTree is a data structure which computes the sum of a bunch of
/// numbers. It can efficiently update that sum (in O(log(N)) time) when one of
/// the numbers is updated. It also allows O(log(N)) search for the index of a
/// partial sum. It has fixed size after initialization.
pub struct BinarySumTree<T: Add<T, Output = T> + Default + Clone + Copy + std::cmp::PartialOrd> {
    data: Vec<T>,
    layer_sizes: Vec<usize>,
    layer_boundaries: Vec<usize>,
}

impl<T: Add<T, Output = T> + Default + Clone + Copy + std::cmp::PartialOrd> BinarySumTree<T> {
    /// Creates a new BinarySumTree with the data provided.
    pub fn new(initial_data: Vec<T>) -> Self {
        let mut layer_sizes: Vec<usize> = Vec::new();
        let mut layer_boundaries: Vec<usize> = Vec::new();
        let mut current_layer_size = initial_data.len();
        layer_boundaries.push(0);
        while current_layer_size > 1 {
            layer_sizes.push(current_layer_size);
            if current_layer_size % 2 == 1 {
                current_layer_size += 1;
            }
            layer_boundaries.push(current_layer_size + layer_boundaries.last().unwrap_or(&0));
            current_layer_size /= 2;
        }
        layer_sizes.push(1);

        let tree_size = layer_boundaries.last().unwrap() + 1;
        let mut data = Vec::with_capacity(tree_size);
        for i in 0..tree_size {
            if i < initial_data.len() {
                data.push(initial_data[i]);
                continue;
            }
            let layer = layer_boundaries.partition_point(|&b| b <= i) - 1;
            let layer_size = layer_sizes[layer];
            if i - layer_boundaries[layer] < layer_size {
                let prev_layer_idx =
                    layer_boundaries[layer - 1] + 2 * (i - layer_boundaries[layer]);
                data.push(data[prev_layer_idx] + data[prev_layer_idx + 1]);
                continue;
            }
            data.push(T::default());
        }
        BinarySumTree {
            data,
            layer_sizes,
            layer_boundaries,
        }
    }

    /// Updates one of the values at the given index and propogates that update
    /// through the tree.
    pub fn update(&mut self, index: usize, new_value: T) {
        self.data[index] = new_value;
        let mut index_in_layer = index;
        for layer in 1..self.layer_sizes.len() {
            index_in_layer >>= 1;
            self.data[self.layer_boundaries[layer] + index_in_layer] = self
                .get(layer - 1, index_in_layer * 2)
                + self.get(layer - 1, index_in_layer * 2 + 1);
        }
    }

    /// Updates multiple values at the given indicies and propogates that update
    /// through the tree.
    pub fn batch_update(&mut self, indicies: &[usize], new_values: &[T]) {
        for (&index, &new_value) in indicies.iter().zip(new_values.iter()) {
            self.data[index] = new_value;
        }
        let mut indicies_in_layer = Vec::from_iter(indicies.iter().map(|idx| idx >> 1));
        let mut next_indicies_in_layer = Vec::with_capacity(indicies_in_layer.len());
        for layer in 1..self.layer_sizes.len() {
            for &index_in_layer in indicies_in_layer.iter() {
                self.data[self.layer_boundaries[layer] + index_in_layer] = self
                    .get(layer - 1, index_in_layer * 2)
                    + self.get(layer - 1, index_in_layer * 2 + 1);
                let next_index_in_layer = index_in_layer >> 1;
                if !next_indicies_in_layer.contains(&next_index_in_layer) {
                    next_indicies_in_layer.push(next_index_in_layer);
                }
            }
            indicies_in_layer.clear();
            (indicies_in_layer, next_indicies_in_layer) =
                (next_indicies_in_layer, indicies_in_layer);
        }
    }

    fn get(&self, layer: usize, index_in_layer: usize) -> T {
        self.data[self.layer_boundaries[layer] + index_in_layer]
    }

    /// Search for the index of a partial sum. If one created an array of the
    /// partial sums of the underlying data, the partition point where that array
    /// by the given value will be the same as the index returned from this
    /// function
    pub fn search(&self, value: T) -> usize {
        if value > *self.data.last().unwrap() {
            return self.layer_sizes[0];
        }
        let mut layer = self.layer_boundaries.len() - 1;
        let mut index = 0;
        let mut sum = T::default();
        while layer > 0 {
            if self.get(layer - 1, 2 * index) + sum < value {
                sum = sum + self.get(layer - 1, 2 * index);
                index = index * 2 + 1;
            } else {
                index = index * 2;
            }
            layer -= 1;
        }
        index
    }

    pub fn sum(&self) -> T {
        *self.data.last().unwrap()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn initialization() {
        let initial_data = vec![1, 3, 10, 7, 2];
        let expected_final_data = vec![
            1, 3, 10, 7, 2, 0, // Layer 0
            4, 17, 2, 0, // Layer 1
            21, 2,  // Layer 2
            23, // Layer 3
        ];
        let tree = BinarySumTree::new(initial_data);
        for (expected, actual) in expected_final_data.iter().zip(tree.data.iter()) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn updating() {
        let initial_data = vec![1, 3, 10, 7, 2];
        let expected_final_data = vec![1, 3, 3, 7, 2, 0, 4, 10, 2, 0, 14, 2, 16];
        let mut tree = BinarySumTree::new(initial_data);
        tree.update(2, 3);
        for (expected, actual) in expected_final_data.iter().zip(tree.data.iter()) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn batch_updating() {
        let initial_data = vec![1, 3, 10, 7, 2];
        let expected_final_data = vec![1, 3, 3, -1, 5, 0, 4, 2, 5, 0, 6, 5, 11];
        let mut tree = BinarySumTree::new(initial_data);
        tree.batch_update(&[2, 3, 4], &[3, -1, 5]);
        assert_eq!(expected_final_data, tree.data);
    }

    #[test]
    fn search() {
        let initial_data = vec![1, 3, 10, 7, 2];
        let tree = BinarySumTree::new(initial_data);

        let to_search_for = [0, 1, 7, 22, 100];
        let expected_indicies = [0, 0, 2, 4, 5];

        for (&expected, &search) in expected_indicies.iter().zip(to_search_for.iter()) {
            assert_eq!(expected, tree.search(search));
        }
    }
}
