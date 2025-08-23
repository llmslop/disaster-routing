use indexed_priority_queue::ArrayMapIPQ;
use pyo3::prelude::*;

pub struct BitSet {
    bits: Vec<u64>,
}

impl BitSet {
    pub fn new(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            bits: vec![u64::MAX; words],
        }
    }

    #[inline]
    fn word_index(pos: usize) -> (usize, u64) {
        (pos / 64, 1u64 << (pos % 64))
    }

    pub fn insert(&mut self, pos: usize) {
        let (wi, mask) = Self::word_index(pos);
        self.bits[wi] |= mask;
    }

    pub fn remove(&mut self, pos: usize) {
        let (wi, mask) = Self::word_index(pos);
        self.bits[wi] &= !mask;
    }

    pub fn contains(&self, pos: usize) -> bool {
        let (wi, mask) = Self::word_index(pos);
        (self.bits[wi] & mask) != 0
    }
}

fn fpga_solve_perm_with_first(
    num_nodes: i32,
    adj_list: &[Vec<i32>],
    start_index: i32,
    num_fses: &[i32],
) -> (Vec<i32>, i32) {
    let mut ordering = vec![start_index];
    ordering.reserve_exact((num_nodes - 1) as _);

    let mut remaining = BitSet::new(num_nodes as _);
    remaining.remove(start_index as _);

    let mut end_indices = vec![0; num_nodes as usize];
    end_indices[start_index as usize] = num_fses[start_index as usize];

    let mut heap = ArrayMapIPQ::<i32>::with_capacity(
        vec![0; num_nodes as usize].into_boxed_slice(),
        vec![usize::MAX; num_nodes as usize].into_boxed_slice(),
        num_nodes as _,
    );
    for node in 0..num_nodes {
        if node != start_index {
            heap.push(node as _, 0);
        }
    }
    for nbr in adj_list[start_index as usize].iter() {
        heap.push(*nbr as _, end_indices[start_index as usize]);
    }

    let mut mofi = end_indices[start_index as usize];

    while let Some((next_index, next_node)) = heap.min_priority().cloned().zip(heap.pop()) {
        let end_index = next_index + num_fses[next_node];
        end_indices[next_node] = end_index;
        mofi = mofi.max(end_index);

        remaining.remove(next_node as _);
        ordering.push(next_node as _);

        // update neighbors
        for &nbr in &adj_list[next_node] {
            if remaining.contains(nbr as _) {
                let mut p = heap.update_up(nbr as usize);
                *p = p.max(end_index);
            }
        }
    }

    (ordering, mofi)
}

#[pyfunction]
fn fpga_solve_for_odsa_perm(
    num_nodes: i32,
    edges: Vec<(i32, i32)>,
    start_indices: Vec<i32>,
    num_fses: Vec<i32>,
) -> PyResult<Vec<i32>> {
    let mut adj: Vec<Vec<i32>> = Vec::new();
    adj.resize_with(num_nodes as usize, Vec::new);
    for (u, v) in edges {
        adj[u as usize].push(v);
        adj[v as usize].push(u);
    }
    let result = start_indices
        .into_iter()
        .map(|first| fpga_solve_perm_with_first(num_nodes, &adj, first, &num_fses))
        .max_by_key(|(_, mofi)| *mofi)
        .map(|(result, _)| result)
        .expect("should not be None");
    Ok(result)
}

#[pymodule]
fn dr_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpga_solve_for_odsa_perm, m)?)?;
    Ok(())
}
