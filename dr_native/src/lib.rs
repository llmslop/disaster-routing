use pyo3::prelude::*;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

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

    let mut cached_max = vec![0; num_nodes as usize];

    for nbr in adj_list[start_index as usize].iter() {
        cached_max[*nbr as usize] = end_indices[start_index as usize];
    }

    // initial cached_max = max over already placed neighbors (only start_index so far)
    let mut heap = BinaryHeap::new();
    for node in 0..num_nodes {
        if node != start_index {
            heap.push(Reverse((cached_max[node as usize], node)));
        }
    }

    let mut mofi = end_indices[start_index as usize];

    while ordering.len() < num_nodes as usize {
        // pop until valid
        let (next_index, next_node);
        loop {
            let Reverse((idx, node)) = heap.pop().unwrap();
            if remaining.contains(node as _) && cached_max[node as usize] == idx {
                next_index = idx;
                next_node = node;
                break;
            }
        }

        let end_index = next_index + num_fses[next_node as usize];
        end_indices[next_node as usize] = end_index;
        mofi = mofi.max(end_index);

        remaining.remove(next_node as _);
        ordering.push(next_node);

        // update neighbors
        for &nbr in &adj_list[next_node as usize] {
            if remaining.contains(nbr as _) && end_index > cached_max[nbr as usize] {
                cached_max[nbr as usize] = end_index;
                heap.push(Reverse((end_index, nbr)));
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
        .min_by_key(|(_, mofi)| *mofi)
        .map(|(result, _)| result)
        .expect("should not be None");
    Ok(result)
}

#[pymodule]
fn dr_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpga_solve_for_odsa_perm, m)?)?;
    Ok(())
}
