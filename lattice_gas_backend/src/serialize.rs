use ndarray::Array2;
use serde::{de::DeserializeOwned, Serialize};
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::{
    analysis::Analyzer, boundary_condition::BoundaryCondition, ending_criterion::EndingCriterion,
    markov_chain::MarkovChain,
};

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

pub const CHAIN_FILE: &str = "chain.msgpack";
pub const BOUNDARY_FILE: &str = "boundary.msgpack";
pub const ENDING_CRITERIA_FILE: &str = "ending_criteria.msgpack";
pub const INITIAL_CONDITIONS_FILE: &str = "initial_conditions.msgpack";
pub const ANALYZER_FILE: &str = "analyzers.msgpack";
pub const DELTA_TIMES_FILE: &str = "delta_times.msgpack";
pub const FINAL_STATE_FILE: &str = "final_state.msgpack";
pub const FINAL_TIME_FILE: &str = "final_time.msgpack";

pub fn save_simulation(
    directory: &str,
    chain: &Box<dyn MarkovChain>,
    boundary: &Box<dyn BoundaryCondition>,
    ending_criteria: &Vec<Box<dyn EndingCriterion>>,
    initial_conditions: &Array2<u32>,
    analyzers: &Vec<Box<dyn Analyzer>>,
    delta_times: &Vec<f64>,
    final_state: &Array2<u32>,
) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let directory = Path::new(directory);
    fs::create_dir(directory)?;
    serialize_object(directory.join(CHAIN_FILE), chain)?;
    serialize_object(directory.join(BOUNDARY_FILE), boundary)?;
    serialize_object(directory.join(ENDING_CRITERIA_FILE), ending_criteria)?;
    serialize_object(directory.join(INITIAL_CONDITIONS_FILE), initial_conditions)?;
    serialize_object(directory.join(ANALYZER_FILE), analyzers)?;
    serialize_object(directory.join(FINAL_STATE_FILE), final_state)?;
    serialize_object(
        directory.join(FINAL_TIME_FILE),
        &delta_times.iter().map(std::convert::identity).sum::<f64>(),
    )?;
    Ok(())
}

pub fn serialize_object(
    path: impl Into<OsString>,
    data: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let data = rmp_serde::to_vec(data)?;

    let mut file = fs::File::create(path.into())?;
    file.write(&data)?;
    Ok(())
}

fn read<'a, T: DeserializeOwned>(
    directory: &Path,
    extension: &'static str,
) -> Result<T, Box<dyn std::error::Error + 'static>> {
    let file = fs::File::open(directory.join(extension))?;
    let data = rmp_serde::from_read(file)?;
    Ok(data)
}

fn py_read<'a, T: DeserializeOwned>(directory: &str, extension: &'static str) -> PyResult<T> {
    match read::<T>(Path::new(directory), extension) {
        Ok(data) => Ok(data),
        Err(err) => Err(PyIOError::new_err(err.to_string())),
    }
}

#[pyfunction]
#[pyo3(name = "chain")]
pub fn py_chain(directory: &str) -> PyResult<Box<dyn MarkovChain>> {
    py_read(directory, CHAIN_FILE)
}

#[pyfunction]
#[pyo3(name = "boundary")]
pub fn py_boundary(directory: &str) -> PyResult<Box<dyn BoundaryCondition>> {
    py_read(directory, BOUNDARY_FILE)
}

#[pyfunction]
#[pyo3(name = "ending_criteria")]
pub fn py_ending_criteria(directory: &str) -> PyResult<Vec<Box<dyn EndingCriterion>>> {
    py_read(directory, ENDING_CRITERIA_FILE)
}

#[pyfunction]
#[pyo3(name = "initial_conditions")]
pub fn py_initial_conditions<'py>(
    py: Python<'py>,
    directory: &str,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let data: Array2<u32> = py_read(directory, INITIAL_CONDITIONS_FILE)?;
    Ok(data.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "analyzers")]
pub fn py_analyzers(directory: &str) -> PyResult<Vec<Box<dyn Analyzer>>> {
    py_read(directory, ANALYZER_FILE)
}

#[pyfunction]
#[pyo3(name = "delta_times")]
pub fn py_delta_times<'py>(
    py: Python<'py>,
    directory: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data: Vec<f64> = py_read(directory, DELTA_TIMES_FILE)?;
    Ok(PyArray1::from_vec(py, data))
}

#[pyfunction]
#[pyo3(name = "final_state")]
pub fn py_final_state<'py>(
    py: Python<'py>,
    directory: &str,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let data: Array2<u32> = py_read(directory, FINAL_STATE_FILE)?;
    Ok(data.to_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "final_time")]
pub fn py_final_time(directory: &str) -> PyResult<f64> {
    py_read(directory, FINAL_TIME_FILE)
}
