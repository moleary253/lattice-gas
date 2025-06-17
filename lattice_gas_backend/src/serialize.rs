use flate2::read::GzDecoder;
use ndarray::Array2;
use serde::{de::DeserializeOwned, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;
use tar::{Archive, Builder, Header};

use numpy::prelude::*;
use numpy::PyArray2;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

pub const TEMP_DIR: &str = "data/current";
pub const CHAIN_FILE: &str = "chain.json";
pub const INITIAL_CONDITIONS_FILE: &str = "initial_conditions.json";
pub const REACTIONS_FILE: &str = "reactions.json";
pub const FINAL_STATE_FILE: &str = "final_state.json";
pub const FINAL_TIME_FILE: &str = "final_time.json";

pub fn serialize_object<W: Write>(
    path: String,
    data: &impl Serialize,
    archive: &mut Builder<W>,
) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let data = serde_json::to_string(data)?.into_bytes();

    let mut header = Header::new_gnu();
    header.set_path(path)?;
    header.set_size(data.len() as u64);
    header.set_mode(0o664);
    header.set_cksum();

    archive.append(&header, &data as &[u8])?;
    Ok(())
}

pub fn unpack_archive(path: String) -> Result<(), Box<dyn std::error::Error + 'static>> {
    let file = fs::File::open(path)?;
    let unzipper = GzDecoder::new(file);
    let mut archive = Archive::new(unzipper);

    archive.unpack(TEMP_DIR)?;
    Ok(())
}

pub fn clean_up() -> Result<(), Box<dyn std::error::Error + 'static>> {
    fs::remove_dir_all(TEMP_DIR)?;
    Ok(())
}

fn read<'a, T: DeserializeOwned>(
    extension: &'static str,
) -> Result<T, Box<dyn std::error::Error + 'static>> {
    let file = fs::File::open(Path::new(TEMP_DIR).join(extension))?;
    let data = serde_json::from_reader(file)?;
    Ok(data)
}

fn py_read<'a, T: DeserializeOwned>(extension: &'static str) -> PyResult<T> {
    match read::<T>(extension) {
        Ok(data) => Ok(data),
        Err(err) => Err(PyIOError::new_err(err.to_string())),
    }
}

#[pyfunction]
pub fn delta_times_and_reactions() -> PyResult<(Vec<f64>, Vec<crate::reaction::BasicReaction<u32>>)>
{
    let data: Vec<(f64, crate::reaction::BasicReaction<u32>)> = py_read(REACTIONS_FILE)?;
    Ok((
        data.iter().map(|(dt, _r)| *dt).collect(),
        data.into_iter().map(|(_dt, r)| r).collect(),
    ))
}

#[pyfunction]
pub fn initial_conditions<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let data: Array2<u32> = py_read(INITIAL_CONDITIONS_FILE)?;
    Ok(data.to_pyarray(py))
}
