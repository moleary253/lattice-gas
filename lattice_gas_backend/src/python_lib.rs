use pyo3::prelude::*;

#[pymodule]
fn lattice_gas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let analysis = PyModule::new(m.py(), "analysis")?;
    use crate::analysis::*;
    analysis.add_function(wrap_pyfunction!(py_commitance, &analysis)?)?;
    analysis.add_function(wrap_pyfunction!(py_cnt_rates, &analysis)?)?;
    analysis.add_class::<Droplets>()?;
    analysis.add_class::<LargestDropletSizeAnalyzer>()?;
    m.add_submodule(&analysis)?;

    let boundary_condition = PyModule::new(m.py(), "boundary_condition")?;
    use crate::boundary_condition::*;
    boundary_condition.add_class::<Periodic>()?;
    m.add_submodule(&boundary_condition)?;

    let ending_criterion = PyModule::new(m.py(), "ending_criterion")?;
    use crate::ending_criterion::*;
    ending_criterion.add_class::<LargestDropletSize>()?;
    ending_criterion.add_class::<ReactionCount>()?;
    ending_criterion.add_class::<ParticleCount>()?;
    ending_criterion.add_class::<TargetState>()?;
    m.add_submodule(&ending_criterion)?;

    let markov_chain = PyModule::new(m.py(), "markov_chain")?;
    use crate::markov_chain::*;
    markov_chain.add_class::<HomogenousChain>()?;
    markov_chain.add_class::<IsingChain>()?;
    markov_chain.add_class::<HomogenousNVTChain>()?;
    markov_chain.add_class::<CNTLadderChain>()?;
    m.add_submodule(&markov_chain)?;

    let simulate = PyModule::new(m.py(), "simulate")?;
    use crate::simulate::*;
    simulate.add_function(wrap_pyfunction!(py_simulate, &simulate)?)?;
    m.add_submodule(&simulate)?;

    let load = PyModule::new(m.py(), "load")?;
    use crate::serialize::*;
    load.add_function(wrap_pyfunction!(py_chain, &load)?)?;
    load.add_function(wrap_pyfunction!(py_boundary, &load)?)?;
    load.add_function(wrap_pyfunction!(py_ending_criteria, &load)?)?;
    load.add_function(wrap_pyfunction!(py_initial_conditions, &load)?)?;
    load.add_function(wrap_pyfunction!(py_analyzers, &load)?)?;
    load.add_function(wrap_pyfunction!(py_delta_times, &load)?)?;
    load.add_function(wrap_pyfunction!(py_final_state, &load)?)?;
    load.add_function(wrap_pyfunction!(py_final_time, &load)?)?;
    m.add_submodule(&load)?;

    Ok(())
}
