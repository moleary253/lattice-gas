import lattice_gas as lg # type: ignore
import multiprocessing as mp
import sys, os
from numpy.random import SeedSequence, default_rng  # type: ignore
import numpy as np  # type: ignore

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import load


DEFAULT_DIR = "data/test_cnt"


def run_simulation(
    output_file,
    random_seed,
    magnetic_field,
    bond_energy,
    initial_state,
    ending_criteria_nsteps,
):

    print(output_file)
    ending_criteria = [lg.ending_criterion.ReactionCount(ending_criteria_nsteps)]
    boundary = lg.boundary_condition.Periodic()

    chain = lg.markov_chain.IsingChain(
        beta=1.0,
        bond_energy=bond_energy,
        magnetic_field=magnetic_field,
    )

    analyzers = [
        lg.analysis.Droplets(
            initial_state,
            boundary,
            [load.BONDING],
        ),
        lg.analysis.LargestDropletSizeAnalyzer(),
    ]

    lg.simulate.simulate(
        initial_state,
        boundary,
        chain,
        analyzers,
        ending_criteria,
        random_seed,
        output_file,
    )


def run():
    data_dir = Path(DEFAULT_DIR)
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir()

    num_trials = 1
    bond_energy = -1.5
    magnetic_field = 5.0

    root_seed = default_rng().integers(low=0, high=1000000, size=1)

    seed_sequence = SeedSequence(root_seed)
    seeds = seed_sequence.spawn(PARALLEL_JOBS)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]

    output_files = [DEFAULT_DIR + f"/external_{magnetic_field:.5f}_{i}" for i in range(PARALLEL_JOBS)]


    initial_state = np.zeros((100, 100), dtype=np.dtype("u4"))
    initial_state[10:20, 10:20] = 2
    ending_criteria_steps = 1_000_000

    jobs = [
        mp.Process(
            target=run_simulation,
            args=(
                output_files[i],
                seeds[i],
                magnetic_field,
                bond_energy,
                initial_state,
                ending_criteria_steps,
            ),
        )
        for i in range(PARALLEL_JOBS)
    ]

    for job in jobs:
        job.start()
    for job in jobs:
        job.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("effect_of_activity")
    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Run simulations",
    )

    parser.add_argument(
        "-n",
        "--cpus",
        type=int,
        default=1,
        help="Number of CPUs to use for parallel processing",
    )
    args = parser.parse_args()

    if args.run:
        global PARALLEL_JOBS
        PARALLEL_JOBS = args.cpus
        run()
