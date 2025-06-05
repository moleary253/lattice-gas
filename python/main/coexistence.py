import lattice_gas as lg
import numpy as np
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import load


def run_coex_simulation(output_file, random_seed, bonding_fugacity):
    initial_state = np.zeros((20, 10), dtype=np.dtype("u4"))
    initial_state[0:initial_state.shape[0]//2, :] = load.BONDING
    boundary = lg.boundary_condition.Periodic()

    chain = lg.markov_chain.HomogenousChain(
        beta=1.0,
        bond_energy=-3.0,
        driving_chemical_potential=0.0,
        inert_fugacity=0.0,
        bonding_fugacity=bonding_fugacity,
        inert_to_bonding_rate=0.0,
    )

    gas_density   = 0.05
    solid_density = 0.95
    gas_ending_criterion   = lg.ending_criterion.ParticleCount(int(initial_state.size * gas_density),   [load.BONDING], "below")
    solid_ending_criterion = lg.ending_criterion.ParticleCount(int(initial_state.size * solid_density), [load.BONDING], "above")

    lg.simulate.simulate(
        initial_state,
        boundary,
        chain,
        [gas_ending_criterion, solid_ending_criterion],
        random_seed,
        output_file,
        save_reactions=True,
    )


if __name__ == "__main__":
    import shutil

    num_trials = 100
    search_range = [0.01, 0.0]
    # z_B_coex = 0.002724609375
    while True:
        bonding_fugacity = np.average(search_range)
        print(f"\n    Checking {bonding_fugacity}:")
        num_gas = 0
        for i in tqdm(range(num_trials)):
            random_seed = i + 57413
            output_file = f"data/coexistence/{bonding_fugacity:.5f}_{i+1}.tar.gz"
            run_coex_simulation(output_file, random_seed, bonding_fugacity)

            directory = load.unpack_natural_input(output_file)
            final_conditions = load.final_state(directory)
            if (np.sum(final_conditions == load.BONDING) / final_conditions.size) < 0.5:
                num_gas += 1
            shutil.rmtree(load.TEMP_ARCHIVE_PATH)
            
        print(f"{num_gas} / {num_trials} ended as gas.")
        if abs(num_gas - num_trials // 2) < 0.5 * np.sqrt(num_trials):
            print(f"Done! Final result:\n\t{bonding_fugacity}")
            break
        elif num_gas < num_trials // 2:
            search_range[0] = bonding_fugacity
            print("Searching higher")
        else:
            search_range[1] = bonding_fugacity
            print("Searching lower")
