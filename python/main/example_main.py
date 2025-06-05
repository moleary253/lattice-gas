import lattice_gas as lg
import numpy as np


if __name__ == "__main__":
    initial_state = np.zeros((20, 20), dtype=np.dtype("u4"))

    chain = lg.markov_chain.HomogenousChain(
        1.0,
        -1.0,
        0.0,
        1.0,
        0.001,
        0.1,
    )
        
    boundary = lg.boundary_condition.Periodic()

    ending_criterion = lg.ending_criterion.ReactionCount(10_000)

    random_seed = 1

    output_file = "data/test.tar.gz"

    lg.simulate.simulate(
        initial_state,
        boundary,
        chain,
        [ending_criterion],
        random_seed,
        output_file,
        save_reactions=True,
    )
