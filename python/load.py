import json
import os, sys, shutil
import tarfile
import numpy as np


CHAIN_FILE = "chain.json"
INITIAL_CONDITIONS_FILE = "initial_conditions.json"
REACTIONS_FILE = "reactions.json"
FINAL_STATE_FILE = "final_state.json"
FINAL_TIME_FILE = "final_time.json"
SIZES_FILE = "sizes.json"

TYPE = "type"
POINT_CHANGE = "PointChange"
DIFFUSION = "Diffusion"

EMPTY = 0
INERT = 1
BONDING = 2
SPECIES = ["Empty", "Inert", "Bonding"]
TRANSLATE = {"Empty": EMPTY, "Inert": INERT, "Bonding": BONDING}

TEMP_ARCHIVE_PATH = "data/current"


def chain(archive):
    """Loads the parameters of the Markov chain governing rates in the simulation.

    :param archive: The tar archive where the data is stored.
    :returns a dictionary of the parameters of the Markov chain.
    """
    with tar_member(archive, CHAIN_FILE) as f:
        chain_json = json.load(f)

    return chain_json


def initial_conditions(archive):
    """Loads the initial conditions of a simulation.

    :param archive: The tar archive where the data is stored.
    :returns a 2d array of integers representing the initial state at each site in the simulation.
    """
    with tar_member(archive, INITIAL_CONDITIONS_FILE) as f:
        initial_conditions_json = json.load(f)
    
    initial_conditions = np.array(
        initial_conditions_json["data"]
    ).reshape(initial_conditions_json["dim"])

    return initial_conditions


def sizes(archive):
    """Loads the series of maximum droplet sizes from a simulation

    :param archive: The tar archive where the data is stored.
    :returns a 2d array of (time, size) points
    """
    with tar_member(archive, SIZES_FILE) as f:
        sizes_json = json.load(f)
    
    return np.array(sizes_json)


def reactions(archive):
    """Loads the reactions that took place in a simulation.

    :param archive: The tar archive where the data is stored.
    :returns a time series of the reactions that occurred during the simulation. This is an array of
    dictionaries of which can take the form
    {
        "dt": float dt,
        "from": int state,
        "to": int state,
        "position": [int x, int y]
    }
    """
    with tar_member(archive, REACTIONS_FILE) as f:
        reactions_json = json.load(f)

    reactions = [
        {
            TYPE: POINT_CHANGE,
            "dt": dt,
            "from": reaction[POINT_CHANGE]["from"],
            "to": reaction[POINT_CHANGE]["to"],
            "position": reaction[POINT_CHANGE]["position"]
        } if POINT_CHANGE in reaction
        else {
            TYPE: DIFFUSION,
            "dt": dt,
            "from": reaction[DIFFUSION]["from"],
            "to": reaction[DIFFUSION]["to"],
        }
        for (dt, reaction) in reactions_json
    ]
    
    return reactions


def final_state(archive):
    """Loads the final state of a simulation.

    :param archive: The tar archive where the data is stored.
    :returns a 2d array of integers representing the initial state at each site in the simulation,
    or None if no final state was stored.
    """
    with tar_member(archive, FINAL_STATE_FILE) as f:
        final_state_json = json.load(f)
    
    final_state= np.array(
        [final_state_json["data"]],
        dtype=np.dtype("u4"),
    ).reshape(final_state_json["dim"])

    return final_state


def final_time(archive):
    """Loads the final time the simulation reached.

    :param archive: The tar archive where the data is stored.
    :returns the time at which the simulation ended,
    or None if no final time was stored.
    """
    with tar_member(archive, FINAL_TIME_FILE) as f:
        final_time_json = json.load(f)
    
    return float(final_time_json)


def read_tar_archive(archive_path):
    """Reads a tar archive into a file that can be used to read from the archive.

    :param archive_path: The path of the archive to be opened.
    :returns a tarfile.TarFile.
    """
    return tarfile.open(archive_path, mode="r:gz")


def list_tar_archive(archive_path):
    """Returns a list of the names of tar files in the archive.

    :param archive_path: The path of the archive to be opened.
    :returns a list of the files in the tar archive
    """
    archive = read_tar_archive(archive_path)
    return archive.getnames()


def tar_member(archive_path, member_path):
    """Returns a readable view to a file in a tar archive.

    :param archive_path: The path of the archive to be opened.
    :param member_path: The path of the desired file inside the tar archive.
    :returns io.BufferedReader view of the tar file.
    """
    archive = read_tar_archive(archive_path)
    return archive.extractfile(member_path)



def apply_reaction(state, reaction):
    """Applies a reaction to the input state in place."""
    if len(reaction) == 3:
        state[tuple(reaction["position"])] = reaction["to"]
    elif len(reaction) == 2:
        (
            state[tuple(reaction["from"])],
            state[tuple(reaction["to"])],
        ) = (
            state[tuple(reaction["to"])],
            state[tuple(reaction["from"])],
        )

def reverse_reaction(state, reaction):
    """Reverses a reaction to the input state in place."""
    if len(reaction) == 3:
        state[tuple(reaction["position"])] = reaction["from"]
    elif len(reaction) == 2:
        (
            state[tuple(reaction["from"])],
            state[tuple(reaction["to"])],
        ) = (
            state[tuple(reaction["to"])],
            state[tuple(reaction["from"])],
        )


def counts(initial_conditions, reactions):
    """Calculates the number of each species initially and after each reaction.

    :param initial_conditions: The initial conditions of the simulation.
    :param reactions: The reactions that took place.
    :returns counts: an array of shape (3, len(reactions) + 1) containing the number of each species
    at each step in the reaction.
    """
    counts = np.zeros((3, len(reactions) + 1))
    for row in initial_conditions:
        for entry in row:
            counts[entry, :] += 1

    reaction_deltas = np.zeros(counts.shape)
    for i, reaction in enumerate(reactions):
        if reaction[TYPE] == DIFFUSION:
            pass
        elif reaction[TYPE] == POINT_CHANGE:
            reaction_deltas[reaction["from"], i+1] = -1
            reaction_deltas[reaction["to"], i+1] = 1

    cum_reaction_deltas = np.cumsum(reaction_deltas, axis=1)
    counts += cum_reaction_deltas
    return counts


def durations_and_times(reactions):
    """Calculates the duration between each reaction and the time each took place.

    :param reactions: The reactions that took place.
    :returns (durations, times), where durations[i] is the time that the system was in state i, and
    times[i] is the last time that the system was in state i. State 0 is the initial state and
    state i > 0 is the state after the i-1th reaction.
    """
    durations = np.array([reaction["dt"] for reaction in reactions] + [0])
    times = np.cumsum(durations)

    return durations, times
