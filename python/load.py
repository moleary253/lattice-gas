import json
import os, sys, shutil
import tarfile
import numpy as np


CHAIN_FILE = "chain.json"
INITIAL_CONDITIONS_FILE = "initial_conditions.json"
REACTIONS_FILE = "reactions.json"
FINAL_STATE_FILE = "final_state.json"

TYPE = "reaction_type"
POINT_CHANGE = "PointChange"
DIFFUSION = "Diffusion"

EMPTY = 0
INERT = 1
BONDING = 2
SPECIES = ["Empty", "Inert", "Bonding"]
TRANSLATE = {"Empty": EMPTY, "Inert": INERT, "Bonding": BONDING}

TEMP_ARCHIVE_PATH = "data/current"


def natural_input(string, clear_directory_after=True):
    """Loads a simulation from a natural text-based input.

    :param string: The natural input. If it is a directory, the simulation in that directory will be
    loaded. If it is of the form '[x].tar.gz/[y]', then the simulation stored in the tarfile [x]
    located at path at [y] will be loaded.
    :param clear_directory_after: Default True. Whether to delete the extracted tar archive, if
    applicable.
    :returns (parameters, initial_conditions, reactions, final_state) where parameters is the
    parameters to the Markov chain for the simulation, initial conditions is the initial conditions
    of the simulation, reactions is a time series of the reactions that occurred during the
    simulation, and final_state is the final state of the simulation.
    """
    if ".tar.gz" in string:
        index = string.find(".tar.gz")+len(".tar.gz")
        tar_archive = string[:index]
        member_location = string[index+1:]
        unpack_tar_archive(tar_archive, TEMP_ARCHIVE_PATH)
        ret = directory(os.path.join(TEMP_ARCHIVE_PATH, member_location))
        if clear_directory_after:
            shutil.rmtree(TEMP_ARCHIVE_PATH)
        return ret
    else:
        return directory(string)


def directory(directory):
    """Loads a simulation from a directory.

    :param directory: The directory the simulation is located in.
    :returns (parameters, initial_conditions, reactions) where parameters is the parameters to the
    Markov chain for the simulation, initial conditions is the initial conditions of the simulation,
    and reactions is a time series of the reactions that occurred during the simulation.
    """
    return (
        chain(directory),
        initial_conditions(directory),
        reactions(directory),
        final_state(directory),
    )


def chain(directory):
    """Loads the parameters of the Markov chain governing rates in the simulation.

    :param directory: The directory the simulation is located in.
    :returns a dictionary of the parameters of the Markov chain.
    """
    with open(os.path.join(directory, CHAIN_FILE)) as f:
        chain_json = json.load(f)

    return chain_json


def initial_conditions(directory):
    """Loads the initial conditions of a simulation.

    :param directory: The directory the simulation is located in.
    :returns a 2d array of integers representing the initial state at each site in the simulation.
    """
    with open(os.path.join(directory, INITIAL_CONDITIONS_FILE)) as f:
        initial_conditions_json = json.load(f)
    
    initial_conditions = np.array([
        TRANSLATE[entry]
        for entry in initial_conditions_json["data"]
    ]).reshape(initial_conditions_json["dim"])

    return initial_conditions


def reactions(directory):
    """Loads the reactions that took place in a simulation.

    :param directory: The directory the simulation is located in.
    :returns a time series of the reactions that occurred during the simulation. This is an array of
    dictionaries of which can take the form
    {
        "dt": float dt,
        "from": int state,
        "to": int state,
        "position": [int x, int y]
    }
    """
    with open(os.path.join(directory, REACTIONS_FILE)) as f:
        reactions_json = json.load(f)

    reactions = [
        {
            TYPE: POINT_CHANGE,
            "dt": dt,
            "from": TRANSLATE[reaction[POINT_CHANGE]["from"]],
            "to": TRANSLATE[reaction[POINT_CHANGE]["to"]],
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


def final_state(directory):
    """Loads the final state of a simulation.

    :param directory: The directory the simulation is located in.
    :returns a 2d array of integers representing the initial state at each site in the simulation,
    or None if no final state was stored.
    """
    if not os.path.exists(os.path.join(directory, FINAL_STATE_FILE)):
        return None

    with open(os.path.join(directory, FINAL_STATE_FILE)) as f:
        final_state_json = json.load(f)
    
    final_state= np.array([
        TRANSLATE[entry]
        for entry in final_state_json["data"]
    ]).reshape(final_state_json["dim"])

    return final_state


def read_tar_archive(archive_path):
    """Reads a tar archive into a file that can be used to read from the archive.

    :param archive_path: The path of the archive to be opened.
    :returns a tarfile.TarFile.
    """
    return tarfile.open(archive_path, mode="r:gz")


def unpack_tar_archive(archive_path, target_path):
    """Unpacks a tar archive into a directory.

    :param archive_path: The path of the archive to be opened.
    :param target_path: Where to put the extracted files.
    """
    archive = read_tar_archive(archive_path)
    archive.extractall(path=target_path, filter="data")


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
    if reaction[TYPE] == POINT_CHANGE:
        state[tuple(reaction["position"])] = reaction["to"]
    elif reaction[TYPE] == DIFFUSION:
        (
            state[tuple(reaction["from"])],
            state[tuple(reaction["to"])],
        ) = (
            state[tuple(reaction["to"])],
            state[tuple(reaction["from"])],
        )

def reverse_reaction(state, reaction):
    """Reverses a reaction to the input state in place."""
    if reaction[TYPE] == POINT_CHANGE:
        state[tuple(reaction["position"])] = reaction["from"]
    elif reaction[TYPE] == DIFFUSION:
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
