import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

import load


IMAGE_FORMAT = {"cmap": "gray", "vmin": 0, "vmax": 2}
E_COLOR = "gray"
I_COLOR = "blue"
B_COLOR = "red"
COLORS = [E_COLOR, I_COLOR, B_COLOR]
E_LINE_FORMAT = {"color": E_COLOR, "linestyle": "-"}
I_LINE_FORMAT = {"color": I_COLOR, "linestyle": "-"}
B_LINE_FORMAT = {"color": B_COLOR, "linestyle": "-"}
LINE_FORMATS = [E_LINE_FORMAT, I_LINE_FORMAT, B_LINE_FORMAT]


def time_slider(axes, times, apply_reaction, reverse_reaction, update, current_index=0):
    time_slider = Slider(
        ax=axes,
        label='Time (sec)',
        valmin=times[0],
        valmax=times[-1],
        valstep=times,
    )

    def set_time(time):
        nonlocal current_index
        target_index = np.searchsorted(times, time)
        while current_index < target_index:
            apply_reaction(reactions[current_index])
            current_index += 1
        while current_index > target_index:
            reverse_reaction(reactions[current_index-1])
            current_index -= 1
        update()

    time_slider.on_changed(set_time)

    return time_slider


def interactive_image(initial_conditions, reactions):
    """An interactive image plot of a simulation.

    :param initial_conditions: The initial conditions of the simulation.
    :param reactions: The reactions that occurred during the simulation.
    """
    fig, ax = plt.subplots()

    current_conditions = initial_conditions.copy()
    axes_image = ax.imshow(current_conditions, **IMAGE_FORMAT)

    times = np.cumsum(np.array([reaction["dt"] for reaction in reactions]))

    time_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
    def apply_reaction(reaction):
        load.apply_reaction(current_conditions, reaction)

    def reverse_reaction(reaction):
        load.reverse_reaction(current_conditions, reaction)

    _slider = time_slider(
        time_ax,
        times,
        apply_reaction,
        reverse_reaction,
        lambda: axes_image.set_data(current_conditions),
    )

    plt.show()


def fractions(initial_conditions, reactions):
    """A plot of the fractions of each species as a function of time.

    :param initial_conditions: The initial conditions of the simulation.
    :param reactions: The reactions that occurred during the simulation.
    """
    fig, ax = plt.subplots()

    counts = load.counts(initial_conditions, reactions)
    fractions = counts / initial_conditions.size

    durations, times = load.durations_and_times(reactions)

    ax.step(times, fractions[load.EMPTY], **E_LINE_FORMAT)
    ax.step(times, fractions[load.INERT], **I_LINE_FORMAT)
    ax.step(times, fractions[load.BONDING], **B_LINE_FORMAT)

    ax.set(xlabel="Time (1/D)", ylabel="$\\phi_i$")

    plt.show()


def fraction_time_correlation(initial_conditions, reactions, species="Bonding", num_samples=1000, num_integration_points=5000, equilibration_time=None):
    """Plots the time correlation of the fraction of a species as a function of time delta.

    :param initial_conditions: The initial conditions of the simulation.
    :param reactions: The reactions that occurred during the simulation.
    :param species: The species to be plotted. Default Bonding.
    :param equilibration_time: The time after which the system should be assumed to be at
    equilibrium. Defaults to half of the simulation time.
    :param num_samples: The number of correlation_samples to graph. Defaults to 800.
    :param window_size: The maximum delta time allowed for a given sample. Default 800.
    """
    fig, ax = plt.subplots()

    counts = load.counts(initial_conditions, reactions)
    fractions = counts / initial_conditions.size

    durations, times = load.durations_and_times(reactions)

    if equilibration_time is None:
        equilibration_index = times.size // 2
    else:
        equilibration_index = np.searchsorted(times, equilibration_time)

    if type(species) == str:
        species = load.TRANSLATE[species]

    fractions = fractions[species, equilibration_index:]
    times = times[equilibration_index:]
    durations = durations[equilibration_index:]

    mean = np.average(fractions, weights=durations)
    fractions = fractions - mean
    stdev = np.sqrt(np.average((fractions) ** 2, weights=durations))
    fractions = fractions / stdev

    time_deltas = np.linspace(0, (np.max(times) - np.min(times)) / 5000, num_samples)
    correlations = np.zeros(time_deltas.size)
    integration_indicies = np.arange(num_integration_points)
    for j, delta_t in enumerate(time_deltas):
        ts = np.min(times) * (1 - integration_indicies / num_integration_points) + (np.max(times) - delta_t) * (integration_indicies / num_integration_points)
        correlations[j] = np.sum(fractions[np.searchsorted(times, ts)] * fractions[np.searchsorted(times, ts + delta_t)]) / num_integration_points

    ax.scatter(time_deltas, correlations, marker=".", c=COLORS[species])

    def decay(t, tau):
        return np.exp(- t / tau)

    popt, pcov = scipy.optimize.curve_fit(decay, time_deltas, correlations, p0=(10.))

    print(popt)

    ax.plot(time_deltas, decay(time_deltas, *popt), "g--")

    plt.show()


def graph_sizes(sizes):
    plt.step(sizes[:,0], sizes[:,1], where="post")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("plot")
    parser.add_argument("-f", "--filename", required=True)
    parser.add_argument(
        "-i", "--interactive-image",
        action="store_true",
        help="An interactive image plot.",
    )
    parser.add_argument(
        "--fractions",
        action="store_true",
        help="A trace of the number of each type of particle.",
    )
    parser.add_argument(
        "--time-correlation",
        choices=load.SPECIES,
        help="Determines the time correlation of the specified species",
        metavar="STATE"
    )
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="A trace of the size of the largest droplet over time",
    )
    args = parser.parse_args()

    print("Loading...")
    directory = load.unpack_natural_input(args.filename)
    initial_conditions = load.initial_conditions(directory)
    reactions = load.reactions(directory)
    final_conditions = load.final_state(directory)

    if args.interactive_image:
        print("Graphing Interactive Image...")
        interactive_image(initial_conditions, reactions)

    if args.fractions:
        print("Graphing Fractions...")
        fractions(initial_conditions, reactions)

    if args.time_correlation is not None:
        print("Graphing Time Correlations...")
        fraction_time_correlation(initial_conditions, reactions, species=args.time_correlation)

    if args.sizes:
        print("Graphing Sizes...")
        sizes = load.sizes(directory)
        graph_sizes(sizes)
        
    import shutil
    shutil.rmtree(load.TEMP_ARCHIVE_PATH)
