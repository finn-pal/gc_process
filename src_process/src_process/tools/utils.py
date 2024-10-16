import os
import sys

import gizmo_analysis as gizmo
import halo_analysis as halo
import numpy as np
from tqdm import tqdm


def block_print():
    """
    Prevents prining of statements. Useful for when using halo tools as a lot of erroneous print statements.
    """

    sys.stdout = open(os.devnull, "w")


def enable_print():
    """
    Enables prints statements.
    """

    sys.stdout = sys.__stdout__


def get_halo_cid(halt, halo_tid: int, fire_dir: str) -> tuple[int, int]:
    """
    Finds the corresponding halo catalogue id (cid) from a provided halo id from the halo tree (tid).

    Args:
        halt (_type_): Halo tree
        halo_tid (int): Halo id from the halo tree
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100")

    Returns:
        tuple[int, int]: Returns the halo catalogue id (cid) and the corresponding snapshot
    """

    # get index of halo in halo tree
    idx = np.where(halt["tid"] == halo_tid)[0][0]
    # get the corresponding snapshot (halo catalogue) and index of the halo in the halo catalogue
    snap = halt["snapshot"][idx]
    halo_idx = halt["catalog.index"][idx]
    # import the relevant halo catalogue
    hal = halo.io.IO.read_catalogs("index", snap, simulation_directory=fire_dir, species=None)
    # get the halo catalogue id (cid)
    halo_cid = hal["id"][halo_idx]
    return halo_cid, snap


def main_prog_halt(halt, main_halo_tid) -> list[int]:
    """
    Get a list of the halo tree ids for the most massive progenitors of the main galaxy from gizmo halo tree

    Args:
        halt (_type_): Halo tree

    Returns:
        list[int]: List of halo tree halo ids (tid) tracing the main progenitors of the most massive galaxy at
        z = 0.
    """
    # main galaxy has index 0 in the halo tree
    main_halo_idx = np.where(halt["tid"] == main_halo_tid)[0][0]
    main_halo_lst = [main_halo_idx]

    # FIRE has 600 snapshots but progenitor has usally not formed much earlier than snapshot 10
    for _ in range(1, 590):
        idx = halt["progenitor.main.index"][main_halo_lst[-1]]
        main_halo_lst.append(idx)

    tid_main_lst = halt["tid"][main_halo_lst]

    return tid_main_lst


def naming_value(flag) -> int:
    """
    Converts flag value to interger for input into file naming for processed data

    Args:
        flag (_type_): Flag value (can be 0, 1, or None)

    Returns:
        int: Flag for file name (0, 1 or 2)
    """
    if flag is None:
        flag_name = 2
    else:
        flag_name = flag
    return flag_name


def get_descendants_halt(halo_tid: int, halt) -> list[int]:
    """
    Get list of descendents (tid's) of the halo in question

    Args:
        halo_tid (int): Halo id from the halo tree
        halt (_type_): Halo tree

    Returns:
        list[int]: A list of descendant halos (list of tid's)
    """
    halo_idx = np.where(halt["tid"] == halo_tid)[0][0]
    halo_snap = halt["snapshot"][halo_idx]
    desc_lst = [halo_idx]

    # get a list of all descendents of the halo up to z = 0
    for _ in range(halo_snap, 600):
        idx = halt["descendant.index"][desc_lst[-1]]
        desc_lst.append(idx)

    return desc_lst


def iteration_name(it: int) -> str:
    """
    Get 3 digit iteration id name from iteration number

    Args:
        it (int): Iteration number.

    Returns:
        str: Iteration id name.
    """
    # ensure group naming is consitent with three digits
    if len(str(it)) == 1:
        it_id = "it00" + str(it)
    elif len(str(it)) == 2:
        it_id = "it0" + str(it)
    else:
        it_id = "it" + str(it)

    return it_id


def get_halo_tree(sim: str, sim_dir: str):
    """
    Given directory to simulation returns halo tree.

    Args:
        sim (str): Simulation of interest (of form "m12i").
        sim_dir (str): Directory of the simulation data.

    Returns:
        _type_: Halo tree.
    """
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    for _ in tqdm(range(1), ncols=150, desc="Retrieving Halo Tree....................."):
        block_print()  # block verbose print statements
        halt = halo.io.IO.read_tree(simulation_directory=fire_dir)
        enable_print()

    return halt


def open_snapshot(snapshot: int, fire_dir: str):
    """
    Get the particle details for a given snapshot. This contains all particles in publicly available snapshots
    with the coordinate frame to be centred on the central galaxy.

    Args:
        snapshot (int): Snapshot.
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100").

    Returns:
        _type_: The particle details for a given snapshot
    """
    for _ in tqdm(range(1), ncols=150, desc="Retrieving Snapshot %d.................." % snapshot):
        block_print()  # block verbose print statements
        part = gizmo.io.Read.read_snapshots("all", "index", snapshot, fire_dir, assign_hosts_rotation=True)
        enable_print()
    return part


def snapshot_name(snap_num: int):
    """
    Get 3 digit iteration id name from snapshot.

    Args:
        snap_num (int): Snapshot.

    Returns:
        str: Snapshot id name.
    """
    # ensure group naming is consitent with three digits
    if len(str(snap_num)) == 1:
        snap_id = "snap00" + str(snap_num)
    elif len(str(snap_num)) == 2:
        snap_id = "snap0" + str(snap_num)
    else:
        snap_id = "snap" + str(snap_num)

    return snap_id
