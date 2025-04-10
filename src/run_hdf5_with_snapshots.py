import argparse
import json
import multiprocessing as mp
import os

import gc_utils  # type: ignore
import h5py
import numpy as np
import pandas as pd

from tools.get_basic_kinematics_at_snap import get_basic_kinematics
from tools.get_mass_at_snap import get_gc_masses_at_snap


def create_hdf5(sim: str, it_lst: list[int], sim_dir: str):
    save_file = sim_dir + sim + "/" + sim + "_processed.hdf5"  # save location

    if not os.path.exists(save_file):
        h5py.File(save_file, "w")

    with h5py.File(save_file, "a") as hdf:
        for it in it_lst:
            it_id = gc_utils.iteration_name(it)
            data_file = sim_dir + sim + "/gc_results/interim/" + it_id + ".json"
            with open(data_file, "r") as sim_json:
                int_data = json.load(sim_json)
            data_dict = int_data[it_id]["source"]

            if it_id in hdf.keys():
                grouping = hdf[it_id]
            else:
                grouping = hdf.create_group(it_id)
            if "source" in grouping.keys():
                source = grouping["source"]
            else:
                source = grouping.create_group("source")
            for key in data_dict.keys():
                if key in source.keys():
                    del source[key]
                if key == "ptype":
                    source.create_dataset(key, data=data_dict[key])
                else:
                    source.create_dataset(key, data=np.array(data_dict[key]))

    # hdf.close()


def add_mass_hdf5(simulation, it_lst: list[int], result_dict: dict, sim_dir: str):
    proc_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "a")  # open processed data file

    for it in it_lst:
        it_id = gc_utils.iteration_name(it)
        if it_id in proc_data.keys():
            it_grouping = proc_data[it_id]
        else:
            it_grouping = proc_data.create_group(it_id)
        if "snapshots" in it_grouping.keys():
            snap_groups = it_grouping["snapshots"]
        else:
            snap_groups = it_grouping.create_group("snapshots")
        for snap_id in result_dict[it_id].keys():
            if snap_id in snap_groups.keys():
                snapshot = snap_groups[snap_id]
            else:
                snapshot = snap_groups.create_group(snap_id)
            for key in result_dict[it_id][snap_id].keys():
                if key in snapshot.keys():
                    del snapshot[key]
                if key == "ptype":
                    snapshot.create_dataset(key, data=result_dict[it_id][snap_id][key])
                else:
                    snapshot.create_dataset(key, data=np.array(result_dict[it_id][snap_id][key]))

    proc_data.close()


def kin_main(
    sim: str,
    it_lst: list[int],
    snapshot: int,
    main_halo_tid: int,
    sim_dir: str,
    add_exsitu_halo_details: bool = True,
    shared_dict: dict = {},
):
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    # only need dark and star as gc can't be gas particles
    part = gc_utils.open_snapshot(snapshot, fire_dir, species=["dark", "star"], assign_hosts_rotation=True)
    get_basic_kinematics(
        part, sim, it_lst, snapshot, main_halo_tid, sim_dir, add_exsitu_halo_details, shared_dict
    )
    del part


def add_kinematics_hdf5(simulation, it_lst: list[int], snap_lst: list[int], result_dict: dict, sim_dir: str):
    proc_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "a")  # open processed data file

    for it in it_lst:
        it_id = gc_utils.iteration_name(it)
        if it_id in proc_data.keys():
            it_grouping = proc_data[it_id]
        else:
            it_grouping = proc_data.create_group(it_id)
        if "snapshots" in it_grouping.keys():
            snap_groups = it_grouping["snapshots"]
        else:
            snap_groups = it_grouping.create_group("snapshots")
        for snap in snap_lst:
            snap_id = gc_utils.snapshot_name(snap)
            if snap_id in snap_groups.keys():
                snapshot = snap_groups[snap_id]
            else:
                snapshot = snap_groups.create_group(snap_id)
            for key in result_dict[snap_id][it_id].keys():
                if key in snapshot.keys():
                    del snapshot[key]
                snapshot.create_dataset(key, data=result_dict[snap_id][it_id][key])

    proc_data.close()


######################################################################################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-a", "--iteration_low_limit", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-b", "--iteration_up_limit", required=True, type=int, help="upper bound iteration")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    parser.add_argument(
        "-e", "--exsitu", required=False, default=True, type=str2bool, help="add the exsitu gc halo details"
    )
    parser.add_argument(
        "-r",
        "--create_hdf5",
        required=False,
        default=True,
        type=str2bool,
        help="create hdf5 and add mass and accretion details",
    )
    parser.add_argument(
        "-k",
        "--get_basic_kinematics",
        required=False,
        default=True,
        type=str2bool,
        help="get basic gc kinematics from the FIRE simulation",
    )
    parser.add_argument(
        "-n",
        "--snapshots",
        required=False,
        nargs="+",
        type=int,
        help="list of snapshots of interest, if None provided will default to all publicly available",
    )
    args = parser.parse_args()

    it_up = args.iteration_up_limit
    it_low = args.iteration_low_limit
    it_rng = it_up - it_low
    it_lst = np.linspace(it_low, it_up, it_rng + 1, dtype=int)

    sim = args.simulation
    location = args.location

    if location == "local":
        sim_dir = "../../simulations/"

    elif location == "katana":
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"

    else:
        raise RuntimeError("Incorrect location provided. Must be local or katana.")

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)

    public_snapshot_file = sim_dir + "snapshot_times_public.txt"
    pub_data = pd.read_table(public_snapshot_file, comment="#", header=None, sep=r"\s+")
    pub_data.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]
    pub_snaps = np.array(pub_data["index"], dtype=int)
    snap_lst = pub_snaps

    cores = args.cores
    if cores is None:
        cores = 8

    if args.create_hdf5:
        create_hdf5(sim, it_lst, sim_dir)

        snap_offset = sim_data[sim]["offset"]

        with mp.Manager() as manager:
            shared_dict = manager.dict()  # Shared dictionary across processes
            mass_args = [(sim, snap_offset, it, snap_lst, sim_dir, shared_dict) for it in it_lst]

            with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
                pool.starmap(get_gc_masses_at_snap, mass_args, chunksize=1)

            final_dict = dict(shared_dict)

        add_mass_hdf5(sim, it_lst, final_dict, sim_dir)

    if args.get_basic_kinematics:
        main_halo_tid = [sim_data[sim]["halo"]]

        kin_snap_lst = args.snapshots
        if kin_snap_lst is None:
            kin_snap_lst = pub_snaps

        add_exsitu_halo_details = args.exsitu

        with mp.Manager() as manager:
            shared_dict = manager.dict()  # Shared dictionary across processes
            # args = [(sim, it_lst, snap_group, sim_dir, data_dir, shared_dict) for snap_group in snap_groups]
            kin_args = [
                (sim, it_lst, snap, main_halo_tid, sim_dir, add_exsitu_halo_details, shared_dict)
                for snap in kin_snap_lst
            ]

            with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
                pool.starmap(kin_main, kin_args, chunksize=1)

            kin_result_dict = dict(shared_dict)

        add_kinematics_hdf5(sim, it_lst, kin_snap_lst, kin_result_dict, sim_dir)
