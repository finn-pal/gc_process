import argparse
import json
import multiprocessing as mp

import gc_utils  # type: ignore
import h5py
import numpy as np
import pandas as pd

from tools.get_basic_kinematics_at_snap import get_basic_kinematics


def main(
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-a", "--iteration_low_limit", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-b", "--iteration_up_limit", required=True, type=int, help="upper bound iteration")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    parser.add_argument("-e", "--exsitu", required=False, type=bool, help="add the exsitu gc halo details")
    parser.add_argument(
        "-n",
        "--snapshots",
        required=False,
        nargs="+",
        type=int,
        help="list of snapshots of interest, if None provided will default to all publicly available",
    )

    args = parser.parse_args()

    it_min = args.iteration_low_limit
    it_max = args.iteration_up_limit
    it_lst = np.linspace(it_min, it_max, it_max - it_min + 1, dtype=int)

    sim = args.simulation
    location = args.location

    if location == "local":
        sim_dir = "../../simulations/"
        # data_dir = "data/"

    elif location == "katana":
        # data_dir = "/srv/scratch/astro/z5114326/gc_process/data/"
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"

    # model_snaps = sim_dir + "model_snapshots.json"
    # with open(model_snaps) as snap_json:
    #     snap_data = json.load(snap_json)

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

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as json_file:
        sim_data = json.load(json_file)

    main_halo_tid = [sim_data[sim]["halo"]]

    snap_lst = args.snapshots
    if snap_lst is None:
        # snap_lst = snap_data["analyse_snapshots"]
        # snap_lst = snap_data["public_snapshots"]
        snap_lst = pub_snaps

    cores = args.cores
    if cores is None:
        # 4 cores is max to run with 64 GB RAM
        # cores = mp.cpu_count()
        cores = 8

    # could default instead but have set it up like this
    add_exsitu_halo_details = args.exsitu
    if add_exsitu_halo_details is None:
        add_exsitu_halo_details = True

    with mp.Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary across processes
        # args = [(sim, it_lst, snap_group, sim_dir, data_dir, shared_dict) for snap_group in snap_groups]
        args = [
            (sim, it_lst, snap, main_halo_tid, sim_dir, add_exsitu_halo_details, shared_dict)
            for snap in snap_lst
        ]

        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            pool.starmap(main, args, chunksize=1)

        result_dict = dict(shared_dict)

    add_kinematics_hdf5(sim, it_lst, snap_lst, result_dict, sim_dir)
