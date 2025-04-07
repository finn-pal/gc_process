import argparse
import json
import multiprocessing as mp

import h5py
import numpy as np
from gc_utils import iteration_name, open_snapshot, snapshot_name  # type: ignore

from tools.get_basic_kinematics_at_snap import get_basic_kinematics


def main(
    simulation: str,
    it_lst: list[int],
    snapshot: int,
    sim_dir: str,
    shared_dict: dict = {},
):
    fire_dir = sim_dir + simulation + "/" + simulation + "_res7100/"

    part = open_snapshot(snapshot, fire_dir)
    get_basic_kinematics(part, simulation, it_lst, snapshot, sim_dir, shared_dict)
    del part


def add_kinematics_hdf5(simulation, it_lst: list[int], snap_lst: list[int], result_dict: dict, sim_dir: str):
    proc_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "a")  # open processed data file

    for it in it_lst:
        it_id = iteration_name(it)
        if it_id in proc_data.keys():
            it_grouping = proc_data[it_id]
        else:
            it_grouping = proc_data.create_group(it_id)
        if "snapshots" in it_grouping.keys():
            snap_groups = it_grouping["snapshots"]
        else:
            snap_groups = it_grouping.create_group("snapshots")
        for snap in snap_lst:
            snap_id = snapshot_name(snap)
            if snap_id in snap_groups.keys():
                snapshot = snap_groups[snap_id]
            else:
                snapshot = snap_groups.create_group(snap_id)
            for key in result_dict[snap_id][it_id].keys():
                if key in snapshot.keys():
                    del snapshot[key]
                else:
                    snapshot.create_dataset(key, data=result_dict[snap_id][it_id][key])

    proc_data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-a", "--iteration_low_limit", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-b", "--iteration_up_limit", required=True, type=int, help="upper bound iteration")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
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
        data_dir = "data/"
        model_snaps = data_dir + "external/model_snapshots.json"

    elif location == "katana":
        data_dir = "/srv/scratch/astro/z5114326/gc_process/data/"
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"
        model_snaps = data_dir + "external/model_snapshots.json"

    with open(model_snaps) as snap_json:
        snap_data = json.load(snap_json)

    snap_lst = args.snapshots
    if snap_lst is None:
        # snap_lst = snap_data["analyse_snapshots"]
        snap_lst = snap_data["public_snapshots"]

    cores = args.cores
    if cores is None:
        # 4 cores is max to run with 64 GB RAM
        cores = mp.cpu_count()

    with mp.Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary across processes
        # args = [(sim, it_lst, snap_group, sim_dir, data_dir, shared_dict) for snap_group in snap_groups]
        args = [(sim, it_lst, snap, sim_dir, shared_dict) for snap in snap_lst]

        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            pool.starmap(main, args, chunksize=1)

        result_dict = dict(shared_dict)

    add_kinematics_hdf5(sim, it_lst, snap_lst, result_dict, sim_dir)
