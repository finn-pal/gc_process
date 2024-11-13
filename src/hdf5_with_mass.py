import argparse
import json
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
from gc_utils import iteration_name  # type: ignore

from tools.get_mass_at_snap import get_gc_masses_at_snap


def create_hdf5(simulation: str, it_lst: list[int], sim_dir: str, data_dir: str):
    save_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"  # save location

    if not os.path.exists(save_file):
        h5py.File(save_file, "w")

    with h5py.File(save_file, "a") as hdf:
        for it in it_lst:
            it_id = iteration_name(it)
            data_file = data_dir + "results/" + simulation + "/interim/" + it_id + ".json"
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

    hdf.close()


def add_mass_hdf5(simulation, it_lst: list[int], result_dict: dict, sim_dir: str):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-a", "--iteration_low_limit", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-b", "--iteration_up_limit", required=True, type=int, help="upper bound iteration")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    args = parser.parse_args()

    it_up = args.iteration_up_limit
    it_low = args.iteration_low_limit
    it_rng = it_up - it_low
    it_lst = np.linspace(it_low, it_up, it_rng + 1, dtype=int)

    sim = args.simulation
    location = args.location

    if location == "local":
        sim_dir = "/Users/z5114326/Documents/simulations/"
        data_dir = "/Users/z5114326/Documents/GitHub/gc_process_katana/data/"
        sim_codes = data_dir + "external/simulation_codes.json"
        model_snaps = data_dir + "external/model_snapshots.json"

    elif location == "katana":
        data_dir = "/srv/scratch/astro/z5114326/gc_process/data/"
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"
        sim_codes = data_dir + "external/simulation_codes.json"
        model_snaps = data_dir + "external/model_snapshots.json"

    else:
        print("Incorrect location provided. Must be local or katana.")
        sys.exit()

    create_hdf5(sim, it_lst, sim_dir, data_dir)

    # start adding masses at all available snapshots
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)

    with open(model_snaps) as snap_json:
        snap_data = json.load(snap_json)

    snap_offset = sim_data[sim]["offset"]
    snap_lst = snap_data["public_snapshots"]

    cores = args.cores
    if cores is None:
        cores = mp.cpu_count()

    with mp.Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary across processes
        args = [(sim, snap_offset, it, snap_lst, sim_dir, data_dir, shared_dict) for it in it_lst]

        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            pool.starmap(get_gc_masses_at_snap, args, chunksize=1)

        final_dict = dict(shared_dict)

    add_mass_hdf5(sim, it_lst, final_dict, sim_dir)
