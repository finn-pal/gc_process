import argparse
import json
import os

import gc_utils  # type: ignore
import numpy as np
import pandas as pd
import utilities as ut


def get_main_halo_tid(halt, final_snapshot: int = 600):
    print("Retrieving main halo id")

    snap_mask = halt["snapshot"] == final_snapshot
    main_halo_idx = halt["host.index"][snap_mask][0]
    main_halo_tid = int(halt["tid"][main_halo_idx])

    return main_halo_tid


def get_sim_offset(halt, main_halo_tid):
    print("Retrieving snapshot offset")

    main_halo_idx = np.where(halt["tid"] == main_halo_tid)[0][0]

    mask = halt["final.index"] == main_halo_idx
    snapshot_offset = int(np.min(halt["snapshot"][mask]))

    return snapshot_offset


def get_cosmology(halt):
    print("Retrieving cosmology")

    Om = float(halt.Cosmology["omega_matter"])
    Ob = float(halt.Cosmology["omega_baryon"])
    h100 = float(halt.Cosmology["hubble"])

    return Om, Ob, h100


def define_halo_centering_snapshots(
    halt, sim: str, sim_dir: str, main_halo_tid: int, central_difference_limit: float = 1
):
    print("Retrieving dm centers")

    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    snap_pub_dir = sim_dir + "/snapshot_times_public.txt"
    snap_pub_data = pd.read_table(snap_pub_dir, comment="#", header=None, sep=r"\s+")
    snap_pub_data.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]
    snap_lst = snap_pub_data["index"]

    dm_center_lst = []
    for snapshot in snap_lst:
        # get halt (dm) center
        halo_at_snap = gc_utils.get_halo_prog_at_snap(halt, main_halo_tid, snapshot)
        halo_idx = np.where(halt["tid"] == halo_at_snap)[0][0]
        halt_center_pos = halt["position"][halo_idx]

        # get centering from star position (how it is done in prop("host.distance"))
        part = gc_utils.open_snapshot(snapshot, fire_dir, species=["star"], assign_hosts_rotation=False)
        part_center_pos = ut.particle.get_center_positions(
            part,
            species_name="star",
            center_positions=halt_center_pos,
            return_single_array=False,
            verbose=False,
        )

        # get distance between centers
        dist = ut.coordinate.get_distances(
            halt_center_pos,
            part_center_pos,
            part.info["box.length"],
            part.snapshot["scalefactor"],
            total_distance=True,
        )  # [kpc physical]

        del part

        # if distance between distances is greater than preset limit then centering should be on halt (dm)
        # center
        if dist > central_difference_limit:
            dm_center_lst.append(int(snapshot))

    return dm_center_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")

    args = parser.parse_args()

    sim = args.simulation
    location = args.location

    if location == "local":
        sim_dir = "../../simulations/"

    elif location == "katana":
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"

    elif location == "one_touch":
        sim_dir = "/Volumes/One Touch/simulations/"

    else:
        raise RuntimeError("Incorrect location provided. Must be local, katana or one_touch.")

    file_path = sim_dir + "simulation_codes.json"
    if not os.path.exists(file_path):
        # Step 2: Create it with an empty list
        with open(file_path, "w") as f:
            json.dump({}, f, indent=2)

    halt = gc_utils.get_halo_tree(sim, sim_dir, assign_hosts_rotation=False)

    main_halo_tid = get_main_halo_tid(halt)
    snap_offset = get_sim_offset(halt, main_halo_tid)
    Om, Ob, h100 = get_cosmology(halt)
    dm_center = define_halo_centering_snapshots(halt, sim, sim_dir, main_halo_tid)

    del halt

    # after
    with open(file_path, "r+") as f:
        data = json.load(f)

    # Add or update keys
    data[sim] = {
        "halo": main_halo_tid,
        "offset": snap_offset,
        "Om": Om,
        "Ob": Ob,
        "h100": h100,
        "dm_center": dm_center,
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
