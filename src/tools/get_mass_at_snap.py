import sys

import h5py
import numpy as np
import pandas as pd
from gc_utils import iteration_name, particle_type, snapshot_name  # type: ignore


def get_gc_masses_at_snap(
    simulation: str,
    offset: int,
    iteration_list: list[int],
    snapshot_list: list[int],
    sim_dir: str,
    data_dir: str,
):
    proc_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "a")  # open processed data file

    for it in iteration_list:
        raw_dir = data_dir + "results/" + simulation + "/raw/it_%d/" % it
        it_id = iteration_name(it)

        gc_id_lst = proc_data[it_id]["source"]["gc_id"]
        analyse_flag_lst = proc_data[it_id]["source"]["analyse_flag"]
        group_id_lst = proc_data[it_id]["source"]["group_id"]
        snap_acc_lst = proc_data[it_id]["source"]["snap_acc"]

        for snapshot in snapshot_list:
            snap_id = snapshot_name(snapshot)

            snap_mass_file = raw_dir + "allcat_s-%d_p2-7_p3-1_k-1.5_logm_snap%d.txt" % (it, snapshot - offset)
            gc_id_file = raw_dir + "allcat_s-%d_p2-7_p3-1_gcid.txt" % it

            gcid_df = pd.read_csv(gc_id_file, sep=" ").drop(["ID"], axis=1)
            gcid_df.columns = ["GC_ID", "quality"]

            mass_snap_df = pd.read_csv(snap_mass_file, header=None)
            mass_snap_df.columns = ["mass"]

            comb_df = pd.concat([gcid_df, mass_snap_df], axis=1)
            filt_comb_df = comb_df[comb_df["mass"] != -1]

            gc_id_snap = [gc_id for gc_id in filt_comb_df["GC_ID"]]
            mass_snap = [mass for mass in filt_comb_df["mass"]]
            ptype_snap = [particle_type(quality) for quality in filt_comb_df["quality"]]
            idx_lst = filt_comb_df.index

            snap_accr_snap = []
            group_id_snap = []
            gc_test_set = []
            analyse_flag_snap = []

            for idx in idx_lst:
                analyse_flag = analyse_flag_lst[idx]
                snap_accr = snap_acc_lst[idx]
                group_id = group_id_lst[idx]
                gc = gc_id_lst[idx]

                analyse_flag_snap.append(analyse_flag)
                snap_accr_snap.append(snap_accr)
                group_id_snap.append(group_id)
                gc_test_set.append(gc)

            for gc in gc_id_snap:
                if gc not in gc_test_set:
                    sys.exit("GC Number mismatch")

            for gc in gc_test_set:
                if gc not in gc_id_snap:
                    sys.exit("GC Number mismatch")

            data_dict = {
                "gc_id": gc_id_snap,
                "ptype": ptype_snap,
                "group_id": group_id_snap,
                "accr_snap": snap_accr_snap,
                "mass": mass_snap,
                "analyse_flag": analyse_flag_snap,
            }

            # now add masses at each snapshot to file.
            if it_id in proc_data.keys():
                grouping = proc_data[it_id]
            else:
                grouping = proc_data.create_group(it_id)
            if "snapshots" in grouping.keys():
                kinematics = grouping["snapshots"]
            else:
                kinematics = grouping.create_group("snapshots")
            if snap_id in kinematics.keys():
                snap_group = kinematics[snap_id]
            else:
                snap_group = kinematics.create_group(snap_id)
            for key in data_dict.keys():
                if key in snap_group.keys():
                    del snap_group[key]
                if key == "ptype":
                    snap_group.create_dataset(key, data=data_dict[key])
                else:
                    snap_group.create_dataset(key, data=np.array(data_dict[key]))
