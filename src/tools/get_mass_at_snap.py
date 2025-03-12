import sys

import h5py
import pandas as pd
from gc_utils import iteration_name, particle_type, snapshot_name  # type: ignore

# lets try this with multiprocessing, combine into a single dictionary and then write another function that
# chucks the dictionary into hdf, this might be a really simple thing to do that will increase processing time
# hugely. I might be able to the implement the same tactic for the kinematics stuff.


def get_gc_masses_at_snap(
    simulation: str,
    offset: int,
    it: int,
    snapshot_list: list[int],
    sim_dir: str,
    data_dir: str,
    data_dict: dict,
):
    proc_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "r")  # open processed data file

    it_id = iteration_name(it)

    # data_dict[it_id] = {}

    raw_dir = data_dir + "results/" + simulation + "/raw/it_%d/" % it

    gc_id_lst = proc_data[it_id]["source"]["gc_id"]
    analyse_flag_lst = proc_data[it_id]["source"]["analyse_flag"]
    group_id_lst = proc_data[it_id]["source"]["group_id"]

    snap_acc_lst = proc_data[it_id]["source"]["snap_acc"]
    snap_form_lst = proc_data[it_id]["source"]["snap_zform"]

    snap_dict = {}
    for snapshot in snapshot_list:
        snap_id = snapshot_name(snapshot)

        snap_mass_file = raw_dir + "allcat_s-%d_p2-7_p3-1_k-1.5_logm_snap%d.txt" % (it, snapshot - offset)
        gc_id_file = raw_dir + "allcat_s-%d_p2-7_p3-1_gcid.txt" % it

        gcid_df = pd.read_csv(gc_id_file, sep=" ").drop(["ID"], axis=1)
        gcid_df.columns = ["GC_ID", "quality"]

        mass_snap_df = pd.read_csv(snap_mass_file, header=None)
        mass_snap_df.columns = ["mass"]

        comb_df = pd.concat([gcid_df, mass_snap_df], axis=1)
        comb_df["snap_form"] = snap_form_lst
        comb_df["analyse_flag"] = analyse_flag_lst  # have included this for now

        filt_comb_df = comb_df[
            (comb_df["mass"] != -1) & (comb_df["snap_form"] <= snapshot) & (comb_df["analyse_flag"] == 1)
        ]

        gc_id_snap = [gc_id for gc_id in filt_comb_df["GC_ID"]]
        mass_snap = [mass for mass in filt_comb_df["mass"]]
        ptype_snap = [particle_type(quality) for quality in filt_comb_df["quality"]]
        idx_lst = filt_comb_df.index

        snap_acc_snap = []
        group_id_snap = []
        gc_test_set = []
        analyse_flag_snap = []
        now_accreted_flag = []
        survived_accretion = []

        for idx in idx_lst:
            analyse_flag = analyse_flag_lst[idx]
            snap_accr = snap_acc_lst[idx]
            group_id = group_id_lst[idx]
            gc = gc_id_lst[idx]

            ######## TEST ADD ALREADY ACCRETED FLAG ########
            ### UPDATE (11/03/2025)
            # I found that in each of these snapshots I have included GCs that have formed
            # but have not yet been accreted. So I need to add a flag showing its been accreted

            if snap_accr <= snapshot:
                # has now been accreted
                now_accreted = 1
            else:
                # has not yet been accreted
                now_accreted = 0

            ###############################################

            ### UPDATE (11/03/2025)
            # add survived accretion flag (if group_id is (-) it did not survive

            # don't need to worry about -2 for analyse flag as only include real GCs
            if group_id == 0:
                # is in-situ (not accreted)
                survive_accreted = -1
            if group_id < 0:
                # did not survive to be accreted
                survive_accreted = 0
            else:
                # survived to accretion
                survive_accreted = 1

            analyse_flag_snap.append(analyse_flag)
            snap_acc_snap.append(snap_accr)
            group_id_snap.append(group_id)
            gc_test_set.append(gc)
            now_accreted_flag.append(now_accreted)
            survived_accretion.append(survive_accreted)

        for gc in gc_id_snap:
            if gc not in gc_test_set:
                sys.exit("GC Number mismatch")

        for gc in gc_test_set:
            if gc not in gc_id_snap:
                sys.exit("GC Number mismatch")

        key_dict = {
            "gc_id": gc_id_snap,
            "ptype": ptype_snap,
            "group_id": group_id_snap,
            "acc_snap": snap_acc_snap,
            "mass": mass_snap,
            "now_accreted": now_accreted_flag,
            "survived_accretion": survived_accretion,
            # "analyse_flag": analyse_flag_snap,
        }

        snap_dict[snap_id] = key_dict

    data_dict[it_id] = snap_dict

    return data_dict
