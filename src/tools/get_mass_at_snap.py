import h5py
import numpy as np
import pandas as pd
from gc_utils import iteration_name, snapshot_name  # type: ignore


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

        gc_id = proc_data[it_id]["source"]["gc_id"]
        analyse_flag = proc_data[it_id]["source"]["analyse_flag"]
        group_id = proc_data[it_id]["source"]["group_id"]

        snap_zform = proc_data[it_id]["source"]["snap_zform"]
        snap_last = proc_data[it_id]["source"]["last_snap"]
        snap_accr = proc_data[it_id]["source"]["snap_acc"]

        ptypes_byte = proc_data[it_id]["source"]["ptype"]
        ptypes = [ptype.decode("utf-8") for ptype in ptypes_byte]

        # need to make a function to state if the gc is alive at the snapshot in question
        gc_id_snap = []
        group_id_snap = []
        ptype_snap = []
        snap_accr_snap = []

        for snapshot in snapshot_list:
            snap_id = snapshot_name(snapshot)
            for gc, group, ptype, a_flag, snap_form, snap_disr, snap_ac in zip(
                gc_id, group_id, ptypes, analyse_flag, snap_zform, snap_last, snap_accr
            ):
                if a_flag == 0:
                    continue

                if (snapshot < snap_form) or (snap_disr < snapshot):
                    continue

                gc_id_snap.append(gc)
                group_id_snap.append(group)
                ptype_snap.append(ptype)
                snap_accr_snap.append(snap_ac)

            if len(gc_id_snap) is None:
                continue

            snap_mass_file = raw_dir + "allcat_s-%d_p2-7_p3-1_k-1.5_logm_snap%d.txt" % (it, snapshot - offset)
            gc_id_file = raw_dir + "allcat_s-%d_p2-7_p3-1_gcid.txt" % it

            gcid_df = pd.read_csv(gc_id_file, sep=" ").drop(["ID"], axis=1)
            gcid_df.columns = ["GC_ID", "quality"]

            mass_snap_df = pd.read_csv(snap_mass_file, header=None)
            mass_snap_df.columns = ["mass"]

            comb_df = pd.concat([gcid_df, mass_snap_df], axis=1)
            filt_comb_df = comb_df[comb_df["mass"] != -1]

            indexes = filt_comb_df[filt_comb_df["GC_ID"].isin(gc_id_snap)].index

            mass_at_snap_lst = []
            for idx in indexes:
                mass_snap = comb_df["mass"][idx]
                mass_at_snap_lst.append(mass_snap)

            data_dict = {
                "gc_id": gc_id_snap,
                "ptype": ptype_snap,
                "group_id": group_id_snap,
                "accr_snap": snap_accr_snap,
                "mass": mass_at_snap_lst,
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
