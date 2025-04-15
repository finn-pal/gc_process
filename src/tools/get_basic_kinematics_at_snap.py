import json
import time

import gc_utils  # type: ignore
import gizmo_analysis as gizmo
import h5py
import numpy as np
import utilities as ut

from tools.params import params

# def get_snap_groups(it_lst: list[int], snapshot: int, proc_data):
#     snap_id = gc_utils.snapshot_name(snapshot)

#     snap_groups = set()

#     for it in it_lst:
#         it_id = gc_utils.iteration_name(it)

#         snap_dat = proc_data[it_id]["snapshots"][snap_id]

#         grp_lst = np.abs(snap_dat["group_id"][()])
#         snap_groups.update(grp_lst)

#     snap_groups = np.array(sorted(snap_groups))

#     return snap_groups


def get_gc_form_groups(it_lst: list[int], snapshot: int, proc_data):
    snap_id = gc_utils.snapshot_name(snapshot)

    snap_groups = set()
    for it in it_lst:
        it_id = gc_utils.iteration_name(it)

        snap_dat = proc_data[it_id]["snapshots"][snap_id]
        group_ids = snap_dat["gc_id"][()]

        # ignore in-situ GCs
        group_mask = group_ids != 0

        halo_zforms = snap_dat["halo_zform"][group_mask]
        snap_groups.update(halo_zforms)

    snap_groups = np.array(sorted(snap_groups))

    return snap_groups


def get_acc_snap(halo_tid, main_halo_tid, halt):
    tid_main_lst = gc_utils.main_prog_halt(halt, main_halo_tid)
    desc_lst = gc_utils.get_descendants_halt(halo_tid, halt)

    idx_lst = np.array([1 if halt["tid"][idx] in tid_main_lst else 0 for idx in desc_lst])
    idx_acc = np.where(idx_lst == 1)[0][0]

    snap_acc = halt["snapshot"][desc_lst[idx_acc]]

    return snap_acc


def get_group_dict(part, halt, proc_data, it_lst, snapshot, main_halo_tid, use_dm_center: bool = False):
    snap_groups = get_gc_form_groups(it_lst, snapshot, proc_data)

    group_dict = {}
    for group in snap_groups:
        # if group == 0:
        #     continue

        halo_tid = gc_utils.get_halo_prog_at_snap(halt, group, snapshot)
        snap_acc = get_acc_snap(halo_tid, main_halo_tid, halt)

        # we are only adding information here of gc not yet accreted
        if snap_acc <= snapshot:
            continue

        if use_dm_center:
            group_dict[group] = gc_utils.get_dm_halo_details(part, halt, halo_tid, snapshot, False)
        else:
            group_dict[group] = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

    return group_dict


def get_correct_gc_part_idx(
    part, halt, proc_data, it, gc_id, snapshot, main_halo_tid, sim, sim_dir, choose_random: bool = False
):
    # this is a temporary solution. I do not know what to do if two possible particles have the same
    # formation snapshot

    it_id = gc_utils.iteration_name(it)
    ana_mask = proc_data[it_id]["source"]["analyse_flag"][()] == 1
    gc_mask = proc_data[it_id]["source"]["gc_id"][()] == gc_id
    gc_snapform = proc_data[it_id]["source"]["snap_zform"][ana_mask & gc_mask][0]

    # do a check to ensure part matches intended snap
    if part.snapshot["index"] != snapshot:
        raise RuntimeError("Part selection does not match intended snapshot for halo details")

    # first step is to see how many GCs match the expected formation snapshot
    # if only one then that is the corresponding GC
    part_idxs = np.where(part["star"]["id"] == gc_id)[0]
    part_snapform = part["star"].prop("form.snapshot", part_idxs)
    matching_snapform_mask = part_snapform == gc_snapform
    part_idxs = part_idxs[matching_snapform_mask]

    if len(part_idxs) == 0:
        raise RuntimeError(
            "No particles have same formation snapshot as GC, cannot determine correct GC particle"
        )

    if len(part_idxs) > 1:
        # code this part up with imports
        r_max = params["rmax_form"]  # kpc

        # if more than one GC has the expected formation snapshot then we next inspect distances from the
        # centre of their parent halo
        group_id = np.abs(proc_data[it_id]["source"]["group_id"][ana_mask & gc_mask][0])

        if group_id == 0:
            halo_tid = main_halo_tid
        else:
            halo_tid = group_id

        # get closest next public snapshot (previously calculated)
        gc_snapform_pub = proc_data[it_id]["source"]["pubsnap_zform"][ana_mask & gc_mask][0]
        halo_tid_snap = gc_utils.get_halo_prog_at_snap(halt, halo_tid, gc_snapform_pub)
        halo_idx = np.where(halt["tid"] == halo_tid_snap)[0][0]

        parent_snap_pub = halt["snapshot"][halo_idx]

        # ensure parent halo matches required snapshot
        if parent_snap_pub != gc_snapform_pub:
            raise RuntimeError("Parent halo does not match intended snapshot")

        parent_halo_pos = halt["position"][halo_idx]

        if snapshot == gc_snapform_pub:
            part_distances = ut.coordinate.get_distances(
                part["star"]["position"][part_idxs],
                parent_halo_pos,
                part.info["box.length"],
                part.snapshot["scalefactor"],
                total_distance=True,
            )  # [kpc physical]

            distance_mask = part_distances < r_max
            part_idxs = part_idxs[distance_mask]

        else:
            # only need to be concerned with star particles
            gc_utils.block_print()
            fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

            part_form_pub = gizmo.io.Read.read_snapshots(
                "star", "index", gc_snapform_pub, fire_dir, assign_pointers=True
            )

            part_snap = gizmo.io.Read.read_snapshots(
                "star", "index", snapshot, fire_dir, assign_pointers=True
            )
            gc_utils.enable_print()

            # get part indices with matching gc_id
            part_form_idxs = np.where(part_form_pub["star"]["id"] == gc_id)[0]

            # get part indices with matching snapform
            part_form_snapform = part_form_pub["star"].prop("form.snapshot", part_form_idxs)
            matching_snapform_mask = part_form_snapform == gc_snapform
            part_form_idxs = part_form_idxs[matching_snapform_mask]

            part_distances = ut.coordinate.get_distances(
                part_form_pub["star"]["position"][part_form_idxs],
                parent_halo_pos,
                part_form_pub.info["box.length"],
                part_form_pub.snapshot["scalefactor"],
                total_distance=True,
            )  # [kpc physical]

            distance_mask = part_distances < r_max
            part_form_idxs = part_form_idxs[distance_mask]

            # now need to map part_form_idxs onto part_idxs
            part_form_pub.Pointer.add_intermediate_pointers(part_snap.Pointer)
            pointers = part_form_pub.Pointer.get_pointers(
                species_name_from="star", species_names_to="star", intermediate_snapshot=True, forward=True
            )

            indices_at_form = part_form_idxs
            indices_at_snap = pointers[indices_at_form]

            part_idxs = indices_at_snap

            if len(part_idxs) > 1:
                if choose_random:
                    # randomly choose one
                    part_idxs = np.random.choice(part_idxs, 1)

                # else take youngest
                else:
                    star_ages = part["star"].prop("age", part_idxs)
                    min_age = np.min(star_ages)

                    # Mask to find all indexes where age == min_age
                    min_age_mask = star_ages == min_age
                    part_idxs = part_idxs[min_age_mask]

                    # if still more than one then just select idx zero
                    if len(part_idxs) > 1:
                        # part_idxs = np.random.choice(part_idxs, 1)
                        part_idxs = part_idxs[0]

    if len(part_idxs) == 0:
        raise RuntimeError(
            "No particles have same formation snapshot as GC, cannot determine correct GC particle"
        )

    # correct_part_idx = part_idxs[matching_snapform][0]

    # final check to confirm only one particle chosen
    if len(part_idxs):
        part_idxs = part_idxs[0]

    else:
        raise RuntimeError("Cannot determine correct GC particle")

    if part["star"]["id"][part_idxs] != gc_id:
        raise RuntimeError("Error in determining correct GC particle index")

    return part_idxs


def remove_duplicates_with_report(arr):
    arr = np.array(arr)  # Convert the list to a numpy array
    unique_elements, counts = np.unique(arr, return_counts=True)  # Get unique elements and their counts
    duplicates = unique_elements[counts > 1]  # Duplicates are those that appear more than once
    unique_list = unique_elements.tolist()  # Convert unique elements back to a list

    return unique_list, duplicates.tolist()  # Return unique and duplicate lists


def create_gc_part_idx_dict(part, halt, proc_data, it, snapshot, main_halo_tid, sim, sim_dir):
    it_id = gc_utils.iteration_name(it)
    snap_id = gc_utils.snapshot_name(snapshot)

    gc_id_snap = proc_data[it_id]["snapshots"][snap_id]["gc_id"][()]

    ptype_byte_snap = proc_data[it_id]["snapshots"][snap_id]["ptype"]
    ptype_snap = [ptype.decode("utf-8") for ptype in ptype_byte_snap]

    # Step 1: group GCs by particle type
    gc_by_ptype = {}
    gc_by_ptype["star"] = []
    gc_by_ptype["dark"] = []

    for gc, ptype in zip(gc_id_snap, ptype_snap):
        gc_by_ptype[ptype].append(gc)

    # Step 2: for each ptype, build a small dict: gc_id → index
    id_idx_map = {}

    for ptype, gc_ids in gc_by_ptype.items():
        ids = part[ptype]["id"]  # potentially millions of entries
        # gc_ids = np.array(gc_ids)  # small subset

        # Check which of these are in the main list
        mask = np.isin(ids, gc_ids)
        idxs = np.nonzero(mask)[0]
        found_ids = ids[idxs]

        # Build small, efficient lookup: GC ID → array index
        id_idx_map[ptype] = dict(zip(found_ids, idxs))

        # concerned abour duplciate star ids
        if ptype == "star":
            _, duplicates_ids = remove_duplicates_with_report(found_ids)

    # only concerned with duplciates in star
    for gc_id in duplicates_ids:
        corrected_idx = get_correct_gc_part_idx(
            part, halt, proc_data, it, gc_id, snapshot, main_halo_tid, sim, sim_dir
        )

        id_idx_map["star"][gc_id] = corrected_idx

    return id_idx_map, gc_id_snap, ptype_snap


#####################################################################################################################
# Main Function
#####################################################################################################################


def get_basic_kinematics(
    halt,
    part,
    sim: str,
    it_lst: list[int],
    snapshot: int,
    main_halo_tid: int,
    sim_dir: str,
    add_exsitu_halo_details: bool = True,
    data_dict: dict = {},
):
    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "r")  # open processed data file

    not_host_snap_lst = gc_utils.get_different_snap_lst(main_halo_tid, halt, sim, sim_dir)

    snap_id = gc_utils.snapshot_name(snapshot)

    # is the MW progenitor is the main host at this snapshot
    is_main_host = snapshot not in not_host_snap_lst

    # check if centering should be on dm halo
    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)

    halt_center_snap_lst = sim_data[sim]["dm_center"]
    use_dm_center = snapshot in halt_center_snap_lst

    # if the halo is not the host at this snapshot or it has been flagged to use dm center at this snapshot
    if (not is_main_host) or (use_dm_center):
        # get MW progenitor halo details at this snapshot
        halo_tid = gc_utils.get_main_prog_at_snap(halt, main_halo_tid, snapshot)

        if use_dm_center:
            halo_detail_dict = gc_utils.get_dm_halo_details(part, halt, halo_tid, snapshot, True)
        else:
            halo_detail_dict = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

    if add_exsitu_halo_details:
        group_dict = get_group_dict(part, halt, proc_data, it_lst, snapshot, main_halo_tid, use_dm_center)

        # if there is nothing to add then don't add it
        if len(group_dict) == 0:
            add_exsitu_halo_details = False

    it_dict = {}
    start = time.time()
    for it in it_lst:
        it_id = gc_utils.iteration_name(it)
        print(snap_id, "-", it_id)

        id_idx_map, gc_id_snap, ptype_snap = create_gc_part_idx_dict(
            part, halt, proc_data, it, snapshot, main_halo_tid, sim, sim_dir
        )

        group_ids = proc_data[it_id]["snapshots"][snap_id]["group_id"][()]
        halo_zform = proc_data[it_id]["snapshots"][snap_id]["halo_zform"][()]

        if len(gc_id_snap) is None:
            continue

        # x_lst = []
        # y_lst = []
        # z_lst = []
        # vx_lst = []
        # vy_lst = []
        # vz_lst = []

        pos_xyz_lst = []
        vel_xyz_lst = []

        pos_cyl_lst = []
        vel_cyl_lst = []

        # r_cyl_lst = []
        # phi_cyl_lst = []
        # vr_cyl_lst = []
        # vphi_cyl_lst = []

        r_lst = []

        ep_fire_lst = []
        ek_lst = []

        # lx_lst = []
        # ly_lst = []
        # lz_lst = []

        l_xyz_lst = []

        # parent_halo_tid_lst = []
        snap_part_idx_lst = []

        for gc, ptype, group_id, halo_gc_form in zip(gc_id_snap, ptype_snap, group_ids, halo_zform):
            idx = id_idx_map[ptype][gc]

            if (not is_main_host) or (use_dm_center):
                pos_xyz, vel_xyz = gc_utils.get_particle_halo_pos_vel(
                    part, idx, ptype, halo_detail_dict, coordinates="cartesian"
                )

                pos_cyl, vel_cyl = gc_utils.get_particle_halo_pos_vel(
                    part, idx, ptype, halo_detail_dict, coordinates="cylindrical"
                )

            # is the MW progenitor is the main host at this snapshot
            else:
                pos_xyz = part[ptype].prop("host.distance.principal", idx)
                vel_xyz = part[ptype].prop("host.velocity.principal", idx)

                pos_cyl = part[ptype].prop("host.distance.principal.cylindrical", idx)
                vel_cyl = part[ptype].prop("host.velocity.principal.cylindrical", idx)

            ep_fir = part[ptype]["potential"][idx]
            ek = 0.5 * np.linalg.norm(vel_xyz) ** 2

            x, y, z = pos_xyz
            vx, vy, vz = vel_xyz

            # r_cyl, phi_cyl, _ = pos_cyl
            # vr_cyl, vphi_cyl, _ = vel_cyl

            r = np.linalg.norm(pos_xyz)

            lx = y * vz - z * vy
            ly = z * vx - x * vz
            lz = x * vy - y * vx

            # x_lst.append(x)
            # y_lst.append(y)
            # z_lst.append(z)
            # vx_lst.append(vx)
            # vy_lst.append(vy)
            # vz_lst.append(vz)

            pos_xyz_lst.append(pos_xyz)
            vel_xyz_lst.append(vel_xyz)

            pos_cyl_lst.append(pos_cyl)
            vel_cyl_lst.append(vel_cyl)

            # r_cyl_lst.append(r_cyl)
            # phi_cyl_lst.append(phi_cyl)
            # vr_cyl_lst.append(vr_cyl)
            # vphi_cyl_lst.append(vphi_cyl)

            r_lst.append(r)

            ep_fire_lst.append(ep_fir)
            ek_lst.append(ek)

            # lx_lst.append(lx)
            # ly_lst.append(ly)
            # lz_lst.append(lz)

            l_xyz_lst.append([lx, ly, lz])

            # if group_id == 0:
            #     group_halo = main_halo_tid
            # else:
            #     group_halo = halo_gc_form
            # parent_tid = gc_utils.get_halo_prog_at_snap(halt, group_halo, snapshot)

            # parent_halo_tid_lst.append(parent_tid)
            snap_part_idx_lst.append(idx)

        kin_dict = {
            # "x": np.array(x_lst),
            # "y": np.array(y_lst),
            # "z": np.array(z_lst),
            # "vx": np.array(vx_lst),
            # "vy": np.array(vy_lst),
            # "vz": np.array(vz_lst),
            # "r_cyl": np.array(r_cyl_lst),
            # "phi_cyl": np.array(phi_cyl_lst),
            # "vr_cyl": np.array(vr_cyl_lst),
            # "vphi_cyl": np.array(vphi_cyl_lst),
            # "halo.tid": np.array(parent_halo_tid_lst),
            "snap_part_idx": np.array(snap_part_idx_lst),
            "pos.xyz": np.array(pos_xyz_lst),
            "vel.xyz": np.array(vel_xyz_lst),
            "pos.cyl": np.array(pos_cyl_lst),
            "vel.cyl": np.array(vel_cyl_lst),
            "r": np.array(r_lst),
            "ep_fire": np.array(ep_fire_lst),
            "ek": np.array(ek_lst),
            # "lx": np.array(lx_lst),
            # "ly": np.array(ly_lst),
            # "lz": np.array(lz_lst),
            "l.xyz": np.array(l_xyz_lst),
        }

        if add_exsitu_halo_details:
            ex_pos_xyz_lst = []
            ex_vel_xyz_lst = []

            ex_pos_cyl_lst = []
            ex_vel_cyl_lst = []

            ex_r_lst = []

            for gc, ptype, group_id, halo_gc_form in zip(gc_id_snap, ptype_snap, group_ids, halo_zform):
                idx = id_idx_map[ptype][gc]

                # need to add an aboslute as gc that die before accretion are noted by negative group ids
                if halo_gc_form not in group_dict:
                    ex_pos_xyz = np.full(3, -1, dtype=int)
                    ex_vel_xyz = np.full(3, -1, dtype=int)

                    ex_pos_cyl = np.full(3, -1, dtype=int)
                    ex_vel_cyl = np.full(3, -1, dtype=int)

                    ex_r = -1

                else:
                    exsitu_halo_details = group_dict[halo_gc_form]

                    ex_pos_xyz, ex_vel_xyz = gc_utils.get_particle_halo_pos_vel(
                        part, idx, ptype, exsitu_halo_details, coordinates="cartesian"
                    )

                    ex_pos_cyl, ex_vel_cyl = gc_utils.get_particle_halo_pos_vel(
                        part, idx, ptype, exsitu_halo_details, coordinates="cylindrical"
                    )

                    ex_r = np.linalg.norm(ex_pos_xyz)

                ex_pos_xyz_lst.append(ex_pos_xyz)
                ex_vel_xyz_lst.append(ex_vel_xyz)

                ex_pos_cyl_lst.append(ex_pos_cyl)
                ex_vel_cyl_lst.append(ex_vel_cyl)

                ex_r_lst.append(ex_r)

            kin_dict["halo.pos.xyz"] = np.array(ex_pos_xyz_lst)
            kin_dict["halo.vel.xyz"] = np.array(ex_vel_xyz_lst)
            kin_dict["halo.pos.cyl"] = np.array(ex_pos_cyl_lst)
            kin_dict["halo.vel.cyl"] = np.array(ex_vel_cyl_lst)
            kin_dict["halo.r"] = np.array(ex_r_lst)

        it_dict[it_id] = kin_dict
        del kin_dict

    proc_data.close()

    end = time.time()
    print(snap_id, "time:", end - start)

    data_dict[snap_id] = it_dict

    return data_dict
