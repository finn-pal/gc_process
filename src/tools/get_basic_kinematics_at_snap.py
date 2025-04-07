import time

import gc_utils  # type: ignore
import h5py
import numpy as np


def get_snap_groups(it_lst: list[int], snapshot: int, proc_data):
    snap_id = gc_utils.snapshot_name(snapshot)

    snap_groups = set()

    for it in it_lst:
        it_id = gc_utils.iteration_name(it)

        snap_dat = proc_data[it_id]["snapshots"][snap_id]

        grp_lst = np.abs(snap_dat["group_id"][()])
        snap_groups.update(grp_lst)

    snap_groups = np.array(sorted(snap_groups))

    return snap_groups


def get_acc_snap(halo_tid, main_halo_tid, halt):
    tid_main_lst = gc_utils.main_prog_halt(halt, main_halo_tid)
    desc_lst = gc_utils.get_descendants_halt(halo_tid, halt)

    idx_lst = np.array([1 if halt["tid"][idx] in tid_main_lst else 0 for idx in desc_lst])
    idx_acc = np.where(idx_lst == 1)[0][0]

    snap_acc = halt["snapshot"][desc_lst[idx_acc]]

    return snap_acc


def get_group_dict(part, halt, proc_data, it_lst, snapshot, main_halo_tid):
    snap_groups = get_snap_groups(it_lst, snapshot, proc_data)

    group_dict = {}
    for group in snap_groups:
        if group == 0:
            continue

        halo_tid = gc_utils.get_halo_prog_at_snap(halt, group, snapshot)
        snap_acc = get_acc_snap(halo_tid, main_halo_tid, halt)

        # we are only adding information here of gc not yet accreted
        if snap_acc <= snapshot:
            continue

        group_dict[group] = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

    return group_dict


def get_basic_kinematics(
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

    halt = gc_utils.get_halo_tree(sim, sim_dir, assign_hosts_rotation=False)
    not_host_snap_lst = gc_utils.get_different_snap_lst(main_halo_tid, halt, sim, sim_dir)

    snap_id = gc_utils.snapshot_name(snapshot)

    # is the MW progenitor is the main host at this snapshot
    is_main_host = snapshot not in not_host_snap_lst

    if not is_main_host:
        # get MW progenitor halo details at this snapshot
        halo_tid = gc_utils.get_main_prog_at_snap(halt, main_halo_tid, snapshot)
        halo_detail_dict = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

    if add_exsitu_halo_details:
        group_dict = get_group_dict(part, halt, proc_data, it_lst, snapshot, main_halo_tid)

        # if there is nothing to add then don't add it
        if len(group_dict) == 0:
            add_exsitu_halo_details = False

    it_dict = {}
    start = time.time()
    for it in it_lst:
        it_id = gc_utils.iteration_name(it)
        print(snap_id, "-", it_id)

        id_idx_map, gc_id_snap, ptype_snap = gc_utils.create_gc_part_idx_dict(part, proc_data, it, snapshot)

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

        for gc, ptype in zip(gc_id_snap, ptype_snap):
            idx = id_idx_map[ptype][gc]

            # is the MW progenitor is the main host at this snapshot
            if is_main_host:
                pos_xyz = part[ptype].prop("host.distance.principal", idx)
                vel_xyz = part[ptype].prop("host.velocity.principal", idx)

                pos_cyl = part[ptype].prop("host.distance.principal.cylindrical", idx)
                vel_cyl = part[ptype].prop("host.velocity.principal.cylindrical", idx)

            else:
                pos_xyz, vel_xyz = gc_utils.get_particle_halo_pos_vel(
                    part, gc, ptype, halo_detail_dict, coordinates="cartesian"
                )

                pos_cyl, vel_cyl = gc_utils.get_particle_halo_pos_vel(
                    part, gc, ptype, halo_detail_dict, coordinates="cylindrical"
                )

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
            group_ids = proc_data[it_id]["snapshots"][snap_id]["group_id"][()]

            ex_pos_xyz_lst = []
            ex_vel_xyz_lst = []

            ex_pos_cyl_lst = []
            ex_vel_cyl_lst = []

            ex_r_lst = []

            for gc, ptype, group_id in zip(gc_id_snap, ptype_snap, group_ids):
                idx = id_idx_map[ptype][gc]

                # need to add an aboslute as gc that die before accretion are noted by negative group ids
                if np.abs(group_id) not in group_dict:
                    ex_pos_xyz = np.full(3, -1, dtype=int)
                    ex_vel_xyz = np.full(3, -1, dtype=int)

                    ex_pos_cyl = np.full(3, -1, dtype=int)
                    ex_vel_cyl = np.full(3, -1, dtype=int)

                    ex_r = -1

                else:
                    exsitu_halo_details = group_dict[np.abs(group_id)]

                    ex_pos_xyz, ex_vel_xyz = gc_utils.get_particle_halo_pos_vel(
                        part, gc, ptype, exsitu_halo_details, coordinates="cartesian"
                    )

                    ex_pos_cyl, ex_vel_cyl = gc_utils.get_particle_halo_pos_vel(
                        part, gc, ptype, exsitu_halo_details, coordinates="cylindrical"
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
