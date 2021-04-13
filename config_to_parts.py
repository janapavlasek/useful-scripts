#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import argparse

from scipy.spatial.transform import Rotation as R

from parse_urdf import parse_urdf


def rot_trans_to_transform(r, t):
    transform = np.eye(4)
    transform[0:3, 0:3] = r
    transform[0, 3] = t[0]
    transform[1, 3] = t[1]
    transform[2, 3] = t[2]

    return transform


def transform_to_rot_trans(transform):
    return transform[0:3, 0:3], transform[0:3, 3]


def transform_parts(parts_tree, r, t):
    obj_to_world = rot_trans_to_transform(r, np.array([0, 0, 0]))
    for name, data in parts_tree.items():
        curr_transform = rot_trans_to_transform(data["rotation"], data["translation"])

        new_transform = obj_to_world.dot(curr_transform)

        translate = rot_trans_to_transform(np.eye(3), t)
        new_transform = translate.dot(new_transform)

        new_r, new_t = transform_to_rot_trans(new_transform)
        data["rotation"] = new_r
        data["translation"] = new_t
        data["frame"] = "world"

    return parts_tree


def build_tree(urdf, config):
    links = {}

    # Get all the links. Put them all at zero.
    for link in urdf.link_names():
        links[link] = {"rotation": np.eye(3),
                       "translation": np.array([0.0, 0.0, 0.0]),
                       "frame": link}

    idx = 0
    # Transform the parts to the correct transforms in the object frame.
    for joint in urdf.joints:
        r = R.from_euler('xyz', joint.rpy).as_dcm()
        t = joint.xyz
        new_transform = rot_trans_to_transform(r, t)

        # Apply the joint value.
        if joint.type == "prismatic":
            q = config[idx]

            if joint.axis[0] == 1:
                prismatic_t = np.array([q, 0, 0])
            elif joint.axis[1] == 1:
                prismatic_t = np.array([0, q, 0])
            elif joint.axis[2] == 1:
                prismatic_t = np.array([0, 0, q])

            prismatic = rot_trans_to_transform(np.eye(3), prismatic_t)
            new_transform = new_transform.dot(prismatic)
            idx += 1

        elif joint.type == "revolute":
            q = config[idx]

            if joint.axis[0] == 1:
                revolute_r = R.from_euler('xyz', np.array([q, 0, 0])).as_dcm()
            elif joint.axis[1] == 1:
                revolute_r = R.from_euler('xyz', np.array([0, q, 0])).as_dcm()
            elif joint.axis[2] == 1:
                revolute_r = R.from_euler('xyz', np.array([0, 0, q])).as_dcm()

            revolute = rot_trans_to_transform(revolute_r, np.array([0, 0, 0]))
            new_transform = new_transform.dot(revolute)
            idx += 1

        # Save the values.
        new_r, new_t = transform_to_rot_trans(new_transform)
        links[joint.child]["rotation"] = new_r
        links[joint.child]["translation"] = new_t
        links[joint.child]["frame"] = joint.parent

    # Now, move everything to the root node frame.
    all_in_parent_frame = all(data["frame"] == urdf.root for name, data in links.items())
    while not all_in_parent_frame:
        for name, data in links.items():
            if data["frame"] != urdf.root:
                to_parent = rot_trans_to_transform(data["rotation"], data["translation"])
                parent = links[data["frame"]]
                parent_transform = rot_trans_to_transform(parent["rotation"], parent["translation"])

                new_transform = parent_transform.dot(to_parent)

                new_r, new_t = transform_to_rot_trans(new_transform)
                links[name]["rotation"] = new_r
                links[name]["translation"] = new_t
                links[name]["frame"] = links[data["frame"]]["frame"]

        all_in_parent_frame = all(data["frame"] == urdf.root for name, data in links.items())

    return links


def calc_part_poses(urdf, r, t, config):
    parts_tree = build_tree(urdf, config)
    r = R.from_quat(r).as_dcm()
    return transform_parts(parts_tree, r, t)


def load_pcd(pcd_file):
    SAMPLE = 0.2
    with open(pcd_file, 'r') as f:
        skip = 0
        while True:
            line = f.readline().decode("utf-8").strip()
            skip += 1
            if line.startswith("DATA"):
                break

    cloud = np.loadtxt(pcd_file, dtype=float, skiprows=skip)
    N = cloud.shape[0]
    idx = np.random.choice(np.arange(N), int(N * SAMPLE), replace=False)
    return cloud[idx]


def draw_results(part_poses, urdf):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    fig.suptitle("Transformed Parts")

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    lim = 0
    for name, data in part_poses.items():
        pcd_file = urdf.links[name].filename.replace(".dae", ".pcd")
        cloud = load_pcd(pcd_file)
        lim = max(abs(cloud.min()), abs(cloud.max()), abs(lim))
        cloud = np.concatenate([cloud, np.ones((cloud.shape[0], 1))], axis=1)
        cloud = rot_trans_to_transform(data['rotation'], data['translation']).dot(cloud.T)
        ax.scatter(cloud[0, :], cloud[1, :], cloud[2, :], s=5)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='This script generates extra annotations '
                                                 'on the raw LabelFusion output.')

    parser.add_argument('--vis', action='store_true',
                        help='Use to visualize the output poses.')
    parser.add_argument('--urdf', required=True,
                        help='Path to URDF file')
    parser.add_argument('--pose', type=float, nargs='+', default=[],
                        help='The pose, in format [x y z qx qy qz qw] or [x y z r p y]')
    parser.add_argument('--config', type=float, nargs='+',
                        help='The configuration.')
    args = parser.parse_args()

    if len(args.pose) == 0:
        args.pos = (0, 0, 0)
        args.rot = (0, 0, 0, 1)
    elif len(args.pose) == 3:
        args.pos = args.pose
        args.rot = (0, 0, 0, 1)
    elif len(args.pose) == 6:
        args.pos = args.pose[:3]
        args.rot = R.from_euler('xyz', args.pose[3:]).as_quat()
    elif len(args.pose) == 7:
        args.pos = args.pose[:3]
        args.rot = args.pose[3:]
    else:
        raise Exception("Incorrect number of pose arguments given.")

    print('\n' + '-' * 79 + '\n')
    print('Running with args:\n')
    print('\n'.join("{} : {}".format(k, v) for k, v in args.__dict__.items()))
    print('\n' + '-' * 79 + '\n')

    return args


if __name__ == '__main__':
    args = parse_args()

    urdf = parse_urdf(args.urdf)

    if len(args.config) != urdf.num_active_joints:
        print("Wrong number of configurations.")
        exit()

    part_poses = calc_part_poses(urdf, args.rot, args.pos, args.config)

    print("RESULT:")
    for key, val in part_poses.items():
        print(key, val)

    if args.vis:
        draw_results(part_poses, urdf)
