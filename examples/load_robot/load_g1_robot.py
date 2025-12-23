"""
load_g1_robot.py

A minimal example to demonstrate position control with mimic joints on G1 robot using SAPIEN.

What you get:
- Load G1 URDF with mimic joints
- Fix root link
- Set PD drive (stiffness/damping) for both regular and mimic joints
- Hold current pose (q0) under gravity with mimic joints properly coupled
- In the sim loop, enforces mimic by updating the mimic joint's PD target each step

Run:
    python examples/load_g1_robot.py

Notes:
- SAPIEN does not automatically handle mimic joints for you in PD control. 
  We parse URDF ourselves and apply coupling each step.
"""

import os
import time
import numpy as np
import xml.etree.ElementTree as ET
import sapien.core as sapien
from sapien.utils import Viewer


# TARGET_JOINT_NAMES = [
#     "body_joint1",
#     "body_joint2",
#     "head_joint1",
#     "head_joint2",
#     "left_joint1",
#     "left_joint2",
#     "left_joint3",
#     "left_joint4",
#     "left_joint5",
#     "left_joint6",
#     "left_joint7",
#     "right_joint1",
#     "right_joint2",
#     "right_joint3",
#     "right_joint4",
#     "right_joint5",
#     "right_joint6",
#     "right_joint7",
#     "left_gripper_joint",
#     "right_gripper_joint",
# ]

ARM_JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]

GRIPPER_JOINT_NAMES = [
    "left_gripper_joint",
    "right_gripper_joint",
]

TARGET_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

# PD settings for arm joints (from g1_wo_gripper.py - 1/128 stiffness, 1/16 damping)
# ARM_PD_SETTINGS = {
#     "left_joint1": (10000.0 , 500.0 / 16, 60.0),
#     "left_joint2": (10000.0 / 128, 500.0 / 16, 60.0),
#     "left_joint3": (10000.0 / 128, 500.0 / 16, 60.0),
#     "left_joint4": (10000.0 / 128, 500.0 / 16, 60.0),
#     "left_joint5": (10000.0 / 128, 500.0 / 16, 30.0),
#     "left_joint6": (10000.0 / 128, 500.0 / 16, 30.0),
#     "left_joint7": (10000.0 / 128, 500.0 / 16, 30.0),
#     "right_joint1": (10000.0 / 128, 500.0 / 16, 60.0),
#     "right_joint2": (10000.0 / 128, 500.0 / 16, 60.0),
#     "right_joint3": (10000.0 / 128, 500.0 / 16, 60.0),
#     "right_joint4": (10000.0 / 128, 500.0 / 16, 60.0),
#     "right_joint5": (10000.0 / 128, 500.0 / 16, 30.0),
#     "right_joint6": (10000.0 / 128, 500.0 / 16, 30.0),
#     "right_joint7": (10000.0 / 128, 500.0 / 16, 30.0),
# }

ARM_PD_SETTINGS = {
    "left_joint1": (10000.0 , 500.0 , 60.0),
    "left_joint2": (10000.0 , 500.0 , 60.0),
    "left_joint3": (10000.0 , 500.0 , 60.0),
    "left_joint4": (10000.0 , 500.0 , 60.0),
    "left_joint5": (10000.0 , 500.0 , 30.0),
    "left_joint6": (10000.0 , 500.0 , 30.0),
    "left_joint7": (10000.0 , 500.0 , 30.0),
    "right_joint1": (10000.0 , 500.0 , 60.0),
    "right_joint2": (10000.0 , 500.0 , 60.0),
    "right_joint3": (10000.0 , 500.0 , 60.0),
    "right_joint4": (10000.0 , 500.0 , 60.0),
    "right_joint5": (10000.0 , 500.0 , 30.0),
    "right_joint6": (10000.0 , 500.0 , 30.0),
    "right_joint7": (10000.0 , 500.0 , 30.0),
}

# ARM_PD_SETTINGS = {
#     "left_joint1": (0 , 0 , 160.0),
#     "left_joint2": (0 , 0 , 160.0),
#     "left_joint3": (0 , 0 , 160.0),
#     "left_joint4": (0 , 0 , 160.0),
#     "left_joint5": (0 , 0 , 130.0),
#     "left_joint6": (0 , 0 , 130.0),
#     "left_joint7": (0 , 0 , 130.0),
#     "right_joint1": (0 , 0 , 160.0),
#     "right_joint2": (0 , 0 , 160.0),
#     "right_joint3": (0 , 0 , 160.0),
#     "right_joint4": (0 , 0 , 160.0),
#     "right_joint5": (0 , 0 , 130.0),
#     "right_joint6": (0 , 0 , 130.0),
#     "right_joint7": (0 , 0 , 130.0),
# }

# PD settings for gripper master joints (from omni_picker.py)
GRIPPER_PD_SETTINGS = {
    "left_gripper_joint": (100.0, 10.0, 50.0),
    "right_gripper_joint": (100.0, 10.0, 50.0),
}

# GRIPPER_PD_SETTINGS = {
#     "left_gripper_joint": (0.0, 0.0, 0.0),
#     "right_gripper_joint": (0.0, 0.0, 0.0),
# }

# PD settings for gripper mimic joints (from omni_picker.py - higher stiffness, zero damping)
# MIMIC_PD_SETTINGS = {
#     "stiffness": 2000.0,
#     "damping": 0.0,
#     "force_limit": 5.0,
# }

MIMIC_PD_SETTINGS = {
    "stiffness": 1.0,
    "damping": 20.0,
    "force_limit": 0.1,
}
# MIMIC_PD_SETTINGS = {
#     "stiffness": 500.0, # Must be > 0 to follow the target!
#     "damping": 5.0,     # Small amount to stop shaking
#     "force_limit": 100.0,
# }

mimic_multipliers = {
    # 'gripper_joint': 1.0,
    # 'narrow2_joint': 0.02,
    # 'narrow3_joint': 0.4,
    # 'narrow_loop_joint': 1.5,
    # 'wide1_joint': 1.0,
    # 'wide2_joint': 0.02,
    # 'wide3_joint': 0.4,
    # 'wide_loop_joint': 1.5,
    # 'left_gripper_joint': 1.0,
    'left_narrow2_joint': 0.02,
    'left_narrow3_joint': 0.4,
    'left_narrow_loop_joint': 1.5,
    'left_wide1_joint': 1.0,
    'left_wide2_joint': 0.02,
    'left_wide3_joint': 0.4,
    'left_wide_loop_joint': 1.5,
    # 'right_gripper_joint': 1.0,
    'right_narrow2_joint': 0.02,
    'right_narrow3_joint': 0.4,
    'right_narrow_loop_joint': 1.5,
    'right_wide1_joint': 1.0,
    'right_wide2_joint': 0.02,
    'right_wide3_joint': 0.4,
    'right_wide_loop_joint': 1.5,
}

def init_robot_PD(robot):
    # set ARM joint PD
    for joint in robot.get_active_joints():
        joint_name = joint.get_name()
        if joint_name in ARM_PD_SETTINGS:
            stiffness, damping, force_limit = ARM_PD_SETTINGS[joint_name]
            joint.set_drive_properties(stiffness=stiffness, damping=damping, force_limit=force_limit)
            print(f"[INFO] Set arm joint PD: {joint_name} stiffness={stiffness}, damping={damping}, force_limit={force_limit}")
        elif joint_name in GRIPPER_PD_SETTINGS:
            stiffness, damping, force_limit = GRIPPER_PD_SETTINGS[joint_name]
            joint.set_drive_properties(stiffness=stiffness, damping=damping, force_limit=force_limit)
            print(f"[INFO] Set gripper joint PD: {joint_name} stiffness={stiffness}, damping={damping}, force_limit={force_limit}")
        elif joint_name in mimic_multipliers.keys():
            # set a very weak PD control for the gripper mimic joints
            # joint.set_drive_properties(stiffness=MIMIC_PD_SETTINGS['stiffness'], 
            #                           damping=MIMIC_PD_SETTINGS['damping'], 
            #                           force_limit=MIMIC_PD_SETTINGS['force_limit'],
            #                           mode="acceleration")
            # print(f"[INFO] Set mimic joint PD: {joint_name} stiffness={MIMIC_PD_SETTINGS['stiffness']}, damping={MIMIC_PD_SETTINGS['damping']}, force_limit={MIMIC_PD_SETTINGS['force_limit']}")        
            pass
        # joints.set_friction(0.05)
    

def set_gripper(robot, type: str, qpos: float, mimic_map: dict, scene):
    """
    Set gripper joint positions based on type, left or right.
    
    The mimic joints will be automatically coupled using the mimic rules.
    
    Args:
        robot: The robot articulation
        type: "left" or "right"
        qpos: Target position for the master gripper joint
        mimic_map: Dictionary mapping mimic joint names to (master_joint_name, multiplier, offset)
    """
    if type == "left":
        MASTER_JOINT_NAME = "left_gripper_joint"
    elif type == "right":
        MASTER_JOINT_NAME = "right_gripper_joint"
    else:
        raise ValueError(f"Unknown gripper type: {type}")
    
    # Build joint name to object mapping
    name_to_joint = {j.get_name(): j for j in robot.get_active_joints()}
    
    # Set the master gripper joint target
    if MASTER_JOINT_NAME in name_to_joint:
        name_to_joint[MASTER_JOINT_NAME].set_drive_target(float(qpos))
    
    # scene.step()  # Step once to apply the master joint change
    
    # Apply mimic rules: mimic_target = multiplier * master_qpos + offset
    for mimic_joint_name, (master_name, mult, offs) in mimic_map.items():
        if master_name == MASTER_JOINT_NAME and mimic_joint_name in name_to_joint:
            mimic_target = mult * qpos + offs
            # need clamp mimic_target within joint limits
            # mimic_target = max(min(mimic_target, name_to_joint[mimic_joint_name].get_upper_limit()), name_to_joint[mimic_joint_name].get_lower_limit())
            # lower="-3.141592653589793" upper="3.141592653589793"
            mimic_target = max(min(mimic_target, 3.141592653589793), -3.141592653589793)
            name_to_joint[mimic_joint_name].set_drive_target(float(mimic_target))
    
    
def set_arm_targets(robot, joint_names: list, qpos_list: list):
    """
    Set arm joint targets.
    """
    name_to_joint = {j.get_name(): j for j in robot.get_active_joints()}
    for jname, qpos in zip(joint_names, qpos_list):
        if jname in name_to_joint:
            joint = name_to_joint[jname]
            joint.set_drive_target(qpos)
        else:
            print(f"[WARN] Joint {jname} not found in robot.")
            
def get_arm_targets(robot, joint_names: list):
    """
    Get arm joint current positions.
    """
    # Build name to qpos index mapping
    name_to_joint = {j.get_name(): j for j in robot.get_active_joints()}
    name_to_qidx = {}
    cursor = 0
    for j in robot.get_active_joints():
        if j.get_dof() > 0:
            name_to_qidx[j.get_name()] = cursor
            cursor += j.get_dof()
    
    qpos = robot.get_qpos()
    qpos_list = []
    for jname in joint_names:
        if jname in name_to_qidx:
            idx = name_to_qidx[jname]
            qpos_list.append(qpos[idx])
        else:
            print(f"[WARN] Joint {jname} not found in robot.")
            qpos_list.append(0.0)
    return qpos_list

def get_gripper_positions(robot, type: str):
    """
    Get gripper joint current positions based on type, left or right.
    """
    if type == "left":
        MASTER_JOINT_NAME = "left_gripper_joint"
    elif type == "right":
        MASTER_JOINT_NAME = "right_gripper_joint"
    else:
        raise ValueError(f"Unknown gripper type: {type}")
    
    # Build name to qpos index mapping
    name_to_qidx = {}
    cursor = 0
    for j in robot.get_active_joints():
        if j.get_dof() > 0:
            name_to_qidx[j.get_name()] = cursor
            cursor += j.get_dof()
    
    if MASTER_JOINT_NAME in name_to_qidx:
        qpos = robot.get_qpos()
        idx = name_to_qidx[MASTER_JOINT_NAME]
        return qpos[idx]
    else:
        print(f"[WARN] Joint {MASTER_JOINT_NAME} not found in robot.")
        return 0.0
    

def parse_mimic_map_from_urdf(urdf_path: str):
    """
    Parse URDF to extract mimic joint relationships.
    
    Returns:
        mimic_map: dict[mimic_joint_name] = (master_joint_name, multiplier, offset)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mimic_map = {}
    for joint in root.findall("joint"):
        jname = joint.get("name")
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        master = mimic.get("joint")
        mult = float(mimic.get("multiplier", "1.0"))
        offs = float(mimic.get("offset", "0.0"))
        mimic_map[jname] = (master, mult, offs)
    return mimic_map

MIMIC_JOINT_NAMES = []

def disable_internal_collisions(robot):
    links = robot.get_links()
    for link in links:
        # Get all collision shapes for the link
        shapes = link.get_collision_shapes()
        for shape in shapes:
            # Group 1: Collision identity
            # Group 2: Collision mask (what it hits)
            # Setting these to [1,1,0,0] effectively makes the link a "ghost" 
            # to check if collisions are causing the explosion.
            shape.set_collision_groups([1, 1, 0, 0]) 
    print("[INFO] Disabled all internal collisions for debugging.")

def main():
    # -------------------
    # 1) Engine & Scene
    # -------------------
    engine = sapien.Engine()
    
    scene = engine.create_scene()
    scene.set_timestep(1 / 240.0)  # Smaller timestep for better stability
    
    # solver settings for better stability
    # get scene config

    
    # Add a ground plane
    scene.add_ground(altitude=0)
    
    # Add lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    
    
    # -------------------
    # 2) Load G1 URDF
    # -------------------
    urdf_path = "robot_descriptions/manipulation/Agibot/agibot_g1_with_gripper_description/agibot_g1_with_omnipicker.debug.urdf"
    
    # Parse mimic joints from URDF (SAPIEN doesn't enforce them automatically)
    if os.path.exists(urdf_path):
        mimic_map = parse_mimic_map_from_urdf(urdf_path)
        print(f"[INFO] Found {len(mimic_map)} mimic joints in URDF")
        if mimic_map:
            print("[INFO] mimic_map =", mimic_map)
    else:
        print(f"[WARN] URDF not found: {urdf_path}")
        mimic_map = {}
        
    MIMIC_JOINT_NAMES.extend(mimic_map.keys())
    
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    robot = loader.load(urdf_path)
    assert robot is not None, f"Failed to load URDF: {urdf_path}"
    
    disable_internal_collisions(robot)

    # Put robot at origin
    robot.set_pose(sapien.Pose([0, 0, 0]))

    for link in robot.get_links():
        if "narrow" in link.get_name() or "wide" in link.get_name():
            # Physical stability hack: boost mass of tiny fingers
            mass = link.get_mass()
            if mass < 0.1:
                # Manually setting a minimum mass for solver stability
                # Note: This requires SAPIEN 2.0+ API
                # If link.set_mass() isn't available, you must edit the URDF <mass> tags
                link.set_mass(0.1)
                print(f"[INFO] Boosted mass of link {link.get_name()} from {mass} to 0.1 kg for stability.")

    # # Disable collisions between arm end and gripper base
    # for link in robot.get_links():
    #     if "gripper" in link.get_name() or "camera_stand" in link.get_name():
    #         # Set a collision group that ignores itself or specific neighbors
    #         collision_shape = link.get_collision_shapes()[0]
    #         collision_shape.set_collision_groups(1, 1, 2, 2) # Group, Mask, etc.
    
    # -------------------
    # 3) PD Drive Setup
    # -------------------
    # # Use conservative PD parameters to prevent instability
    # stiffness = 1000.0  # Reduced from 3000
    # damping = 50.0      # Reduced from 100
    # force_limit = 87.0
    
    # make all non-target joints passive (no PD control), only target joints get PD control
    # Mimic joints should also be passive - they don't get direct PD control
    init_robot_PD(robot)
    
    # Initialize at q0 (initial configuration)
    q0_arm = get_arm_targets(robot, ARM_JOINT_NAMES)
    q0_arm_left = q0_arm[:7]
    q0_arm_right = q0_arm[7:]
    q0_gripper_left = get_gripper_positions(robot, "left")
    q0_gripper_right = get_gripper_positions(robot, "right")
    print(f"[INFO] q0_arm: {q0_arm}")
    print(f"[INFO] q0_gripper_left: {q0_gripper_left}")
    print(f"[INFO] q0_gripper_right: {q0_gripper_right}")
    
    # # IMPORTANT: Set initial drive targets to current positions before simulation starts
    # # First, zero out all velocities
    # robot.set_qvel(np.zeros(robot.dof))
    # print("[INFO] Zeroed all velocities")
    
    # Set qpos again to ensure proper initialization
    qpos_full = robot.get_qpos()
    robot.set_qpos(qpos_full)
    print(f"[INFO] Set full qpos to: {qpos_full}")
    
    # Set drive targets to current positions
    set_arm_targets(robot, ARM_JOINT_NAMES, q0_arm)
    set_gripper(robot, "left", q0_gripper_left, mimic_map, scene)
    set_gripper(robot, "right", q0_gripper_right, mimic_map, scene)
    print("[INFO] Initial drive targets set to q0 positions")
    
    # Run simulation steps with forced zero velocity to let robot stabilize
    print("[INFO] Letting robot settle with zero velocity constraint...")
    robot.set_qvel(np.zeros(robot.dof))  # Force zero velocity
    # for i in range(300):
    #     scene.step()
    #     if i % 100 == 0:
    #         print(f"[INFO] Settling step {i}/300")
    # print("[INFO] Robot settled")
    
    
    # -------------------
    # 4) Hold at q0 pose with mimic joints
    # -------------------
    # We hold the robot at its q0 (initial) configuration
    # The mimic joints will be automatically coupled each simulation step

    
    # Simple smoothing for target interpolation (optional)
    alpha = 0.02
    q_cmd = np.array(q0_arm_left + q0_arm_right + [q0_gripper_left] + [q0_gripper_right])  # Initial command at q0
    q_des = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.04, 0.04])  # Arms at zero, grippers slightly open
    # -------------------
    # 5) Simulation Loop
    # -------------------
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.2, y=0.0, z=0.8)
    viewer.set_camera_rpy(r=0.0, p=-0.4, y=np.radians(180))    
    
    last_time = time.time()
    
    counter = 0
    while not viewer.closed:
        viewer.render()
        if counter == 0:
            time.sleep(1.0)  # Initial pause to observe
        else:
            time.sleep(0.01)
        counter += 1
        # Update drive target every step with smoothing (for filtered target joints)
        q_cmd = (1 - alpha) * q_cmd + alpha * q_des
        
        # Apply targets to target joints and mimic coupling
        set_arm_targets(robot, ARM_JOINT_NAMES, q_cmd[:14])
        set_gripper(robot, "left", q_cmd[14], mimic_map, scene)
        set_gripper(robot, "right", q_cmd[15], mimic_map, scene)
        
        # Step physics
        scene.step()
        scene.update_render()
        viewer.render()
        
        # Print status periodically
        now = time.time()
        if now - last_time > 1.0:
            last_time = now
            qpos_current = get_arm_targets(robot, ARM_JOINT_NAMES)
            gripper_left_pos = get_gripper_positions(robot, "left")
            gripper_right_pos = get_gripper_positions(robot, "right")
            print(f"[INFO] Current arm qpos: {qpos_current}")
            print(f"[INFO] Current left gripper pos: {gripper_left_pos:.4f}, right gripper pos: {gripper_right_pos:.4f}")
            
            # # Verify mimic joints are coupled correctly
            # if mimic_pairs:
            #     for mimic_joint, master_joint, mimic_qidx, master_qidx, mult, offs in mimic_pairs[:2]:  # Print first 2
            #         expected_mimic = mult * qpos_current[master_qidx] + offs
            #         actual_mimic = qpos_current[mimic_qidx]
            #         print(f"  Mimic check: {mimic_joint.get_name()} = {actual_mimic:.4f} "
            #               f"(expected {expected_mimic:.4f} from {master_joint.get_name()})")
        
if __name__ == "__main__":
    main()    


