import sapien.core as sapien
import numpy as np

scene = sapien.Scene()
scene.set_timestep(1 / 240)  # Smaller timestep for stability
scene.add_ground(altitude=0)


# Add a robot (URDF)
loader = scene.create_urdf_loader()
# loader.fix_root_link = True  # Fix the base to prevent falling
# loader.load_multiple_collisions_from_file = True  # Better collision handling
# robot = loader.load("robot_descriptions/Panda/panda.urdf")  # Replace
# robot = loader.load("robot_descriptions/manipulation/Agibot/agibot_omni_description/urdf/omni_picker.urdf")
robot = loader.load("robot_descriptions/manipulation/Agibot/agibot_g1_with_gripper_description/agibot_g1_with_omnipicker.urdf")

assert robot is not None

# Set initial pose above ground
robot.set_root_pose(sapien.Pose([0, 0, 0.5]))
# robot.set_root_pose(sapien.Pose([0, 0, 0.5], [0, 0, 1, 0]))

# Disable self-collision to prevent finger collisions
for link in robot.get_links():
    for shape in link.get_collision_shapes():
        # Set collision groups to avoid self-collision
        shape.set_collision_groups([1, 1, 2, 0])  # Only collide with group 2 (ground)


# Inspect:
for joint in robot.get_joints():
    print(f"Joint: {joint.get_name()}, Type: {joint.get_type()}")

# Set initial joint positions by directly setting qpos (more stable)
# The gripper has mimic joints - set them all to initial position
gripper_value = 0  # Initial gripper opening (0=closed, 1=open)
mimic_multipliers = {
    # 'gripper_joint': 1.0,
    'narrow2_joint': 0.02,
    'narrow3_joint': 0.4,
    'narrow_loop_joint': 1.5,
    'wide1_joint': 1.0,
    'wide2_joint': 0.02,
    'wide3_joint': 0.4,
    'wide_loop_joint': 1.5,
}
dualarm_mimic_multipliers = {}
# left mimic multiplier is mimic key attached with 'left_' prefix and same value
for key in list(mimic_multipliers.keys()):
    left_key = 'left_' + key
    dualarm_mimic_multipliers[left_key] = mimic_multipliers[key]
# right mimic multiplier is mimic key attached with 'right_' prefix and same value
for key in list(mimic_multipliers.keys()):
    right_key = 'right_' + key
    dualarm_mimic_multipliers[right_key] = mimic_multipliers[key]

# Build qpos array with mimic joint values
qpos = []
for joint in robot.get_active_joints():
    joint_name = joint.get_name()
    if joint_name in dualarm_mimic_multipliers:
        qpos.append(gripper_value * dualarm_mimic_multipliers[joint_name])
    else:
        qpos.append(0.0)


# Optional: Add weak PD control for stability
for joint in robot.get_active_joints():
    joint_name = joint.get_name()
    # if joint_name in mimic_multipliers.keys():
    if joint_name in dualarm_mimic_multipliers.keys():
        # Mimic joints: High stiffness but WITH force limit and damping to prevent overpowering arm
        # joint.set_drive_properties(stiffness=2000.0, damping=0.0, force_limit=100.0)
        joint.set_drive_properties(stiffness=1000.0, damping=0.0)
        print(f"[MIMIC] {joint_name}: stiffness=1000.0, damping=0.0, force_limit=100.0")
    elif "gripper_joint" in joint_name:
        # Master gripper joint: moderate control
        # joint.set_drive_properties(stiffness=2000.0, damping=100.0)
        joint.set_drive_properties(stiffness=100.0, damping=10.0, force_limit=50.0)
        print(f"[GRIPPER] {joint_name}: stiffness=100.0, damping=10.0, force_limit=50.0")
    else:
        # Arm joints: strong control to resist gripper forces
        joint.set_drive_properties(stiffness=2000, damping=100)

        #### VERY IMPORTANT: if not set manually the arm will not stay at target position dragged away by gripper
        joint.set_armature([0.5])
        # joint.set_friction(1.0)
        print(f"[ARM] {joint_name}: stiffness={joint.stiffness}, damping=1000.0, force_limit=200.0")
# Set initial position directly (more stable than PD control)
pass

robot.set_qpos(np.array(qpos))
print("Initial qpos set to:", robot.get_qpos())

# Viewer
import sapien.core as sapien
from sapien.utils import Viewer
viewer = Viewer()
viewer.set_scene(scene)
# Create a look at camera
viewer.set_camera_xyz(x=1.0, y=1.0, z=1.0)
viewer.set_camera_rpy(r=0, p=-np.pi / 4, y=np.pi / 4)
# Add lighting
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
time = 0.0
dt = 1 / 240.0
scene.set_timestep(dt)


# print joint name with order
for j_idx, j in enumerate(robot.get_active_joints()):
    j_name = j.get_name()
    print(f"{j_idx}: Joint '{j_name}'")
"""
0: Joint 'body_joint1'
1: Joint 'body_joint2'
2: Joint 'head_joint1'
3: Joint 'head_joint2'
4: Joint 'left_joint1'
5: Joint 'right_joint1'
6: Joint 'left_joint2'
7: Joint 'right_joint2'
8: Joint 'left_joint3'
9: Joint 'right_joint3'
10: Joint 'left_joint4'
11: Joint 'right_joint4'
12: Joint 'left_joint5'
13: Joint 'right_joint5'
14: Joint 'left_joint6'
15: Joint 'right_joint6'
16: Joint 'left_joint7'
17: Joint 'right_joint7'
18: Joint 'left_gripper_joint'
19: Joint 'left_narrow_loop_joint'
20: Joint 'left_wide1_joint'
21: Joint 'left_wide_loop_joint'
22: Joint 'right_gripper_joint'
23: Joint 'right_narrow_loop_joint'
24: Joint 'right_wide1_joint'
25: Joint 'right_wide_loop_joint'
26: Joint 'left_narrow2_joint'
27: Joint 'left_wide2_joint'
28: Joint 'right_narrow2_joint'
29: Joint 'right_wide2_joint'
30: Joint 'left_narrow3_joint'
31: Joint 'left_wide3_joint'
32: Joint 'right_narrow3_joint'
33: Joint 'right_wide3_joint'
"""


# -----------------------------
# Mount Camera on Head and Arm
# -----------------------------
link = next(l for l in robot.get_links() if l.get_name() == "head_camera")

# 1) Head Camera
head_camera = scene.add_mounted_camera(
    name="wrist_cam",
    mount=link.entity,           # Use link.entity as the mount point
    # # +90 degree around Y axis
    # pose=sapien.Pose(p=[0, 0, 0], q=[0.707, 0, 0.707, 0]),  # Identity pose relative to link
    pose =sapien.Pose(
        p=[0.0, 0.0, 0.0],
        q=[1, 0.0, 0, 0.0]  # 90 degrees around Y axis
    ),
    width=1280,
    height=720,
    fovy=np.deg2rad(58.0),
    near=0.01,
    far=100.0
)

print(f"Camera mounted to link: head_camera")

# 2) Left and Right Wrist Cameras
rpy = [0, -1.117, 1.5708] # Ref: https://github.com/fiveages-sim/robot_descriptions/blob/main/humanoid/Agibot/agibot_g1_description/xacro/omnipicker_camera_stand.xacro
# convert rpy to quaternion
def rpy_to_quat(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

    
def get_link_by_name(robot, link_name):
    for link in robot.get_links():
        if link.name == link_name:
            return link
    return None

left_wrist_camera = scene.add_mounted_camera(
    name="left_wrist_camera",
    mount=get_link_by_name(robot, "left_camera_stand").entity,  # Mount to the left_wrist_camera link
    pose=sapien.Pose(p=[0, -0.07754, 0.028618], 
                     q=rpy_to_quat(*rpy)
                    #  q=[1, 0, 0, 0]
                     ),  # Identity pose relative to link
    width=640,
    height=480,
    fovy=np.pi / 2,
    near=0.01,
    far=100,
)

right_wrist_camera = scene.add_mounted_camera(
    name="right_wrist_camera",
    mount=get_link_by_name(robot, "right_camera_stand").entity,  # Mount to the right_wrist_camera link
    pose=sapien.Pose(p=[0, -0.07754, 0.028618], q=rpy_to_quat(*rpy)),  # Identity pose relative to link
    width=640,
    height=480,
    fovy=np.pi / 2,
    near=0.01,
    far=100,
)
print(f"   ✓ Camera added: {head_camera.name}")
print(f"   Camera type: {type(head_camera)}")
print(f"   ✓ Camera added: {left_wrist_camera.name}")
print(f"   Camera type: {type(left_wrist_camera)}")
print(f"   ✓ Camera added: {right_wrist_camera.name}")
print(f"   Camera type: {type(right_wrist_camera)}")



while not viewer.closed:
    scene.step()
    scene.update_render()
    viewer.render()

    # ------------------------------
    # Take pictures from the cameras
    # ------------------------------
    head_camera.take_picture()
    left_wrist_camera.take_picture()
    right_wrist_camera.take_picture()
    # Get the RGB image
    rgb_head = head_camera.get_picture("Color")
    rgb_left = left_wrist_camera.get_picture("Color")
    rgb_right = right_wrist_camera.get_picture("Color")

    # Process and Save the images as needed
    # For example, you can use matplotlib to display or save the images

    import matplotlib.pyplot as plt
    plt.imsave("head_camera_image.png", rgb_head)
    plt.imsave("left_wrist_camera_image.png", rgb_left)
    plt.imsave("right_wrist_camera_image.png", rgb_right)


    head_body_q_des = np.array([
        0.27,
        0.5235985,
        0.0,
        0.43633231,
    ], dtype=np.float32)
    # back and forward arm in loop
    arm_q_des = np.array([
        -1.074,
        1.075,
        0.6106,
        -0.61137,
        0.2808,
        -0.2807,
        -1.2839,
        1.2839,
        0.72,
        -0.7319,
        1.495,
        -1.495,
        -0.186,
        0.1876,
    ], dtype=np.float32)

    # in loop open/close the gripper
    time += dt
    gripper_value = 0.5 + 0.5 * np.sin(time)
    arm_value = arm_q_des * 0.5 + arm_q_des * 0.5 * np.sin(time * 0.5)
    headbody_value = head_body_q_des * 0.5 + head_body_q_des * 0.5 * np.sin(time * 0.5)

    # gripper limits to [0.1, 0.8]
    gripper_value = np.clip(gripper_value, 0.1, 0.7)

    qpos = []
    for joint_idx, joint in enumerate(robot.get_active_joints()):
        joint_name = joint.get_name()
        if joint_name in dualarm_mimic_multipliers:
            qpos.append(gripper_value * dualarm_mimic_multipliers[joint_name])
        elif joint_name.startswith("right_joint") or joint_name.startswith("left_joint"):
            # qpos.append(0.0)
            qpos.append([arm_value[joint_idx-len(head_body_q_des)]])
        elif joint_name.startswith("head_joint") or joint_name.startswith("body_joint"):
            qpos.append([headbody_value[joint_idx]])
        else:
            qpos.append(gripper_value)  # keep arm joints at gripper_value for testing
    # robot.set_qpos(np.array(qpos))
    # Instead of directly setting qpos, use drive targets for smooth motion
    for i, joint in enumerate(robot.get_active_joints()):
        print(f"Setting joint {joint.get_name()} drive target to {qpos[i]}")
        joint.set_drive_target(qpos[i])
    
