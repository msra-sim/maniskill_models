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
robot = loader.load("robot_descriptions/manipulation/Agibot/agibot_g1_with_gripper_description/agibot_g1_with_omnipicker.singlearm.urdf")

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
    'gripper_joint': 1.0,
    'narrow2_joint': 0.02,
    'narrow3_joint': 0.4,
    'narrow_loop_joint': 1.5,
    'wide1_joint': 1.0,
    'wide2_joint': 0.02,
    'wide3_joint': 0.4,
    'wide_loop_joint': 1.5,
}

# Build qpos array with mimic joint values
qpos = []
for joint in robot.get_active_joints():
    joint_name = joint.get_name()
    if joint_name in mimic_multipliers:
        qpos.append(gripper_value * mimic_multipliers[joint_name])
    else:
        qpos.append(0.0)


# Optional: Add weak PD control for stability
for joint in robot.get_active_joints():
    # joint.set_drive_property(stiffness=20.0, damping=5.0)
    # joint.set_drive_properties(stiffness=100.0, damping=10.0, force_limit=50) # in isaac sim, usd is set to stiffness:100, damping: 10, no force limit
    # joint.set_drive_properties(stiffness=200.0, damping=20.0, force_limit=50) 
    # joint.set_drive_properties(stiffness=400.0, damping=40.0) 
    joint_name = joint.get_name()
    if joint_name in mimic_multipliers.keys():
        # set a very weak PD control for the gripper joints
        joint.set_drive_properties(stiffness=2000.0, damping=0.0) # big stiffness and zero damping on mimic joint to avoid shaking
    else:
        joint.set_drive_properties(stiffness=100.0, damping=10.0, force_limit=50)
    # set a normal stiffness and damping for the arm joints
    # joint.set_drive_property(stiffness=1000.0, damping=40.0)
# Set initial position directly (more stable than PD control)


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
while not viewer.closed:
    scene.step()
    scene.update_render()
    viewer.render()

    # in loop open/close the gripper
    time += dt
    gripper_value = 0.5 + 0.5 * np.sin(time)
    qpos = []
    for joint in robot.get_active_joints():
        joint_name = joint.get_name()
        if joint_name in mimic_multipliers:
            qpos.append(gripper_value * mimic_multipliers[joint_name])
        else:
            # qpos.append(0.0)
            qpos.append(gripper_value)  # keep arm joints at gripper_value for testing
    # robot.set_qpos(np.array(qpos))
    # Instead of directly setting qpos, use drive targets for smooth motion
    for i, joint in enumerate(robot.get_active_joints()):
        joint.set_drive_target(qpos[i])
    
