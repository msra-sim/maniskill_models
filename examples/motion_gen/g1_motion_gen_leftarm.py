import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to suppress warnings

import sapien.core as sapien
import mplib
import numpy as np

# Load robot in SAPIEN
scene = sapien.Scene()
loader = scene.create_urdf_loader()
robot = loader.load("robot_descriptions/manipulation/Agibot/agibot_g1_description/urdf/agibot_g1_omni-picker.leftarm.urdf")  # Replace with your URDF path

print([joint.get_name() for joint in robot.get_active_joints()])
print([link.get_name() for link in robot.get_links()])
# Create mplib planner
planner = mplib.Planner(
    urdf="robot_descriptions/manipulation/Agibot/agibot_g1_description/urdf/agibot_g1_omni-picker.leftarm.urdf",
    srdf="robot_descriptions/manipulation/Agibot/agibot_g1_description/urdf/agibot_g1_omni-picker.leftarm_mplib.srdf",  # Specifies collision pairs to ignore
    # srdf=None,
#    user_link_names=[link.get_name() for link in robot.get_links()],
    # user_joint_names=[joint.get_name() for joint in robot.get_active_joints()],
    # user_joint_names=["left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6", "left_joint7"],
    move_group="left_end_link",  # Planning group defined in SRDF
    # joint_vel_limits=np.ones(7),  # Max joint velocities
    # joint_acc_limits=np.ones(7)   # Max joint accelerations
)


# Create Pinocchio model for IK computation
model = robot.create_pinocchio_model()

model.compute_forward_kinematics([0.1] * len(robot.get_active_joints()))  # Optional: compute FK at neutral pose
pose0 = model.get_link_pose(12)  # Example: link index 9 for Panda end-effector
print(f"End-effector initial position: {pose0.p}, orientation: {pose0.q}")

# Define target pose [x, y, z, qw, qx, qy, qz] in robot base frame
# target_pose = mplib.Pose([0.4, 0.5, 1], [1, 0, 0, 0])
target_pose = mplib.Pose([pose0.p[0], pose0.p[1], pose0.p[2]], pose0.q)

# Get current joint positions
current_qpos = robot.get_qpos()

# Plan collision-free path
result = planner.plan_pose(
    target_pose,
    current_qpos,
    time_step=0.1,      # Time between waypoints
    rrt_range=0.1,      # RRT sampling range
    planning_time=1.0,   # Time limit for planning
    mask=[True, True, False, False, False, False, False, False, False]  # If value at a given index is True, the joint is no tused in IK
)


if result['status'] == "Success":
    print(f"Path found! Duration: {result['duration']:.2f}s")
    # result['position']: (n x m) waypoints in joint space
    # result['velocity']: (n x m) joint velocities
    # result['acceleration']: (n x m) joint accelerations
    # result['time']: (n,) timestamps
    print(f"qpos waypoints: {result['position']}")
else:
    print(f"Planning failed: {result['status']}")
    raise RuntimeError("Motion planning failed")


# Set PD controller properties
for joint in robot.get_active_joints():
    joint.set_drive_property(stiffness=1000, damping=200)

# Execute path
print("Executing path...")
for i in range(len(result['time'])):
    qpos_target = result['position'][i]
    qvel_target = result['velocity'][i]
    
    # Set drive targets (only for first 7 joints - the arm)
    for j, joint in enumerate(robot.get_active_joints()[:7]):
        if j < len(qpos_target):
            joint.set_drive_target(qpos_target[j])
            joint.set_drive_velocity_target(qvel_target[j])
    
    # Compensate passive forces
    qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
    robot.set_qf(qf)
    
    # Step simulation
    scene.step()
    scene.update_render()

print("✓ Path execution completed!")