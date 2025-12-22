"""
minimal_panda_pd_hold.py

A minimal example to demonstrate position control with hold mode on Panda robot using SAPIEN.

What you get:
- Load Panda URDF
- Fix root link
- Set PD drive (stiffness/damping)
- Hold current pose under gravity
- Optionally move to a desired qpose smoothly

Run:
    python examples/load_robot/g1_wo_gripper.py
"""

import time
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer


def main():
    # -------------------
    # 1) Engine & Scene
    # -------------------
    engine = sapien.Engine()

    
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    
    # Add a ground plane
    scene.add_ground(altitude=0)
    
    # Add lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    
    # -------------------
    # 2) Load Panda URDF
    # -------------------
    loader = scene.create_urdf_loader()
    loader.fix_root_link=True
    
    # NOTE: replace the path below with the correct path to your panda.urdf
    urdf_path = "robot_descriptions/manipulation/Agibot/agibot_g1_description/urdf/agibot_g1_omni-picker.urdf"
    robot = loader.load(urdf_path)
    assert robot is not None, f"Failed to load URDF: {urdf_path}"
    
    # Put robot above ground a bit (if needed)
    # Some URDFs already have correct base pose; adjust if it starts intersecting the ground
    robot.set_pose(sapien.Pose([0, 0, 0]))
    
    
    # -------------------
    # 3) PD Drive Setup
    # -------------------
    # "Soft noodle" usually means stiffness too low / no drive.
    # Panda typical stable range: stiffness 1000-5000, damping 50-200
    stiffness = 3000.0
    damping = 100.0
    force_limit = 87.0  # Panda's max torque per joint; tune if needed
    
    active_joints = robot.get_active_joints()
    for j in active_joints:
        print(f"Set {j.get_name()} PD drive: stiffness={stiffness}, damping={damping}, force_limit={force_limit}")
    dof = robot.get_dof()
    
    # Initialize targets at current qpos so it holds immediately
    q0 = robot.get_qpos()
    assert len(q0) == dof
    
    for i, j in enumerate(active_joints):
        # Some Sapien versions support different arg names; common pattern:
        j.set_drive_property(stiffness=stiffness, damping=damping, force_limit=force_limit)
        # Set drive mode to position if your version requires it (often default is fine):
        # j.set_drive_mode(sapien.JointDriveMode.POSITION)
        j.set_drive_target(q0[i])
    
    
    # -------------------
    # 4) (Optional) define a desired pose and move there smoothly
    # -------------------
    # If you only want "hold", q_des=q0
    q_des = q0.copy()
    if dof >= 7:
        # A common "ready" pose for Panda (7 arm joints)
        # Keep gripper joint (if present) unchanged
        # q_des[:7] = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=np.float32)
        pass # keep q0
        
    # A simple first-order smoothing on target (not physics PD; just target interpolation)
    alpha = 0.02  # smoothing factor; smaller = slower
    
    # -------------------
    # 5) Simulation Loop
    # -------------------
    # If you use rendering: create a viewer
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.2, y=0.0, z=0.8)
    viewer.set_camera_rpy(r=0.0, p=-0.4, y=np.radians(180))    
    
    q_cmd = q0.copy()
    last_time = time.time()
    
    while not viewer.closed:
        # Update drive target every step (this is key!)
        q_cmd = (1 - alpha) * q_cmd + alpha * q_des
        for i, joint in enumerate(active_joints):
            joint.set_drive_target(q_cmd[i])
        
        # Step physics
        scene.step()
        scene.update_render()
        viewer.render()
        
        # Print fps (optional)
        now = time.time()
        if now - last_time > 1.0:
            last_time = now
            # If robot "sags" slowly: increase stiffness; If it jitters: increase damping
            print(f"qpos[0:7]:", robot.get_qpos()[:7])
            pass
        
if __name__ == "__main__":
    main()
     
    


