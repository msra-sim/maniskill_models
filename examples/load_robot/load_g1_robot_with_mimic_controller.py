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

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

# -----------------------------
# Config
# -----------------------------

@dataclass
class MimicPDConfig:
    stiffness: float = 200.0
    damping: float = 20.0
    force_limit: float = 200.0
    friction: float = 0.0  # if your SAPIEN build supports joint friction setter
    drive_mode: str = "force"  # or "acceleration" depending on your SAPIEN
    use_delta: bool = False
    interpolate: bool = False
    sim_steps_per_action: int = 1  # e.g., 4 if you apply action every 4 sim steps


# -----------------------------
# Controller
# -----------------------------

class MimicPDController:
    """
    A SAPIEN mimic PD controller (ManiSkill-style):
    - Action space is ONLY control joints (masters).
    - Mimic joints are driven followers via q_m = q_c * mul + off.
    - Drive targets are set for ALL active joints every step (or per action).
    """

    def __init__(
        self,
        articulation,
        joint_name_list: Sequence[str],
        mimic_map: Dict[str, Dict[str, float]],
        config: MimicPDConfig = MimicPDConfig(),
        master_config: Optional[MimicPDConfig] = None,
        device: str = "cpu",
    ):
        """
        articulation: sapien.Articulation (or equivalent)
        joint_name_list: all joints you want this controller to manage (controls + mimics)
        mimic_map:
            {
              "mimic_joint_name": {"joint": "control_joint_name", "multiplier": k, "offset": b},
              ...
            }
        """
        self.art = articulation
        self.cfg = config
        self.master_cfg = master_config

        # Resolve joint objects (active joints only)
        # NOTE: API differs slightly across SAPIEN versions; adjust if needed.
        self.joints = []
        self.joint_names = []
        joints_map = {j.get_name(): j for j in self.art.get_joints()}  # includes fixed/passive too
        for name in joint_name_list:
            if name not in joints_map:
                raise KeyError(f"Joint '{name}' not found in articulation.")
            self.joints.append(joints_map[name])
            self.joint_names.append(name)

        # Active index mapping
        # Many SAPIEN builds: joint.get_dof() or joint.get_qpos() etc.
        # We'll build indices by reading articulation dof ordering.
        # Robust approach: map name -> active_dof_index by scanning active joints list.
        self.active_joints = list(self.art.get_active_joints())
        self.active_name_to_index = {j.get_name(): i for i, j in enumerate(self.active_joints)}

        # Build mimic indices (in "active joint index" space)
        self.mimic_joint_indices = []
        self.control_joint_indices_for_mimic = []
        self.multiplier = []
        self.offset = []

        if not mimic_map:
            raise ValueError("mimic_map is empty. Provide mapping mimic_joint -> control_joint.")

        # Sanity checks + parse map
        for mimic_joint_name, data in mimic_map.items():
            control_joint_name = data["joint"]
            k = float(data.get("multiplier", 1.0))
            b = float(data.get("offset", 0.0))

            if mimic_joint_name not in self.active_name_to_index:
                raise ValueError(f"Mimic joint '{mimic_joint_name}' is not an ACTIVE joint in SAPIEN.")
            if control_joint_name not in self.active_name_to_index:
                raise ValueError(f"Control joint '{control_joint_name}' is not an ACTIVE joint in SAPIEN.")

            self.mimic_joint_indices.append(self.active_name_to_index[mimic_joint_name])
            self.control_joint_indices_for_mimic.append(self.active_name_to_index[control_joint_name])
            self.multiplier.append(k)
            self.offset.append(b)

        self.mimic_joint_indices = np.array(self.mimic_joint_indices, dtype=np.int32)
        self.control_joint_indices_for_mimic = np.array(self.control_joint_indices_for_mimic, dtype=np.int32)
        self.multiplier = np.array(self.multiplier, dtype=np.float32)
        self.offset = np.array(self.offset, dtype=np.float32)

        self.control_joint_indices = np.unique(self.control_joint_indices_for_mimic).astype(np.int32)
        self.effective_dof = len(self.control_joint_indices)

        # Cache limits for clamping
        self.qlimits = self._get_qlimits()  # shape (ndof, 2)

        # State
        self._start_qpos = None
        self._target_qpos = None
        self._step = 0
        self._step_size = None

    # --------- helpers ---------

    def _get_qpos(self) -> np.ndarray:
        # SAPIEN: articulation.get_qpos() returns (ndof,)
        return np.array(self.art.get_qpos(), dtype=np.float32)

    def _set_drive_targets(self, qpos_targets: np.ndarray):
        # SAPIEN API differs:
        # - Some versions: joint.set_drive_target(float)
        # - Some versions: articulation.set_drive_target(array)
        #
        # We'll do per-joint for robustness:
        for i, j in enumerate(self.active_joints):
            # For revolute/prismatic DOF joints:
            j.set_drive_target(float(qpos_targets[i]))

    def _get_qlimits(self) -> np.ndarray:
        # Many versions: articulation.get_qlimits() -> (ndof, 2)
        ql = np.array(self.art.get_qlimits(), dtype=np.float32)
        if ql.ndim == 3:
            # some batched API returns (1, ndof, 2)
            ql = ql[0]
        return ql

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        lo = self.qlimits[:, 0]
        hi = self.qlimits[:, 1]
        # Some joints might have (0,0) or huge; still safe
        return np.minimum(np.maximum(q, lo), hi)

    # --------- public API ---------

    def set_drive_properties(self):
        """
        Sets drive properties for ALL active joints, including mimic joints.
        This is important for stability: mimic joints must not be free.
        """
        for i, j in enumerate(self.active_joints):
            # SAPIEN joint drive property call names vary.
            # Common pattern:
            #   j.set_drive_property(stiffness, damping, force_limit=..., mode=...)
            
            is_mimic_j = i in self.mimic_joint_indices
            if not is_mimic_j and self.master_cfg is not None:
                j.set_drive_property(
                    self.master_cfg.stiffness,
                    self.master_cfg.damping,
                    force_limit=self.master_cfg.force_limit,
                    mode=self.master_cfg.drive_mode,
                )
                print(f"Set drive for master joint '{j.get_name()}': stiffness={self.master_cfg.stiffness}, damping={self.master_cfg.damping}, force_limit={self.master_cfg.force_limit}, mode={self.master_cfg.drive_mode}")
            else:
                j.set_drive_property(
                    self.cfg.stiffness,
                    self.cfg.damping,
                    force_limit=self.cfg.force_limit,
                    mode=self.cfg.drive_mode,
                )
                print(f"Set drive for joint '{j.get_name()}': stiffness={self.cfg.stiffness}, damping={self.cfg.damping}, force_limit={self.cfg.force_limit}, mode={self.cfg.drive_mode}")
            # # Optional friction (if supported):
            # try:
            #     j.set_friction(self.cfg.friction)
            # except Exception:
            #     pass

    def reset(self):
        qpos = self._get_qpos()
        self._start_qpos = qpos.copy()
        self._target_qpos = qpos.copy()
        self._step = 0
        self._step_size = None

    def action_space_shape(self) -> Tuple[int]:
        # action is only for control joints
        return (self.effective_dof,)

    def set_action(self, action: np.ndarray):
        """
        action: shape (effective_dof,)
        - if use_delta: action is delta on control joints
        - else: action is absolute targets for control joints
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        assert action.shape[0] == self.effective_dof, (
            f"Expected action dim {self.effective_dof}, got {action.shape[0]}"
        )

        self._step = 0
        qpos = self._get_qpos()
        self._start_qpos = qpos.copy()
        if self._target_qpos is None:
            self._target_qpos = qpos.copy()

        # write control targets
        if self.cfg.use_delta:
            self._target_qpos[self.control_joint_indices] = (
                qpos[self.control_joint_indices] + action
            )
        else:
            self._target_qpos[self.control_joint_indices] = action

        # compute mimic targets from control targets
        self._target_qpos[self.mimic_joint_indices] = (
            self._target_qpos[self.control_joint_indices_for_mimic] * self.multiplier
            + self.offset
        )

        # clamp everything
        self._target_qpos = self._clamp(self._target_qpos)

        # apply now or setup interpolation
        if self.cfg.interpolate and self.cfg.sim_steps_per_action > 1:
            self._step_size = (self._target_qpos - self._start_qpos) / float(self.cfg.sim_steps_per_action)
        else:
            self._target_qpos[self.control_joint_indices] = 0
            self._set_drive_targets(self._target_qpos)

    def before_simulation_step(self):
        """
        Call this once per sim step (if interpolate=True).
        """
        if not self.cfg.interpolate or self.cfg.sim_steps_per_action <= 1:
            return
        self._step += 1
        t = min(self._step, self.cfg.sim_steps_per_action)
        q = self._start_qpos + self._step_size * float(t)
        q = self._clamp(q)
        self._set_drive_targets(q)

    def debug_print(self):
        print("MimicPDController:")
        print("  effective_dof:", self.effective_dof)
        print("  control_joint_indices:", self.control_joint_indices.tolist())
        print("  mimic_joint_indices:", self.mimic_joint_indices.tolist())
        # print mapping pairs
        for mi, ci, k, b in zip(
            self.mimic_joint_indices,
            self.control_joint_indices_for_mimic,
            self.multiplier,
            self.offset,
        ):
            print(f"    {self.active_joints[mi].get_name()} <= {self.active_joints[ci].get_name()} * {k} + {b}")


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

TARGET_JOINT_NAMES = GRIPPER_JOINT_NAMES

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
    urdf_path = "robot_descriptions/manipulation/Agibot/agibot_g1_with_gripper_description/agibot_g1_with_omnipicker.urdf"
    
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
    
    # disable_internal_collisions(robot)

    # Put robot at origin
    robot.set_pose(sapien.Pose([0, 0, 0]))

    # for link in robot.get_links():
    #     if "narrow" in link.get_name() or "wide" in link.get_name():
    #         # Physical stability hack: boost mass of tiny fingers
    #         mass = link.get_mass()
    #         if mass < 0.1:
    #             # Manually setting a minimum mass for solver stability
    #             # Note: This requires SAPIEN 2.0+ API
    #             # If link.set_mass() isn't available, you must edit the URDF <mass> tags
    #             link.set_mass(0.1)
    #             print(f"[INFO] Boosted mass of link {link.get_name()} from {mass} to 0.1 kg for stability.")

    # # Disable collisions between arm end and gripper base
    # for link in robot.get_links():
    #     if "gripper" in link.get_name() or "camera_stand" in link.get_name():
    #         # Set a collision group that ignores itself or specific neighbors
    #         collision_shape = link.get_collision_shapes()[0]
    #         collision_shape.set_collision_groups(1, 1, 2, 2) # Group, Mask, etc.
    
    # -------------------
    # 3) Setup MimicPDController
    # -------------------
    
    # Convert mimic_map format from (master, mult, offset) to controller format
    mimic_map_controller = {}
    for mimic_joint_name, (master_joint_name, mult, offs) in mimic_map.items():
        mimic_map_controller[mimic_joint_name] = {
            "joint": master_joint_name,
            "multiplier": mult,
            "offset": offs
        }
    
    # Get all active joint names (control joints + mimic joints)
    all_joint_names = TARGET_JOINT_NAMES + list(mimic_map.keys())
    
    # Create controller config
    controller_config = MimicPDConfig(
        stiffness=2000.0,
        damping=0.0,
        force_limit=10000.0,
        friction=0.0,
        drive_mode="force",
        use_delta=False,  # Use absolute targets
        interpolate=True,  # Enable smooth interpolation
        sim_steps_per_action=10  # Interpolate over 10 steps
    )

    master_controller_config = MimicPDConfig(
        stiffness=100.0,
        damping=10.0,
        force_limit=50.0,
        friction=0.0,
        drive_mode="force",
        use_delta=False,
        interpolate=True,
        sim_steps_per_action=10
    )
    
    # Initialize controller
    controller = MimicPDController(
        articulation=robot,
        joint_name_list=all_joint_names,
        mimic_map=mimic_map_controller,
        config=controller_config,
        master_config=master_controller_config,
    )
    
    # Set drive properties for all joints
    controller.set_drive_properties()
    controller.debug_print()
    
    # Reset controller to initialize at current qpos
    controller.reset()
    print(f"[INFO] Controller initialized with action space: {controller.action_space_shape()}")
    
    # Get initial qpos for control joints only
    qpos_full = robot.get_qpos()
    robot.set_qpos(qpos_full)
    robot.set_qvel(np.zeros(robot.dof))
    print(f"[INFO] Set full qpos to: {qpos_full}")
    
    # Define desired action (only for control joints: 14 arm + 2 gripper = 16 DOF)
    q0_control = qpos_full[controller.control_joint_indices]
    # q_des_control = np.array([0.04, 0.04])  # Arms at zero, grippers slightly open
    
    # Set initial action to hold current position
    controller.set_action(q0_control)
    # -------------------
    # 4) Simulation Loop
    # -------------------
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.2, y=0.0, z=0.8)
    viewer.set_camera_rpy(r=0.0, p=-0.4, y=np.radians(180))    
    
    last_time = time.time()
    action_timer = 0
    action_interval = 120  # Set new action every 120 steps (0.5 seconds at 240Hz)
    
    counter = 0
    time_dt = 0.0
    while not viewer.closed:
        # viewer.render()
        
        
        # Update controller before each simulation step (handles interpolation)
        controller.before_simulation_step()
        
        # Step physics
        scene.step()
        scene.update_render()
        viewer.render()
        
        counter += 1
        # in loop open/close the gripper
        time_dt += 1/240.0
        gripper_value = 0.5 + 0.5 * np.sin(time_dt)
        
        q_des_control = np.array([gripper_value, gripper_value])  # open/close gripper periodically
        # Set new action periodically
        # if counter % action_interval == 0:
        #     if counter == 0:
        #         # Hold initial position
        #         controller.set_action(q0_control)
        #         print("[INFO] Action: Hold initial position")
        #     else:
        # Move to desired position
        controller.set_action(q_des_control)
        print(f"[INFO] Action: Move to target position")

        # Print status periodically
        now = time.time()
        if now - last_time > 1.0:
            last_time = now
            qpos_current = robot.get_qpos()
            control_qpos = qpos_current[controller.control_joint_indices]
            print(f"[INFO] Current control qpos: {control_qpos}")
            print(f"[INFO] Step: {counter}")
        
if __name__ == "__main__":
    main()    


