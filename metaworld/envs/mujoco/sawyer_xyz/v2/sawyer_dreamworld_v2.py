import mujoco
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDreamworldEnvV2(SawyerXYZEnv):
    def __init__(self, tasks=None, render_mode=None):
        self.max_dist = 0.03

        hand_low = (-0.5, 0.4, 0.05)
        hand_high = (0.5, 1.0, 0.5)

        # range of coffee machine position and base for other the objects
        obj_low = (-0.1, 0.8, -0.001)
        obj_high = (0.1, 0.9, +0.001)
        # goal_low[3] would be .1, but objects aren't fully initialized until a
        # few steps after reset(). In that time, it could be .01
        goal_low = obj_low + np.array([-0.001, -0.22 + self.max_dist, 0.299])
        goal_high = obj_high + np.array([+0.001, -0.22 + self.max_dist, 0.301])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_pos": np.array([0, 0.9, 0.28]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
        self.goal = np.array([0, 0.78, 0.33])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_dreamworld.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.02),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [("goal", self._target_pos)]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        return np.hstack(
            (self._get_site_pos("buttonStart"), self.get_body_com("mug_obj").copy(), self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.05]))
        )

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("mug").xmat.reshape(3, 3)
        mugquat = Rotation.from_matrix(geom_xmat).as_quat()
        return np.hstack(
            (np.array([1.0, 0.0, 0.0, 0.0]), mugquat, np.zeros(4))
        )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        # set mug position
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0

        # set drawer position (open drawer)
        qpos[self.model.jnt_qposadr[self.model.body_jntadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer_link")]]] = -0.15

        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self._get_state_rand_vec() # random init position of coffee machine and base for other objects
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
        ] = self.obj_init_pos

        # Set _target_pos to current drawer position (closed)
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos
        self._target_pos = (self.obj_init_pos
                            + np.array([0.3, 0, 0]) # offset of drawer relative to coffee machine
                            + np.array([0.0, -0.16, 0.09])) # offset of drawer_link relative to drawer

        pos_mug = self.obj_init_pos + np.array([0.0, -0.22, 0.0])
        self._set_obj_xyz(pos_mug) # set mug position and open drawer
        self.obj_init_pos = self._get_pos_objects()[2] # set obj_init_pos as initial position of drawer_link (like in drawer-close env)

        # set button target
        #pos_button = self.obj_init_pos + np.array([0.0, -0.22, 0.3])
        #self._target_pos = pos_button + np.array([0.0, self.max_dist, 0.0])


        return self._get_obs()

    '''
    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid="long_tail",
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self.max_dist,
            sigmoid="long_tail",
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed)
    '''
    def compute_reward(self, action, obs):
        obj = obs[18:21]

        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj - target
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = self.obj_init_pos - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        handle_reach_radius = 0.005
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS + 0.015:
            reward = 1.0

        reward *= 10

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)


class TrainDreamworldv2(SawyerDreamworldEnvV2):
    tasks = None

    def __init__(self):
        SawyerDreamworldEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestDreamworldv2(SawyerDreamworldEnvV2):
    tasks = None

    def __init__(self):
        SawyerDreamworldEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
