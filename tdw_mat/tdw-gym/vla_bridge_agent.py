"""
VLA Bridge Agent

Replaces CoELA's LLM-as-planner with physicai's VLM planner.
Keeps CoELA's AgentMemory for spatial reasoning + A* pathfinding.
Produces discrete actions compatible with tdw_gym.step().
"""

import os
import sys
import copy
import logging
import random
import numpy as np

from agent_memory import AgentMemory

PHYSICAI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PHYSICAI_ROOT not in sys.path:
    sys.path.insert(0, PHYSICAI_ROOT)

logger = logging.getLogger(__name__)

CELL_SIZE = 0.125
ANGLE = 15


class VLABridgeAgent:
    """
    Hybrid agent: physicai VLM planner + CoELA spatial memory.

    Plan generation: VLM planner (or heuristic fallback) decides WHAT to do.
    Plan execution: AgentMemory's A* pathfinding decides HOW to get there.
    Action output: discrete actions for CoELA's tdw_gym.
    """

    def __init__(self, agent_id, logger, max_frames, args=None, output_dir='results'):
        self.agent_id = agent_id
        self.agent_type = 'vla_bridge'
        self.logger = logger
        self.max_frames = max_frames
        self.output_dir = output_dir

        self.map_size = (240, 120)
        self._scene_bounds = {
            "x_min": -15, "x_max": 15,
            "z_min": -7.5, "z_max": 7.5
        }
        self.max_nav_steps = 80
        self.max_move_steps = 150

        self._vlm_planner = None
        self._vlm_url = getattr(args, 'vlm_url', None) if args else None
        self._dsm = None
        self.communication = getattr(args, 'communication', False) if args else False

        self.agent_memory = None
        self.obs = None
        self.plan = None
        self.target_pos = None
        self.explore_count = 0
        self.local_step = 0
        self.steps = 0
        self.num_frames = 0
        self.invalid_count = 0

        self.goal_objects = None
        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.object_map = None
        self.id_map = None
        self.color2id = {}

        self.rooms_name = None
        self.rooms_explored = {}
        self.current_room = None
        self.env_api = None
        self.holding_objects_id = []
        self.satisfied = []
        self.with_character = []
        self.with_oppo = []
        self.dropping_object = []
        self.last_action = None
        self.rotated = None
        self.action_history = []
        self.save_img = True
        self.gt_mask = True
        self.position = None
        self.forward = None

    def reset(self, obs=None, goal_objects=None, output_dir=None, env_api=None,
              rooms_name=None, agent_color=None, agent_id=0, gt_mask=True, save_img=True):
        if agent_color is None:
            agent_color = [-1, -1, -1]
        self.obs = obs
        self.env_api = env_api
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.gt_mask = gt_mask
        self.save_img = save_img
        self.goal_objects = goal_objects or {}
        if output_dir:
            self.output_dir = output_dir

        self.agent_memory = AgentMemory(
            agent_id=agent_id, agent_color=agent_color,
            output_dir=output_dir, gt_mask=gt_mask, gt_behavior=True,
            env_api=env_api, constraint_type=None,
            map_size=self.map_size, scene_bounds=self._scene_bounds
        )
        self.object_map = np.zeros(self.map_size, np.int32)
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.color2id = {}
        self.holding_objects_id = []
        self.satisfied = []
        self.with_character = []
        self.with_oppo = []
        self.dropping_object = []
        self.rooms_explored = {}
        self.plan = None
        self.target_pos = None
        self.last_action = None
        self.steps = 0
        self.num_frames = 0
        self.invalid_count = 0
        self.local_step = 0
        self.action_history = []

        if obs:
            self.position = obs["agent"][:3]
            self.forward = obs["agent"][3:]
            self.current_room = env_api['belongs_to_which_room'](self.position)
            self.action_history.append(f"go to {self.current_room} at initial step")

        self._init_vlm_planner()

    def _init_vlm_planner(self):
        try:
            from simulator.vla.vlm_planner import VLMPlanner
            self._vlm_planner = VLMPlanner(url=self._vlm_url)
            logger.info("VLMPlanner initialized")
        except Exception:
            logger.info("VLMPlanner unavailable, using heuristic fallback")
            self._vlm_planner = None

    # -- main loop --

    def act(self, obs):
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)
        self.num_frames = obs['current_frames']
        self.steps += 1

        if obs.get('valid') is False:
            self._handle_invalid()

        self._update_agent_state()
        self._update_memory()

        if self.obs['status'] == 0:
            return {'type': 'ongoing'}

        self._scan_objects()
        self._build_object_list()

        action = None
        retries = 0
        while action is None and retries < 4:
            if self.plan is None:
                self.target_pos = None
                self.plan = self._generate_plan()
                self.action_history.append(f"{self.plan} at step {self.num_frames}")
                retries += 1
            action = self._execute_plan()

        self.last_action = action
        return action or {'type': 'ongoing'}

    # -- plan generation --

    def _generate_plan(self):
        """VLM planner or heuristic fallback."""
        if self._vlm_planner:
            return self._vlm_plan()
        return self._heuristic_plan()

    def _vlm_plan(self):
        """Ask VLM planner for next high-level action."""
        context = self._build_vlm_context()
        try:
            from simulator.vla.vlm_planner import VLMPlanner
            result = self._vlm_planner.plan(
                image=self.obs['rgb'],
                instruction=self._goal_description(),
                context=context
            )
            return self._parse_vlm_result(result)
        except Exception as e:
            logger.warning(f"VLM plan failed: {e}, falling back to heuristic")
            return self._heuristic_plan()

    def _heuristic_plan(self):
        """Priority-based fallback: transport > grasp > explore."""
        held = self.obs['held_objects']
        has_target = any(h['type'] == 0 for h in held if h['id'] is not None)
        has_container = any(h['type'] == 1 for h in held if h['id'] is not None)

        if has_target and has_container:
            if held[0]['type'] == 1 and held[0].get('contained', [None])[
                -1] is None and held[1]['type'] == 0:
                return f"put <{held[1]['name']}> ({held[1]['id']}) into container"
            if held[1]['type'] == 1 and held[1].get('contained', [None])[
                -1] is None and held[0]['type'] == 0:
                return f"put <{held[0]['name']}> ({held[0]['id']}) into container"

        if (has_target or has_container) and len(self.object_list[2]) > 0:
            return "transport objects I'm holding to the bed"

        # Grasp targets
        if len(self.object_list[0]) > 0 and (held[0]['id'] is None or held[1]['id'] is None):
            obj = self.object_list[0][0]
            return f"go grasp target object <{obj['name']}> ({obj['id']})"

        # Grasp container
        if len(self.object_list[1]) > 0 and not has_container and (
                held[0]['id'] is None or held[1]['id'] is None):
            obj = self.object_list[1][0]
            return f"go grasp container <{obj['name']}> ({obj['id']})"

        # Explore unexplored rooms
        for room in (self.rooms_name or []):
            if room not in self.rooms_explored or self.rooms_explored[room] != 'all':
                if room != self.current_room:
                    return f"go to {room}"
                return f"explore current room {room}"

        return "[wait]"

    def _parse_vlm_result(self, result):
        """Convert VLM planner output to CoELA plan string."""
        mode = getattr(result, 'mode', 'NAV')
        target = getattr(result, 'target', None)
        if mode == 'MANIP' and target:
            return f"go grasp target object <{target}>"
        if mode == 'NAV' and target:
            return f"go to {target}"
        return self._heuristic_plan()

    def _build_vlm_context(self):
        return {
            'goal': self._goal_description(),
            'current_room': self.current_room,
            'rooms_explored': self.rooms_explored,
            'holding': [h for h in self.obs['held_objects'] if h['id'] is not None],
            'visible_targets': len(self.object_list[0]),
            'visible_containers': len(self.object_list[1]),
            'satisfied_count': len(self.satisfied),
            'step': self.num_frames,
        }

    # -- plan execution (reuses AgentMemory pathfinding) --

    def _execute_plan(self):
        if self.plan is None:
            return None
        if self.plan.startswith('go to'):
            return self._exec_goto()
        if self.plan.startswith('explore'):
            return self._exec_explore()
        if self.plan.startswith('go grasp'):
            return self._exec_grasp()
        if self.plan.startswith('put'):
            return self._exec_putin()
        if self.plan.startswith('transport'):
            return self._exec_transport()
        if self.plan.startswith('[wait]'):
            return {'type': 'ongoing'}
        self.plan = None
        return None

    def _exec_goto(self):
        target_room = ' '.join(self.plan.split(' ')[2:4])
        if target_room[-1] == ',':
            target_room = target_room[:-1]
        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room:
            self.plan = None
            return None
        return self._move_to(target_pos)

    def _exec_explore(self):
        target_room = ' '.join(self.plan.split(' ')[-2:])
        target_pos = self.env_api['center_of_room'](target_room)
        self.explore_count += 1
        d = np.linalg.norm(np.array(self.position[[0, 2]]) - np.array([target_pos[0], target_pos[2]]))
        if d < 1 + self.explore_count / 50:
            if self.rotated is None:
                self.rotated = 0
            if self.rotated >= 16:
                self.rooms_explored[target_room] = 'all'
                self.plan = None
                self.rotated = None
                return None
            self.rotated += 1
            return {"type": 1}
        return self._move_to(target_pos)

    def _exec_grasp(self):
        parts = self.plan.split(' ')
        try:
            target_id = int(parts[-1].strip('()'))
        except ValueError:
            self.plan = None
            return None
        if target_id in self.holding_objects_id:
            self.plan = None
            return None
        if target_id not in self.object_info:
            self.plan = None
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_info[target_id]['position'])
        d = np.linalg.norm(np.array(self.position[[0, 2]]) - np.array([self.target_pos[0], self.target_pos[2]]))
        if d > 1.0:
            return self._move_to(self.target_pos)
        arm = 'left' if self.obs['held_objects'][0]['id'] is None else 'right'
        return {"type": 3, "object": target_id, "arm": arm}

    def _exec_putin(self):
        return {"type": 4}

    def _exec_transport(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            return None
        if not self.object_list[2]:
            self.plan = None
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_list[2][0]['position'])
        d = np.linalg.norm(np.array(self.position[[0, 2]]) - np.array([self.target_pos[0], self.target_pos[2]]))
        if d > 1.5:
            return self._move_to(self.target_pos)
        if self.obs['held_objects'][0]['type'] is not None:
            return {"type": 5, "arm": "left"}
        return {"type": 5, "arm": "right"}

    def _move_to(self, target_pos):
        self.local_step += 1
        action, _ = self.agent_memory.move_to_pos(target_pos)
        return action

    # -- state management --

    def _handle_invalid(self):
        if self.last_action and 'object' in self.last_action:
            oid = self.last_action['object']
            self.object_map[np.where(self.id_map == oid)] = 0
            self.id_map[np.where(self.id_map == oid)] = 0
            self.satisfied.append(oid)
        self.invalid_count += 1
        self.plan = None

    def _update_agent_state(self):
        self.position = np.array(self.obs["agent"][:3])
        self.forward = np.array(self.obs["agent"][3:])
        room = self.env_api['belongs_to_which_room'](self.position)
        if room:
            self.current_room = room
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'

        self.holding_objects_id = []
        self.with_character = [self.agent_id]
        self.with_oppo = []
        for h in self.obs['held_objects']:
            if h['id'] is not None:
                self.holding_objects_id.append(h['id'])
                self.with_character.append(h['id'])
                for c in (h.get('contained') or []):
                    if c is not None:
                        self.with_character.append(c)
        for h in self.obs['oppo_held_objects']:
            if h['id'] is not None:
                self.with_oppo.append(h['id'])
                for c in (h.get('contained') or []):
                    if c is not None:
                        self.with_oppo.append(c)
        for oid in self.with_oppo:
            if oid not in self.satisfied:
                self.satisfied.append(oid)
                self.object_map[np.where(self.id_map == oid)] = 0
                self.id_map[np.where(self.id_map == oid)] = 0

    def _update_memory(self):
        ignore_ids = self.with_character + self.with_oppo + self.satisfied
        ignore_obs = self.with_character + self.satisfied
        self.agent_memory.update(
            self.obs, ignore_ids=ignore_ids,
            ignore_obstacles=ignore_obs, save_img=self.save_img
        )

    def _scan_objects(self):
        for o in self.obs['visible_objects']:
            if o['id'] is None or o['id'] in self.satisfied or o['type'] is None:
                continue
            if o['type'] >= 3:
                continue
            self.color2id[o['seg_color']] = o['id']
            pos, _ = self.agent_memory.cal_object_position(o)
            if pos is None:
                continue
            self.object_info[o['id']] = {**o, 'position': pos}
            x, z = pos[0], pos[2]
            i, j = self.agent_memory.pos2map(x, z)
            if 0 <= i < self.map_size[0] and 0 <= j < self.map_size[1]:
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = o['type'] + 1
                    self.id_map[i, j] = o['id']

    def _build_object_list(self):
        self.object_list = {0: [], 1: [], 2: []}
        for otype in [0, 1, 2]:
            idxs = np.where(self.object_map == otype + 1)
            for k in range(idxs[0].shape[0]):
                oid = self.id_map[idxs[0][k], idxs[1][k]]
                if oid in self.satisfied or oid in self.holding_objects_id:
                    continue
                if oid in self.object_info and self.object_info[oid] not in self.object_list[otype]:
                    self.object_list[otype].append(self.object_info[oid])

    def _goal_description(self):
        parts = [f"{c} {n}{'s' if c > 1 else ''}" for n, c in (self.goal_objects or {}).items()]
        return f"Transport {', '.join(parts)} to the bed."
