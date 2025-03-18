"""
This code is partially adapted from:
https://github.com/lbaa2022/LLMTaskPlanning/blob/main/src/alfred/thor_connector.py

ThorConnector Class
------------------
Provides an interface between AI2-THOR and the LLM planning system. Extends the base
ThorEnv with higher-level interaction methods for navigation, object manipulation,
and environment querying. Translates natural language instructions into environment
actions, handling 3D navigation and object interaction complexities.
"""

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

import math
import textwrap
import numpy as np
from scipy import spatial
from PIL import Image, ImageDraw, ImageFont
import logging

from env.thor_env import ThorEnv
from gen import constants
from alfred.utils import natural_word_to_ithor_name

# Get a logger for this module
log = logging.getLogger(__name__)


class ThorConnector(ThorEnv):
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        log.debug("Initializing ThorConnector")
        super().__init__(x_display, player_screen_height, player_screen_width, quality, build_path)
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
        self.agent_height = 0.9
        self.cur_receptacle = None
        self.reachable_positions, self.reachable_position_kdtree = None, None
        self.sliced = False
        log.debug("ThorConnector initialized")

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        log.debug("Restoring scene")
        super().restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        self.cur_receptacle = None
        log.debug("Scene restored")

    def get_reachable_positions(self):
        log.debug("Getting reachable positions")
        free_positions = super().step(dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        log.debug(f"Found {len(free_positions)} reachable positions")
        return free_positions, kd_tree

    def write_step_on_img(self, cfg, idx, description):
        img = Image.fromarray(self.last_event.frame)
        text = str(idx) + ':' + description['action']
        lines = textwrap.wrap(text, width=20)
        y_text = 6
        draw = ImageDraw.Draw(img)
        for line in lines:
            width, height = self.font.getsize(line)
            draw.text((6, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += height
        if cfg is True:
            if not description['success']:
                text_msg = 'error : ' + description['message']
                lines = textwrap.wrap(text_msg, width=20)
                for line in lines:
                    width, height = self.font.getsize(line)
                    draw.text((6, y_text + 6), line, font=self.font, fill=(255, 0, 0))
                    y_text += height
        return img


    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def llm_skill_interact(self, instruction: str):
        """
        Execute actions based on instructions using the specified action names:
        OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject, Navigation
        """
        log.debug(f"Executing instruction: {instruction}")
        
        # Reset receptacle for PutObject if not relevant
        if instruction.startswith("PutObject ") or instruction.startswith("OpenObject "):
            pass
        else:
            self.cur_receptacle = None

        # Handle Navigation
        if instruction.startswith("Navigation "):
            obj_name = instruction.replace('Navigation ', '')
            self.cur_receptacle = obj_name
            ret = self.nav_obj(natural_word_to_ithor_name(obj_name), self.sliced)
        
        # Handle PickupObject
        elif instruction.startswith("PickupObject "):
            obj_name = instruction.replace('PickupObject ', '')
            ret = self.pick(natural_word_to_ithor_name(obj_name))
        
        # Handle PutObject
        elif instruction.startswith("PutObject "):
            parts = instruction.replace('PutObject ', '').split()
            
            if len(parts) > 1:
                # Format: PutObject [object] [receptacle]
                receptacle = parts[-1]
                self.cur_receptacle = receptacle
                ret = self.put(natural_word_to_ithor_name(receptacle))
            else:
                # Just drop if no receptacle specified
                if self.cur_receptacle is None:
                    ret = self.drop()
                else:
                    ret = self.put(natural_word_to_ithor_name(self.cur_receptacle))

            if len(ret) > 16:
                # if put down failed, then drop the object
                ret = self.drop()
                self.last_event.metadata['lastActionSuccess'] = False
        
        # Handle OpenObject
        elif instruction.startswith("OpenObject "):
            obj_name = instruction.replace('OpenObject ', '')
            ret = self.open(natural_word_to_ithor_name(obj_name))
        
        # Handle CloseObject
        elif instruction.startswith("CloseObject "):
            obj_name = instruction.replace('CloseObject ', '')
            ret = self.close(natural_word_to_ithor_name(obj_name))
        
        # Handle ToggleObjectOn
        elif instruction.startswith("ToggleObjectOn "):
            obj_name = instruction.replace('ToggleObjectOn ', '')
            ret = self.toggleon(natural_word_to_ithor_name(obj_name))
        
        # Handle ToggleObjectOff
        elif instruction.startswith("ToggleObjectOff "):
            obj_name = instruction.replace('ToggleObjectOff ', '')
            ret = self.toggleoff(natural_word_to_ithor_name(obj_name))
        
        # Handle SliceObject
        elif instruction.startswith("SliceObject "):
            obj_name = instruction.replace('SliceObject ', '')
            ret = self.slice(natural_word_to_ithor_name(obj_name))
            self.sliced = True
        
        # Handle drop as a fallback
        elif instruction.startswith("drop"):
            ret = self.drop()
        
        else:
            print(f"Instruction: {instruction}")
            assert False, 'instruction not supported'

        if not self.last_event.metadata['lastActionSuccess']:
            log.warning(f"llm_skill_interact failed")
            log.warning(f"errorMessage: {self.last_event.metadata['errorMessage']}")
            log.warning(f"returned msg: {ret}")
        else:
            log.debug(f"Last action succeeded")

        ret_dict = {
            'action': instruction,
            'success': len(ret) <= 0,
            'message': ret
        }

        return ret_dict

    def get_object_prop(self, name, prop, metadata):
        for obj in metadata['objects']:
            if name in obj['objectId']:
                return obj[prop]
        return None

    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))
    def nav_obj(self, target_obj: str, prefer_sliced=False):
        log.debug(f'Navigation to {target_obj} (prefer_sliced={prefer_sliced})')
        objects = self.last_event.metadata['objects']
        action_name = 'object navigation'
        ret_msg = ''
        log.info(f'{action_name} ({target_obj})')

        # get the object location
        obj_id, obj_data = self.get_obj_id_from_name(target_obj, priority_in_visibility=True, priority_sliced=prefer_sliced)

        # find object index from id
        obj_idx = -1
        for i, o in enumerate(objects):
            if o['objectId'] == obj_id:
                obj_idx = i
                break

        if obj_idx == -1:
            ret_msg = f'Cannot find {target_obj}'
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            max_attempts = 20
            teleport_success = False

            # get obj location
            loc = objects[obj_idx]['position']
            obj_rot = objects[obj_idx]['rotation']['y']

            # do not move if the object is already visible and close
            if objects[obj_idx]['visible'] and objects[obj_idx]['distance'] < 1.0:
                log.info('Object is already visible')
                max_attempts = 0
                teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (target_obj == 'Fridge' or target_obj == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position([loc['x'], loc['y'], loc['z']], reachable_pos_idx)

                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]), loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (target_obj == 'Fridge' or target_obj == 'Microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if target_obj == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0], loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport
                super().step(dict(action="TeleportFull",
                                  x=closest_loc[0], y=self.agent_height, z=closest_loc[2],
                                  rotation=rot_angle, horizon=-hor_angle))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                ret_msg = f'Cannot move to {target_obj}'

        return ret_msg

    def get_obj_id_from_name(self, obj_name, only_pickupable=False, only_toggleable=False, priority_sliced=False, get_inherited=False,
                             parent_receptacle_penalty=True, priority_in_visibility=False, exclude_obj_id=None):
        obj_id = None
        obj_data = None
        min_distance = 1e+8
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == exclude_obj_id:
                continue

            if (only_pickupable is False or obj['pickupable']) and \
                    (only_toggleable is False or obj['toggleable']) and \
                    obj['objectId'].split('|')[0].casefold() == obj_name.casefold() and \
                    (get_inherited is False or len(obj['objectId'].split('|')) == 5):
                if obj["distance"] < min_distance:
                    penalty_advantage = 0  # low priority for objects in closable receptacles such as fridge, microwave
                    if parent_receptacle_penalty and obj['parentReceptacles']:
                        for p in obj['parentReceptacles']:
                            is_open = self.get_object_prop(p, 'isOpen', self.last_event.metadata)
                            openable = self.get_object_prop(p, 'openable', self.last_event.metadata)
                            if openable is True and is_open is False:
                                penalty_advantage += 100000
                                break

                    if obj_name.casefold() == 'stoveburner':
                        # try to find an empty stove
                        if len(obj['receptacleObjectIds']) > 0:
                            penalty_advantage += 10000

                    if priority_in_visibility and obj['visible'] is False:
                        penalty_advantage += 1000

                    if priority_sliced and '_Slice' in obj['name']:
                        penalty_advantage += -100  # prefer sliced objects; this prevents picking up non-sliced objects

                    if obj["distance"] + penalty_advantage < min_distance:
                        min_distance = obj["distance"] + penalty_advantage
                        obj_data = obj
                        obj_id = obj["objectId"]

        return obj_id, obj_data

    def pick(self, obj_name):
        log.debug(f'Picking up {obj_name}')
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, only_pickupable=True, priority_in_visibility=True, priority_sliced=self.sliced)
        ret_msg = ''
        log.info(f'pick {obj_id}')

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to pick up'
        else:
            if obj_data['visible'] is False and obj_data['parentReceptacles'] is not None and len(obj_data['parentReceptacles']) > 0:
                recep_name = obj_data["parentReceptacles"][0].split('|')[0]
                ret_msg = f'{obj_name} is not visible because it is in {recep_name}'

                # try anyway
                super().step(dict(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False
                ))
            else:
                super().step(dict(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False
                ))
                
                if not self.last_event.metadata['lastActionSuccess']:
                    if len(self.last_event.metadata['inventoryObjects']) == 0:
                        ret_msg = f'Robot is not holding any object'
                    else:
                        # check if the agent is holding the object
                        holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
                        holding_obj_type = self.last_event.metadata['inventoryObjects'][0]['objectType']
                        ret_msg = f'Robot is currently holding {holding_obj_type}'

            if self.last_event.metadata['lastActionSuccess']:
                ret_msg = ''

        return ret_msg

    def put(self, receptacle_name):
        log.debug(f'Putting object on {receptacle_name}')
        # assume the agent always put the object currently holding
        ret_msg = ''

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'Robot is not holding any object'
            return ret_msg
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']

        halt = False
        last_recep_id = None
        exclude_obj_id = None
        for k in range(2):  # try closest and next closest one
            for j in range(7):  # move/look around or rotate obj
                for i in range(2):  # try inherited receptacles too (e.g., sink basin, bath basin)
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id  # previous recep id

                    if i == 0:
                        recep_id, _ = self.get_obj_id_from_name(receptacle_name, exclude_obj_id=exclude_obj_id)
                    else:
                        recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True, exclude_obj_id=exclude_obj_id)

                    if not recep_id:
                        ret_msg = f'Cannot find {receptacle_name}'
                        continue

                    log.info(f'put {holding_obj_id} on {recep_id}')

                    # look up (put action fails when a receptacle is not visible)
                    if j == 1:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                    elif j == 2:
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                    elif j == 3:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="MoveBack"))
                    elif j == 4:
                        super().step(dict(action="MoveAhead"))
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                    elif j == 5:
                        for r in range(8):
                            super().step(dict(action="MoveLeft"))
                    elif j == 6:
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                        super().step(dict(  # this somehow make putobject success in some cases
                            action="RotateHand",
                            x=40
                        ))

                    super().step(dict(
                        action="PutObject",
                        objectId=holding_obj_id,
                        receptacleObjectId=recep_id,
                        forceAction=True
                    ))
                    last_recep_id = recep_id

                    if not self.last_event.metadata['lastActionSuccess']:
                        log.warning(f"PutObject action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                        ret_msg = f'Putting the object on {receptacle_name} failed'
                    else:
                        ret_msg = ''
                        halt = True
                        break
                if halt:
                    break
            if halt:
                break

        return ret_msg

    def drop(self):
        log.info(f'drop')
        ret_msg = ''
        super().step(dict(
            action="DropHandObject",
            forceAction=True
        ))

        if not self.last_event.metadata['lastActionSuccess']:
            if len(self.last_event.metadata['inventoryObjects']) == 0:
                ret_msg = f'Robot is not holding any object'
            else:
                ret_msg = f"Drop action failed"
        else:
            ret_msg = 'put down failed'

        return ret_msg

    def open(self, obj_name):
        log.info(f'open {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to open'
        else:
            for i in range(4):
                super().step(dict(
                    action="OpenObject",
                    objectId=obj_id,
                ))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"OpenObject action failed: {self.last_event.metadata['errorMessage']}, moving backward and trying again...")
                    ret_msg = f"Open action failed"

                    # move around to avoid self-collision
                    if i == 0:
                        super().step(dict(action="MoveBack"))
                    elif i == 1:
                        super().step(dict(action="MoveBack"))
                        super().step(dict(action="MoveRight"))
                    elif i == 2:
                        super().step(dict(action="MoveLeft"))
                        super().step(dict(action="MoveLeft"))
                else:
                    ret_msg = ''
                    break

        return ret_msg

    def close(self, obj_name):
        log.info(f'close {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to close'
        else:
            super().step(dict(
                action="CloseObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Close action failed"

        return ret_msg

    def toggleon(self, obj_name):
        log.info(f'toggle on {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn on'
        else:
            super().step(dict(
                action="ToggleObjectOn",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Turn on action failed"

        return ret_msg

    def toggleoff(self, obj_name):
        log.info(f'toggle off {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn off'
        else:
            super().step(dict(
                action="ToggleObjectOff",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Turn off action failed"

        return ret_msg

    def slice(self, obj_name):
        log.info(f'slice {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to slice'
        else:
            super().step(dict(
                action="SliceObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Slice action failed"

        return ret_msg

    def get_visible_objects(self):
        log.debug("Getting visible objects")
        visible_objects = []
        
        for obj in self.last_event.metadata['objects']:
            if obj['visible']:
                # Convert from iTHOR object name to natural language
                obj_name = natural_word_to_ithor_name(obj['objectType'])
                
                # Avoid duplicates (e.g. multiple apples)
                if obj_name not in visible_objects:
                    visible_objects.append(obj_name)
        
        # Sort for consistency
        visible_objects.sort()
        
        # Return as comma-separated string
        result = ", ".join(visible_objects)
        log.debug(f"Visible objects: {result}")
        return result
