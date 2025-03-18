"""
High-Level Planner for LLM-based Task Planning
----------------------------------------------
Implements the planning system that uses large language models to generate plans for
embodied AI tasks. Features KNN-based example retrieval, prompt engineering, and
plan formatting.
"""

import pandas as pd
import random
import logging
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import GPT2Tokenizer

# Get a logger for this module
log = logging.getLogger(__name__)

ACT_TO_STR = {
    'OpenObject': "Open",
    'CloseObject': "Close",
    'PickupObject': "Pickup",
    'PutObject': "Put",
    'ToggleObjectOn': "Toggle on",
    'ToggleObjectOff': "Toggle off",
    'SliceObject': "Slice",
    'Navigation': "Navigate"
}


class LLM_Planner():
    def __init__(self, knn_data_path, emb_model_name='paraphrase-MiniLM-L6-v2', debug=False):
        self.sentence_embedder = SentenceTransformer(emb_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.knn_set = pd.read_pickle(knn_data_path)
        self.debug=debug


    def knn_retrieval(self, curr_task, k):
        # Find K train examples with closest sentence embeddings to test example
        traj_emb = self.sentence_embedder.encode(curr_task["task_instr"], show_progress_bar=False)
        topK = []
        for idxTrain, trainItem in self.knn_set.iterrows():

            train_emb = self.sentence_embedder.encode(trainItem["task_instr"], show_progress_bar=False)

            dist = -1 * cos_sim(traj_emb, train_emb)

            if len(topK) < k:
                topK.append((trainItem["task"], dist))
                topK = sorted(topK, key = lambda x : x[1])
            else:
                if float(dist) < topK[-1][1]:
                    if (trainItem["task"], dist) not in topK:
                        topK.append((trainItem["task"], dist))
                        topK = sorted(topK, key = lambda x : x[1])
                        topK = topK[:k]

        return [entry[0] for entry in topK]


    def generate_prompt(self, curr_task, k, removeNav=False, naturalFormat=False, includeLow=False, dynamic=True):
        log.debug(f"Generating prompt with {k} examples, includeLow={includeLow}, dynamic={dynamic}")
        # Define the valid objects list
        VALID_OBJECTS = [
            'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
            'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl', 'Box', 'Bread',
            'ButterKnife', 'Cabinet', 'Candle', 'Cart', 'CD', 'CellPhone', 'Chair',
            'Cloth', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'Curtains',
            'Desk', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'FloorLamp',
            'Footstool', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel',
            'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
            'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LightSwitch',
            'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan',
            'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow',
            'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe',
            'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'ShowerGlass', 'Sink',
            'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle',
            'Statue', 'StoveBurner', 'StoveKnob', 'DiningTable', 'CoffeeTable', 'SideTable',
            'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet',
            'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'Towel',
            'TowelHolder', 'TVStand', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'
        ]
        
        # Format the valid objects list for the prompt
        valid_objects_str = ", ".join(VALID_OBJECTS)
        
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions. Follow the exact output format described in the examples. Only output the next steps of the plan. Always try to navigate to the object before interacting with it."
        
        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        # Add the valid objects constraint
        prompt += f"\n\nIMPORTANT: Only use objects from this list: {valid_objects_str}"
        
        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:])
            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str
                
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str
            
            # Only include visible objects if dynamic planning is enabled
            if dynamic:
                if not in_context_obj_str == "":
                    prompt += "\nVisible objects are " + in_context_obj_str
            
            prompt += "\nNext Plans: " + next_plans_str
        
        # Add the task prompt for GPT
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]

        task_high_level_str = str(curr_task["task_instr"][0])
        
        # Fix for step_instr being a list of lists or empty
        if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list):
            # Either flatten the list
            step_instr_flat = [step for sublist in curr_task["step_instr"] for step in sublist]
            task_step_by_step_str = '. '.join(step_instr_flat)
        elif curr_task["step_instr"]:
            # Original code if it's already a flat list
            task_step_by_step_str = '. '.join(curr_task["step_instr"])
        else:
            # Handle empty step_instr
            task_step_by_step_str = ""
        
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

        prompt += "\n\nTask description: " + task_high_level_str

        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str
        
        # Only include visible objects if dynamic planning is enabled
        if dynamic:
            if not task_obj_str == "":
                prompt += "\nVisible objects are " + task_obj_str
        
        prompt += "\nNext Plans:"
            
        curr_task["Prompts"] = prompt
        
        # Only store vis_objs if dynamic planning is enabled
        if dynamic:
            curr_task["vis_objs"] = vis_objs
        
        log.debug("Prompt generation complete")
        return prompt

    # Main point of entry for LLM HLP prompt generator, use in run_eval
    def generate_gpt_prompt(self, curr_task, k, includeLow=False, dynamic=True):
        prompt = self.generate_prompt(curr_task, k, removeNav=False, naturalFormat=False, includeLow=includeLow, dynamic=dynamic)
        return prompt

    
    # Below are helper functions 

    # Change object list into object string:
    ## Example: ['Drawer', 'ButterKnife'] -> Drawer, ButterKnife
    def format_object_str(self, obj_list):
        """
        Format a list of objects or a comma-separated string into a properly formatted string.
        
        Args:
            obj_list: Either a list of object names or a comma-separated string of object names
            
        Returns:
            A properly formatted comma-separated string of object names
        """
        # If input is already a string, split it into a list
        if isinstance(obj_list, str):
            # Split by comma and strip whitespace
            obj_list = [obj.strip() for obj in obj_list.split(',')]
            # Filter out empty strings
            obj_list = [obj for obj in obj_list if obj]
        
        # If the list is empty, return an empty string
        if not obj_list:
            return ""
        
        # Join the list with commas and spaces
        obj_str = ", ".join(obj_list)
        
        return obj_str

    # Change plan list into plan string:
    ## Example: [('Navigation','Shelf'), ('PickupObject', 'knife')] -> Navigation Shelf, PickupObject Knife
    def format_plan_str(self, plan_list):
        if not plan_list:
            return ""

        # Lowercase object names in (action, plan) tuple
        formatted_plans = []
        for item in plan_list:
            if isinstance(item, tuple):
                # Handle tuples (normal case)
                item_list = list(item)
                # Make object names lowercase
                if len(item_list) > 1:
                    item_list[1] = item_list[1].lower()
                if len(item_list) > 2:
                    item_list[2] = item_list[2].lower()
                # Join action and object with a space
                formatted_plans.append(" ".join(item_list))
            else:
                # Handle string items (in case they're already formatted)
                formatted_plans.append(item)

        # Join plans with comma and space
        plan_str = ', '.join(formatted_plans)

        return plan_str