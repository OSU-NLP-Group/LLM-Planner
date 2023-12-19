# ALL in one script to run ALFRED with ALFWORLD backbone

import os
import base64
import sys
import json
import yaml
import cv2
import argparse
from openai import OpenAI

import alfworld.agents.environment
from hlp_planner import LLM_HLP_Generator

CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def llm(prompt, engine, images=None, stop=["\n"]):
    
    if engine == 'gpt-4':
        response = CLIENT.chat.completions.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can plan household tasks."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response.choices[0].message.content
    
    elif engine == 'gpt-4v':

        response = CLIENT.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    else:
        response = CLIENT.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response.choices[0].text


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def encode_image(numpy_img):
    _, JPEG = cv2.imencode('.jpeg', numpy_img)
    return base64.b64encode(JPEG).decode('utf-8')


def eval_single_task(prompt, env, hlp_generator, curr_task, engine, dynamic=False, to_print=True, vision=False,  ob=''):

    # Get initial frames if GPT-4V is used
    if engine == "gpt-4v":
        vision == True

    encoded_frames = []
    if vision:
        init_frames = env.get_frames()
        encoded_frames = [encode_image(frame) for frame in init_frames]

    completed_plans = []
    seen_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
    
    # Get initial high-level plan
    llm_out = llm(prompt, engine=engine, images=encoded_frames, stop=['\n'])
    high_level_plans = llm_out.split(',')

    # Run until high-level plans are exhausted
    while high_level_plans:
        plan = high_level_plans.pop(0).strip()
        observation, reward, done, info = env.step([plan])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

        if done:
            return reward
        
        # High level plan has an error
        if observation == "Nothing happens.":
            # Dynamic re-planning
            if dynamic:
                curr_vis_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
                seen_objs += curr_vis_objs
                curr_task = {
                        "task_instr": high_instrs[0],
                        "step_instr": step_instrs[0],
                        "vis_objs": init_vis_objs, 
                        "completed_plans": []
                    }
                curr_task["vis_objs"] = curr_vis_objs
                curr_task["completed_plans"] = completed_plans
                new_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=9)

                if vision:
                    curr_frames = env.get_frames()
                    encoded_frames = [encode_image(frame) for frame in curr_frames]

                # Generate new plans if dynamic
                high_level_plans = llm(new_prompt, images=encoded_frames, stop=['\n']).split(',')
        else:
            completed_plans.append(plan)
        
        if to_print:
            print(f'Act {i}: {plan}\nObs {i}: {observation}')
            sys.stdout.flush()
            
    return 0


if __name__ == '__main__':

    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()

    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution"

    # Start simulator and set up environment
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    num_games = len(env.json_file_list)

    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    cnts = [0] * 6
    rs = [0] * 6

    hlp_generator = LLM_HLP_Generator(knn_data_path=config["llm_planner"]["knn_dataset_path"], emb_model_name=config["llm_planner"]["emb_model_name"], debug=config["llm_planner"]["debug"])

    # Main eval loop
    for _ in range(num_games):
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-2:-1])

        with open(os.path.join(info['extra.gamefile'][0], "traj_data.json"), "r") as f:
            traj_data = json.load(f)

        # Retrieve all instructions for the current enviornment setup
        high_instrs = [ann["task_desc"] for ann in traj_data["turk_annotations"]["anns"]]
        step_instrs = [ann["high_descs"] for ann in traj_data["turk_annotations"]["anns"]]

        init_vis_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]

        # If there are multiple annotations for the same environment setup, evaluate on all of them
        for i, high_instr in enumerate(high_instrs):
            curr_task = {
                        "task_instr": [high_instr],
                        "step_instr": step_instrs[i],
                        "vis_objs": init_vis_objs, 
                        "completed_plans": []
                    }
            
            init_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=config["llm_planner"]["num_in_context_examples"])

            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    
                    r = eval_single_task(init_prompt, env, hlp_generator, curr_task, config["llm_planner"]["engine"], ob=ob)
                    rs[i] += r
                    cnts[i] += 1
                    break
            print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
            print('------------\n')
