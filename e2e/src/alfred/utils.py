"""
This code is partially adapted from:
https://github.com/lbaa2022/LLMTaskPlanning/blob/main/src/alfred/thor_connector.py
"""

import os, json, re
import string


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_task_json(task):
    '''
    load preprocessed json from disk
    '''
    json_path = os.path.join('alfred/data/json_2.1.0', task['task'], 'pp',
                             'ann_%d.json' % task['repeat_idx'])
    with open(json_path) as f:
        data = json.load(f)
    return data


def ithor_name_to_natural_word(w):
    # e.g., RemoteController -> remote controller
    if w == 'CD':
        return w
    else:
        return re.sub(r"(\w)([A-Z])", r"\1 \2", w).lower()


def natural_word_to_ithor_name(w):
    # e.g., floor lamp -> FloorLamp
    if w == 'CD':
        return w
    else:
        return ''.join([string.capwords(x) for x in w.split()])


def find_indefinite_article(w):
    # simple rule, not always correct
    w = w.lower()
    if w[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    else:
        return 'a'
