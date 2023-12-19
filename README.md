# LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models 

Code for [LLM-Planner](https://arxiv.org/abs/2212.04088).

Check [project website](https://dki-lab.github.io/LLM-Planner/) for an overview and a demo.

## News:
- Dec 23: LLM-Planner with support for an oracle low-level planner with a new easy-setup framework with ALFOWLRD backbone. We hope this codebase can serve as a foundation for building LLM or LMM based methods with ALFRED.

## Release process:
- [x] High level planner
  - [x] KNN dataset
  - [x] KNN retriever
- [ ] Low level planner
  - [x] Oracle low level planner
  - [ ] HLSM low-evel planner
- [ ] Fine-grained control over visualization
- [ ] Support for non-OpenAI foundation models

## Quickstart

Clone repo:

```
git clone https://github.com/OSU-NLP-Group/LLM-Planner
cd LLM-Planner
export ALFWORLD_DATA="$(pwd)/alfworld/data"
```

Install requirements: 

```
# Conda or Python enviornment recommended 
# Install requirements for the AI2Thor simulator and ALFRED
cd alfworld
pip install .
# Install requirements for LLM-Planner
cd ../src
pip install -r requirements.txt
```

Download data:

```
cd ../alfworld
alfworld-download
```

Sanity check on AI2Thor simulator
```
python scripts/check_thor.py
# This should return successful, if not your AI2Thor simulator is not set up correctly.
```

Start evaluation with GPT-4

```
export OPENAI_KEY=<Your OpenAI Key>
cd ../src
python run_eval.py --config gpt4_base_config.yaml
```


## Common Questions and Answers

Coming soon.



<!-- Check `QA.md` for a complete list of questions and answers. -->

## Hardware

Tested on:
- Mac M1
- Ubuntu 18.04

## Citation Information

If you find this code useful, please consider citing our paper:

```
@InProceedings{song2023llmplanner,
  author    = {Song, Chan Hee and Wu, Jiaman and Washington, Clayton and Sadler, Brian M. and Chao, Wei-Lun and Su, Yu},
  title     = {LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
}
```
