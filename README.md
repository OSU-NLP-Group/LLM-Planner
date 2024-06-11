# LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models 

Code for [LLM-Planner](https://arxiv.org/abs/2212.04088).

Check [project website](https://dki-lab.github.io/LLM-Planner/) for an overview and a demo.

## News:
- Jun 24: Due to an object grounding error in the simulator we are using, we revert the code into high-level plan generation only before fix is implemented.

## Quickstart
`
python hlp_planner.py
`

This commands uses the KNN dataset to generate a high-level plan for an example task.
Check out the code for more details.

<!-- Check `QA.md` for a complete list of questions and answers. -->

## Hardware

Tested on:
- Mac M1
- Ubuntu 18.04

## Citation Information

```
@InProceedings{song2023llmplanner,
  author    = {Song, Chan Hee and Wu, Jiaman and Washington, Clayton and Sadler, Brian M. and Chao, Wei-Lun and Su, Yu},
  title     = {LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
}
```

## Acknowledgements

We thank the authors of [ALFWORLD](https://github.com/alfworld/alfworld/tree/master) for releasing their code.

## License

- LLM-Planner - MIT License
- ALFWorld - MIT License

## Contact

Questions or issues? File an issue or contact [Luke Song](https://chanh.ee)

