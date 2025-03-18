# End-to-End Agent

An end-to-end agent that uses a LLM-Planner.


## What's Here

- Fully end-to-end agent that uses a LLM-Planner.
- Supports dynamic re-planning, image input, and more.

## Usage

### Setup   

Please be in the `e2e/` directory for the following commands~

```
conda create -n llm-planner python=3.8 -y
conda activate llm-planner
pip install -r requirements.txt
```

### Download ALFRED Dataset

```
cd alfred/data
bash download_data.sh json
```

### Sanity Checks
#### Simulator Check

```
python check_thor.py
```
This will check if your THOR environment is setup correctly.

#### Agent Check

```
python src/run_eval.py --config config/config_alfred.yaml --dry-run
```
This will run the agent on 3 tasks and save the results to `results/`.

**NOTE**: It is recommended to dry run the agent first since it will preprocess the dataset.

### Full Evaluation

```
export OPENAI_API_KEY=<your-openai-api-key>
python src/run_eval.py --config config/config_alfred.yaml
```
This will run the agent on all tasks and save the results to `results/`.

Check out the `config/config_alfred.yaml` for more options.

## FAQs

**Q1:** AssertionError: Invalid DISPLAY 0 - cannot find X server with xdpyinfo

**A1:** Check your display number with `xdpyinfo` and set it by running `export DISPLAY=:<your-display-number>` before executing the script.

**Q2:** How does the LLM generated plans get grounded?

**A2:** The function `llm_skill_interact` in `src/alfred/thor_connector.py` is the function that grounds the LLM generated plans.

**Q3:** What system was this tested on?

**A3:** This was tested on a MacBook Pro with an M1 Pro chip and on Ubuntu 22.04.


