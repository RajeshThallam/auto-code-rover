# Run `auto-code-rover` with Vertex Gemini APIs


## STEP 1. Build Docker image
```
# git clone https://github.com/RajeshThallam/auto-code-rover.git

docker build -f Dockerfile -t acr .
```

## STEP 2. Shell into Docker image
```
mkdir -p ~/runs
docker run -it -p 3000:3000 -p 5000:5000 --volume ~/runs:/opt/scratch --volume ${PWD}:/opt/auto-code-rover acr
```

## STEP 3. Set up SWE-bench

In the docker container, first set up the tasks to run in SWE-bench (e.g., `astropy__astropy-12907`). The list of all tasks can be found in [`conf/swe_lite_tasks.txt`](conf/swe_lite_tasks.txt).

The tasks need to be put in a file, one per line:

```
cd /opt/SWE-bench
echo astropy__astropy-12907 > /opt/auto-code-rover/conf/autodev_exp_tasks.txt
```

Then, set up these tasks by running:

```
cd /opt/SWE-bench
conda activate swe-bench
python harness/run_setup.py --log_dir /opt/scratch/logs --testbed /opt/scratch/testbed --result_dir /opt/scratch/setup_result --subset_file /opt/auto-code-rover/conf/autodev_exp_tasks.txt
```

Once the setup for this task is completed, the following two lines will be printed:

```
setup_map is saved to setup_result/setup_map.json
tasks_map is saved to setup_result/tasks_map.json
```

The `testbed` directory will now contain the cloned source code of the target project. A conda environment will also be created for this task instance.

_If you want to set up multiple tasks together, put their ids in `tasks.txt` and follow the same steps._

## STEP 4. Run in SWE-bench

### Run a single task in SWE-bench

Before running the task (`astropy__astropy-12907` here), make sure it has been set up as mentioned in STEP 3.

```
cd /opt/auto-code-rover
conda deactivate
conda activate auto-code-rover
export PROJECT_ID="[your-project]"
PYTHONPATH=. python app/main.py swe-bench --model "vertex_ai/gemini-pro-experimental" --model-temperature=0.2 --setup-map /opt/scratch/setup_result/setup_map.json --tasks-map /opt/scratch/setup_result/tasks_map.json --output-dir /opt/scratch/output --task astropy__astropy-12907 --enable-validation
```

The output of the run can then be found in `output/`. For example, the patch generated for `astropy__astropy-12907` can be found at a location like this: `output/applicable_patch/astropy__astropy-12907_yyyy-MM-dd_HH-mm-ss/extracted_patch_1.diff` (the date-time field in the directory name will be different depending on when the experiment was run).

### Run multiple tasks in SWE-bench

First, put the id's of all tasks to run in a file, one per line. Suppose this file is `autodev_exp_tasks.txt` in `/opt/auto-code-rover/conf/`, the tasks can be run with

```
cd /opt/auto-code-rover
conda activate auto-code-rover
PYTHONPATH=. python app/main.py swe-bench --model "vertex_ai/gemini-pro-experimental" --model-temperature=0.2 --setup-map /opt/scratch/setup_result/setup_map.json --tasks-map /opt/scratch/setup_result/tasks_map.json --output-dir /opt/scratch/output --task-list-file /opt/auto-code-rover/conf/autodev_exp_tasks.txt
```

**NOTE**: make sure that the tasks in `autodev_exp_tasks.txt` have all been set up in SWE-bench. See STEP 3.

### Run in SWE-bench using a config file

Alternatively, a config file can be used to specify all parameters and tasks to run. See `conf/autodev-exp.conf` for an example.
Also see [EXPERIMENT.md](EXPERIMENT.md) for the details of the items in a conf file.
A config file can be used by:

```
cd /opt/auto-code-rover
conda activate auto-code-rover
python scripts/run.py conf/autodev-exp.conf
```