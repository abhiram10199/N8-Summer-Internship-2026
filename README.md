## To run on Barkla2:
Code should be inside `/mnt/scratch/groups/powersys2/ALFD`
1. Set the dimension in Line 4 of the config.py file.
```bash
cd /mnt/scratch/groups/powersys2/ALFD
nano config.py --line
# Set SELECT_DIMENSION to the desired dimension.
```
2. Run the following command:
```bash
sbatch run_experiment.sh
```

- Output files are located in outputs/ folder
- Error files are located in errors/ folder
- Logs file are located in logs/ folder

## To run the code for a specific dimension, follow the following steps:

1. Set the dimension in the config file.
2. Run the following command:
```bash
python3 Lyapunov_test_dso.py
```


## Changes Made By Abhi

1. ### Typo in utils/train.py Line 723 : Variable `p_collection_reward` -> `gp_collection_reward`

2. ### Deprecated commands fixed using uv:
Repo uses python version 3.9. To set up a local virtual environment, I used `uv` & `pip` package manager.
Run the following one after another in the terminal. 
(You can use PIP only as well if you prefer, just remove all `uv` commands)

```
uv venv --python 3.9
uv pip install "setuptools<58.0.0" wheel cython "numpy<2.0.0"
pip install -r requirements.txt --no-build-isolation
cd libs/sd3/dso
rm -rf .eggs/ build/ dso.egg-info/
python setup.py build_ext --inplace
cd ../../..
export PYTHONPATH=$(pwd):$(pwd)/libs/sd3/dso
```


3. ### CUDA Setup:
The author originally hardcoded the code to run on CPU or 2 GPUs. To make it more flexible, I changed every presence of the phrase 'CUDA 1' to 'DEVICE' to match the users' setup. 
```python
utils/train.py Line 1339-1341
``` 
Example:
```python
utils/train.py Line 1339
sympy_torch = sympytorch.SymPyModule(expressions=[sympy]).to('cuda:1')
# Replace with
sympy_torch = sympytorch.SymPyModule(expressions=[sympy]).to(DEVICE)
```
4. ### 9d does not exist. There is no 9D, it's a copy of 8D!


---
# Analytical-Lyapunov-Function-Discovery
Offical Pytorch implementation for the paper **"Analytical Lyapunov Function Discovery: An RL-based Generative Approach"**, presented at ICML 2025. We introduce the first RL-based framework for directly discovering analytical Lyapunov functions for nonlinear dynamical systems, bypassing the need for supervised learning with large-scale datasets. Our framework succeeds on various non-polynomial dynamics, like the simple pendulum, quadrotor, and power system frequency control, and notably scales to a 10-D system and discovers a valid local Lyapunov function for power system frequency control with lossy transmission lines, which is previously unknown in the literature. For details, see [**Analytical Lyapunov Function Discovery: An RL-based Generative Approach**](https://arxiv.org/abs/2502.02014).


## Installation
Clone this repository and install the required Conda environment and dependencies by running:

```bash
run install.sh
```

## Input Test Dynamics and Training Parameters

Before training, you need to specify the test dynamics and parameters defining the desirable output expressions. Follow these steps:

### Define the Symbolic ODEs

In [config.py](config.py), edit the function ``dynamics()`` to define your system's ODEs. Specify the state-space variables and return the dynamics in symbolic form.

### Specify the State Space for Local Stability Analysis


Modify the state space domain $\mathcal{D}$ and the size of training set $\mathcal{X}$ in [libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv](libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv). Assign the corresponding entry name (e.g., fn_d_all_x) to the variable ``conf.exp.benchmark`` in
[Lyapunov_test_dso.py](Lyapunov_test_dso.py) at Line 67.


### Set Training Parameters


In [config.py](config.py), update the ``config_factory()`` function to customize training hyperparameters, such as:

* symbolic library $\mathcal{L}$,

* length of sampled candidates (max \& min),

* learning rates of training,

* hyperparameter $\epsilon$ of risk-seeking policy gradient and so on.

**Note**: In folder example_config, you can find a few examples on how to define system dynamics and relevant training parameters.

## Training
Once your dynamics and training domain are configured, start training by running:
```bash
python Lyapunov_test_dso.py
```

Intermediate training statistics and final training result (if training converges) can be found in the log directory ``./log/{$RUN}``.

To configure and test the experiments with different initialization, please modify at the top of [Lyapunov_test_dso.py](Lyapunov_test_dso.py)
```python
conf.exp.seed_start = 13
```


## Citation
If you found this work useful or interesting for your own research, we would appreciate if you could cite our work:
```
@article{zou2025analytical,
  title={Analytical Lyapunov Function Discovery: An RL-based Generative Approach},
  author={Zou, Haohan and Feng, Jie and Zhao, Hao and Shi, Yuanyuan},
  journal={arXiv preprint arXiv:2502.02014},
  year={2025}
}
```

Feel free to leave any questions in the issues of Github or email the author at hazou@ucsd.edu.
