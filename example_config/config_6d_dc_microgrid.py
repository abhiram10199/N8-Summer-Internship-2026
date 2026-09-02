import collections

from omegaconf import OmegaConf
import sympy as sym


def config_factory():
    return {
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "sin", "cos", "n2"],  # Standard Koza library
        },
        "training": {
            "n_samples": 2000000,
            "batch_size": 500,
            "epsilon": 0.1,
            "n_cores_batch": 1,
        },
        "controller": {
            "learning_rate": 0.0003,
            "entropy_weight": 0.005,
            "entropy_gamma": 0.8,
        },
        "prior": {
            "length": {
                "min_": 6,
                "max_": 15,
                "on": True,
            },
            "repeat": {"tokens": "const", "min_": None, "max_": 5, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": False},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
        },
    }


def dynamics():
    """6-D DC Microgrid with 2 Constant Power Loads (CPLs).

    Derived from Zitian Qiu/DC microgrid/Copy_of_DC_grid_2CPLs_simplified.m.
    States:
        x1, x2, x3: Inductor current deviations Delta i_L1, Delta i_L2, Delta i_L3 [A]
        x4, x5, x6: Capacitor voltage deviations Delta v_C1, Delta v_C2, Delta v_C3 [V]
    Equilibrium (SEP):
        i_e = [7.72595415, 4.65880530, 3.06714885] A
        v_e = [196.90961834, 193.18257410, 195.62141583] V
    """
    x1, x2, x3, x4, x5, x6 = sym.symbols("x1, x2, x3, x4, x5, x6")
    state_variables = [x1, x2, x3, x4, x5, x6]

    # Circuit Parameters
    # R = [0.4, 0.8, 0.42] Ohm
    # L = [17.3e-3, 40.0e-3, 19.6e-3] H
    # C = [1.05e-3, 1.0e-3, 1.05e-3] F
    # Constant Power Loads: p = [900.0, 600.0] W at buses 2 and 3
    # Equilibrium voltages for CPL terms:
    v_e2 = 193.18257410
    v_e3 = 195.62141583

    # Shifted ODEs in deviation coordinates: dot(x) = f(x) with f(0) = 0
    # dot(Delta i1) = (-R1*Delta i1 - Delta v1) / L1
    # dot(Delta i2) = (-R2*Delta i2 + Delta v1 - Delta v2) / L2
    # dot(Delta i3) = (-R3*Delta i3 + Delta v1 - Delta v3) / L3
    # dot(Delta v1) = (Delta i1 - Delta i2 - Delta i3) / C1
    # dot(Delta v2) = (Delta i2 - p1/(v_e2 + Delta v2) + p1/v_e2) / C2
    # dot(Delta v3) = (Delta i3 - p2/(v_e3 + Delta v3) + p2/v_e3) / C3

    dynamics_ode = [
        (-0.4 / 17.3e-3) * x1 - (1.0 / 17.3e-3) * x4,
        (-0.8 / 40.0e-3) * x2 + (1.0 / 40.0e-3) * (x4 - x5),
        (-0.42 / 19.6e-3) * x3 + (1.0 / 19.6e-3) * (x4 - x6),
        (1.0 / 1.05e-3) * (x1 - x2 - x3),
        (1.0 / 1.0e-3) * (x2 - 900.0 / (v_e2 + x5) + 900.0 / v_e2),
        (1.0 / 1.05e-3) * (x3 - 600.0 / (v_e3 + x6) + 600.0 / v_e3),
    ]

    return state_variables, dynamics_ode


def train_config_factory():
    return OmegaConf.create(
        {
            "architecture": {
                "sinuisodal_embeddings": False,
                "dec_pf_dim": 32,
                "dec_layers": 1,
                "dim_hidden": 32,
                "lr": 0.0001,
                "dropout": 0,
                "num_features": 2,
                "ln": True,
                "N_p": 0,
                "num_inds": 50,
                "activation": "relu",
                "bit16": True,
                "norm": True,
                "linear": False,
                "input_normalization": False,
                "src_pad_idx": 0,
                "trg_pad_idx": 0,
                "length_eq": 20,
                "n_l_enc": 5,
                "mean": 0.5,
                "std": 0.5,
                "dim_input": 6,
                "num_heads": 2,
                "output_dim": 10,
            },
        }
    )


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_config(skip_cli=True):
    base_conf = OmegaConf.load("config.yaml")
    if skip_cli:
        return base_conf
    flat_base_conf = flatten(base_conf)
    cli_conf = OmegaConf.from_cli()
    cli_conf = OmegaConf.create(
        {(k[2:] if k[:2] == "--" else k): v for k, v in cli_conf.items()}
    )
    flat_cli_conf = flatten(cli_conf)

    list_cond = [k in flat_base_conf for k in flat_cli_conf.keys()]
    contains_all_keys_bool = all(list_cond)
    assert contains_all_keys_bool, (
        f"Input CLI keys that cannot be set {set(flat_cli_conf) - set(flat_base_conf)}"
    )
    conf = OmegaConf.merge(base_conf, cli_conf)
    return conf
