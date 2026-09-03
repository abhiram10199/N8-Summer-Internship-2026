import collections

from omegaconf import OmegaConf
import sympy as sym


def config_factory():
    return {
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "n2"],  # Standard Koza library
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
                "min_": 12,
                "max_": 20,
                "on": True,
            },
            "repeat": {"tokens": "const", "min_": None, "max_": 5, "on": True},
            "inverse": {"on": True},
            "trig": {"on": False},
            "const": {"on": False},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 15, "scale": 5, "on": True},
        },
    }


def dynamics():
    """6-D DC Microgrid with 2 Constant Power Loads (CPLs) in Normalized Energy Coordinates.

    Derived from Zitian Qiu/DC microgrid/Copy_of_DC_grid_2CPLs_simplified.m.
    Physical Circuit Parameters:
        R = [0.4, 0.8, 0.42] Ohm
        L = [17.3e-3, 40.0e-3, 19.6e-3] H
        C = [1.05e-3, 1.0e-3, 1.05e-3] F
        CPL power: p1 = 900.0 W, p2 = 600.0 W at buses 2 and 3
        Equilibrium voltages: v_e2 = 193.18257410 V, v_e3 = 195.62141583 V

    Coordinate Normalization (Energy State Space):
        To allow symbolic discovery without arbitrary float weights (since const=False),
        we normalize states by their physical storage factors:
            x1 = sqrt(L1) * Delta i_L1
            x2 = sqrt(L2) * Delta i_L2
            x3 = sqrt(L3) * Delta i_L3
            x4 = sqrt(C1) * Delta v_C1
            x5 = sqrt(C2) * Delta v_C2
            x6 = sqrt(C3) * Delta v_C3
        In these coordinates, the reactive LC power exchange terms (+x1*x4 and -x1*x4)
        are skew-symmetric and cancel out identically, making standard polynomial
        and quadratic candidates V = sum(xi^2) physically consistent Lyapunov candidates.
    """
    x1, x2, x3, x4, x5, x6 = sym.symbols("x1, x2, x3, x4, x5, x6")
    state_variables = [x1, x2, x3, x4, x5, x6]

    L = [17.3e-3, 40.0e-3, 19.6e-3]
    C = [1.05e-3, 1.0e-3, 1.05e-3]
    R = [0.4, 0.8, 0.42]
    v_e2 = 193.18257410
    v_e3 = 195.62141583

    inv_sqrt_L1C1 = 1.0 / sym.sqrt(L[0] * C[0])
    inv_sqrt_L2C1 = 1.0 / sym.sqrt(L[1] * C[0])
    inv_sqrt_L2C2 = 1.0 / sym.sqrt(L[1] * C[1])
    inv_sqrt_L3C1 = 1.0 / sym.sqrt(L[2] * C[0])
    inv_sqrt_L3C3 = 1.0 / sym.sqrt(L[2] * C[2])

    inv_sqrt_C2 = 1.0 / sym.sqrt(C[1])
    inv_sqrt_C3 = 1.0 / sym.sqrt(C[2])

    # Normalized ODEs: dot(x) = f(x) with f(0) = 0
    dynamics_ode = [
        (-R[0] / L[0]) * x1 - inv_sqrt_L1C1 * x4,
        (-R[1] / L[1]) * x2 + inv_sqrt_L2C1 * x4 - inv_sqrt_L2C2 * x5,
        (-R[2] / L[2]) * x3 + inv_sqrt_L3C1 * x4 - inv_sqrt_L3C3 * x6,
        inv_sqrt_L1C1 * x1 - inv_sqrt_L2C1 * x2 - inv_sqrt_L3C1 * x3,
        inv_sqrt_L2C2 * x2 - inv_sqrt_C2 * (900.0 / (v_e2 + x5 * (1.0 / inv_sqrt_C2)) - 900.0 / v_e2),
        inv_sqrt_L3C3 * x3 - inv_sqrt_C3 * (600.0 / (v_e3 + x6 * (1.0 / inv_sqrt_C3)) - 600.0 / v_e3),
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
