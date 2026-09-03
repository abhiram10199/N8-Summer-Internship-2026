import collections

from omegaconf import OmegaConf
import sympy as sym


def config_factory():
    return {
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "sin", "cos", "n2"],
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
                "min_": 10,
                "max_": 30,
                "on": True,
            },
            "repeat": {"tokens": "const", "min_": None, "max_": 5, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": False},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 20, "scale": 5, "on": True},
        },
    }


def dynamics():
    """49-D Full IEEE 39-bus system with 10 generators (2nd order) and 29 dynamic loads (1st order).

    Derived from Zitian Qiu/39 bus system/IEEE_39bus_system.m and Function_CCT_Succesive_TwoStage.m.
    States:
        x1 ... x29:  Load bus angle deviations Delta delta_1 ... Delta delta_29 [rad]
        x30 ... x39: Generator bus angle deviations Delta delta_30 ... Delta delta_39 [rad]
        x40 ... x49: Generator angular speed deviations omega_1 ... omega_10 [rad/s]
    """
    x_symbols = sym.symbols(", ".join([f"x{i+1}" for i in range(49)]))
    state_variables = list(x_symbols)

    # Bus equilibrium angles [rad] from power flow
    Bus_delta = [
        -0.16458406, -0.12025318, -0.16982054, -0.18378317, -0.16371139,
        -0.15149369, -0.1891937,  -0.19792034, -0.19460421, -0.11013028,
        -0.12426744, -0.12461652, -0.12252202, -0.1511446,  -0.15812599,
        -0.13369075, -0.15097007, -0.16563126, -0.05305801, -0.0776672,
        -0.09180436, -0.0143117,  -0.01780236, -0.13159635, -0.09616774,
        -0.11815878, -0.15323901, -0.05707227, -0.00890118, -0.07801622,
         0.0,         0.02844887,  0.03804818,  0.01291544,  0.07225663,
         0.11920599,  0.02199115,  0.11431907, -0.19128809
    ]

    # Generator inertia constants Hg
    Hg = [42.0, 30.3, 35.8, 28.6, 26.0, 34.8, 26.4, 24.3, 34.5, 500.0]
    dl = 1.0 / min(Hg)

    # Line connections: (from_bus, to_bus, bi, theta0) [0-indexed buses]
    lines = [
        (0, 1, 24.331580, 0.0),
        (0, 38, 40.0, 0.0),
        (1, 2, 65.419082, 0.0),
        (1, 24, 114.739778, 0.0),
        (1, 29, 52.338780, 0.0),
        (2, 3, 44.423984, 0.0),
        (2, 17, 73.805562, 0.0),
        (3, 4, 73.889397, 0.0),
        (3, 13, 73.916843, 0.0),
        (4, 7, 85.034873, 0.0),
        (5, 4, 389.030582, 0.0),
        (5, 6, 107.561578, 0.0),
        (5, 10, 121.282914, 0.0),
        (6, 7, 215.340243, 0.0),
        (7, 8, 27.271167, 0.0),
        (8, 38, 41.280000, 0.0),
        (9, 10, 235.158586, 0.0),
        (9, 12, 235.312941, 0.0),
        (9, 31, 46.857317, 0.0),
        (11, 10, 22.868772, 0.0),
        (11, 12, 22.803875, 0.0),
        (12, 13, 98.665798, 0.0),
        (13, 14, 46.591039, 0.0),
        (14, 15, 105.772545, 0.0),
        (15, 16, 111.458633, 0.0),
        (15, 18, 50.871891, 0.0),
        (15, 20, 74.341142, 0.0),
        (15, 23, 169.570144, 0.0),
        (16, 17, 122.955404, 0.0),
        (16, 26, 56.666986, 0.0),
        (18, 32, 65.345866, 0.0),
        (18, 19, 65.864756, 0.0),
        (19, 33, 49.336444, 0.0),
        (20, 21, 71.979069, 0.0),
        (21, 22, 103.743788, 0.0),
        (21, 34, 69.953503, 0.0),
        (22, 23, 27.973305, 0.0),
        (22, 35, 38.675000, 0.0),
        (24, 25, 30.686523, 0.0),
        (24, 36, 42.115403, 0.0),
        (25, 26, 68.324204, 0.0),
        (25, 27, 21.050633, 0.0),
        (25, 28, 16.326269, 0.0),
        (27, 28, 66.861730, 0.0),
        (28, 37, 62.484252, 0.0),
        (30, 5, 43.834167, 0.0)
    ]

    dynamics_ode = []

    # 1-29: Load bus angle deviations
    for i in range(29):
        line_flow_sum = 0
        for f, t, b_k, theta_k in lines:
            delta_f_sym = state_variables[f]
            delta_t_sym = state_variables[t]
            delta_f_star = Bus_delta[f]
            delta_t_star = Bus_delta[t]
            if f == i:
                line_flow_sum += b_k * (sym.sin(delta_f_sym - delta_t_sym + delta_f_star - delta_t_star + theta_k) - sym.sin(delta_f_star - delta_t_star + theta_k))
            elif t == i:
                line_flow_sum -= b_k * (sym.sin(delta_f_sym - delta_t_sym + delta_f_star - delta_t_star + theta_k) - sym.sin(delta_f_star - delta_t_star + theta_k))
        dynamics_ode.append(-dl * line_flow_sum)

    # 30-39: Generator bus angle deviations
    for i in range(10):
        dynamics_ode.append(state_variables[39 + i])

    # 40-49: Generator speed deviations
    for i in range(10):
        bus_idx = 29 + i
        M_i = 2.0 * Hg[i]
        D_i = 0.5 * M_i
        gen_speed_var = state_variables[39 + i]
        line_flow_sum = 0
        for f, t, b_k, theta_k in lines:
            delta_f_sym = state_variables[f]
            delta_t_sym = state_variables[t]
            delta_f_star = Bus_delta[f]
            delta_t_star = Bus_delta[t]
            if f == bus_idx:
                line_flow_sum += b_k * (sym.sin(delta_f_sym - delta_t_sym + delta_f_star - delta_t_star + theta_k) - sym.sin(delta_f_star - delta_t_star + theta_k))
            elif t == bus_idx:
                line_flow_sum -= b_k * (sym.sin(delta_f_sym - delta_t_sym + delta_f_star - delta_t_star + theta_k) - sym.sin(delta_f_star - delta_t_star + theta_k))
        accel = -(D_i / M_i) * gen_speed_var - (1.0 / M_i) * line_flow_sum
        dynamics_ode.append(accel)

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
                "dim_input": 49,
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
