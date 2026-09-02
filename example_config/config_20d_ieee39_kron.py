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
                "max_": 25,
                "on": True,
            },
            "repeat": {"tokens": "const", "min_": None, "max_": 5, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": False},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 15, "scale": 5, "on": True},
        },
    }


def dynamics():
    """20-D Kron-reduced IEEE 39-bus / New England 10-generator system.

    Derived from Zitian Qiu/39 bus system - Kron reduction/main_IEEE39_EN20.m.
    States:
        x1 ... x10:  Rotor angle deviations Delta delta_1 ... Delta delta_10 [rad] (where Delta delta_i = delta_i - delta_eq_i)
        x11 ... x20: Angular frequency deviations omega_1 ... omega_10 [rad/s]
    """
    x_symbols = sym.symbols(", ".join([f"x{i+1}" for i in range(20)]))
    state_variables = list(x_symbols)

    delta_vars = state_variables[:10]
    omega_vars = state_variables[10:]

    omega_R = 2.0 * 3.141592653589793 * 60.0

    H = [42.0, 30.3, 35.8, 28.6, 26.0, 34.8, 26.4, 24.3, 34.5, 500.0]
    d = [0.0267, 0.0161, 0.0209, 0.0243, 0.0014, 0.0277, 0.0140, 0.0116, 0.0002, 0.3979]
    A_const = [
        1.2922104567, 6.2053650830, 5.7785570690, 4.9010913642, 4.4328575390,
        5.5031460833, 4.7988776652, 4.5059193236, 6.4623859140, -1.7747490940
    ]

    K = [
        [0.0,          1.1387164000, 1.3606040523, 1.2791332133, 0.5725319953, 1.2913871707, 1.0516770292, 3.1750701870, 1.7979613231, 5.1949755422],
        [1.1387164000, 0.0,          2.2804016761, 0.7847815847, 0.3512633101, 0.7922997071, 0.6452312839, 0.6032208818, 0.5339053142, 3.1851373576],
        [1.3606040523, 2.2804016761, 0.0,          1.0379374403, 0.4645742816, 1.0478807683, 0.8533708235, 0.7284145884, 0.6661286587, 3.2071328686],
        [1.2791332133, 0.7847815847, 1.0379374403, 0.0,          2.7465184964, 2.0251245471, 1.6492164518, 0.7610195438, 0.9034713379, 1.6209805845],
        [0.5725319953, 0.3512633101, 0.4645742816, 2.7465184964, 0.0,          0.9064330326, 0.7381789293, 0.3406275697, 0.4043880984, 0.7255407323],
        [1.2913871707, 0.7922997071, 1.0478807683, 2.0251245471, 0.9064330326, 0.0,          4.1242393036, 0.7683100284, 0.9121264952, 1.6365094025],
        [1.0516770292, 0.6452312839, 0.8533708235, 1.6492164518, 0.7381789293, 4.1242393036, 0.0,          0.6256946224, 0.7428155587, 1.3327369094],
        [3.1750701870, 0.6032208818, 0.7284145884, 0.7610195438, 0.3406275697, 0.7683100284, 0.6256946224, 0.0,          1.4515272868, 2.5534966067],
        [1.7979613231, 0.5339053142, 0.6661286587, 0.9034713379, 0.4043880984, 0.9121264952, 0.7428155587, 1.4515272868, 0.0,          1.6531544900],
        [5.1949755422, 3.1851373576, 3.2071328686, 1.6209805845, 0.7255407323, 1.6365094025, 1.3327369094, 2.5534966067, 1.6531544900, 0.0]
    ]

    gamma = [
        [0.0,          -0.4249931491, -0.4075416737, -0.4550597204, -0.5427884708, -0.4474450943, -0.4486560321, -0.0968012888, -0.3126340711, -0.2601641059],
        [-0.4249931491, 0.0,          -0.2515597806, -0.5070038969, -0.5947326473, -0.4993892708, -0.5006002086, -0.3969204981, -0.5701273029, -0.3205739609],
        [-0.4075416737, -0.2515597806, 0.0,          -0.4632916359, -0.5510203863, -0.4556770098, -0.4568879476, -0.3834666381, -0.5463551270, -0.3338989568],
        [-0.4550597204, -0.5070038969, -0.4632916359, 0.0,          -0.2802950779, -0.3680153597, -0.3692262975, -0.4661925337, -0.5439061662, -0.5125445312],
        [-0.5427884708, -0.5947326473, -0.5510203863, -0.2802950779, 0.0,          -0.4557441101, -0.4569550479, -0.5539212841, -0.6316349166, -0.6002732816],
        [-0.4474450943, -0.4993892708, -0.4556770098, -0.3680153597, -0.4557441101, 0.0,          -0.1869338036, -0.4585779076, -0.5362915401, -0.5049299051],
        [-0.4486560321, -0.5006002086, -0.4568879476, -0.3692262975, -0.4569550479, -0.1869338036, 0.0,          -0.4597888454, -0.5375024779, -0.5061408429],
        [-0.0968012888, -0.3969204981, -0.3834666381, -0.4661925337, -0.5539212841, -0.4585779076, -0.4597888454, 0.0,          -0.4234409820, -0.1920804108],
        [-0.3126340711, -0.5701273029, -0.5463551270, -0.5439061662, -0.6316349166, -0.5362915401, -0.5375024779, -0.4234409820, 0.0,          -0.4247828402],
        [-0.2601641059, -0.3205739609, -0.3338989568, -0.5125445312, -0.6002732816, -0.5049299051, -0.5061408429, -0.1920804108, -0.4247828402, 0.0]
    ]

    delta_eq = [
        -0.0614913039,
         0.3995991222,
         0.3064793949,
         0.2552451612,
         0.4656559771,
         0.2941195896,
         0.3064097491,
         0.2562535902,
         0.4856044847,
        -0.1974415869
    ]

    dynamics_ode = []

    # 1-10: Rotor angle rates = speed deviations
    for i in range(10):
        dynamics_ode.append(omega_vars[i])

    # 11-20: Speed acceleration equations:
    # dot(omega_i) = (1/M_i) * ( A_i - d_i * omega_i - sum_{j != i} K_ij * sin(Delta delta_i - Delta delta_j + delta_eq_i - delta_eq_j - gamma_ij) )
    for i in range(10):
        M_i = 2.0 * H[i] / omega_R
        coupling_terms = 0
        for j in range(10):
            if i != j:
                angle_offset = delta_eq[i] - delta_eq[j] - gamma[i][j]
                coupling_terms += K[i][j] * sym.sin(delta_vars[i] - delta_vars[j] + angle_offset)
        
        accel_i = (1.0 / M_i) * (A_const[i] - d[i] * omega_vars[i] - coupling_terms)
        dynamics_ode.append(accel_i)

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
                "dim_input": 20,
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
