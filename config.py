import importlib
import os

# 2d, 3d, 4d, 6d_poly, 6d_quad, 8d, 10d, 12d
SELECT_DIMENSION = "6d_quad"
# SELECT_DIMENSION = os.getenv("SELECT_DIMENSION", "2d")


def CURRENT_CONFIG():
    CONFIGS = {
        "2d": "example_config.config_2d_simple_pendulum",
        "3d": "example_config.config_3d_trig",
        "4d": "example_config.config_4d_2bus_lossy_power",
        "6d_poly": "example_config.config_6d_poly",
        "6d_quad": "example_config.config_6d_quadrator",
        "8d": "example_config.config_8d_poly",
        # "9d": "example_config.config_9d_syn",
        "10d": "example_config.config_10d_poly",
        "12d": "example_config.config_12d_bus",
        "6d_dc": "example_config.config_6d_dc_microgrid",
        "20d_kron": "example_config.config_20d_ieee39_kron",
        "49d_full": "example_config.config_49d_ieee39_full",
    }
    
    module_name = CONFIGS[SELECT_DIMENSION]
    return importlib.import_module(module_name)

# Gets the benchmark name based on the dimension for Lyapunov file
def get_benchmark_name():
    dimension_benchmark_name = {
        "2d": "fn_d_all_y",
        "3d": "fn_d_all_t",
        "4d": "fn_d_all_o",
        "6d_poly": "fn_d_all_m",
        "6d_quad": "fn_d_all_z",
        "8d": "fn_d_all_n",
        # "9d": "fn_d_all_l",
        "10d": "fn_d_all_x",
        "12d": "fn_d_all_12d",
        "6d_dc": "fn_d_all_6dc",
        "20d_kron": "fn_d_all_kron",
        "49d_full": "fn_d_all_full",
    }
    return dimension_benchmark_name[SELECT_DIMENSION]


# --- Get the config and assign the functions ---
CONFIG = CURRENT_CONFIG()
get_config = CONFIG.get_config
config_factory = CONFIG.config_factory
dynamics = CONFIG.dynamics
train_config_factory = CONFIG.train_config_factory


if __name__ == "__main__":
    conf = get_config()
    print("priority_queue_training: ", conf.exp.priority_queue_training)
    print("seed_runs: ", conf.exp.seed_runs)
    print("")
