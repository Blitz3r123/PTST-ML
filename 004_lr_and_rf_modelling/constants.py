ALL_MODELS_DIRPATH = "./all_models/"

DDS_METRICS = [
    'avg_lost_samples',
    'avg_lost_samples_percentage',
    'avg_received_samples',
    'avg_received_samples_percentage',
    'avg_samples_per_sec',
    'avg_throughput_mbps',
    'latency_us',
    'total_lost_samples',
    'total_lost_samples_percentage',
    'total_received_samples',
    'total_received_samples_percentage',
    'total_samples_per_sec',
    'total_throughput_mbps'
]

STATS = [
    'mean', 'std',
    'min', 'max',
    '1', '2', '5', '10', 
    '25', '30', '40', '50', '60', '70', '75', '80', 
    '90', '95', '99'
]

STANDARDISATION_FUNCTIONS = ["none", "z_score", 'min_max', 'robust_scaler',]

TRANSFORM_FUNCTIONS = [
    "none",
    "log",
    "log10",
    "log2",
    "log1p",
    "sqrt",
]

ERROR_METRICS = [
    "rmse", 
    "mse", 
    "mae", 
    "mape", 
    "r2", 
    "medae", 
    "explained_variance"
]