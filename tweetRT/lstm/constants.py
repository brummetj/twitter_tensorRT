hyper_parameter = {
    'NUM_DIMENSIONS': 300,
    'MAX_SEQ_LENGTH': 250,
    'BATCH_SIZE': 24,
    'LSTM_UNITS': 64,
    'NUM_CLASSES': 2,
    'ITERATIONS': 1000,
    'NUM_LAYERS': 3
}

cluster_specification = {
    "ps": ["localhost:2222"],  # list of parameter servers,
    "worker": ["localhost:2223", "localhost:2224"]  # list of workers
}