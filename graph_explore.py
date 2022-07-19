import h5py
import numpy as np
import os

DATA_DIR = '/Users/aronjansen/Documents/grainsData/graph_test_data/'
FILENAME = 'simState_drained_200_10000_0_gnn_data.hdf5'
SAVE_DIR = '/Users/aronjansen/Documents/grainsData/graph_test_data/graphs.hdf5'
SAVE_DIR_SHORT = '/Users/aronjansen/Documents/grainsData/graph_test_data/graphs_short.hdf5'

# just focus on this case for now
PRESSURE = '0.1e6'
EXPERIMENT_TYPE = 'drained'
NUM_PARTICLES = '10000'

file_names = [fn for fn in os.listdir(DATA_DIR) if fn.endswith('_gnn_data.hdf5')]

def count_max_interactions():
    interactions = []
    for fname in file_names:
        f = h5py.File(DATA_DIR + fname, 'r')
        f = f[PRESSURE][EXPERIMENT_TYPE][NUM_PARTICLES]
        num_interactions = f['outputs_inters'][:][0].shape[1]
        interactions.append(num_interactions)
    interactions = np.array(interactions)

    max_interactions = np.max(interactions)
    return max_interactions


def write_to_file(num_steps, filename):
    if os.path.exists(filename):
        os.remove(filename)
    f_all = h5py.File(filename, 'a')
    f_all['num_steps'] = num_steps
    for step in range(0, num_steps):
        fname = f'simState_drained_200_10000_{step}_gnn_data.hdf5'
        f_input = h5py.File(DATA_DIR + fname, 'r')
        f_input = f_input[PRESSURE][EXPERIMENT_TYPE][NUM_PARTICLES]
        interactions_raw = f_input['outputs_inters'][0]
        # raw interactions are of the form (source, destination, features...)
        src = interactions_raw[0, :].astype(int)
        dst = interactions_raw[1, :].astype(int)
        e = interactions_raw[2:]
        e = np.transpose(e, (1, 0))
        n = np.transpose(f_input['outputs_bodies'][0], (1, 0))

        if step == 0:
            f_all['contact_params'] = f_input['contact_params'][:]

        f_step = f_all.require_group(f'{step}')
        f_step['sources'] = src
        f_step['destinations'] = dst
        f_step['edge_features'] = e
        f_step['node_features'] = n
        f_step['input_features'] = f_input['inputs'][:][0]

write_to_file(num_steps=201, filename=SAVE_DIR)
write_to_file(num_steps=5, filename=SAVE_DIR_SHORT)
