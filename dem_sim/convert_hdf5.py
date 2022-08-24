# Make the following changes to the format of the hdf5 file
# 1. Join the timesteps into a single tensor
# 2. Apart from the edges, stored as 'edges/sources/3' etc.
# 3. Change the precision of radius to normal, and take out of metadata
# 4. Put the metadata together into 'properties', apart from num_steps which is separate.
# 5. Make sure the positions are between 0 and the domain size.

import h5py
import numpy as np


def convert(old_path, new_path=None, include_edges=False):
    old_file = h5py.File(old_path, 'r')
    if not new_path:
        new_path = old_path[:-5] + '_reformatted.hdf5'
    new_file = h5py.File(new_path, 'w')

    for attribute in list(old_file.attrs):
        new_file.attrs[attribute] = old_file.attrs[attribute]
    property_list = ['compressive_strain_rate', 'initial_friction', 'pressure', 'shear_strain_rate']
    new_file.attrs['sample_properties'] = property_list

    new_file.require_group('metadata')
    for key in old_file['metadata'].keys():
        new_file['metadata'][key] = old_file['metadata'][key][()]

    samples = [key for key in old_file.keys() if key != 'metadata']
    for sample_key in samples:
        old_sample = old_file[sample_key]
        new_sample = new_file.require_group(sample_key)

        new_sample['radius'] = np.array(old_sample['metadata/radius'], dtype=np.float32)
        num_steps = old_sample['metadata/num_steps'][()]
        # NOTE: seems to be one step less here
        if sample_key == '2_4':
            num_steps = num_steps - 1
        new_sample['num_steps'] = num_steps

        new_sample['properties'] = np.stack([old_sample['metadata'][key][()] for key in property_list]).astype(np.float32)

        macro_input_features = []
        macro_output_features = []
        node_features = []
        time = []
        for t in range(num_steps):
            old_step = old_sample[f'time_sequence/{t}']
            if include_edges:
                new_sample[f'edges/{t}/sources'] = old_step['sources'][()]
                new_sample[f'edges/{t}/destinations'] = old_step['destinations'][()]

            macro_input_features.append(old_step['macro_input_features'][()])
            macro_output_features.append(old_step['macro_output_features'][()])
            node_features.append(old_step['node_features'][()])
            time.append(old_step['time'][()])

        macro_output_features = np.stack(macro_output_features)
        new_sample['macro_output_features'] = macro_output_features / 1e6
        new_sample['time'] = np.stack(time)

        domains = np.stack(macro_input_features)
        node_features = np.stack(node_features)
        positions = node_features[:, :, :3]
        velocities = node_features[:, :, 3:]
        positions = np.remainder(positions, np.expand_dims(domains, axis=1))
        node_features = np.concatenate([positions, velocities], axis=-1)

        new_sample['macro_input_features'] = domains
        new_sample['node_features'] = node_features


def check_steps_present(path):
    old_file = h5py.File(path, 'r')
    samples = [key for key in old_file.keys() if key != 'metadata']
    for sample_key in samples:
        old_sample = old_file[sample_key]
        num_steps = old_sample['metadata/num_steps'][()]
        for t in range(num_steps):
            try:
                old_step = old_sample[f'time_sequence/{t}']
            except:
                print(f"For sample {sample_key}, with {num_steps} steps, actually step {t} doesn't exist?")

if __name__ == '__main__':
    original_path = '/Users/aronjansen/Documents/grainsData/raw/simState_path_sampling_5000_graphs.hdf5'
    convert(original_path, include_edges=False)
