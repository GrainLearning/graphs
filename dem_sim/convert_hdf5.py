# Make the following changes to the format of the hdf5 file
# 1. Join the timesteps into a single tensor
# 2. Apart from the edges, stored as 'edges/sources/3' etc.
# 3. Change the precision of radius to normal, and take out of metadata
# 4. Put the metadata together into 'properties', apart from num_steps which is separate.
# 5. Make sure the positions are between 0 and the domain size.
# 6. Convert 'metadata' at sample level also to 32 bit floats
# 7. Divide macro_output_features by 1e6 to make them order 1
# 8. Rename macro_input/output_features to 'domain' and 'stress' respectively
# 9. Replace 'mean_radius' and 'dispersion_radius' with 'radius_min' and 'radius_max' computed from those
# 10. In 'time', remove the step count, just keeping the time interval
# 11. Split 'node_features' into 'positions' and 'velocities'

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
        if key != 'mean_radius' and key != 'dispersion_radius':
            new_file['metadata'][key] = old_file['metadata'][key][()]
    radius_min = old_file['metadata/mean_radius'][()] - 0.5 * old_file['metadata/dispersion_radius'][()]
    radius_max = old_file['metadata/mean_radius'][()] + 0.5 * old_file['metadata/dispersion_radius'][()]
    new_file['metadata/radius_min'] = radius_min
    new_file['metadata/radius_max'] = radius_max

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

        new_sample['sample_properties'] = np.stack([old_sample['metadata'][key][()] for key in property_list]).astype(np.float32)

        domains = []
        stresses = []
        node_features = []
        time = []
        for t in range(num_steps):
            old_step = old_sample[f'time_sequence/{t}']
            if include_edges:
                new_sample[f'edges/{t}/sources'] = old_step['sources'][()]
                new_sample[f'edges/{t}/destinations'] = old_step['destinations'][()]

            domains.append(old_step['macro_input_features'][()])
            stresses.append(old_step['macro_output_features'][()])
            node_features.append(old_step['node_features'][()])
            time.append(old_step['time'][:1])  # only take the time interval, not the step count

        stresses = np.stack(stresses)
        new_sample['stress'] = stresses / 1e6
        new_sample['time'] = np.stack(time)

        domains = np.stack(domains)
        node_features = np.stack(node_features)
        positions = node_features[:, :, :3]
        velocities = node_features[:, :, 3:-3]
        angular_velocities = node_features[:, :, -3:]
        positions = np.remainder(positions, np.expand_dims(domains, axis=1))

        new_sample['domain'] = domains
        new_sample['positions'] = positions
        new_sample['velocities'] = velocities
        new_sample['angular_velocities'] = angular_velocities


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
