"""
Preprocess data so that it can be used as input in an RNN.
Data consists of:
- contact_params (samples, 5)
- input_params (samples, sequence_length, x)
- output_params (samples, sequence_length, y)
To run this file yade must be installed and then run: yade GetNetworkdata.py
"""
import numpy as np
import os, glob
import h5py

contact_keys = {
        'young_modulus':'O.materials[0].young',  # young's modulus  = 10^E
        'possion_ratio':'O.materials[0].poisson',  # poisson's ratio
        'friction_angle':'O.materials[0].frictionAngle',  # sliding friction
        'particle_density':'O.materials[0].density',  # density
        }

particle_keys = {
        'radius':'shape.radius', #  particle radius
#        'mass':'shape.mass', #  particle mass can be computed as 4/3*density*pi*radius**3
        }

sampling_keys = [
        'pressure',  # confining pressure
        'initial_friction',  # initial friction (for generating microstructure with different void ratio)
        'compressive_strain_rate',  # rate of compressive strain d{\epsilon_p}
        'shear_strain_rate',  # rate of shear strain d{\epsilon_q}
        ]

## The following list contains macro-variables. For drained triaxial, the macro-input and -output are like below. Note that for the initial load step, all macro-variables are input.
macro_keys = {
        ## macroscopic input (strain can be computed from log(l_x^i/l_x^0) and void ratio from the sum of particle volume)
        'domain_size_z':'O.cell.hSize[2,2]',  # domain size in z
        'stress_x':'getStress()[0,0]',  # stress in x
        'stress_y':'getStress()[1,1]',  # stress in y
        ## macroscopic output
        'domain_size_x':'O.cell.hSize[0,0]',  # domain size in x
        'domain_size_y':'O.cell.hSize[1,1]',  # domain size in y
        'stress_z':'getStress()[2,2]',  # stress in z
        ## others
        'time':'O.time',  # time increment
        'num_time_steps':'O.iter',  # number of time steps
}

micro_keys_bodies = {
        ## particle info (in b.shape and b.state)
        'position_x':'state.pos[0]',
        'position_y':'state.pos[1]',
        'position_z':'state.pos[2]',
        'velocity_x':'state.vel[0]',
        'velocity_y':'state.vel[1]',
        'velocity_z':'state.vel[2]',
        'angular_velocity_x':'state.angVel[0]',
        'angular_velocity_y':'state.angVel[1]',
        'angular_velocity_z':'state.angVel[2]',
        }

micro_keys_inters = {
        ## interaction connectivity info (in inter)
        'contact_pair_ID1':'id1',
        'contact_pair_ID2':'id2',
        ## interaction geometry info (in inter.geom)
        'tangential_stiffness':'phys.ks',  # tangential stiffness
        'tangential_stiffness':'phys.kn',  # normal stiffness
        'overlap':'geom.penetrationDepth', # overlap between spheres
        'shear_increment_x':'geom.shearInc[0]',  # shear increment x between particles
        'shear_increment_y':'geom.shearInc[1]',  # shear increment y between particles
        'shear_increment_z':'geom.shearInc[2]',  # shear increment z between particles
        'contact_point_x':'geom.contactPoint[0]', # x, y, z, in the cross section of the overlap
        'contact_point_y':'geom.contactPoint[1]',
        'contact_point_z':'geom.contactPoint[2]',
        ## interaction physics info (in inter.phys)
        'elastic_shear_x':'phys.usElastic[0]',  # elastic component of the shear displacement x
        'elastic_shear_y':'phys.usElastic[1]',  # elastic component of the shear displacement y
        'elastic_shear_z':'phys.usElastic[2]',  # elastic component of the shear displacement z
        'total_shear_x':'phys.usTotal[0]',  # total shear displacement x
        'total_shear_y':'phys.usTotal[1]',  # total shear displacement y
        'total_shear_z':'phys.usTotal[2]',  # total shear displacement z
        'shear_force_x':'phys.shearForce[0]',  # shear foce x
        'shear_force_y':'phys.shearForce[1]',  # shear foce y
        'shear_force_z':'phys.shearForce[2]',  # shear foce z
        'normal_force_x':'phys.normalForce[0]',  # normal force x
        'normal_force_y':'phys.normalForce[1]',  # normal force y
        'normal_force_z':'phys.normalForce[2]',  # normal force z
        }

unused_keys_sequence = [
]

unused_keys_constant = [
]


TARGET_DIR = '/home/cheng/DataGen/'
DATA_DIR = '/home/cheng/DataGen/'
STORE_EDGE_FEATURES = False

# load the sampling variables
state_sampling = [log10(0.1e6), log10(0.01)]
path_sampling = [0, 0]

def main(pressure, experiment_type,numParticles):
    data_dir = DATA_DIR + f'{pressure}/{experiment_type}/{numParticles}/'
    if not os.listdir(data_dir):
        print(f"Directory {data_dir} is empty.")
        return

    # get DEM state file names
    simStateName = 'simState_' + experiment_type + f'_*_{numParticles}_0.yade.gz'
    file_names = glob.glob(data_dir + '/' + f'{simStateName}')

    # get the list of sample IDs
    samples = [int(f.split('.yade.gz')[0].split('_')[-3]) for f in file_names]
    simStateName = data_dir + '/' + 'simState_' + experiment_type
    print(f'Number of samples is {len(samples)}')

    # name the HDF5 file
    h5file_name = data_dir + '/' + 'simState_' + experiment_type + f'_all_{numParticles}' + '_graphsClean.hdf5'
    if os.path.exists(h5file_name):
        os.remove(h5file_name)
    f_all = h5py.File(h5file_name, 'a')

    # store the variable keys
    f_all.attrs['contact_keys'] = list(contact_keys.keys())
    f_all.attrs['radius'] = list(particle_keys.keys())
    f_all.attrs['sampling_variables'] = sampling_keys
    f_all.attrs['macro_input_features'] = [list(macro_keys.keys())[i] for i in [0,1,2]]
    f_all.attrs['macro_output_features'] = [list(macro_keys.keys())[i] for i in [3,4,5]]
    f_all.attrs['time'] = list(macro_keys.keys())[6:]
    f_all.attrs['node_features'] = list(micro_keys_bodies.keys())
    f_all.attrs['edge_features'] = list(micro_keys_inters.keys())

    # load a simulation state to get the contact parameters
    O.load(f'{simStateName}_1_{numParticles}_1.yade.gz')
    contact_tensor = np.array([float(eval(key)) for key in contact_keys.values()])

    # store metadata universal to all samples
    f_all_meta = f_all.create_group('metadata')
    f_all_meta['contact_params'] = contact_tensor
    f_all_meta['num_nodes'] = int(numParticles)
    f_all_meta['mean_radius'] = 0.5
    f_all_meta['dispersion_radius'] = 0.4

    # load YADE and store data in f_all
    for sample in samples[:2]:
        # load the initial time step
        try:
            O.load(f'{simStateName}_{sample}_{numParticles}_0.yade.gz'); O.step()
        except RuntimeError:
            print(f'RuntimeError encountered for {simStateName}_{sample}_{numParticles}_0.yade.gz')
            continue       

        ## get DEM state file names
        file_names = glob.glob(f'{simStateName}_{sample}_{numParticles}_*.yade.gz')
        # get the list of loadstep IDs
        steps = sorted([int(f.split('.yade.gz')[0].split('_')[-1]) for f in file_names])
        print(f'Steps: {steps}')
        
        # create a group per sample
        f_sample = f_all.create_group(f'{sample}')
        # store metadata specific to each sample
        f_sample_meta = f_sample.create_group('metadata')
        for m,key in enumerate(sampling_keys): f_sample_meta[key] = (state_sampling+path_sampling)[m]
        f_sample_meta['num_steps'] = len(steps)
        radius = np.array([float(eval('b.'+const_body_key)) for b in O.bodies for const_body_key in particle_keys.values()]).reshape([int(numParticles),1])
        f_sample_meta['radius'] = radius.astype('float32')
        # store the time sequence of each sample
        f_sample_time = f_sample.create_group('time_sequence')
        # loop over time
        for i,step in enumerate(steps[:2]):
            # load YADE at a given time step
            try:
                O.load(f'{simStateName}_{sample}_{numParticles}_{step}.yade.gz'); O.step()
            except RuntimeError:
                print(f'RuntimeError encountered for {simStateName}_{sample}_{numParticles}_{step}.yade.gz')
                continue
            ## macroscopic data (domain size, stress, etc)
            macro_tensor = np.array([float(eval(key)) for key in  macro_keys.values()])

            ## microscopic body data (particle size, position, etc)
            micro_bodies_data = []
            for bodyKey in micro_keys_bodies.values():
                micro_bodies_data.append([float(eval('b.'+bodyKey)) for b in O.bodies])
            micro_bodies_data = np.array(micro_bodies_data)

            ## macroscopic interaction data (contact pairs, interaction forces, etc)
            micro_inters_data = []
            if not STORE_EDGE_FEATURES: keys = list(micro_keys_inters.values())[:2]
            for key in keys:
                micro_inters_data.append([float(eval('i.'+key)) for i in O.interactions if i.isReal])
            micro_inters_data = np.array(micro_inters_data)            

            ## add to DGL library format (from Aron)
            src = micro_inters_data[0, :].astype(int)
            dst = micro_inters_data[1, :].astype(int)
            if STORE_EDGE_FEATURES:
                e = micro_inters_data[2:]
                e = np.transpose(e, (1, 0))
            n = np.transpose(micro_bodies_data, (1, 0))
            
            f_step = f_sample_time.create_group(f'{i}')
            f_step['sources'] = src
            f_step['destinations'] = dst
            if STORE_EDGE_FEATURES: f_step['edge_features'] = e
            f_step['node_features'] = n.astype('float32')
            f_step['macro_input_features'] = macro_tensor[0:3]
            f_step['macro_output_features'] = macro_tensor[3:6]
            f_step['time'] = macro_tensor[6:]

    print(f'Added data to {h5file_name}')


for pressure in ['0.1e6']:
    for experiment_type in ['drained']:
        for numParticles in ['5000']:
            main(pressure, experiment_type, numParticles)
