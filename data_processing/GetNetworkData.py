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

contact_keys = [
        'O.materials[0].young', # young's modulus  = 10^E
        'O.materials[0].poisson', # poisson's ratio
        'O.materials[0].frictionAngle', # sliding friction
        ]

## The following list contains macro-variables. For drained triaxial, the macro-input and -output are like below. Note that for the initial load step, all macro-variables are input.
macro_keys = [
        ## macroscopic input (strain can be computed from log(l_x^i/l_x^0) and void ratio from the sum of particle volume)
        'O.cell.hSize[2,2]',  # domain size in z
        'getStress()[0,0]',  # stress in x
        'getStress()[1,1]',  # stress in y
        ## macroscopic output
        'O.cell.hSize[0,0]',  # domain size in x
        'O.cell.hSize[1,1]',  # domain size in y
        'getStress()[2,2]',  # stress in z
        ## others
        'O.dt',  # time increment
        'O.iter',  # number of iterations
]

micro_keys_bodies = [
        ## particle info (in b.shape and b.state)
        'id',
        'shape.radius',
        'state.mass',
        'state.pos[0]',
        'state.pos[1]',
        'state.pos[2]',
        'state.vel[0]',
        'state.vel[1]',
        'state.vel[2]',
        'state.angVel[0]',
        'state.angVel[1]',
        'state.angVel[2]',
        'state.refPos[0]', # position at t=0, with respect to the origin
        'state.refPos[1]',
        'state.refPos[2]',
        ]
micro_keys_inters = [
        ## interaction connectivity info (in inter)
        'id1',
        'id2',
        ## interaction geometry info (in inter.geom)
        'phys.ks',  # tangential stiffness
        'phys.kn',  # normal stiffness
        'geom.penetrationDepth', # overlap between spheres
        'geom.shearInc[0]',  # shear increment x between particles
        'geom.shearInc[1]',  # shear increment y between particles
        'geom.shearInc[2]',  # shear increment z between particles
        'geom.contactPoint[0]', # x, y, z, in the cross section of the overlap
        'geom.contactPoint[1]',
        'geom.contactPoint[2]',
        ## interaction physics info (in inter.phys)
        'phys.usElastic[0]',  # elastic component of the shear displacement x
        'phys.usElastic[1]',  # elastic component of the shear displacement y
        'phys.usElastic[2]',  # elastic component of the shear displacement z
        'phys.usTotal[0]',  # total shear displacement x
        'phys.usTotal[1]',  # total shear displacement y
        'phys.usTotal[2]',  # total shear displacement z
        'phys.shearForce[0]',  # shear foce x
        'phys.shearForce[1]',  # shear foce y
        'phys.shearForce[2]',  # shear foce z
        'phys.normalForce[0]',  # normal force x
        'phys.normalForce[1]',  # normal force y
        'phys.normalForce[2]',  # normal force z
        ]

unused_keys_sequence = [
]

unused_keys_constant = [
]


TARGET_DIR = '/home/cheng/DataGen/'
DATA_DIR = '/home/cheng/DataGen/'
STORE_EDGE_FEATURES = False

    
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

    # name the HDF5 file
    h5file_name = data_dir + '/' + 'simState_' + experiment_type + f'_all_{numParticles}' + '_graphs.hdf5'
    if os.path.exists(h5file_name):
        os.remove(h5file_name)
    f_all = h5py.File(h5file_name, 'a')

    # store the keys and constant parameter and
    f_all['contact_keys'] = contact_keys
    f_all['macro_keys'] = macro_keys
    f_all['micro_keys_bodies'] = micro_keys_bodies
    f_all['micro_keys_inters'] = micro_keys_inters
    O.load(f'{simStateName}_1_{numParticles}_1.yade.gz')
    contact_tensor = np.array([float(eval(key)) for key in contact_keys])
    f_all['contact_params'] = contact_tensor

    # load YADE and store data in f_all
    for sample in samples:
        simStateName = data_dir + '/' + 'simState_' + experiment_type
        # load the initial time step
        try:
            O.load(f'{simStateName}_{sample}_{numParticles}_0.yade.gz'); O.step()
        except IOError:
            print('IOError', f, pressure)
            continue       

        ## get DEM state file names
        file_names = glob.glob(f'{simStateName}_{sample}_{numParticles}_*.yade.gz')
        # get the list of loadstep IDs
        steps = sorted([int(f.split('.yade.gz')[0].split('_')[-1]) for f in file_names])

        f_sample = f_all.create_group(f'{sample}')
        f_sample['num_steps'] = len(steps)

        for step in steps[1:]:
            # load YADE at a given time step
            try:
                O.load(f'{simStateName}_{sample}_{numParticles}_{step}.yade.gz'); O.step()
            except IOError:
                print('IOError', f, pressure)
                continue
            ## macroscopic data (domain size, stress, etc)
            macro_tensor = np.array([float(eval(key)) for key in  macro_keys])

            ## microscopic body data (particle size, position, etc)
            micro_bodies_data = []
            for bodyKey in micro_keys_bodies:
                micro_bodies_data.append([float(eval('b.'+bodyKey)) for b in O.bodies])
            micro_bodies_data = np.array(micro_bodies_data)

            ## macroscopic interaction data (contact pairs, interaction forces, etc)
            micro_inters_data = []
            if not STORE_EDGE_FEATURES: keys = micro_keys_inters[:2]
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
            
            f_step = f_sample.create_group(f'{step}')
            f_step['sources'] = src
            f_step['destinations'] = dst
            if STORE_EDGE_FEATURES: f_step['edge_features'] = e
            f_step['node_features'] = n
            f_step['macro_input_features'] = macro_tensor[:3]
            f_step['macro_output_features'] = macro_tensor[4:7]
            f_step['other_features'] = macro_tensor[8:]

    print(f'Added data to {h5file_name}')

for pressure in ['0.1e6']:
    for experiment_type in ['drained']:
        for numParticles in ['15000']:
            main(pressure, experiment_type, numParticles)
