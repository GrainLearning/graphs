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
        'E', # young's modulus  = 10^E
        'v', # poisson's ratio
        'mu', # sliding friction
        'sample', # sample ID
        'step', # loadsep ID
        ]

input_keys = [
        'e',  # initial void ratio
        'conf',  # confining pressure (stored as group name already)
        'e_x',  # radial strain in x
        'e_y',  # radial strain in y
        'e_z',  # axial strain in z
        'l_x',  # domain size in x
        'l_y',  # domain size in y
        'l_z',  # domain size in z
        'num',  # number of particles
]

output_keys_bodies = [
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
output_keys_inters = [
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

    # name the HDF5 file
    h5file_name = data_dir + '/' + 'simState_' + experiment_type + f'_all_{numParticles}' + '_graphsNew.hdf5'
    if os.path.exists(h5file_name):
        os.remove(h5file_name)
    f_all = h5py.File(h5file_name, 'a')

    # load YADE and store data in f_all
    for sample in samples:
        simStateName = data_dir + '/' + 'simState_' + experiment_type
        # load the initial time step
        try:
            O.load(f'{simStateName}_{sample}_{numParticles}_0.yade.gz'); O.step()
        except IOError:
            print('IOError', f, pressure)
            continue
        ### contact parameters        
        contact_tensor = np.array([
            O.materials[0].young,
            O.materials[0].poisson,
            O.materials[0].frictionAngle,
            ])

        # get DEM state file names
        file_names = glob.glob(f'{simStateName}_{sample}_{numParticles}_*.yade.gz')
        # get the list of loadstep IDs
        steps = sorted([int(f.split('.yade.gz')[0].split('_')[-1]) for f in file_names])

        f_sample = f_all.create_group(f'{sample}')
        f_sample['contact_params'] = contact_tensor
        f_sample['num_steps'] = len(steps)

        for step in steps:
            # load YADE at a given time step
            try:
                O.load(f'{simStateName}_{sample}_{numParticles}_{step}.yade.gz'); O.step()
            except IOError:
                print('IOError', f, pressure)
                continue
            ### input data (void ratio e, mean pressure, number of particles)
            inputs_tensor = np.array([
                porosity()/(1-porosity()),
                getStress().trace()/3,
                triax.strain[0],
                triax.strain[1],
                triax.strain[2],
                O.cell.hSize[0,0],
                O.cell.hSize[1,1],
                O.cell.hSize[2,2],
                len(O.bodies)
                ])
            ### output data
            ## particle info
            bodies_data = []
            for bodyKey in output_keys_bodies:
                bodies_data.append([float(eval('b.'+bodyKey)) for b in O.bodies])
            bodies_data = np.array(bodies_data)
            ## interaction info
            inters_data = []
            for interKey in output_keys_inters:
                inters_data.append([float(eval('i.'+interKey)) for i in O.interactions if i.isReal])
            inters_data = np.array(inters_data)

            ## add to DGL library format (from Aron)
            src = inters_data[0, :].astype(int)
            dst = inters_data[1, :].astype(int)
            e = inters_data[2:]
            e = np.transpose(e, (1, 0))
            n = np.transpose(bodies_data, (1, 0))
            
            f_step = f_sample.create_group(f'{step}')
            f_step['sources'] = src
            f_step['destinations'] = dst
            f_step['edge_features'] = e
            f_step['node_features'] = n
            f_step['input_features'] = inputs_tensor

    print(f'Added data to {h5file_name}')

for pressure in ['0.1e6']:
    for experiment_type in ['drained']:
        for numParticles in ['15000']:
            main(pressure, experiment_type, numParticles)
