# Gathering Data from YADE files

## What is this for?
`GetNetworkData.py` collects the data from yade simulation and stores it in in an hdf5 file.
Such file can be read in python using the package h5py as follows: 

```python
import h5py
dataset = h5py.File(path_to_file, 'r')
```

## Data format
When importing the generated hdf5 file to python you will get a dictionary with the following keys:

* **metadata**: Data that is common for all simulations in the dataset:
    * *contact_params*: array containing Young modulus $E$, poisson ratio $\nu$, friction angle $\mu$ in degrees and particle density.
    * *num_nodes*: Number of particles
    * *radius_max*: Maximum radius of the spherical particles.
    * *radius_min*: Minimum radius

* **SampleID_LoadPath**: inside each one of these groups:
    * *radius*: List of the particle radii `[num_nodes]`
    * *sample_properties*: List of: Pressure $P$, compressive strain rate $\dot{\varepsilon_p}$, shear strain rate $\dot{\varepsilon_q}$ and initial friction $\mu_0$ `[4]`
    * *num_steps*: Number of time steps in the simulation `[int]`
    * *domain*: Domain size in x,y,z `[num_steps, 3]`
    * *stress*: Sample stress in x,y,z `[num_steps, 3]`
    * *velocities*: Particle velocity x,y,z `[num_steps, num_nodes, 3]`
    * *positions*: Particle positions x,y,z `[num_steps, num_nodes, 3]`
    * *time*: Physical time at each step `[num_steps]`

Labels of the fields can be found the attribute `attrs` of the hdf5 file:
```python
dataset.attrs.keys()
<KeysViewHDF5 ['contact_keys', 'edge_features', 'macro_input_features', 'macro_output_features', 'node_features', 'radius', 'sample_properties', 'sampling_variables', 'time']>
```

## How to run it
To run `GetNetworkData.py` you must have [yade](https://yade-dem.org/doc/installation.html) installed and then in a command line run:
```bash
yade GetNetworkData.py
```

## How to contribute
If you have DEM simulation data from another software package, you can write your own parser (i.g. equivalet to GetNetworkData.py) that produces an hdf5 file having the **same format**. 
1. Create an issue explaining to the community what you are working on.
2. Create a branch or fork the repository.
3. Test your code locally.
4. Once you're ready create a pull request, the maintainers of the code will evaluate if your contribution can be directly merged or requires changes.  