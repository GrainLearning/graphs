# Gathering Data from YADE files

## What is this for?
`GetNetworkData.py` collects the data from yade simulation and stores it in in an hdf5 file.

## Data stored

When importing the generated hdf5 file to python you will get a dictionary with the following keys:
* `contact_params` : array of size 5 containing Young modulus $E$, poisson ratio $\nu$, friction angle $\mu$ in degrees, sample ID (same as the seed), and loadstep ID
* ints between 0 and 200: The key name is the index of the saved state. Each dictionary contains the following dictionaries with keys:
1. `sources` : list of size (number_edges,) with the indices of the source node of each edge.
2. `destinations` : list of size (number_edges,) with the indices of the destination node of each edge.
Look [here](https://distill.pub/2021/gnn-intro/) for more information about adjacency matrix. 
4. `input_features` : Parameters controlling the triaxial experiment.

Table 1. Input parameter:
|index |Imput parameter|	
|:----:|:-------|
|[0]| initial void ratio $e$|
|[1]| confining pressure $\sigma_3$|
|[2]| strain in x direction $\varepsilon_x$ |
|[3]| strain in y direction $\varepsilon_y$ |
|[4]| strain in z direction $\varepsilon_z$ |
|[5]| domain size in x direction $ l_x$     |
|[6]| domain size in x direction $ l_y$     |
|[7]| domain size in x direction $ l_z$     |
|[8]| number of particles|

4. `node_features` : Particle features

Table 2. Node parameters:
|index |Node parameter|
|:----:|:-------|
|[0] | ID    |
|[1] | radius|
|[2] | mass  |
|[3] | position x coordinate|
|[4] | position y coordinate|
|[5] | position z coordinate|
|[6] | velocity x component|
|[7] | velocity y component|
|[8] | velocity z component|
|[9] | angular velocity 1 in YADE | 
|[10] | angular velocity 2 in YADE | 
|[11] | angular velocity 3 in YADE | 
|[12] | x coordinate at t=0, with respect to the origin|
|[13] | y coordinate at t=0, with respect to the origin|
|[14] | z coordinate at t=0, with respect to the origin|

5. `edge_features` : Contact features

Table 3. Edge parameters:
|index|Edge parameter|
| ----|-------|
|[0] |radius particle 1 |
|[1] |radius particle 2|
|[2] |tangential stiffness ks |
|[3] |normal stiffness kn |
|[4] |overlapping depth between spheres|
|[5] |shear increment x between particles|
|[6] |shear increment y between particles|
|[7] |shear increment z between particles|
|[8] |contact normal vector x component (can get rid of and just keep normal force)|
|[9] |contact normal vector y component|
|[10] |contact normal vectorz component|
|[11] |contactPoint x coordinate in the cross section of the overlap|
|[12] |contactPoint y coordinate|
|[13] |contactPoint z coordinate|
|[14]| elastic component of the shear displacement x|
|[15]| elastic component of the shear displacement y|
|[16]| elastic component of the shear displacement z|
|[17]| total shear displacement x|
|[18]| total shear displacement y|
|[19]| total shear displacement z|
|[20]| shear force x|
|[21]| shear force y|
|[22]| shear force z|
|[23]| normal force x|
|[24]| normal force y|
|[25]| normal force z|

## How to run it
To run `GetNetworkData.py` you must have [yade](https://yade-dem.org/doc/installation.html) installed and then simply run `yade GetNetworkData.py`.
