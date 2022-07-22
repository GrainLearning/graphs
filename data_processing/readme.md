# Gathering Data from YADE files

## What is this for?
`GetNetworkData.py` collects the data from yade simulation and stores it in in an hdf5 file.

## Data stored

When importing the generated hdf5 file to python you will get a dictionary with the following keys:
* `contact_params` : array of size 3 containing Young modulus $E$, poisson ratio $\nu$, and friction angle $\mu$ in degrees.
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
|[5]| number of particles|

4. `node_features` : Particle features

Table 2. Node parameters:
|index |Node parameter|
|:----:|:-------|
| [0] | radius|
| [1] | mass  |
|[2] | position x coordinate|
|[3] | position y coordinate|
|[4] | position z coordinate|
|[5] | velocity x component|
|[6] | velocity y component|
|[7] | velocity z component|
|[8] | angular velocity 1 in YADE | 
|[9] | angular velocity 2 in YADE | 
|[10] | angular velocity 3 in YADE | 
|[11] | angular momentum 1 in YADE|
|[12] | angular momentum 2 in YADE|
|[13] | angular momentum 3 in YADE|
|[14] | Inertia 1 in YADE|
|[15] | Inertia 2 in YADE|
|[16] | Inertia 3 in YADE|
|[17] | x coordinate at t=0, with respect to the origin|
|[18] | y coordinate at t=0, with respect to the origin|
|[19] | z coordinate at t=0, with respect to the origin|
|[20] | orientation angle 1 in YADE at t=0, with respect to the origin|
|[21] | orientation angle 2 in YADE at t=0, with respect to the origin|
|[22] | orientation angle 3 in YADE at t=0, with respect to the origin|
|[23] | orientation angle 4 in YADE at t=0, with respect to the origin|

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
|[14] |elastic component of the shear (tangential) force x (redundant equal to shearForce)|
|[15] |elastic component of the shear (tangential) force y|
|[16] |elastic component of the shear (tangential) force z|
|[17]| elastic component of the shear displacement x|
|[18]| elastic component of the shear displacement y|
|[19]| elastic component of the shear displacement z|
|[20]| total shear displacement x|
|[21]| total shear displacement y|
|[22]| total shear displacement z|
|[23]| shear force x|
|[24]| shear force y|
|[25]| shear force z|
|[26]| normal force x|
|[27]| normal force y|
|[28]| normal force z|

## How to run it
To run `GetNetworkData.py` you must have [yade](https://yade-dem.org/doc/installation.html) installed and then simply run `yade GetNetworkData.py`.
