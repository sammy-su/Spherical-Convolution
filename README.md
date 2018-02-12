# Spherical Convolution

## Code Structure

A brief introduction for the purpose of each file.

### rf.yaml

The updated receptive field for each kernel. It contains a python dictionary with the following structure.

Resolution -> Layer -> Row

Resolution: SphereH[h]Ks[k], where h is the height of the input image and k is the resolution for 65.5° FOV.

Layer: {1_1,...,5_3}

Row: {0,...,h/2}, because the receptive field is symmetric about the equator.

### SphereProjection.py

Utility class / function for sphere to tangent plane projection.

### SphericalLayers.py

Caffe layers for spherical convolution.

SplitRowLayer: split the 2D image to multiple rows for each kernel.

MergeRowLayer: merge the convolution output for each kernel.

The other layers are for SphConv specific data fetching and loss.

### util/rf.py

Functions for computing and loading rf.yaml

Several parameters are hand coded in this files, including

kernel_sizes: the receptive field for the kernels in each layer

top_down: order of the layers, with (key, val) := (top layer, bottom layer).

strides: pixel size (in terms of FOV) relative to the input image for each layer. For example, 4 indicates that the pixel in the current layer is 4x larger than the input pixels. The structure is: Ks -> SphereH -> Layer

### bin/generate_sources.py

Generate the target values for spherical convolution. It will compute the exact convolution result for each pixel and stores the outputs in HDF5 binary.

### bin/generate_targets.py

Similar to generate_sources.py, but only compute the values for a subset of pixels. It stores the spherical convolution target value and pixel location in pkl file.

### bin/crop_srcs.py

Generate the training data for each layer by reading the output of generate_sources.py and generate_targets.py. It reads the spherical convolution target values from the output of generate_targets.py and crops the corresponding input values from the output of generate_sources.py. The output is a HDF5 file containing the target and source value for the kernel.

### bin/generate_proto.py

Generate the Caffe prototxt file of each kernel for kernel-wise training.

### bin/generate_sphconv.py

Generate the Caffe prototxt file for the entire spherical convolution network.

### bin/solve_net.py

Perform kernel-wise training. It requires the output of crop_srcs.py and generate_proto.py.

### bin/solve_sphconv.py

Training full spherical convolution network.

