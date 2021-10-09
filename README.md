# wino_stream
Streaming architecture for winograd convolution

### Demo Video
[![Demo Video](https://img.youtube.com/vi/r0CQHH15nIY/0.jpg)](https://www.youtube.com/watch?v=r0CQHH15nIY)

## Slide_fast branch
To test my own sliding window method and compare with the Xilinx FINN method.


### Block Diagram
![Block Diagram](/media/block_diagram.png)


# Lenet MNIST Results

### Neural Network Architecture
![Resources](/media/mnist_lenet/lenet_labelled.png)


### Without Slicer
![Resources](/media/mnist_lenet/resources_noSlicer.png)
##### FPS
	17.1k images/sec

### With Slicer
![Resources](/media/mnist_lenet/resources.png)
##### FPS
	22.2k images/sec but uses 28 more BRAMs

### Electric Sliding window
![Resources](/media/mnist_lenet/electric_slide.png)
##### FPS
	20.15k images/sec and slightly more resources than Xilinx sliding window



# CNV Cifar10 Results

### Neural Network Architecture
![Resources](/media/cifar10_cnv/cifar10_cnv.png)


### Without Slicer
![Resources](/media/cifar10_cnv/orig_smallest_footprint.png)
##### FPS
	772.04 images/sec



