# ANNetGPGPU - Neural network library with GPU support #

# We moved to: https://github.com/ANNetGPGPU/ANNetGPGPU #

## Donation ##
If someone has left some piece of a bitcoin: 15YR45hEhm2Adb9e1Za1Hatm2Uw3DvYy7B

![https://annetgpgpu.googlecode.com/files/ANNetGPGPU_bitcoin.png](https://annetgpgpu.googlecode.com/files/ANNetGPGPU_bitcoin.png)

## Introduction ##
ANNet is a small library to create neural nets. A hallmark of the project are implementations of several neural network models with usage of [OpenMP](http://openmp.org/wp/) and/or [Thrust](http://code.google.com/p/thrust/). See [quickstart guide](QuickStartGuide.md) to learn how to use them. Especially self organizing maps (SOMs) benefit strongly from calculations on GPUs and speed-ups by a factor of 100 can easily be achieved for bigger networks (>256x256 nodes) even on old graphic hardware. The GPU implementation of SOMs is also supporting asynchronous calculation on more than one device. So computation on clusters is possible.

## Build ##
I wrote some cmake-scripts to make it easier to build the library. For building the library with all features you need to have installed:
  * [Qt4](http://qt.nokia.com/products/) (just for some examples and designer required)
  * [SWIG for python bindings](http://www.swig.org/) (just required for python bindings)
  * [CUDA](http://www.nvidia.de/object/cuda_home_new_de.html)/[Thrust](http://code.google.com/p/thrust/) (shipped with CUDA; just required for GPGPU implementation)
  * [Doxygen](http://www.stack.nl/~dimitri/doxygen/) (required for documentation generation)
  * [OpenMP](http://openmp.org/wp/) (required if multi CPU support is wished)
  * Lib [bzip2](http://www.bzip.org/) (**required**)
  * [CMake](http://www.cmake.org/) (required if you want to use the cmake scripts)
  * A C++ compiler ([GCC](http://gcc.gnu.org/) or [MinGW](http://www.mingw.org/); required)

How you build the library:
  1. Clone the repository with git
```
> git clone https://github.com/ANNetGPGPU/ANNetGPGPU.git
```
  1. Create a build directory, where your compiler stores the objects
```
> cd annetgpgpu
> mkdir build
> cd build
```
  1. Run cmake and make to build.  Dependent on the installed libraries, either all or just some example programs will be built.
```
> cmake .. && make
```

## Examples ##
### 1.) Classical backpropagation networks ###
#### a) Perceptrons ####
The perceptron is an algorithm for supervised classification of an input into one of two possible outputs. It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector describing a given input.
Here you see how easy it is to create a perceptron.
```
ANN::BPLayer layer1(64*64, ANN::ANLayerInput); // input layer of size: 64X64
ANN::BPLayer layer2(16, ANN::ANLayerOutput);   // output layer of size: 16

layer1.ConnectLayer(&layer2);

ANN::BPNet cpu;
cpu.AddLayer(&layer1);
cpu.AddLayer(&layer2);
cpu.SetNetFunction(&Functions::fcn_linear);
```
It should be noted, that `AddLayer` connects all neurons of each layer together. This isn't necessry because the network is handled as a linked list. You also could create assymetric nets by defining the connections manually. Member functions for this exist already.
#### b) Artificial neural (_back propagation_) network ####
An Artificial Neural Network (ANN), usually called neural network (NN), is a mathematical model or computational model that is inspired by the structure and/or functional aspects of biological neural networks. A neural network consists of an interconnected group of artificial neurons, and it processes information using a connectionist approach to computation. In most cases an ANN is an adaptive system that changes its structure based on external or internal information that flows through the network during the learning phase. Modern neural networks are non-linear statistical data modeling tools. They are usually used to model complex relationships between inputs and outputs or to find patterns in data.
Even more simple it is to create this kind of networks, because it is the standard implementation of the "BPNet" classes.
```
ANN::BPLayer layer1(64*64, ANN::ANLayerInput); // input layer of size: 64X64
ANN::BPLayer layer2(64, ANN::ANLayerHidden);   // hidden layer of size: 64
ANN::BPLayer layer3(16, ANN::ANLayerOutput);   // output layer of size: 16

layer1.ConnectLayer(&layer2);
layer2.ConnectLayer(&layer3);

ANN::BPNet cpu;
cpu.AddLayer(&layer1);
cpu.AddLayer(&layer2);
cpu.AddLayer(&layer3);
```
### 2.) Hopfield networks ###
A Hopfield network is a form of recurrent artificial neural network invented by John Hopfield. Hopfield nets serve as content-addressable memory systems with binary threshold units. They are guaranteed to converge to a local minimum, but convergence to one of the stored patterns is not guaranteed. Furthermore, Hopfield networks provide a model for understanding human memory.
My implementation of hopfield networks are slightly different, because they only consist of one layer and training isn't necessary but could directly calculated from the input. So the creation is a no-brainer.
```
ANN::HFNet cpu;
cpu.Resize(16,16); // create a hopfield net of size: 16X16
```
### 3.) Self organizing maps (SOMs) ###
#### a) CPU implementation ####
A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), discretized representation of the input space of the training samples, called a map. Self-organizing maps are different from other artificial neural networks in the sense that they use a neighborhood function to preserve the topological properties of the input space.
Creation of this kind of neural network is as easy as the example above. The evaluation NOT.
```
ANN::SOMNet cpu;
cpu.SetTrainingSet(input);
cpu.CreateSOM(3, 1, 128, 128); // create a SOM of size: 128X128
```
#### b) GPGPU implementation ####
At the moment it is possible to speed up your calculation of SOMs with GPU. You can train, save and load the net with CPU as well and later continue with GPU and the other way round. Simply use the proper CTOR:
```
ANN::SOMNet cpu;
cpu.CreateSOM(3, 1, 128, 128); // input w, h, net w, h

ANN::SOMNetGPU gpu(&cpu);      // use copy CTOR or create it like above
// .. 
cpu.ExpToFS(foo.bar);
gpu.ImpFromFS(foo.bar);
```

## News ##
### 06/2013 Device function pointer support for plugin system ###
An advantage of the library is it's modular layout. You can implement your own e.g. decay or distance/transfer functions and pass them as function pointers. For decay functions the GPU-implementation of SOMs behaves like the CPU-implementation and implementation of you own ones is very easy. Unfortunately the distance function belongs to device space. So more recent hardware with SM 2.0 is necessary for that. I created a new example to illustrate how you can implement your own device distance function. To add this support I had to figure out a lot of problems. One was in the cmake module "FindCUDA.cmake". I reported this [bug](http://www.cmake.org/Bug/view.php?id=14238) (compiler flags are not passed correctly) and it should be corrected soon. With the current cmake version this example will not compile without changes in the named cmake module. But the library is already supporting this.
### 06/2013 Implementation of multi GPU-support for SOMs ###
I recently optimized the implementation of SOMs calculated by GPUs. As a result I was able to reduce the number of matrix operations, which led to a reduced calculation time and probably to a reduced amount of VRAM. Furthermore the multi-GPU support should now work correctly and is activated by default. The calculation is done asynchronously and can work on different devices (for example: GTX 260 and GTX 570). But maybe the support of SM 1.3 will fall in later releases, due to lacking support of function pointers.
### 05/2013 Implementation of a Python Interface ###
Added python bindings. Made SOM-part more suitable for usage of clustering analysis as an alternative for the kmeans-algorithm. The condensed python example below shows what's new: Centroids as result of the clusterization approach (here with four clusters) instead of a whole map of 128x128 nodes as known in the previous [example](https://code.google.com/p/annetgpgpu/wiki/QuickStartGuide). Furthermore I implemented standard python output as binding extension for many container-classes.
```
# .. input definition ..
..
# .. end input definition ..

trainSet = TrainingSet()
trainSet.AddInput(red)
trainSet.AddInput(black)
trainSet.AddInput(white)
trainSet.AddInput(blue)

widthMap = 4
heightMap = 1
inpWidth = 3
inpHeight = 1

SOM = SOMNet(inpWidth,inpHeight,widthMap,heightMap)
SOM.SetTrainingSet(trainSet)
SOM.Training(1000)

centroids = SOM.GetCentroidList()

for i in centroids:
  print i
```
Output:
```
# White
1
1
1
# Red
1
7.24969e-023
7.24969e-023
# Black
9.65357e-018
1.4013e-045
3.06738e-018
# Blue
2.91553e-022
1.4013e-045
1
```
In this example a three-dimensional input was chosen. Possible would be any N-dimensional input vector. As you see, the network is converging very well in this case :)

### 12/2012 Starting implementation of multi-gpu support for SOMs ###
Implemented experimental support for multiple GPUs to calculate SOMs. Unfortunately I have yet no possibility to test it.

### 11/2012 Implemented _adapted_ conscience mechanism for cpu/gpgpu kohonen maps ###
Furthermore I fixed a performance problem while connecting bigger kohonen maps. This was caused by usage of a potentially slower function, inheriting a unnecessary validation. Commit follows ..

### 08/2012 Finished initial implementation of CUDA-Thrust support for back propagation networks ###
Multi layer calculations should work fine on GPU now. Performance difference to CPU will grow higher and higher, when networks becoming bigger. Nevertheless the difference is  smaller in comparison to SOMs, where VRAM access is minimized and calculations are simpler. E.g. the error calculations part of BP nets, which uses a reduction algorithm is on it's own more complex then the complete calculation of SOMs.

### 07/2012 Started implementation of CUDA-Thrust support for back propagation networks ###
At the moment i have somewhat little time (cause of my PhD which hasn't to do at least a bit with programming). ~~Nevertheless it seems to be very likely that `-arch=sm_20` will be needed to build that part of code. This means you need a GTX4/5XX or above.~~ After implementation is finished I try to implement a GPU-accelerated version of a [Neocognitron](http://en.wikipedia.org/wiki/Neocognitron). Hopefully this wouldn't be a very big deal. At the moment it looks like much of the code could get re-cycled for that.

### 06/07/2012 front end _finished_ ###
I recently added a front end for the handling of back propagation neural networks (as you see below). Take a look in the yet not finished [guide](http://code.google.com/p/annetgpgpu/wiki/FrontendGuide) for more screenshots. With the current input/output editor it is possible to create instances of networks and train them. This GUI is thought to assist the design process of networks by testing the so builded topology on the fly. Nevertheless I suggest, that input handling/output handling (e.g.: images, libraries,..) should be done by the backend or a seperate tool (see [quickstart guide](QuickStartGuide.md)), if a special network was found to be able to process the content as expected.
![http://annetgpgpu.googlecode.com/files/Designer.png](http://annetgpgpu.googlecode.com/files/Designer.png)

## Plans for the future ##
  1. Done: ~~Multi GPU support~~
  1. Done: ~~Rework save/load to filesystem to make it modular~~
  1. Done: ~~QGraphicsView based network designer frontend for backpropagation networks~~
  1. Discarded: ~~Usage of linked lists instead of std::vector for handling connections between neurons~~
  1. Done: ~~GPGPU implementation for the other kinds of networks (at least for back propagation networks)~~
  1. Done: ~~Basic python bindings for existing implementation~~

## Contributors ##
At the moment the only constributor is **Daniel** _dgrat_ **Frenzel** <dgdanielf@gmail.com>