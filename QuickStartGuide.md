# Introduction #

With this library it is very easy to create new kinds or modify existing types of nets to process data.
One example to use this small library is to implement neuronal nets.
Some types of neuronal nets are already implemented and could be overloaded to process specific problems. The goal was to keep the library as abstract as possible to achieve a maximum of independence from the topology of the networks.

# Backpropagation networks #
The first I do in this example is to define a desired input and/or output, saved into an array. std::vectors are supported too.
```
float fInp1[3];
fInp1[0] = 0;
fInp1[1] = 0;
fInp1[2] = 0;
float fInp2[3];
fInp2[0] = 0;
fInp2[1] = 1;
fInp2[2] = 0;

float fOut1[6];
fOut1[0] = 0.1;
fOut1[1] = 0.2;
fOut1[2] = 0.3;
fOut1[3] = 0.4;
fOut1[4] = 0.5;
fOut1[5] = 0.6;
float fOut2[6];
fOut2[0] = 0;
fOut2[1] = 1;
fOut2[2] = 0;
fOut2[3] = 0;
fOut2[4] = 0;
fOut2[5] = 0;
```
After that, we put the content into a proper data structure.
```
ANN::TrainingSet input;
input.AddInput(fInp1, 3);
input.AddOutput(fOut1, 6);
input.AddInput(fInp2, 3);
input.AddOutput(fOut2, 6);
```

Then we define the sizes of the layers of the network.
```
ANN::BPNet net;
ANN::BPLayer layer1(3, ANN::ANLayerInput);
layer1.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer2(64, ANN::ANLayerHidden);
layer2.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer3(64, ANN::ANLayerHidden);
layer3.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer4(64, ANN::ANLayerHidden);
layer4.AddFlag(ANN::ANBiasNeuron);
ANN::BPLayer layer5(6, ANN::ANLayerOutput);
```
Every neuron could get connected with every neuron of another layer simply by calling the following functions. Differnt connection patterns are supported too by usage of std::vector as function argument. Then you have to adress the ID of each source and destination neuron.
```
layer1.ConnectLayer(&layer2);
layer2.ConnectLayer(&layer3);
layer3.ConnectLayer(&layer4);
layer4.ConnectLayer(&layer5);
```
Now we are ready to add all layers to the network. This is necessary, because the algorithms to process the network is defined there.
```
net.AddLayer(&layer1);
net.AddLayer(&layer2);
net.AddLayer(&layer3);
net.AddLayer(&layer4);
net.AddLayer(&layer5);
```
One of the last steps is to modify some settings. This isn't necessary but could influence the later results.
```
std::vector<float> errors;
net.SetLearningRate(0.2);
net.SetMomentum(0.9);
net.SetWeightDecay(0);
net.SetTrainingSet(input);
```
After that it is possible to train the network, save it to filesystem or load it from there. Furthermore you are able to plot the learning curve stored in "errors".
```
std::vector<float> errors = net.TrainFromData(5000, 0.001);
std::cout<< net <<std::endl;

net.ExpToFS("foo.bar");
net.ImpFromFS("foo.bar");
net.SetTrainingSet(input);

std::cout<< net <<std::endl;
```
# Self organizing maps #
Again the first we do in this example is to define a desired demo input or output and save it into an array. To demonstrate, we use std::vectors instead of plain arrays.
```
std::vector<float> red, green, blue, yellow, orange;

red.push_back(1);
red.push_back(0);
red.push_back(0);

green.push_back(0);
green.push_back(1);
green.push_back(0);

blue.push_back(0);
blue.push_back(0);
blue.push_back(1);

yellow.push_back(1);
yellow.push_back(1);
yellow.push_back(0.2);

orange.push_back(1);
orange.push_back(0.4);
orange.push_back(0.25);

```
Like in backpropagation networks we put the content into a proper data structure.
```
ANN::TrainingSet input;
input.AddInput(red);
input.AddInput(green);
input.AddInput(blue);
input.AddInput(yellow);
input.AddInput(orange);
```
It is possible to use CPU or GPU for processing SOMs. GPU is 100X and more quicker. Here i demonstrate both ways by usage of the copy constructor. "3" and "1" are the dimensions of the input. "128X128" is the size of our map.
```
ANN::SOMNet cpu;
cpu.SetTrainingSet(input);
cpu.CreateSOM(3, 1, 128, 128);

ANN::SOMNetGPU gpu(&cpu);
```
Training is done by calling the calling the pre-implemented Training function. GPU is usually quicker even if you want to train much more times...
```
cpu.Training(20);
gpu.Training(1000);
```

The last thing is, that we may want to save our result. I implemented a very basic Qt-tool to show the way you may create your own ones.
```
SOMReader w(128, 128, 2);
for(int x = 0; x < 128*128; x++) {
	ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
	vCol[0] = pNeur->GetConI(0)->GetValue();
	vCol[1] = pNeur->GetConI(1)->GetValue();
	vCol[2] = pNeur->GetConI(2)->GetValue();

	w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
}
w.Save("GPU.png");
```

| **A** | **B** |
|:------|:------|
| ![http://annetgpgpu.googlecode.com/files/CPU.png](http://annetgpgpu.googlecode.com/files/CPU.png) | ![http://annetgpgpu.googlecode.com/files/GPU.png](http://annetgpgpu.googlecode.com/files/GPU.png) |

**Fig. 1:** These pictures show the training result with some more colors. The calculation with the CPU is much slower so less iterations could be done in the same time. **A)** This is what you obtain after 9 cycles calculated by the CPU. The picture still looks like the random noise pattern after initialization. **B)** In the same time you would be able to calculate over 1000 cycles with the GPU. The result looks pretty fine. Colors got classified by the network and often similiar colors lie next to each other.

# Hopfield networks #
Hopfield networks are a little bit differnt to implement. I made this implementation to demonstrate how it could work if you want to do something similar and show how to use the existing implementation.
At first we define _again_ our samples and store them into the known structure.
```
float TR[16];
TR[0] 	= -1;
TR[1] 	= 1;
TR[2] 	= -1;
TR[3] 	= 1;
TR[4] 	= -1;
TR[5] 	= 1;
TR[6] 	= -1;
TR[7] 	= 1;
TR[8] 	= -1;
TR[9] 	= 1;
TR[10] 	= -1;
TR[11] 	= 1;
TR[12] 	= -1;
TR[13] 	= 1;
TR[14] 	= -1;
TR[15] 	= 1;

float fInpHF[16];
fInpHF[0] = 1;
fInpHF[1] = -1;
fInpHF[2] = -1;
fInpHF[3] = 1;
fInpHF[4] = -1;
fInpHF[5] = 1;
fInpHF[6] = -1;
fInpHF[7] = 1;
fInpHF[8] = -1;
fInpHF[9] = 1;
fInpHF[10]= 1;
fInpHF[11]= 1;
fInpHF[12]= -1;
fInpHF[13]= 1;
fInpHF[14]= -1;
fInpHF[15]= 1;

ANN::TrainingSet input;
input.AddInput(TR, 16);
```
Then we create the network and train it. A typical training as known from the other types of networks is not necessary, because the perfect learning matrix could be calculated directly. So I did NOT defined a "training" function and yust overloaded PropagateBW() as demonstrated in the [implementation guide](ImplementationGuide.md).
```
ANN::HFNet hfnet;
hfnet.Resize(16,1);
hfnet.SetTrainingSet(input);
hfnet.PropagateBW();
```

Output for a specific sample could be achieved by doing something like that or overload the std::ostream operator:
```
hfnet.SetInput(fInpHF);
for(int k = 0; k < 1; k++) {
	hfnet.PropagateFW();

	for(int k = 0; k < hfnet.GetOutput().size(); k++) {
		std::cout<<"outp: "<<hfnet.GetOutput().at(k)<<std::endl;
	}
	std::cout<<std::endl;
}
```