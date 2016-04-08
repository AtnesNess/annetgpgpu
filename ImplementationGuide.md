# Introduction #
Usually a network consists of nodes and edges. To deal with all these notes it was useful to implement a container class too. Most implementations of neural networks store their data in a plain array. Calculation is simple with this approach, but asymmetric networks get hard to implement and new functionality is hard to add.

To create more complex variants of networks it makes sense to put the information of the network in the edges and the functionality to calculate parts of the network into the nodes. This makes it easier to include new functionality and to re-use old code.

Because information flow in neuronal networks is often directed, container classes help to keep an order. The network class on the other hand calls learning or training functions and implements the principal learning procedure, e. g. switching training patterns or break the learning procedure if a certain error limit was hit.

To make the long story short, the three important classes to derive from are:
  1. `AbsNeuron`
  1. `AbsLayer`
  1. `AbsNet`

## 1.) `AbsNeuron` ##
To decrease the probability that someone forgets to overload some important functions i declared some functions as pure abstract. So you would get an compiler error if you derive from my abstract classes without implementing the proper functions. This method approach also makes it possible the implement a new neuron class and use it with already implemented layer or network classes.
```
virtual void AdaptEdges() 	= 0;

virtual void CalcValue() 	= 0;
```
It doesn't make sense to implement them in every case (e. g. in Hopfield networks). So you keep the functions not needed empty.
```
void HFNeuron::AdaptEdges() {
}
```
In other cases (e. g. back propagation networks) things are a little bit different and functionality could get stored out very well.
In `CalcValue()` you have to implement the algorithm to calculate the data you want to store in a certain neuron in the network.
Every neuron (or node) in the network has a list of edges which direct to neurons of another (or the same) layer. This example shows you how to run through this list to implement a neuron in a back propagation network.
```
void BPNeuron::CalcValue() {
	if(GetConsI().size() == 0)
		return;

	// bias neuron
	float fBias = 0.f;
	SetValue( 0.f );
	if(GetBiasEdge() ) {
		fBias = GetBiasEdge()->GetValue();
		SetValue(fBias);
	}

	// sum from product of all incoming neurons with their weights
	AbsNeuron *from;
	for(unsigned int i = 0; i < GetConsI().size(); i++) {
		from = GetConI(i)->GetDestination(this);
		SetValue(GetValue() + (from->GetValue() * GetConI(i)->GetValue()));
	}

	float fVal = GetNetFunction()->normal( GetValue(), fBias );
	SetValue(fVal);
}
```
It is a common property of neuronal networks, that they are NOT static. So it makes sense to change it. The algorithm to do this is implemented in `AdaptEdges()`. Again we use the internal list to run through all edges (outgoing ones) the neuron is connected with.
```
void BPNeuron::AdaptEdges() {
	if(GetConsO().size() == 0)
		return;

	AbsNeuron *pCurNeuron;
	Edge 		*pCurEdge;
	float 		fVal;

	// calc error deltas
	fVal = GetErrorDelta();
	for(unsigned int i = 0; i < GetConsO().size(); i++) {
		pCurNeuron = GetConO(i)->GetDestination(this);
		fVal += pCurNeuron->GetErrorDelta() * GetConO(i)->GetValue();
	}
	fVal *= GetNetFunction()->derivate( GetValue(), 0.f );
	SetErrorDelta(fVal);

	// adapt weights
	for(unsigned int i = 0; i < GetConsO().size(); i++) {
		pCurEdge = GetConO(i);
		if(pCurEdge->GetAdaptationState() == true) {
			fVal = 0.f;	// delta for momentum
			// stdard backpropagation
			fVal += pCurEdge->GetDestination(this)->GetErrorDelta() * m_fLearningRate * GetValue()
			// weight decay term
			- m_fWeightDecay * pCurEdge->GetValue()
			// momentum term
			+ m_fMomentum * pCurEdge->GetMomentum();

			pCurEdge->SetMomentum( fVal );
			pCurEdge->SetValue( fVal+pCurEdge->GetValue() );
		}
	}
}
```

## 2.) `AbsLayer` ##

The next bigger thing to reimplement is the `AbsLayer` class. This would only be necessary if you decide not to use the other implementations of this abstract class. If you decide to write your own one, then you have to implement the `Resize()` function. This could be useful especially if you have strange layer topologies (e.g. 2 dimensional or 3 dimensional ones) or you implemented your own Neuron-class previously.
```
virtual void Resize(const unsigned int &iSize) = 0;
```
This is the proper implementation example.
```
void BPLayer::Resize(const unsigned int &iSize) {
	EraseAll();
	for(unsigned int i = 0; i < iSize; i++) {
		AbsNeuron *pNeuron = new BPNeuron(this);
		pNeuron->SetID(i);
		m_lNeurons.push_back(pNeuron);
	}
}
```

## 3.) `AbsNet` ##
The last class you may want to derive from is `AbsNet`. Here are three functions you have to overload. Usually one function is used to calculate the output of the network: `PropagateFW()` and one is used to change the network (e. g. the edges): `PropagateBW()`. The last one is needed for the export functionality of net to the filesystem. This function is protected and you shouldn't deal much with it.
```
virtual void PropagateFW() = 0;

virtual void PropagateBW() = 0;

virtual void AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) = 0;
```
Here I show you how to implement these functions in a back propagation network. Also you see at this point how nice this implementation looks if you store the algorithm in the nodes of the network.
```
void BPNet::PropagateFW() {
	for(unsigned int i = 1; i < m_lLayers.size(); i++) {
		BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
		#pragma omp parallel for
		for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->CalcValue();
		}
	}
}

void BPNet::PropagateBW() {
	for(int i = m_lLayers.size()-1; i >= 0; i--) {
		BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
		#pragma omp parallel for
		for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->AdaptEdges();
		}

		#pragma omp parallel
		if(curLayer->GetBiasNeuron() != NULL) {
			curLayer->GetBiasNeuron()->AdaptEdges();
		}
	}
}
```
Different Implementations use different types of layers. This is why it is necessary to overload the pure virtual function `AddLayer()` which gets called by `AbsNet::CreateNet()`. You need to tell the network which type of layer it has to use. This function is used for the import functionality of the net from the filesystem.
```
void BPNet::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	AbsNet::AddLayer( new BPLayer(iSize, flType) );
}
```
# Save custom models to filesystem #
## Back propagation networks ##
Sometimes you may want to create some kinds of data types which add some fancy kind of features. The standard implementation saves _all_ the content of the classes you derive from. Nevertheless if you decide to add features you may want to overload the `ExpToFS()` and `ImpFromFS()` functions as well. Calling the virtual base ensures to save the base content of the class you derived from.
The following example shows you how easy it is to add support for a bias neuron, which has to get handled a little bit different by the network. At first the save function to export the dcontent of the neural net to the filesystem. I prefer to use bzip2 compression, because bigger nets may alloc much space.
```
void BPLayer::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	std::cout<<"Save BPLayer to FS()"<<std::endl;
	AbsLayer::ExpToFS(bz2out, iBZ2Error);

	unsigned int iNmbOfConnects 	= 0;
	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;

	bool bHasBias = false;
	(GetBiasNeuron() == NULL) ? bHasBias = false : bHasBias = true;
	BZ2_bzWrite( &iBZ2Error, bz2out, &bHasBias, sizeof(bool) );

	if(bHasBias) {
		AbsNeuron *pCurNeur = GetBiasNeuron();
		iNmbOfConnects = pCurNeur->GetConsO().size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
		for(unsigned int k = 0; k < iNmbOfConnects; k++) {
			Edge *pCurEdge = pCurNeur->GetConO(k);
			iDstLayerID = pCurEdge->GetDestination(pCurNeur)->GetParent()->GetID();
			iDstNeurID = pCurEdge->GetDestinationID(pCurNeur);
			fEdgeValue = pCurEdge->GetValue();
			BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
			BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
			BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(float) );
		}
	}
}
```
Now the other way round, we load the content from the filesystem..
```
int BPLayer::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table) {
	std::cout<<"Load BPLayer from FS()"<<std::endl;
	int iLayerID = AbsLayer::ImpFromFS(bz2in, iBZ2Error, Table);

	unsigned int iNmbOfConnects 	= 0;
	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;

	bool bHasBias = false;

	BZ2_bzRead( &iBZ2Error, bz2in, &bHasBias, sizeof(bool) );

	if(bHasBias) {
		BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
		for(unsigned int j = 0; j < iNmbOfConnects; j++) {
			BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
			BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
			BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(float) );
			ConDescr	cCurCon;
			cCurCon.m_fVal 			= fEdgeValue;
			cCurCon.m_iDstNeurID 	= iDstNeurID;
			cCurCon.m_iSrcLayerID 	= iLayerID;
			cCurCon.m_iDstLayerID 	= iDstLayerID;
			Table.BiasCons.push_back(cCurCon);
		}
	}

	return iLayerID;
}
```

Last thing is to overload the `CreateNet()` function. Here  the content loaded from the filesystem is used to create a copy of the net in the RAM. The base implementation creates the layers and the connections of the network, so we yust have to implement the bias neuron. Here the complete example of `CreateNet(const ConTable &Net)`:
```
std::cout<<"Create BPNet"<<std::endl;

/*
 * Init
 */
unsigned int iDstNeurID 	= 0;
unsigned int iDstLayerID 	= 0;
unsigned int iSrcLayerID 	= 0;

float fEdgeValue		= 0.f;

AbsLayer *pDstLayer 		= NULL;
AbsLayer *pSrcLayer 		= NULL;
AbsNeuron *pDstNeur 		= NULL;
AbsNeuron *pSrcNeur 		= NULL;
```
Then we call the base implementation to connect the layers:
```
/*
 * For all nets necessary: Create Connections (Edges)
 */
AbsNet::CreateNet(Net);
```
Then we handle our special bias neuron which we previously added to our back propagation network:
```
/*
 * Only for back propagation networks
 */
if(Net.NetType == ANNetBP) {
	for(unsigned int i = 0; i < Net.BiasCons.size(); i++) {
		iDstNeurID = Net.BiasCons.at(i).m_iDstNeurID;
		iDstLayerID = Net.BiasCons.at(i).m_iDstLayerID;
		iSrcLayerID = Net.BiasCons.at(i).m_iSrcLayerID;
		if(iDstNeurID < 0 || iDstLayerID < 0 || GetLayers().size() < iDstLayerID || GetLayers().size() < iSrcLayerID) {
			return;
		}
			else {
			fEdgeValue 	= Net.BiasCons.at(i).m_fVal;

			pDstLayer 	= ( (BPLayer*)GetLayer(iDstLayerID) );
			pSrcLayer 	= ( (BPLayer*)GetLayer(iSrcLayerID) );
			pSrcNeur 	= ( (BPLayer*)pSrcLayer)->GetBiasNeuron();

			pDstNeur 	= pDstLayer->GetNeuron(iDstNeurID);
			Connect(pSrcNeur, pDstNeur, fEdgeValue, 0.f, true);
		}
	}
}
```
## Self organizing maps ##
Here is another example for SOMs. Only the import of the positions has to be added.
```
std::cout<<"Create SOMNet"<<std::endl;

/*
 * For all nets necessary: Create Connections (Edges)
 */
AbsNet::CreateNet(Net);

/*
 * Set Positions
 */
for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
	int iLayerID 	= Net.Neurons.at(i).m_iLayerID;
	int iNeurID 	= Net.Neurons.at(i).m_iNeurID;
	std::vector<float> vPos = Net.Neurons.at(i).m_vPos;

	GetLayer(iLayerID)->GetNeuron(iNeurID)->SetPosition(vPos);
}
```
## Overloading the net description struct ##
The transient struct which stores the network could get extended by overloading. You are free to extend it for your needs and use it with the functions shown in this guide.

```
struct ConTable {
	NetTypeFlag 			NetType;
	unsigned int 			NrOfLayers;

	std::vector<unsigned int> 	SizeOfLayer;
	std::vector<LayerTypeFlag> 	TypeOfLayer;

	std::vector<NeurDescr> 		Neurons;

	std::vector<ConDescr> 		BiasCons;
	std::vector<ConDescr> 		NeurCons;
};
```