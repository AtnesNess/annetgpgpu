/*
 * BPLayer.cpp
 *
 *  Created on: 02.06.2009
 *      Author: Xerces
 */

#include <cassert>
//own classes
#include <math/ANFunctions.h>
#include <basic/ANEdge.h>
#include <basic/ANAbsNeuron.h>
#include <ANBPNeuron.h>
#include <ANBPLayer.h>

#include <containers/ANConTable.h>

using namespace ANN;


BPLayer::BPLayer() {
	m_pBiasNeuron = NULL;
}

BPLayer::BPLayer(const BPLayer *pLayer) {
	int iNumber 			= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();
	m_pBiasNeuron 			= NULL;

	Resize(iNumber);
	SetFlag(fType);
}

BPLayer::BPLayer(const unsigned int &iNumber, LayerTypeFlag fType) {
	Resize(iNumber);
	m_pBiasNeuron = NULL;
	SetFlag(fType);
}

BPLayer::~BPLayer() {
	if(m_pBiasNeuron) {
		delete m_pBiasNeuron;
	}
}

void BPLayer::Resize(const unsigned int &iSize) {
	EraseAll();
	for(unsigned int i = 0; i < iSize; i++) {
		AbsNeuron *pNeuron = new BPNeuron(this);
		pNeuron->SetID(i);
		m_lNeurons.push_back(pNeuron);
	}
}

void BPLayer::ConnectLayer(AbsLayer *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron *pSrcNeuron;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "destLayer"
	 */
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		pSrcNeuron = m_lNeurons[i];
		Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
	}

	if(m_pBiasNeuron) {
		Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

void BPLayer::ConnectLayer(
		AbsLayer *pDestLayer,
		std::vector<std::vector<int> > Connections,
		const bool bAllowAdapt)
{
	AbsNeuron *pSrcNeuron;

	assert( Connections.size() != m_lNeurons.size() );
	for(unsigned int i = 0; i < Connections.size(); i++) {
		std::vector<int> subArray = Connections.at(i);
		pSrcNeuron = GetNeuron(i);
		assert(i != pSrcNeuron->GetID() );

		for(unsigned int j = 0; j < subArray.size(); j++) {
			assert( j < pDestLayer->GetNeurons().size() );
			AbsNeuron *pDestNeuron = pDestLayer->GetNeuron(j);
			assert( j < pDestNeuron->GetID() );
			Connect(pSrcNeuron, pDestNeuron, bAllowAdapt);
		}
	}

	if(m_pBiasNeuron) {
		Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

BPNeuron *BPLayer::GetBiasNeuron() const {
	return m_pBiasNeuron;
}

void BPLayer::SetFlag(const LayerTypeFlag &fType) {
	m_fTypeFlag = fType;
	if( (m_fTypeFlag & ANBiasNeuron) && m_pBiasNeuron == NULL ) {
		m_pBiasNeuron = new BPNeuron(this);
		m_pBiasNeuron->SetValue(1.0f);
	}
}

void BPLayer::AddFlag(const LayerTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
	m_fTypeFlag |= fType;
	if( (m_fTypeFlag & ANBiasNeuron) && m_pBiasNeuron == NULL ) {
		m_pBiasNeuron = new BPNeuron(this);
		m_pBiasNeuron->SetValue(1.0f);
	}
}

void BPLayer::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetLearningRate(fVal);
	}
}

void BPLayer::SetMomentum(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetMomentum(fVal);
	}
}

void BPLayer::SetWeightDecay(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetWeightDecay(fVal);
	}
}

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

/*
 * AUSGABEOPERATOR
 * OSTREAM
 */
std::ostream& operator << (std::ostream &os, BPLayer &op)
{
	if(op.GetBiasNeuron() != 0)
	os << "Bias neuron: \t" << op.GetBiasNeuron()->GetValue() 	<< std::endl;
    os << "Nr. neurons: \t" << op.GetNeurons().size() 					<< std::endl;
    return os;     // Ref. auf Stream
}

std::ostream& operator << (std::ostream &os, BPLayer *op)
{
	if(op->GetBiasNeuron() != 0)
	os << "Bias neuron: \t" << op->GetBiasNeuron()->GetValue()	<< std::endl;
    os << "Nr. neurons: \t" << op->GetNeurons().size() 					<< std::endl;
    return os;     // Ref. auf Stream
}
