#ifndef _SOMKERNELS_
#define _SOMKERNELS_

#include "include/math/Random.h"
#include "include/math/Functions.h"
#include "include/gpgpu/Kernels.h"
#include "include/gpgpu/Functors.h"
#include "include/gpgpu/helper_cuda.h"

#include <cfloat>
#include <cassert>
#include <cmath>

#include <omp.h>

#include <thrust/extrema.h>
#include <thrust/distance.h>
#include <thrust/device_vector.h>

using namespace ANNGPGPU;


// new reference implementation
ANNGPGPU::BMUExport hostGetMin(std::vector<ANNGPGPU::BMUExport> &vec) {
	assert(vec.size() > 0);
	if(vec.size() > 1) {
		std::sort(vec.begin(), vec.end() );
	}
	return *vec.begin();
}

// fast when maps are big
std::pair<float, unsigned int> devGetMin(const thrust::device_vector<float> &vec) {
	thrust::device_vector<float>::const_iterator d_min = thrust::min_element(vec.begin(), vec.end() );
	unsigned int iID = thrust::distance(vec.begin(), d_min);
	return std::pair<float, unsigned int>(*d_min, iID);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Layout of SOMEdgeF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		toNeur1	toNeur1	toNeur1	..
 * ROW2		toNeur2	toNeur2	toNeur2	..
 * ROW3		toNeur3	toNeur3	toNeur3	..
 * ROW(n+1)	..		..		..
 */
BMUExport
hostSOMFindBMNeuronID(std::vector<SplittedNetExport*> &SExp,
		const float &fConscRate)
{
	BMUExport resBMU;
	std::vector<ANNGPGPU::BMUExport> vBMUExp(SExp.size() );

	assert(SExp.size() > 0);
	assert(vBMUExp.size() == SExp.size() );

	omp_set_num_threads(SExp.size() );  							// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 									// for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->f2dEdges.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->f2dEdges.GetH();

		assert(iWidth 	> 0);
		assert(iHeight 	> 0);

		thrust::device_vector<float> dvRes(iWidth, 0.f);

		for(int y = 0; y < static_cast<int>(iHeight); y++) {               
			thrust::transform(SExp.at(iDevID)->f2dEdges.GetRowBegin(y),
				SExp.at(iDevID)->f2dEdges.GetRowEnd(y),
				dvRes.begin(),
				dvRes.begin(),
				spowAmXpY_functor((*SExp.at(iDevID)->dvInput)[y]) );
		}

		if(fConscRate > 0.f) { 								// Implementation of conscience mechanism
			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->dvConscience->begin(),				// input
				dvRes.begin(),							// result
				sXmAmY_functor(1.f/(float)iWidth) );				// functor

			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->dvConscience->begin(),				// input
				SExp.at(iDevID)->dvConscience->begin(),				// result
				sAXmY_functor(fConscRate) );					// functor
		}

		std::pair<float, unsigned int> pCurBMUVal = devGetMin(dvRes);
		BMUExport BMU(pCurBMUVal.first, pCurBMUVal.second, iDevID);
		vBMUExp[iDevID] = BMU;
	}

	resBMU = hostGetMin(vBMUExp);
	checkCudaErrors(cudaSetDevice(resBMU.iDeviceID) );
	resBMU.dvBMUPos = SExp.at(resBMU.iDeviceID)->f2dPositions.GetSubArrayY(resBMU.iBMUID);

	return resBMU;
}

/*
 * Layout of SOMPositionF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		Xpos	Xpos	Xpos	..
 * ROW2		Ypos	Ypos	Ypos	..
 * ROW3		Zpos	Zpos	Zpos	..
 * ROW(n+1)	..		..		..		..
 */
template<typename BinaryFunction>
void hostSOMPropagateBW( std::vector<SplittedNetExport*> &SExp,
		const BMUExport &BMU,
		const float &fSigmaT,
		const float &fLearningRate,
		const BinaryFunction &binaryDistFunc
		)
{
	omp_set_num_threads(SExp.size() );  							// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 									// for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->f2dPositions.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->f2dPositions.GetH();

		thrust::device_vector<float> dvTmp(iWidth, pow(fSigmaT, 2) ); 			// temporary
		thrust::device_vector<float> dvInfl(iWidth, 0.f);
		thrust::device_vector<float> dvDist(iWidth, 0.f);

		// 1. Calc distances for all neurons to BMNeuron: Distance = sqrt(pow(x,2)+pow(y,2)+pow(z,2)+pow(n+1,2) )
		for(int y = 0; y < static_cast<int>(iHeight); y++) { 				// for each coordinate position of the neuron
			thrust::transform(
				SExp.at(iDevID)->f2dPositions.GetRowBegin(y),
				SExp.at(iDevID)->f2dPositions.GetRowEnd(y),
				dvDist.begin(),
				dvDist.begin(),
				spowAmXpY_functor(BMU.dvBMUPos[y]) );
		}

		// 2. Calculate the influence for each neuron
		thrust::transform( dvDist.begin(),						// input
			dvDist.end(), 								// input
			dvInfl.begin(), 							// result
			binaryDistFunc );							// functor

		// 3. Only handle neurons in radius:
		// 3a. Make stencil
		thrust::transform( dvDist.begin(), 						// input
			dvDist.end(),								// input
			dvTmp.begin(),								// input
			dvTmp.begin(), 								// result
			thrust::less<float>() 							// functor
		);
		// 3b. Use stencil to modify only neurons inside the radius
		iWidth 	= SExp.at(iDevID)->f2dEdges.GetW();
		iHeight = SExp.at(iDevID)->f2dEdges.GetH();
		for(int y = 0; y < static_cast<int>(iHeight); y++) {				// for each edge of the neuron
			thrust::transform_if( SExp.at(iDevID)->f2dEdges.GetRowBegin(y),		// input 1
				SExp.at(iDevID)->f2dEdges.GetRowEnd(y), 			// input 1
				dvInfl.begin(),							// input 2
				dvTmp.begin(),							// stencil
				SExp.at(iDevID)->f2dEdges.GetRowBegin(y), 			// result
				hebbian_functor(fLearningRate, (*SExp.at(iDevID)->dvInput)[y]), // functor
				thrust::identity<int>() ); 					// predicate
		}
	}
}

void hostSOMTraining( std::vector<SplittedNetExport*> &SExp,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles,
		const float &fSigma0, 
		const float &fLearningRate0,
		const float &fConscRate,
		const ANN::DistFunction &DistFunc )
{
	float fLambda 	= iCycles / log(fSigma0);

	int iMin 		= 0;
	int iMax 		= InputSet.GetNrElements()-1;
	int iProgCount 		= 1;
	float fSigmaT 		= fSigma0;
	float fLearningRate 	= fLearningRate0;

	for(int i = 0; i < static_cast<int>(iCycles); i++) {
		if(iCycles >= 10) {
			if(((i+1) / (iCycles/10)) == iProgCount && (i+1) % (iCycles/10) == 0) {
				std::cout<<"Current training progress calculated by the GPU is: "<<iProgCount*10.f<<"%/Step="<<i+1<<std::endl;
				iProgCount++;
			}
		} 
		else {
			std::cout<<"Current training progress calculated by the CPU is: "<<(float)(i+1.f)/(float)iCycles*100.f<<"%/Step="<<i+1<<std::endl;
		}

		// Set Input
		std::vector<float> vCurInput = InputSet.GetInput(ANN::RandInt(iMin, iMax) );
		for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
			checkCudaErrors(cudaSetDevice(iDevID) );

			thrust::device_vector<float> *p_dvInputVector = new thrust::device_vector<float>(vCurInput.size() );
			thrust::copy(vCurInput.begin(), vCurInput.end(), p_dvInputVector->begin() );
			SExp[iDevID]->dvInput = p_dvInputVector;
		}

		// Calc fSigmaT if conscience is _not_ used
		fSigmaT 	= DistFunc.rad_decay(fSigma0, i, iCycles); 			// SM 1.3
		fLearningRate 	= DistFunc.lrate_decay(fLearningRate0, i, iCycles); 		// SM 1.3

		// Find BMNeuron 
		BMUExport BMUExp = hostSOMFindBMNeuronID(SExp, fConscRate);
 
		// Propagate BW SM 2.0
#if __CUDA_ARCH__ >= 200
#warning Compiling with SM 2.0 or higher.
		external_device_func_t *dvFuncPtr = NULL;
		*dvFuncPtr 	= DistFunc.distance; 						// SM 2.0
		assert(dvFuncPtr != NULL);
		hostSOMPropagateBW( SExp,
			BMUExp,
			fSigmaT,
			fLearningRate,
			sm20distance_functor(fSigmaT, dvFuncPtr));
#else //__CUDA_ARCH__ < 200
#warning Compiling with SM 1.3 or less.
		// Propagate BW SM 1.3
		if (strcmp (DistFunc.name, "gaussian") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,							// const
					fSigmaT,						// const
					fLearningRate,						// const
					sm13gaussian_functor(fSigmaT)); 			// const
		} else if (strcmp (DistFunc.name, "mexican") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,							// const
					fSigmaT,						// const
					fLearningRate,						// const
					sm13mexican_functor(fSigmaT)); 				// const
		} else if (strcmp (DistFunc.name, "bubble") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,							// const
					fSigmaT,						// const
					fLearningRate,						// const
					sm13bubble_functor(fSigmaT)); 				// const
		} else if (strcmp (DistFunc.name, "cutgaussian") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,							// const
					fSigmaT,						// const
					fLearningRate,						// const
					sm13cut_gaussian_functor(fSigmaT)); 			// const
		} else if (strcmp (DistFunc.name, "epanechicov") == 0) {
			hostSOMPropagateBW( SExp,
					BMUExp,							// const
					fSigmaT,						// const
					fLearningRate,						// const
					sm13epanechicov_functor(fSigmaT)); 			// const
		}
#endif
	}
}

#endif
