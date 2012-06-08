/*
 * Matrix.h
 *
 *  Created on: 05.04.2012
 *      Author: dgrat
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <thrust/device_vector.h>
#include <cassert>


namespace ANN {

/*
 * Host classes
 */
class Matrix : public thrust::device_vector<float> {
private:
	unsigned int iWidth;
	unsigned int iHeight;

public:
	Matrix();
	Matrix(unsigned int width, unsigned int height, float val);
	Matrix(unsigned int width, unsigned int height, thrust::host_vector<float> vec);

	thrust::device_vector<float> GetCol(const unsigned int x) const;

	iterator getRowBegin(const unsigned int &y) {
		assert(y < iHeight);
		return begin()+y*iWidth;
	}
	iterator getRowEnd(const unsigned int &y) {
		assert(y < iHeight);
		return begin()+y*iWidth+iWidth;
	}

	const_iterator getRowBegin(const unsigned int &y) const {
		assert(y < iHeight);
		return begin()+y*iWidth;
	}
	const_iterator getRowEnd(const unsigned int &y) const {
		assert(y < iHeight);
		return begin()+y*iWidth+iWidth;
	}

	unsigned int GetW() const {
		return iWidth;
	}
	unsigned int GetH() const {
		return iHeight;
	}
};

}

#endif /* MATRIX_H_ */