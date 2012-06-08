#ifdef __CUDACC__
	#include <cuda.h>
#else
	#include <iostream>
	#include <math/ANFunctions.h>

	using namespace ANN;

	/*
	 * BP
	 */
	const TransfFunction
	Functions::fcn_tanh = {
		(char*)"tanh",
		fcn_tanh_normal,
		fcn_tanh_derivate
	};

	const TransfFunction
	Functions::fcn_log = {
		(char*)"log",
		fcn_log_normal,
		fcn_log_derivate
	};

	const TransfFunction
	Functions::fcn_linear = {
		(char*)"linear",
		fcn_linear_normal,
		fcn_linear_derivate
	};

	const TransfFunction
	Functions::fcn_binary = {
		(char*)"binary",
		fcn_binary_normal,
		fcn_binary_derivate
	};

	/*
	 * SOM
	 */
	const DistFunction
	Functions::fcn_gaussian = {
		(char*)"gaussian",
		fcn_gaussian_bell,
		fcn_decay
	};

	const DistFunction
	Functions::fcn_mexican = {
		(char*)"mexican",
		fcn_mexican_hat,
		fcn_decay
	};

	const TransfFunction*
	Functions::ResolveTransfFByName (const char *name) {
		if (strcmp (name, "tanh") == 0)
			return (&fcn_tanh);
		if (strcmp (name, "log") == 0)
			return (&fcn_log);
		if (strcmp (name, "linear") == 0)
			return (&fcn_linear);
		if (strcmp (name, "binary") == 0)
			return (&fcn_binary);
		return (NULL);
	}

	const DistFunction*
	Functions::ResolveDistFByName (const char *name) {
		if (strcmp (name, "gaussian") == 0)
			return (&fcn_gaussian);
		if (strcmp (name, "mexican") == 0)
			return (&fcn_mexican);
		return (NULL);
	}
#endif
