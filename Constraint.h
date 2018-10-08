#pragma once

#include <thrust/device_vector.h>

#include "InputData.h"

class GNSolver;

class Constraint
{
public:
	Constraint(): m_gnSolver{nullptr}, m_weight{0}, m_inputData{nullptr}
	{
	};

	virtual ~Constraint()
	{
	}

	virtual char* ctype() { return "constraint_base"; }
	virtual bool init(GNSolver* gnSolver, float weight) = 0;
	virtual bool init() = 0;
	virtual void setWeight(float weight) { m_weight = sqrtf(weight); }

	virtual float val(float3* dVars)
	{
		b(dVars);
		thrust::multiplies<float> op;
		thrust::transform(m_dB.begin(), m_dB.end(), m_dB.begin(), m_dB.begin(), op);
		return sqrtf(thrust::reduce(m_dB.begin(), m_dB.end()));
	}

	void directiveJTJ(float* dJTJ_a, int* dJTJ_ia)
	{
	}

	void directiveJTb(thrust::device_vector<float>& dJTb)
	{
	}

	virtual void b(float3* dVars) = 0;
	virtual void getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars) = 0;

	GNSolver* m_gnSolver;
	float m_weight;
	thrust::device_vector<float> m_dB;
	InputData* m_inputData;
	int m_iter;
};

