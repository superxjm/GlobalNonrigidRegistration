#pragma once

#include "Constraint.h"

class SmoothConstraint : public Constraint {
public:
	SmoothConstraint()
	{
	}

	~SmoothConstraint()
	{
	}

	char *ctype() override { return "smooth"; }
	bool init(GNSolver *gnSolver, float weight) override;
	bool init() override;
	void b(float3* dVars) override;
	void directiveJTJ(float *JTJ_a, int *JTJ_ia);
	void directiveJTb(thrust::device_vector<float> &JTb);

	void getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars) override;
};
 