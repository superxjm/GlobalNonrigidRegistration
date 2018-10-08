#pragma once

#include "Constraint.h"

class PhotoConstraint : public Constraint {
public:
	PhotoConstraint()
	{
	}

	~PhotoConstraint()
	{
	}

    char *ctype() override { return "data_photo"; }
    bool init(GNSolver *gnSolver, float weight) override;
	bool init() override;
	void b(float3* dVars) override;
	void directiveJTJ(float *dJTJ_a, int *dJTJ_ia);
	void directiveJTb(thrust::device_vector<float> &JTb);

	void getJTJAndJTb(float* dJTJ_a, int* dJTJ_ia, thrust::device_vector<float>& dJTb, float3* dVars) override;
};


