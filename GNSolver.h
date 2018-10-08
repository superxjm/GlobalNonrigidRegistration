#pragma once

#include "SparseMatrix.h"
#include "Helpers/xUtils.h"

class LinearSolver;
class PcgLinearSolver;
class Constraint;

struct SolverPara
{
	float m_dataConsWeight1;
	bool m_hasDataCons1;

	float m_dataConsWeight2;
	bool m_hasDataCons2;

	float m_smoothConsWeight;
	bool m_hasSmoothCons;

	float m_rotConsWeight;
	bool m_hasRotCons;

	SolverPara(bool hasDataCons1, float dataConsWeight1,
		bool hasDataCons2, float dataConsWeight2,
		bool hasRegCons1, float regConsWeight1,
		bool hasRegCons2, float regConsWeight2)
	{
		m_dataConsWeight1 = dataConsWeight1;
		m_hasDataCons1 = hasDataCons1;

		m_smoothConsWeight = regConsWeight1;
		m_hasSmoothCons = hasRegCons1;

		m_rotConsWeight = regConsWeight2;
		m_hasRotCons = hasRegCons2;

		m_hasDataCons2 = hasDataCons2;
		m_dataConsWeight2 = dataConsWeight2;
	}
};

class InputData;

class GNSolver
{
public:
	GNSolver(InputData* inputData, const SolverPara &para);
	~GNSolver();

	bool initVars();
	void initJtj();
	bool initCons(const SolverPara &para);
	bool initCons();
	bool next(int iter);
	void updateVboVec(VBOType* dVboCuda);
	void updateCameraPoses();

	InputData* m_inputData;
	PcgLinearSolver* m_pcgLinearSolver;

	std::vector<Constraint *> m_cons;
	SparseMatrixCsrGpu m_dJTJ;
	thrust::device_vector<float> m_dJTb;
	thrust::device_vector<float> m_dDelta;
	thrust::device_vector<float> m_dVars;
	thrust::device_vector<float> m_dRsTransInv;
	thrust::device_vector<float> m_dPreconditioner;
	int m_varsNum;
	int m_maxVarNum;
	int m_iter;
	float m_residual;
	thrust::device_vector<float> m_dInitVarPatternVec;
	thrust::device_vector<float> m_dCameraToNodeWeightVec;

	double m_residualBeforeOpt;
	double m_residualAfterOpt;
};


