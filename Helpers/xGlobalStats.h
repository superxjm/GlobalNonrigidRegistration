#pragma once

enum GeoRegistrationType
{
	POINT_TO_POINT,
	POINT_TO_PLAIN,
	BOTH,
	NO_GEOREGISTATION
};

enum PhotoRegistrationType
{
	PHOTOREGISTATION,
	NO_PHOTOREGISTATION
};

enum DescriptorType
{
	SIFT,
	GMS,
	NO_DESCRIPTOR
};
enum GeoMatchingType
{
	KNN,
	PERSPECTIVE,
	KNNWITHCONSISTANTCHECK,
	PERSPECTIVEFORMVIRTUAL,
	NO_GEOMATCHING
};

namespace gs
{
	static GeoRegistrationType geoRegistrationType = POINT_TO_PLAIN;
	static PhotoRegistrationType photoRegistrationType = PHOTOREGISTATION;
	static DescriptorType descriptorType = NO_DESCRIPTOR;// GMS;
	static GeoMatchingType geoMatchingType = PERSPECTIVE;

	static float weightScale = 100.0;
	static float weightGeo = 1.3		* weightScale;
	static float weightPhoto = 0.00001 * weightScale;
	//static float weightPhoto = 0.00001	* weightScale;
	//static float weightReg = 20.0f	* weightScale;//10.0	* weightScale;
	static float weightReg = 3.0f	* weightScale;//10.0	* weightScale;
	//static float weightReg = 0.0f	* weightScale;//10.0	* weightScale;
	//static float weightRot = 0.0000f;//	*weightScale;
	//static float weightRot = 10.0	* weightScale;
	static float weightRot = 3.0f	* weightScale;
	static float weightTrans = 0.0000f	* weightScale;

#if 0
	static float weightScale =	1000.0f;
	static float weightGeo =	0.1f		* weightScale;
	static float weightPhoto =	0.00001f	* weightScale;
	static float weightReg =	10.0f	* weightScale;
	static float weightRot =	10.0f	* weightScale;
	static float weightTrans =	0.0000f	* weightScale;
#endif
}