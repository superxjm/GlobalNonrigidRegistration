#include "wUtils.h"

__global__ void CreateLocalVertexFromDepthKernel(
	int _max_num,
	float* _d_depth_map,
	float4* _d_local_vertex_map,
	float4* _d_camera_fxfycxcx,
	int _width,
	int _height,
	int _camera_num)
{

	int th = blockDim.x * blockIdx.x + threadIdx.x;

	if (th >= _max_num) return;

	int camera_id = th / (_width*_height);
	int u = th % (_width*_height) % _width;
	int v = th % (_width*_height) / _width;
	float z = _d_depth_map[th];
	float fx_inv = 1.0f / _d_camera_fxfycxcx[camera_id].x;
	float fy_inv = 1.0f / _d_camera_fxfycxcx[camera_id].y;
	float cx = _d_camera_fxfycxcx[camera_id].z;
	float cy = _d_camera_fxfycxcx[camera_id].w;

	if (z != 0)
	{
		float vx = z * (u - cx) * fx_inv;
		float vy = z * (v - cy) * fy_inv;
		float vz = z;

		_d_local_vertex_map[th].x = vx;
		_d_local_vertex_map[th].y = vy;
		_d_local_vertex_map[th].z = vz;
		_d_local_vertex_map[th].w = 1.0f;
	}
	else
	{
		_d_local_vertex_map[th].x = __int_as_float(0x7fffffff); //CUDART_NAN_F
	}
}

__global__ void CreataLocalNormalFromLocalVertexKernel(
	int _max_num, 
	float4* _d_local_vertex_map,
	float4* _d_local_normal_map, 
	int *_d_index_map,
	int _width,
	int _height,
	int _camera_num)
{
	int th = blockDim.x * blockIdx.x + threadIdx.x;

	if (th >= _max_num) return;

	int camera_id = th / (_width*_height);
	int offset = camera_id*_width*_height;
	int u = th % (_width*_height) % _width;
	int v = th % (_width*_height) / _width;

	_d_index_map[offset + v*_width + u] = -1;
	_d_local_normal_map[offset + v*_width + u].x = __int_as_float(0x7fffffff); //CUDART_NAN_F

	int range = 1;

	if (u >= _width - range || v >= _height - range || u < range || v < range)
	{
		return;
	}

	float3 v00_0, v01_0, v10_0, v00_1, v01_1, v10_1;
	v00_0.x = _d_local_vertex_map[offset + v*_width + u].x;
	v01_0.x = _d_local_vertex_map[offset + v*_width + (u - range)].x;
	v10_0.x = _d_local_vertex_map[offset + (v - range)*_width + u].x;

	v00_1.x = v00_0.x;
	v01_1.x = _d_local_vertex_map[offset + v*_width + (u + range)].x;
	v10_1.x = _d_local_vertex_map[offset + (v + range)*_width + u].x;

	if (!isnan(v00_0.x) && !isnan(v01_0.x) && !isnan(v01_0.x) && !isnan(v00_1.x) && !isnan(v01_1.x) && !isnan(v01_1.x))
	{
		v00_0.y = _d_local_vertex_map[offset + v*_width + u].y;
		v01_0.y = _d_local_vertex_map[offset + v*_width + (u - range)].y;
		v10_0.y = _d_local_vertex_map[offset + (v - range)*_width + u].y;

		v00_1.y = v00_0.y;
		v01_1.y = _d_local_vertex_map[offset + v*_width + (u + range)].y;
		v10_1.y = _d_local_vertex_map[offset + (v + range)*_width + u].y;

		v00_0.z = _d_local_vertex_map[offset + v*_width + u].z;
		v01_0.z = _d_local_vertex_map[offset + v*_width + (u - range)].z;
		v10_0.z = _d_local_vertex_map[offset + (v - range)*_width + u].z;

		v00_1.z = v00_0.z;
		v01_1.z = _d_local_vertex_map[offset + v*_width + (u + range)].z;
		v10_1.z = _d_local_vertex_map[offset + (v + range)*_width + u].z;

		float3 r = normalized(cross(v01_0 - v00_0, v10_0 - v00_0) + cross(v01_1 - v00_1, v10_1 - v00_1));

		_d_local_normal_map[offset + v*_width + u].x = r.x;
		_d_local_normal_map[offset + v*_width + u].y = r.y;
		_d_local_normal_map[offset + v*_width + u].z = r.z;
		_d_local_normal_map[offset + v*_width + u].w = 0.0f;
		_d_index_map[offset + v*_width + u] = offset + v*_width + u;
	}
}

void CreateLocalVertexAndNoramlFromDepthFromDeviceData(
	float* _d_depth_map, 
	float4* _d_local_vertex_map, 
	float4* _d_local_normal_map, 
	int* _d_index_map,
	float4* _d_camera_fxfycxcx,
	int _width, 
	int _height, 
	int _camera_num)
{
	int block = 256;
	int grid = DivUp(_width*_height*_camera_num, block);

	CreateLocalVertexFromDepthKernel << <grid, block >> > (
		_width*_height*_camera_num, 
		_d_depth_map, 
		_d_local_vertex_map, 
		_d_camera_fxfycxcx, 
		_width, 
		_height, 
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	CreataLocalNormalFromLocalVertexKernel << <grid, block >> > (
		_width*_height*_camera_num,
		_d_local_vertex_map,
		_d_local_normal_map,
		_d_index_map,
		_width,
		_height,
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

void CreateLocalVertexAndNoramlFromDepthFromHostData(
	float* _h_depth_map,
	float4* _h_local_vertex_map,
	float4* _h_local_normal_map,
	int* _h_index_map,
	float4* _h_camera_fxfycxcy,
	int _width,
	int _height,
	int _camera_num)
{
	thrust::device_vector<float>    d_depth_map;
	thrust::device_vector<float4>   d_local_vertex_map;
	thrust::device_vector<float4>   d_local_normal_map;
	thrust::device_vector<int>      d_index_map;
	thrust::device_vector<float4>   d_camera_fxfycxcy;
	d_depth_map.resize(_camera_num*_width*_height);
	d_local_vertex_map.resize(_camera_num*_width*_height);
	d_local_normal_map.resize(_camera_num*_width*_height);
	d_index_map.resize(_camera_num*_width*_height);
	d_camera_fxfycxcy.resize(_camera_num);

	checkCudaErrors(cudaMemcpy(RAW_PTR(d_depth_map), _h_depth_map, _camera_num*_width*_height * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_camera_fxfycxcy), _h_camera_fxfycxcy, _camera_num * sizeof(float4), cudaMemcpyHostToDevice));

	//cv::Mat tmp(_height, _width, CV_32FC1);
	//checkCudaErrors(cudaMemcpy(tmp.data, RAW_PTR(d_depth_map), _camera_num*_width*_height * sizeof(float), cudaMemcpyDeviceToHost));

	int block = 256;
	int grid = DivUp(_width*_height*_camera_num, block);

	CreateLocalVertexFromDepthKernel << <grid, block >> > (
		_width*_height*_camera_num,
		RAW_PTR(d_depth_map),
		RAW_PTR(d_local_vertex_map),
		RAW_PTR(d_camera_fxfycxcy),
		_width,
		_height,
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//cv::Mat tmp_vertex_map(_height, _width, CV_32FC4);
	//checkCudaErrors(cudaMemcpy(tmp_vertex_map.data, RAW_PTR(d_local_vertex_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));

	CreataLocalNormalFromLocalVertexKernel << <grid, block >> > (
		_width*_height*_camera_num,
		RAW_PTR(d_local_vertex_map),
		RAW_PTR(d_local_normal_map),
		RAW_PTR(d_index_map),
		_width,
		_height,
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(_h_local_vertex_map, RAW_PTR(d_local_vertex_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_h_local_normal_map, RAW_PTR(d_local_normal_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_h_index_map, RAW_PTR(d_index_map), _camera_num*_width*_height * sizeof(int), cudaMemcpyDeviceToHost));
}

__global__ void CreateGlobalVertexAndNormalFromLocalVertexAndNormalKernel(
	int _max_num,
	float4* _d_local_vertex_map,
	float4* _d_local_normal_map,
	float4* _d_global_vertex_map,
	float4* _d_global_normal_map,
	float4* _d_camera_pose_t,
	int _width,
	int _height,
	int _camera_num)
{
	int th = blockDim.x * blockIdx.x + threadIdx.x;

	if (th >= _max_num) return;

	int camera_id = th / (_width*_height);
	float4 *camera_pose_t = _d_camera_pose_t + camera_id * 4;
	float4 local_vertex = _d_local_vertex_map[th];
	float4 local_normal = _d_local_normal_map[th];

	_d_global_vertex_map[th] = { __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), 1.0f };
	_d_global_normal_map[th] = { __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), 0.0f };

	if (!isnan(_d_local_vertex_map[th].x))
	{
		_d_global_vertex_map[th] =
			camera_pose_t[0] * local_vertex.x
			+ camera_pose_t[1] * local_vertex.y
			+ camera_pose_t[2] * local_vertex.z
			+ camera_pose_t[3] * 1.0f;
	}

	if (!isnan(_d_local_normal_map[th].x))
	{
		_d_global_normal_map[th] =
			camera_pose_t[0] * local_normal.x
			+ camera_pose_t[1] * local_normal.y
			+ camera_pose_t[2] * local_normal.z;
	}
}

void CreateGlobalVertexAndNormalFromLocalVertexAndNormalFromDeviceData(
	float4* _d_local_vertex_map,
	float4* _d_local_normal_map, 
	float4* _d_global_vertex_map,
	float4* _d_global_normal_map,
	float4* _d_camera_pose_t, 
	int _width,
	int _height,
	int _camera_num)
{
	int block = 256;
	int grid = DivUp(_width*_height*_camera_num, block);

	CreateGlobalVertexAndNormalFromLocalVertexAndNormalKernel << <grid, block >> >(
		_width*_height*_camera_num, 
		_d_local_vertex_map,
		_d_local_normal_map, 
		_d_global_vertex_map,
		_d_global_normal_map,
		_d_camera_pose_t, 
		_width, 
		_height,
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

void CreateGlobalVertexAndNormalFromLocalVertexAndNormalFromHostData(
	float4* _h_local_vertex_map,
	float4* _h_local_normal_map,
	float4* _h_global_vertex_map,
	float4* _h_global_normal_map,
	float4* _h_camera_pose_t,
	int _width,
	int _height,
	int _camera_num)
{
	thrust::device_vector<float4> d_local_vertex_map;
	thrust::device_vector<float4> d_local_normal_map;
	thrust::device_vector<float4> d_global_vertex_map;
	thrust::device_vector<float4> d_global_normal_map;
	thrust::device_vector<float4> d_camera_pose_t;
	d_local_vertex_map.resize(_camera_num*_width*_height);
	d_local_normal_map.resize(_camera_num*_width*_height);
	d_global_vertex_map.resize(_camera_num*_width*_height);
	d_global_normal_map.resize(_camera_num*_width*_height);
	d_camera_pose_t.resize(_camera_num * 4);

	checkCudaErrors(cudaMemcpy(RAW_PTR(d_local_vertex_map), _h_local_vertex_map, _camera_num*_width*_height * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_local_normal_map), _h_local_normal_map, _camera_num*_width*_height * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(RAW_PTR(d_camera_pose_t), _h_camera_pose_t, _camera_num * 4 * sizeof(float4), cudaMemcpyHostToDevice));

	//cv::Mat local_vertex_map(_height, _width, CV_32FC4);
	//checkCudaErrors(cudaMemcpy(local_vertex_map.data, RAW_PTR(d_local_vertex_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));

	int block = 256;
	int grid = DivUp(_width*_height*_camera_num, block);

	CreateGlobalVertexAndNormalFromLocalVertexAndNormalKernel << <grid, block >> >(
		_width*_height*_camera_num,
		RAW_PTR(d_local_vertex_map),
		RAW_PTR(d_local_normal_map),
		RAW_PTR(d_global_vertex_map),
		RAW_PTR(d_global_normal_map),
		RAW_PTR(d_camera_pose_t),
		_width,
		_height,
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//cv::Mat tmp(_height, _width, CV_32FC4);
	//checkCudaErrors(cudaMemcpy(tmp.data, RAW_PTR(d_global_vertex_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));


	checkCudaErrors(cudaMemcpy(_h_global_vertex_map, RAW_PTR(d_global_vertex_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_h_global_normal_map, RAW_PTR(d_global_normal_map), _camera_num*_width*_height * sizeof(float4), cudaMemcpyDeviceToHost));
}

__global__ void CudaProjectVertexToDepthImageKernel(
	int _vertex_num, 
	float4 *_d_vertex, 
	float *_d_depth, 
	float4 *_d_camera_pose_inv_t,
	float4 _d_camera_fxfycxcy,
	int _width,
	int _height)
{
	int th = blockDim.x * blockIdx.x + threadIdx.x;

	if (th >= _vertex_num) return;

	float4 vertex = _d_vertex[th];
	float4 local_vertex = _d_camera_pose_inv_t[0] * vertex.x
		+ _d_camera_pose_inv_t[1] * vertex.y
		+ _d_camera_pose_inv_t[2] * vertex.z
		+ _d_camera_pose_inv_t[3];

	int u = __float2int_rn((local_vertex.x * _d_camera_fxfycxcy.x) / local_vertex.z + _d_camera_fxfycxcy.z);
	int v = __float2int_rn((local_vertex.y * _d_camera_fxfycxcy.y) / local_vertex.z + _d_camera_fxfycxcy.w);

	if (u < 0 || u >= _width || v < 0 || v >= _height) return;
	if (_d_depth[v*_width + u] == 0 || (_d_depth[v*_width + u] != 0 && local_vertex.z < _d_depth[v*_width + u]))
	{
		_d_depth[v*_width + u] = local_vertex.z;
	}
}

void CudaProjectVertexToDepthImage(
	int _vertex_num, 
	float4 *_d_vertex,
	float* _d_depth, 
	float4 *_d_camera_pose_inv_t,
	float4 _d_camera_fxfycxcy, 
	int _width, int _height)
{
	checkCudaErrors(cudaMemset(_d_depth, 0, _width*_height * sizeof(float)));

	int block = 256;
	int grid = DivUp(_vertex_num, block);
	CudaProjectVertexToDepthImageKernel << <grid, block >> > (
		_vertex_num, 
		_d_vertex, 
		_d_depth,
		_d_camera_pose_inv_t,
		_d_camera_fxfycxcy,
		_width,
		_height);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}


__global__ void CudaProjectVertexToDepthImageKernel(
	int _vertex_num,
	float4 *_d_vertex,
	float *_d_depth,
	int   *_d_index,
	float4 *_d_camera_pose_inv_t,
	float4 _d_camera_fxfycxcy,
	int _width,
	int _height)
{
	int th = blockDim.x * blockIdx.x + threadIdx.x;

	if (th >= _vertex_num) return;

	float4 vertex = _d_vertex[th];
	float4 local_vertex = _d_camera_pose_inv_t[0] * vertex.x
		+ _d_camera_pose_inv_t[1] * vertex.y
		+ _d_camera_pose_inv_t[2] * vertex.z
		+ _d_camera_pose_inv_t[3];

	int u = __float2int_rn((local_vertex.x * _d_camera_fxfycxcy.x) / local_vertex.z + _d_camera_fxfycxcy.z);
	int v = __float2int_rn((local_vertex.y * _d_camera_fxfycxcy.y) / local_vertex.z + _d_camera_fxfycxcy.w);

	if (u < 0 || u >= _width || v < 0 || v >= _height) return;
	if (_d_depth[v*_width + u] == 0 || (_d_depth[v*_width + u] != 0 && local_vertex.z < _d_depth[v*_width + u]))
	{
		_d_depth[v*_width + u] = local_vertex.z;
		_d_index[v*_width + u] = th;
	}
}

void CudaProjectVertexToDepthImage(
	int _vertex_num,
	float4 *_d_vertex,
	float* _d_depth,
	int*   _d_index,
	float4 *_d_camera_pose_inv_t,
	float4 _d_camera_fxfycxcy,
	int _width, int _height)
{
	checkCudaErrors(cudaMemset(_d_depth, 0, _width*_height * sizeof(float)));
	checkCudaErrors(cudaMemset(_d_index, 0xFF, _width*_height * sizeof(int)));

	int block = 256;
	int grid = DivUp(_vertex_num, block);
	CudaProjectVertexToDepthImageKernel << <grid, block >> > (
		_vertex_num,
		_d_vertex,
		_d_depth,
		_d_index,
		_d_camera_pose_inv_t,
		_d_camera_fxfycxcy,
		_width,
		_height);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}

__global__ void FindMatchingPointsPerspectiveFromMultiCameraKernel(
	int     _max_num,
	float4* _d_deform_vertex,
	float4* _d_target_vertex,
	float4* _d_camera_pose_inv_t,
	float4* _d_camera_fxfycxcy,
	int* _d_index_map,
	int* _d_matching_points,
	float* _d_min_matching_dist,
	int _matching_point_num,
	int _width,
	int _height,
	int _camera_num)
{
	int th = blockDim.x * blockIdx.x + threadIdx.x;

	if (th >= _max_num) return;

	int camera_id = th / _matching_point_num;
	int vertex_idx = th % _matching_point_num;
	float4 deform_vertex;
	deform_vertex = _d_deform_vertex[vertex_idx];

	float4 *target_camera_pose_in_t = _d_camera_pose_inv_t + camera_id * 4;
	float4 target_camera_fxfycxcy = _d_camera_fxfycxcy[camera_id];
	int* target_index_map = _d_index_map + _width*_height*camera_id;

	float4 to_target_local_vertex;
	to_target_local_vertex = target_camera_pose_in_t[0] * deform_vertex.x
		+ target_camera_pose_in_t[1] * deform_vertex.y
		+ target_camera_pose_in_t[2] * deform_vertex.z
		+ target_camera_pose_in_t[3];
	
	int u_to_target = __float2int_rn((to_target_local_vertex.x * target_camera_fxfycxcy.x) / to_target_local_vertex.z + target_camera_fxfycxcy.z);
	int v_to_target = __float2int_rn((to_target_local_vertex.y * target_camera_fxfycxcy.y) / to_target_local_vertex.z + target_camera_fxfycxcy.w);

	float dist;
	int steps[4] = { 1, 3, 6, 9 };
	if (u_to_target < steps[3] || v_to_target < steps[3] || u_to_target >= _width - steps[3] || v_to_target >= _height - steps[3])
	{
		return;
	}

	int vertex_ind_target[33];
	int step = steps[0];
	vertex_ind_target[0] = *(target_index_map + v_to_target * _width + u_to_target);
	vertex_ind_target[1] = *(target_index_map + (v_to_target - step) * _width + (u_to_target - step));
	vertex_ind_target[2] = *(target_index_map + (v_to_target + step) * _width + (u_to_target - step));
	vertex_ind_target[3] = *(target_index_map + (v_to_target - step) * _width + (u_to_target + step));
	vertex_ind_target[4] = *(target_index_map + (v_to_target + step) * _width + (u_to_target + step));
	vertex_ind_target[5] = *(target_index_map + (v_to_target - step) * _width + (u_to_target));
	vertex_ind_target[6] = *(target_index_map + (v_to_target + step) * _width + (u_to_target));
	vertex_ind_target[7] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	vertex_ind_target[8] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	step = steps[1];
	vertex_ind_target[9] = *(target_index_map + (v_to_target - step) * _width + (u_to_target - step));
	vertex_ind_target[10] = *(target_index_map + (v_to_target + step) * _width + (u_to_target - step));
	vertex_ind_target[11] = *(target_index_map + (v_to_target - step) * _width + (u_to_target + step));
	vertex_ind_target[12] = *(target_index_map + (v_to_target + step) * _width + (u_to_target + step));
	vertex_ind_target[13] = *(target_index_map + (v_to_target - step) * _width + (u_to_target));
	vertex_ind_target[14] = *(target_index_map + (v_to_target + step) * _width + (u_to_target));
	vertex_ind_target[15] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	vertex_ind_target[16] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	step = steps[2];
	vertex_ind_target[17] = *(target_index_map + (v_to_target - step) * _width + (u_to_target - step));
	vertex_ind_target[18] = *(target_index_map + (v_to_target + step) * _width + (u_to_target - step));
	vertex_ind_target[19] = *(target_index_map + (v_to_target - step) * _width + (u_to_target + step));
	vertex_ind_target[20] = *(target_index_map + (v_to_target + step) * _width + (u_to_target + step));
	vertex_ind_target[21] = *(target_index_map + (v_to_target - step) * _width + (u_to_target));
	vertex_ind_target[22] = *(target_index_map + (v_to_target + step) * _width + (u_to_target));
	vertex_ind_target[23] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	vertex_ind_target[24] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	step = steps[3];
	vertex_ind_target[25] = *(target_index_map + (v_to_target - step) * _width + (u_to_target - step));
	vertex_ind_target[26] = *(target_index_map + (v_to_target + step) * _width + (u_to_target - step));
	vertex_ind_target[27] = *(target_index_map + (v_to_target - step) * _width + (u_to_target + step));
	vertex_ind_target[28] = *(target_index_map + (v_to_target + step) * _width + (u_to_target + step));
	vertex_ind_target[29] = *(target_index_map + (v_to_target - step) * _width + (u_to_target));
	vertex_ind_target[30] = *(target_index_map + (v_to_target + step) * _width + (u_to_target));
	vertex_ind_target[31] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));
	vertex_ind_target[32] = *(target_index_map + (v_to_target)* _width + (u_to_target + step));

	int vertex_ind_target_nearest = -1;
	float minDist = _d_min_matching_dist[vertex_idx];
	float4 target_vertex;
	for (int i = 0; i < 33; ++i)
	{
		if (vertex_ind_target[i] >= 0)
		{
			target_vertex = _d_target_vertex[vertex_ind_target[i]];
			if (!isnan(target_vertex.x))
			{
				dist = norm(deform_vertex - target_vertex);
				if (dist < minDist)
				{
					vertex_ind_target_nearest = vertex_ind_target[i];
					minDist = dist;
				}
			}
		}
	}

	if (vertex_ind_target_nearest == -1) return;
	_d_matching_points[vertex_idx] = vertex_ind_target_nearest;
	_d_min_matching_dist[vertex_idx] = minDist;
}

void FindMatchingPointsPerspectiveFromMultiCamera(
	float4* _d_deform_vertex,
	float4* _d_target_vertex,
	float4* _d_camera_pose_inv_t,
	float4* _d_camera_fxfycxcy, 
	int* _d_index_map,
	int* _d_matching_points,
	float* _d_min_matching_dist,
	int _matching_point_num,
	int _width, 
	int _height,
	int _camera_num)
{
	int max_num = _matching_point_num*_camera_num;
	int block = 256;
	int grid = DivUp(max_num, block);

	FindMatchingPointsPerspectiveFromMultiCameraKernel << <grid, block >> > (
		max_num,
		_d_deform_vertex,
		_d_target_vertex,
		_d_camera_pose_inv_t,
		_d_camera_fxfycxcy,
		_d_index_map,
		_d_matching_points,
		_d_min_matching_dist,
		_matching_point_num,
		_width,
		_height,
		_camera_num);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
}