/**********************************************
 *      Filename: innoreal_timer.hpp
 *        Coding: utf-8
 *        Author: wuqing
 *         Email: wuqing@galasports.net
 *      HomePage: http://www.innoreal.net
 *  Copyright(C): InnoReal Inc.
 *       Version: 0.0.1
 *    CreateDate: 2016-07-21 17:03:30
 *    LastChange: 2016-12-13 17:19:43
 *          Desc: 
 *       History: 
 **********************************************/
#ifndef _INNOREAL_TIMER_HPP_
#define _INNOREAL_TIMER_HPP_
#ifdef __cplusplus
extern "C"
{
#endif

#include <time.h>
#if defined(_WIN32)
#include <Windows.h>
#define getlocaltime(timep, ret_tm) localtime_s(ret_tm, timep)
#elif defined(__linux) || defined(ANDROID)
#define getlocaltime(timep, ret_tm) localtime_r(timep, ret_tm)
#endif
namespace innoreal {
	class InnoRealTimer {
#if defined(_WIN32)
		private:
			LARGE_INTEGER Freq;
			LARGE_INTEGER StartTickCount;
			LARGE_INTEGER EndTickCount;
			double timegap_ms;
		public:
			InnoRealTimer() {
				QueryPerformanceFrequency(&Freq);
				timegap_ms = 0;
			}
			double Get_TimeRes_in_ms(){
				return 1000.0f / Freq.QuadPart;
			}
			void TimeStart() {
				QueryPerformanceCounter(&StartTickCount);
			}
			void TimeEnd() {
				QueryPerformanceCounter(&EndTickCount);
			}
			double TimeGap_in_ms() {
				timegap_ms = (double)(EndTickCount.QuadPart - StartTickCount.QuadPart) * 1000.0f / Freq.QuadPart;
				return timegap_ms;
			}
#elif defined(__linux) || defined(ANDROID)
		private:
			clockid_t _clk_id;
			struct timespec Res, StartTime, EndTime;
			double timegap_ms;
		public:
			InnoRealTimer(clockid_t clk_id = CLOCK_MONOTONIC) {
				_clk_id = clk_id;
				timegap_ms = 0;
			}
			double Get_TimeRes_in_ms(){
				clock_getres(_clk_id, &Res);
				return double(Res.tv_sec) * 1000.0 + double(Res.tv_nsec) / 1.0e6;
			}
			void TimeStart() {
				clock_gettime(_clk_id, &StartTime);
			}
			void TimeEnd() {
				clock_gettime(_clk_id, &EndTime);
			}
			double TimeGap_in_ms() {
				timegap_ms = double(EndTime.tv_sec - StartTime.tv_sec) * 1000.0 + double(EndTime.tv_nsec - StartTime.tv_nsec) / 1.0e6;
				return timegap_ms;
			}
#endif
	};
} /*! namespace innoreal */

#ifdef __cplusplus
}
#endif
#endif // _INNOREAL_TIMER_HPP_
