#include "imu_gps_localizer/imu_gps_localizer.h"

#include <glog/logging.h>

#include "imu_gps_localizer/utils.h"

namespace ImuGpsLocalization {

ImuGpsLocalizer::ImuGpsLocalizer(const double acc_noise, const double gyro_noise,
                                 const double acc_bias_noise, const double gyro_bias_noise,
                                 const Eigen::Vector3d& I_p_Gps) 
    : initialized_(false){
    initializer_ = std::make_unique<Initializer>(I_p_Gps); //[0, 0, 0]
    imu_processor_ = std::make_unique<ImuProcessor>(acc_noise, gyro_noise, 
                                                    acc_bias_noise, gyro_bias_noise,
                                                    Eigen::Vector3d(0., 0., -9.81007)); //默认IMU为ENU型
    gps_processor_ = std::make_unique<GpsProcessor>(I_p_Gps); //[0, 0, 0]
}


bool ImuGpsLocalizer::ProcessImuData(const ImuDataPtr imu_data_ptr, State* fused_state) {
    if (!initialized_) {
        initializer_->AddImuData(imu_data_ptr);
        return false;
    }
    
    //state is initialized
    //注意: 作者此处的系统状态不是error-state, 而是state。

    //每来一次imu对filter状态[p v q ba bg]进行一次积分，以及对应的方差进行一次propagate
    imu_processor_->Predict(state_.imu_data_ptr, imu_data_ptr, &state_); 

    // Convert ENU state to lla.
    ConvertENUToLLA(init_lla_, state_.G_p_I, &(state_.lla));
    //通过GeographicLib库得到imu预测的p在ECEF(WGS84坐标系)下的位置

    *fused_state = state_;
    return true;
}


bool ImuGpsLocalizer::ProcessGpsPositionData(const GpsPositionDataPtr gps_data_ptr) {
    if (!initialized_) {
        //第一帧gps available时, 用imu_buff队列计算平均imu加速度来init filter
        //p=0, v=0, ba=0, bg=0
        //初始时刻的q的roll, pitch由imu acc测量方程计算, 默认在初始化时刻imu处于静止状态，且在水平面内。即G_a = [0 0 0]^T， G_g = [0 0 -g]
        //yaw设置为0，对应的cov比roll, pitch大一点
        if (!initializer_->AddGpsPositionData(gps_data_ptr, &state_)) {//初始化时刻imu没有处于静止状态
            return false;
        }

        // Initialize the initial gps point used to convert lla to ENU.
        init_lla_ = gps_data_ptr->lla;
        
        initialized_ = true;

        LOG(INFO) << "[ProcessGpsPositionData]: System initialized!";
        return true;
    }

    // Update.
    gps_processor_->UpdateStateByGpsPosition(init_lla_, gps_data_ptr, &state_);
    
    //jxl: corrected position of state by gps(lla data), not published by author
    //ConvertENUToLLA(init_lla_, state_.G_p_I, &(state_.lla));
    //*fused_state = state_;

    return true;
}

}  // namespace ImuGpsLocalization