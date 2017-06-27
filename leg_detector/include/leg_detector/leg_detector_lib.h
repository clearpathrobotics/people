#ifndef LEG_DETECTOR_LEG_DETECTOR_LIB
#define LEG_DETECTOR_LEG_DETECTOR_LIB

#include <ros/ros.h>

#include <leg_detector/laser_processor.h>
#include <leg_detector/calc_leg_features.h>

#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/ml.h>

#include <people_msgs/PositionMeasurement.h>
#include <people_msgs/PositionMeasurementArray.h>
#include <people_msgs/PersonPositionMeasurement.h>
#include <sensor_msgs/LaserScan.h>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>

#include <people_tracking_filter/tracker_kalman.h>
#include <people_tracking_filter/state_pos_vel.h>
#include <people_tracking_filter/rgb.h>

#include <algorithm>
#include "leg_detector/saved_feature.h"
#include "leg_detector/matched_feature.h"

// leg detector class
class LegDetector
{
public:
  laser_processor::ScanMask mask_;

  int mask_count_;

  cv::Ptr<cv::ml::RTrees> forest;

  float connected_thresh_;

  int feat_count_;

  char save_[100];

  std::list<std::shared_ptr<SavedFeature>> saved_features_;
  boost::mutex saved_mutex_;

  int feature_id_;

  int next_p_id_;
  double leg_reliability_limit_;
  int min_points_per_group;

  LegDetector(const std::string& model_path);

  ~LegDetector();

  double distance(std::list<std::shared_ptr<SavedFeature>>::iterator it1,  std::list<std::shared_ptr<SavedFeature>>::iterator it2) const;

  void pairLegs();

  std::vector<people_msgs::PersonPositionMeasurement> processLaserScan(const sensor_msgs::LaserScan& scan, tf::TransformListener& tfl);

private:
  double no_observation_timeout_s_;
  double max_second_leg_age_s_;
  double max_track_jump_m_;
  double max_meas_jump_m_;
  double leg_pair_separation_m_;
  std::string fixed_frame_;

  double kal_p_, kal_q_, kal_r_;
  bool use_filter_;
};

#endif
