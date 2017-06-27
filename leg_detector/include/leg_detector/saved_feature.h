#ifndef LEG_DETECTOR_SAVED_FEATURE_H
#define LEG_DETECTOR_SAVED_FEATURE_H

#include <ros/ros.h>

#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <opencv/ml.h>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>

#include <people_tracking_filter/tracker_kalman.h>
#include <people_tracking_filter/state_pos_vel.h>
#include <people_tracking_filter/rgb.h>

#include <algorithm>

class SavedFeature
{
public:
  static int nextid;
  tf::TransformListener& tfl_;

  BFL::StatePosVel sys_sigma_;
  estimation::TrackerKalman filter_;

  std::string id_;
  std::string object_id;
  ros::Time time_;
  ros::Time meas_time_;

  double reliability, p;

  tf::Stamped<tf::Point> position_;
  std::shared_ptr<SavedFeature> other;
  float dist_to_person_;

  // .ctor
  SavedFeature(tf::Stamped<tf::Point> loc, tf::TransformListener& tfl, std::string fixed_frame, bool use_filter, double kal_p, double kal_q, double kal_r);

  // one leg tracker
  void propagate(ros::Time time);

  void update(tf::Stamped<tf::Point> loc, double probability);

  double getLifetime();
  
  double getReliability();

private:
  void updatePosition();

  std::string fixed_frame_;

  bool use_filter_;

  double kal_p_, kal_q_, kal_r_;
};

#endif
