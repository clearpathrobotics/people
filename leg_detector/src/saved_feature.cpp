#include "leg_detector/saved_feature.h"

int SavedFeature::nextid = 0;

SavedFeature::SavedFeature(tf::Stamped<tf::Point> loc, tf::TransformListener& tfl, std::string fixed_frame, bool use_filter, double kal_p, double kal_q, double kal_r)
    : tfl_(tfl),
      sys_sigma_(tf::Vector3(0.05, 0.05, 0.05), tf::Vector3(1.0, 1.0, 1.0)),
      filter_("tracker_name", sys_sigma_),
      reliability(-1.), p(4),
      fixed_frame_(fixed_frame),
      use_filter_(use_filter),
      kal_p_(kal_p),
      kal_q_(kal_q),
      kal_r_(kal_r)
  {
    id_ = "legtrack" + std::to_string(nextid++);

    object_id = "";
    time_ = loc.stamp_;
    meas_time_ = loc.stamp_;
    other = NULL;

    try
    {
      tfl_.transformPoint(fixed_frame_, loc, loc);
    }
    catch (...)
    {
      ROS_WARN("TF exception spot 6.");
    }
    tf::StampedTransform pose(tf::Pose(tf::Quaternion(0.0, 0.0, 0.0, 1.0), loc), loc.stamp_, id_, loc.frame_id_);
    tfl_.setTransform(pose);

    BFL::StatePosVel prior_sigma(tf::Vector3(0.1, 0.1, 0.1), tf::Vector3(0.0000001, 0.0000001, 0.0000001));
    filter_.initialize(loc, prior_sigma, time_.toSec());

    BFL::StatePosVel est;
    filter_.getEstimate(est);

    updatePosition();
  }

  void SavedFeature::propagate(ros::Time time)
  {
    time_ = time;

    filter_.updatePrediction(time.toSec());

    updatePosition();
  }

  void SavedFeature::update(tf::Stamped<tf::Point> loc, double probability)
  {
    tf::StampedTransform pose(tf::Pose(tf::Quaternion(0.0, 0.0, 0.0, 1.0), loc), loc.stamp_, id_, loc.frame_id_);
    tfl_.setTransform(pose);

    meas_time_ = loc.stamp_;
    time_ = meas_time_;

    MatrixWrapper::SymmetricMatrix cov(3);
    cov = 0.0;
    cov(1, 1) = 0.0025;
    cov(2, 2) = 0.0025;
    cov(3, 3) = 0.0025;

    filter_.updateCorrection(loc, cov);

    updatePosition();

    if (reliability < 0 || !use_filter_)
    {
      reliability = probability;
      p = kal_p_;
    }
    else
    {
      p += kal_q_;
      double k = p / (p + kal_r_);
      reliability += k * (probability - reliability);
      p *= (1 - k);
    }
  }

  double SavedFeature::getLifetime()
  {
    return filter_.getLifetime();
  }

  double SavedFeature::getReliability()
  {
    return reliability;
  }

  void SavedFeature::updatePosition()
  {
    BFL::StatePosVel est;
    filter_.getEstimate(est);

    position_[0] = est.pos_[0];
    position_[1] = est.pos_[1];
    position_[2] = est.pos_[2];
    position_.stamp_ = time_;
    position_.frame_id_ = fixed_frame_;
    double nreliability = fmin(1.0, fmax(0.1, est.vel_.length() / 0.5));
    //reliability = fmax(reliability, nreliability);
  }

