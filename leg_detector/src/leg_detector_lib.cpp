#include "leg_detector/leg_detector_lib.h"

 LegDetector::LegDetector(const std::string& model_path) :
    mask_count_(0),
    feat_count_(0),
    next_p_id_(0),
    no_observation_timeout_s_(0.5),
    max_second_leg_age_s_(2.0),
    max_track_jump_m_(1.0),
    max_meas_jump_m_(1.0),
    leg_pair_separation_m_(1.0),
    fixed_frame_("odom"),
    kal_p_(4),
    kal_q_(0.002),
    kal_r_(10),
    use_filter_(true),
    connected_thresh_(0.06),
    min_points_per_group(5),
    leg_reliability_limit_(0.7)
  {
    forest = cv::ml::StatModel::load<cv::ml::RTrees>(model_path);
    feat_count_ = forest->getVarCount();
    printf("Loaded forest with %d features: %s\n", feat_count_, model_path.c_str());
    feature_id_ = 0;
  }


  LegDetector::~LegDetector()
  {
  }

  double LegDetector::distance(std::list<std::shared_ptr<SavedFeature>>::iterator it1,  std::list<std::shared_ptr<SavedFeature>>::iterator it2) const
  {
    const tf::Stamped<tf::Point> one = (*it1)->position_, two = (*it2)->position_;
    const double dx = one[0] - two[0], dy = one[1] - two[1], dz = one[2] - two[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
  }

  void LegDetector::pairLegs()
  {
    // Deal With legs that already have ids
    std::list<std::shared_ptr<SavedFeature>>::iterator begin = saved_features_.begin();
    std::list<std::shared_ptr<SavedFeature>>::iterator end = saved_features_.end();
    std::list<std::shared_ptr<SavedFeature>>::iterator leg1, leg2, best, it;

    for (leg1 = begin; leg1 != end; ++leg1)
    {
      // If this leg has no id, skip
      if ((*leg1)->object_id == "")
        continue;

      leg2 = end;
      best = end;
      double closest_dist = leg_pair_separation_m_;
      for (it = begin; it != end; ++it)
      {
        if (it == leg1) continue;

        if ((*it)->object_id == (*leg1)->object_id)
        {
          leg2 = it;
          break;
        }

        if ((*it)->object_id != "")
          continue;

        double d = distance(it, leg1);
        if (((*it)->getLifetime() <= max_second_leg_age_s_)
            && (d < closest_dist))
        {
          closest_dist = d;
          best = it;
        }

      }

      if (leg2 != end)
      {
        double dist_between_legs = distance(leg1, leg2);
        if (dist_between_legs > leg_pair_separation_m_)
        {
          (*leg1)->object_id = "";
          (*leg1)->other = NULL;
          (*leg2)->object_id = "";
          (*leg2)->other = NULL;
        }
        else
        {
          (*leg1)->other = *leg2;
          (*leg2)->other = *leg1;
        }
      }
      else if (best != end)
      {
        (*best)->object_id = (*leg1)->object_id;
        (*leg1)->other = *best;
        (*best)->other = *leg1;
      }
    }

    // Attempt to pair up legs with no id
    for (;;)
    {
      std::list<std::shared_ptr<SavedFeature>>::iterator best1 = end, best2 = end;
      double closest_dist = leg_pair_separation_m_;

      for (leg1 = begin; leg1 != end; ++leg1)
      {
        // If this leg has an id or low reliability, skip
        if ((*leg1)->object_id != ""
            || (*leg1)->getReliability() < leg_reliability_limit_)
          continue;

        for (leg2 = begin; leg2 != end; ++leg2)
        {
          if (((*leg2)->object_id != "")
              || ((*leg2)->getReliability() < leg_reliability_limit_)
              || (leg1 == leg2)) continue;
          double d = distance(leg1, leg2);
          if (d < closest_dist)
          {
            best1 = leg1;
            best2 = leg2;
          }
        }
      }

      if (best1 != end)
      {
        std::string object_id_str = "Person" + std::to_string(next_p_id_++);
        (*best1)->object_id = object_id_str;
        (*best2)->object_id = object_id_str;
        (*best1)->other = *best2;
        (*best2)->other = *best1;
      }
      else
      {
        break;
      }
    }
  }

  std::vector<people_msgs::PersonPositionMeasurement> LegDetector::processLaserScan(const sensor_msgs::LaserScan& scan, tf::TransformListener& tfl)
  {
    laser_processor::ScanProcessor processor(scan, mask_);

    processor.splitConnected(connected_thresh_);
    processor.removeLessThan(5);

    cv::Mat tmp_mat = cv::Mat(1, feat_count_, CV_32FC1);

    // if no measurement matches to a tracker in the last <no_observation_timeout>  seconds: erase tracker
    ros::Time purge = scan.header.stamp + ros::Duration().fromSec(-no_observation_timeout_s_);
    std::list<std::shared_ptr<SavedFeature>>::iterator sf_iter = saved_features_.begin();
    while (sf_iter != saved_features_.end())
    {
      if ((*sf_iter)->meas_time_ < purge)
      {
        if ((*sf_iter)->other)
          (*sf_iter)->other->other = NULL;
        sf_iter = saved_features_.erase(sf_iter);
      }
      else
        ++sf_iter;
    }


    // System update of trackers, and copy updated ones in propagate list
    std::list<std::shared_ptr<SavedFeature>> propagated;
    for (auto& sf : saved_features_)
    {
      sf->propagate(scan.header.stamp);
      propagated.push_back(sf);
    }


    // Detection step: build up the set of "candidate" clusters
    // For each candidate, find the closest tracker (within threshold) and add to the match list
    // If no tracker is found, start a new one
    BFL::multiset<MatchedFeature> matches;
    for (std::list<laser_processor::SampleSet*>::iterator i = processor.getClusters().begin();
         i != processor.getClusters().end();
         i++)
    {
      std::vector<float> f = calcLegFeatures(*i, scan);

      for (int k = 0; k < feat_count_; k++)
        tmp_mat.data[k] = (float)(f[k]);

      float probability = 0.5 - forest->predict(tmp_mat, cv::noArray(), cv::ml::RTrees::PREDICT_SUM) / forest->getRoots().size();
      tf::Stamped<tf::Point> loc((*i)->center(), scan.header.stamp, scan.header.frame_id);
      try
      {
        tfl.transformPoint(fixed_frame_, loc, loc);
      }
      catch (...)
      {
        ROS_WARN("TF exception spot 3.");
      }

      std::list<std::shared_ptr<SavedFeature>>::iterator closest = propagated.end();
      float closest_dist = max_track_jump_m_;

      for (std::list<std::shared_ptr<SavedFeature>>::iterator pf_iter = propagated.begin();
           pf_iter != propagated.end();
           pf_iter++)
      {
        // find the closest distance between candidate and trackers
        float dist = loc.distance((*pf_iter)->position_);
        if (dist < closest_dist)
        {
          closest = pf_iter;
          closest_dist = dist;
        }
      }
      // Nothing close to it, start a new track
      if (closest == propagated.end())
      {
        std::list<std::shared_ptr<SavedFeature>>::iterator new_saved = saved_features_.insert(saved_features_.end(), std::make_shared<SavedFeature>(loc, tfl, fixed_frame_, use_filter_, kal_p_, kal_q_, kal_r_));
      }
      // Add the candidate, the tracker and the distance to a match list
      else
        matches.insert(MatchedFeature(*i, *closest, closest_dist, probability));
    }

    // loop through _sorted_ matches list
    // find the match with the shortest distance for each tracker
    while (!matches.empty())
    {
      BFL::multiset<MatchedFeature>::iterator matched_iter = matches.begin();
      bool found = false;
      std::list<std::shared_ptr<SavedFeature>>::iterator pf_iter = propagated.begin();
      while (pf_iter != propagated.end())
      {
        // update the tracker with this candidate
        if (matched_iter->closest_ == *pf_iter)
        {
          // Transform candidate to fixed frame
          tf::Stamped<tf::Point> loc(matched_iter->candidate_->center(), scan.header.stamp, scan.header.frame_id);
          try
          {
            tfl.transformPoint(fixed_frame_, loc, loc);
          }
          catch (...)
          {
            ROS_WARN("TF exception spot 4.");
          }

          // Update the tracker with the candidate location
          matched_iter->closest_->update(loc, matched_iter->probability_);

          // remove this match and
          matches.erase(matched_iter);
          propagated.erase(pf_iter++);
          found = true;
          break;
        }
        // still looking for the tracker to update
        else
        {
          pf_iter++;
        }
      }

      // didn't find tracker to update, because it was deleted above
      // try to assign the candidate to another tracker
      if (!found)
      {
        tf::Stamped<tf::Point> loc(matched_iter->candidate_->center(), scan.header.stamp, scan.header.frame_id);
        try
        {
          tfl.transformPoint(fixed_frame_, loc, loc);
        }
        catch (...)
        {
          ROS_WARN("TF exception spot 5.");
        }

        std::list<std::shared_ptr<SavedFeature>>::iterator closest = propagated.end();
        float closest_dist = max_track_jump_m_;

        for (std::list<std::shared_ptr<SavedFeature>>::iterator remain_iter = propagated.begin();
             remain_iter != propagated.end();
             remain_iter++)
        {
          float dist = loc.distance((*remain_iter)->position_);
          if (dist < closest_dist)
          {
            closest = remain_iter;
            closest_dist = dist;
          }
        }

        // no tracker is within a threshold of this candidate
        // so create a new tracker for this candidate
        if (closest == propagated.end())
          std::list<std::shared_ptr<SavedFeature>>::iterator new_saved = saved_features_.insert(saved_features_.end(), std::make_shared<SavedFeature>(loc, tfl, fixed_frame_, use_filter_, kal_p_, kal_q_, kal_r_));
        else
          matches.insert(MatchedFeature(matched_iter->candidate_, *closest, closest_dist, matched_iter->probability_));
        matches.erase(matched_iter);
      }
    }

    pairLegs();

    // Publish Data!
    int i = 0;
    std::vector<people_msgs::PersonPositionMeasurement> people;
    std::vector<people_msgs::PositionMeasurement> legs;

    for (std::list<std::shared_ptr<SavedFeature>>::iterator sf_iter = saved_features_.begin();
         sf_iter != saved_features_.end();
         sf_iter++, i++)
    {
      // reliability
      double reliability = (*sf_iter)->getReliability();

      if ((*sf_iter)->getReliability() > leg_reliability_limit_)
      {
        people_msgs::PositionMeasurement pos;
        pos.header.stamp = scan.header.stamp;
        pos.header.frame_id = fixed_frame_;
        pos.name = "leg_detector";
        pos.object_id = (*sf_iter)->id_;
        pos.pos.x = (*sf_iter)->position_[0];
        pos.pos.y = (*sf_iter)->position_[1];
        pos.pos.z = (*sf_iter)->position_[2];
        pos.reliability = reliability;
        pos.covariance[0] = pow(0.3 / reliability, 2.0);
        pos.covariance[1] = 0.0;
        pos.covariance[2] = 0.0;
        pos.covariance[3] = 0.0;
        pos.covariance[4] = pow(0.3 / reliability, 2.0);
        pos.covariance[5] = 0.0;
        pos.covariance[6] = 0.0;
        pos.covariance[7] = 0.0;
        pos.covariance[8] = 10000.0;
        pos.initialization = 0;
        legs.push_back(pos);
      }

      std::shared_ptr<SavedFeature> other = (*sf_iter)->other;
      if (other != NULL && other < (*sf_iter))
      {
        double dx = ((*sf_iter)->position_[0] + other->position_[0]) / 2,
               dy = ((*sf_iter)->position_[1] + other->position_[1]) / 2,
               dz = ((*sf_iter)->position_[2] + other->position_[2]) / 2;

        reliability = reliability * other->reliability;
        people_msgs::PersonPositionMeasurement pos;
        pos.header.stamp = (*sf_iter)->time_;
        pos.header.frame_id = fixed_frame_;
        pos.name = (*sf_iter)->object_id;;
        pos.object_id = (*sf_iter)->id_ + "|" + other->id_;
        pos.pos.x = dx;
        pos.pos.y = dy;
        pos.pos.z = dz;
  
        pos.leg1_pos.x = (*sf_iter)->position_[0];
        pos.leg1_pos.y = (*sf_iter)->position_[1];
        pos.leg1_pos.z = (*sf_iter)->position_[2];
  
        pos.leg2_pos.x = other->position_[0];
        pos.leg2_pos.y = other->position_[1];
        pos.leg2_pos.z = other->position_[2];
 
        pos.reliability = reliability;
        pos.covariance[0] = pow(0.3 / reliability, 2.0);
        pos.covariance[1] = 0.0;
        pos.covariance[2] = 0.0;
        pos.covariance[3] = 0.0;
        pos.covariance[4] = pow(0.3 / reliability, 2.0);
        pos.covariance[5] = 0.0;
        pos.covariance[6] = 0.0;
        pos.covariance[7] = 0.0;
        pos.covariance[8] = 10000.0;
        pos.initialization = 0;
        people.push_back(pos);
      }
    }
    return people;
 }
