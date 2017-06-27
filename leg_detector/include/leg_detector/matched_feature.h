#ifndef LEG_DETECTOR_MATCHED_FEATURE
#define LEG_DETECTOR_MATCHED_FEATURE

#include <leg_detector/laser_processor.h>

class MatchedFeature
{
public:
  laser_processor::SampleSet* candidate_;
  std::shared_ptr<SavedFeature> closest_;
  float distance_;
  double probability_;

  MatchedFeature(laser_processor::SampleSet* candidate, std::shared_ptr<SavedFeature> closest, float distance, double probability)
    : candidate_(candidate)
    , closest_(closest)
    , distance_(distance)
    , probability_(probability)
  {}

  inline bool operator< (const MatchedFeature& b) const
  {
    return (distance_ <  b.distance_);
  }
};

#endif
