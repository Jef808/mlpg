#ifndef CONFIG_H
#define CONFIG_H

#include "Eigen/Core"

#include <vector>


namespace simple {

      struct Config {
          Eigen::Index InputSize;
          Eigen::Index OutputSize;
          std::vector<Eigen::Index> HiddenLayers;
          double LearningRate;
        double MomentumGain;
        Eigen::Index batch_size;
  };



namespace er {

    struct Config {
          Eigen::Index InputSize;
          Eigen::Index OutputSize;
          std::vector<Eigen::Index> HiddenLayers;
          double LearningRate;
      double MomentumGain;
      Eigen::Index batch_size;
  };

}

}

#endif /* CONFIG_H */
