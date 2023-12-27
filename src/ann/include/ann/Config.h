#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>
#include <vector>

#include "Eigen/Dense"

namespace simple {

  class TemplateAnn;

      struct Config {

        Eigen::Index InputSize;
        Eigen::Index OutputSize;
        std::vector<Eigen::Index> HiddenLayers {};

        size_t n_data;
        double training_ratio      { 0.70 };
        double validation_ratio    { 0.15 };
        double testing_ratio       { 0.15 };

        ptrdiff_t batch_size { 1 };

        double LearningRate;
        double MomentumGain  { 0 };
        double L2RegCoeff    { 0 };

        bool monitor_loss_per_batch_while_training { false };
        bool monitor_accuracy_while_training { false };
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
