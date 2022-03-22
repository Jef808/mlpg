#include "LinearAnn.h"
#include "Helper.h"
#include "utils/stopwatch.h"

#include <Eigen/Core>


#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <numeric>


namespace simple {

utils::Stopwatch timer;


  void myANN::setup(Config&& config) {
    config = std::move(config);
    std::cout << "Entering constructor" << std::endl;

    // Add the bias extra row
    Eigen::Vector<float, 4> input_layer{};
    Eigen::Vector<float, 2> layer1{ 0.5 };
    Eigen::Vector<float, 2> output_layer{};

    size_t w1 = config.InputSize;
  }

}
