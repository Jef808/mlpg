#ifndef HELPER_H_
#define HELPER_H_

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

template<typename InputDerived, typename OutputDerived>
void ReLu(const & input, Eigen::ArrayBase<InputDerived>& output) {
    std::transform(input.begin(), input.end(), output.begin(),
                 [](auto el){ return el > 0.0 ? el : 0.0; });
}

template<typename InputDerived, typename OutputDerived>
static void Sigmoid(const Eigen::ArrayBase<InputDerived>& x, Eigen::ArrayBase<OutputDerived>& y) {
    y = 1.0 / (1 + Eigen::exp(-x));
}

template<typename InputDerived, typename OutputDerived>
static void SigmoidDeriv(const Eigen::ArrayBase<InputDerived>& y, Eigen::ArrayBase<OutputDerived>& x) {
    x = y % (1.0 - y);
}


#endif // HELPER_H_
