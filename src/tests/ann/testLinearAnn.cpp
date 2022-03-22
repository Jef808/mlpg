#include "LinearAnn.h"
#include <iostream>

#include <Eigen/Core>

using namespace simple;

int main(int argc, char *argv[]) {

  std::cout << "\n\nHello..." << std::endl;
  Config config;
  myANN NN;

  config.InputSize = 4;
  config.OutputSize = 1;
  config.HiddenLayers  = { 2, 2 };

  NN.setup(config);
  NN.print(std::cout);

  // Eigen::VectorXd inputs;

  // NN.Forward(inputs);

  std::cout << "\n\nByebye...  " << std::endl;

  return 0;
}
