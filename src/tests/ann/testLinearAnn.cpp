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
  config.LearningRate = 0.01;

  NN.setup(config);
  std::cout << "======= After Setup =======" << std::endl;
  NN.print(std::cout);

  Eigen::VectorXd inputs(4);
  inputs << 0.20, 0.40, 0.60, 0.80;

  NN.Forward(inputs);
  std::cout << "====== After forward ======" << std::endl;
  NN.print(std::cout);

  Eigen::VectorXd outputs(1);
  outputs << 1.0;


  // NN.CalculateErrors(outputs);
  // std::cout << "====== After CalculateErrors ======" << std::endl;
  // NN.print(std::cout);

  // NN.UpdateWeights();
  // std::cout << "====== After update ======" << std::endl;
  // NN.print(std::cout);

  NN.Backward(outputs);
  std::cout << "====== After backward ======" << std::endl;
  NN.print(std::cout);

  std::cout << "\n\nByebye...  " << std::endl;

  return 0;
}
