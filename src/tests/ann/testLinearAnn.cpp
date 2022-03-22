#include "LinearAnn.h"
#include <iostream>

#include <Eigen/Core>

using namespace simple;

int main(int argc, char *argv[]) {


  Eigen::VectorXf vec = Eigen::VectorXf::Zero(100);

  std::cout << "Hello..." << std::endl;

  myANN network{};
  Config conf;

  network.setup(std::move(conf));

  std::cout << "Byebye...  " << std::endl;

  return 0;
}
