#include "LinearAnn.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <iterator>

#include "Eigen/Dense"

using namespace simple;

Eigen::IOFormat Clean(4, 0, ", ", "\n", "[", "]");


template<typename OutputIter>
void whitespace(OutputIter& out, size_t N) {
  std::generate_n(std::ostream_iterator<char>(out), N, []{ return ' '; });
}

template<typename F, typename Derived>
struct Fn {
  Fn(F f_) : f(f_) {}
  void operator()(const Eigen::MatrixBase<Derived>& in, Eigen::MatrixBase<Derived>& out) {
    in.colwise();
  }
  F f;
};


template<typename Derived>
void plot(std::ostream& out,
          const Eigen::ArrayBase<Derived>& sample_x,
          const Eigen::ArrayBase<Derived>& sample_y,
          const double xmin,
          const double xmax) {

  auto x_size = sample_x.size();
  auto x_max = xmax > 80 ? 80 : x_size;
  auto y_max = (x_size) > 50 ? 50 : x_size;
  auto y_min = 2;

  // f -= -std::abs(f.minCoeff()) - 0.00001;

  // f -= -std::abs(f.minCoeff()) - y_min - 0.001;

  Eigen::ArrayXd data_y = sample_y;
  data_y -= -std::abs(data_y.minCoeff() - 2.0001);

  data_y *= 0.75 * (y_max - y_min) / (data_y.maxCoeff());

  assert(y_max - data_y.maxCoeff() - 1 > 0  && "Error: found with f(x) above limit");
  assert(data_y.minCoeff() - 1 > 0 && "Error: found f(x) <= 0");

  std::vector<std::string> plot;
  Eigen::ArrayXi heights = Eigen::ArrayXi::Constant(xmax-xmin, 0);

  heights = data_y.template cast<int>();

  assert(y_max - heights.maxCoeff() - 1 > 0  && "Error: found height above limit");
  assert(heights.minCoeff() - 1 > 0 && "Error: found height <= 0");

  for (Eigen::Index i = 0; i < x_size; i += x_size / x_max) {
    std::ostringstream ss;
    whitespace(ss, y_max - heights[i] - 1);
    ss << 'o';
    whitespace(ss, heights[i] - 1);
    ss << '_';
    whitespace(ss, y_min);
    plot.emplace_back(ss.str());
  }

  bool all_same = std::all_of(plot.begin(), plot.end(),
                              [n=plot.front().size()](const auto& line) { return line.size() == n; });
  assert(all_same && "Error: Got verticals of different lengths in plot");

  int nx = plot.size();
  int ny = plot.back().size();

  for (int j=0; j<ny; ++j) {
     for (int i=0; i<nx; ++i) {
       out << plot[plot.size() - 1 - i][j];
     }
     out << '\n';
  }
  out << std::endl;
}


template<typename F, typename Derived>
void build_training(const Fn<F, Derived>& fun,
                    const double xmin,
                    const double xmax,
                    typename Eigen::DenseBase<Derived>& train_x,
                    typename Eigen::DenseBase<Derived>& train_y)
{
  constexpr Eigen::Index tr_size = Eigen::MatrixBase<Derived>::ColsAtCompileTime;

  train_x.setLinSpaced(xmin, xmax);
  fun(train_x, train_y);

  assert(train_x.cols() == tr_size && "Failed to initialize sample_x");
  assert(train_y.cols() == tr_size && "Failed to initialize sample_y");
}


template<typename Derived1, typename Derived2>
void sample(const Eigen::ArrayBase<Derived1>& train_x,
            const Eigen::ArrayBase<Derived1>& train_y,
            const double epsilon,
            const Eigen::Index minibatch_size,
            const Eigen::ArrayBase<Derived2>& sample_x,
            const Eigen::ArrayBase<Derived2>& sample_y)
{
  assert(train_x.size() > 0 && train_y.size() == train_x.size() && "Error: training dataset is not defined");

  std::cout << "\nIn sample:\n"
            << "\nTrain_x;\n" << train_x.transpose().format(Clean)
            << "\nTrain_y:\n" << train_y.transpose().format(Clean) << std::endl;

  using Scalar = typename Derived2::Scalar;
  using RowVectorType = typename Eigen::internal::plain_row_type<Derived2>::type;

  static std::vector<Eigen::Index> ndx(train_x.size(), 0);
  static std::random_device rd;
  static std::mt19937 eng{ rd() };
  std::normal_distribution<> d{0, 1};


  Eigen::ArrayXd rand = Eigen::ArrayXd::Random(train_x.size());

  std::iota(ndx.begin(), ndx.end(), 0);
  static std::vector<double> noise;
  noise.clear();

  for (int i = 0; i < minibatch_size; ++i) {
    noise.push_back(d(eng));
  }

  std::partial_sort(ndx.begin(), ndx.begin() + minibatch_size, ndx.end(),
                    [&rand](const auto a, const auto b) { return rand[a] < rand[b]; });

  auto n = ndx.begin();
  for (Eigen::Index i = 1; i < minibatch_size; ++i, ++n) {

    const_cast<Eigen::ArrayBase<Derived2>&>(sample_x).coeffRef(i) = train_x.coeff(*n);
    const_cast<Eigen::ArrayBase<Derived2>&>(sample_y).coeffRef(i) = train_y.coeff(*n) + epsilon * noise[i];
  }
}


int main(int argc, char *argv[]) {

  std::cout << "\n\nHello..." << std::endl;
  Config config;
  myANN NN;

  config.InputSize = 1;
  config.OutputSize = 1;
  config.HiddenLayers  = { 10, 10 };
  config.LearningRate = 0.1;

  double lo = -2 * 3.1456;
  double hi = 2 * 3.1456;
  
  constexpr Eigen::Index tr_size = 1000;
  constexpr Eigen::Index minibatch_size = 100;

  Eigen::ArrayXd train_x = Eigen::ArrayXd::LinSpaced(tr_size, lo, hi);
  Eigen::ArrayXd train_y = Eigen::cos(train_x.array());

  std::cout << "Training data constructed" << std::endl;

  Eigen::Index n_epochs = 30;

  for (Eigen::Index i = 0; i < n_epochs; ++i) {
    Eigen::ArrayXd sample_x = train_x.head(minibatch_size);
    Eigen::ArrayXd sample_y = train_y.head(minibatch_size);

    sample(train_x,
           train_y,
           0.4,
           minibatch_size,
           sample_x,
           sample_y);

    std::cout << "Constructed sample_x and sample_y" << std::endl;

    // plot(std::cout,
    //    train_x,
    //    train_y,
    //    lo,
    //    hi);

    }




  // NN.setup(config);
  // std::cout << "======= After Setup =======" << std::endl;
  // NN.print(std::cout);

  // Eigen::VectorXd inputs(4);
  // inputs << 0.20, 0.40, 0.60, 0.80;

  // NN.Forward(inputs);
  // std::cout << "====== After forward ======" << std::endl;
  // NN.print(std::cout);

  // Eigen::VectorXd outputs(1);
  // outputs << 1.0;

  // NN.Backward(outputs);
  // std::cout << "====== After backward ======" << std::endl;
  // NN.print(std::cout);

  std::cout << "\n\nByebye...  " << std::endl;

  return 0;
}
