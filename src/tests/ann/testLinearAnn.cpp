#include "LinearAnn.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <iterator>

#include "Eigen/Core"
#include "Eigen/StdVector"

using namespace simple;

Eigen::IOFormat Clean(4, 0, ", ", "\n", "[", "]");

/**
 * Wrap a function of one variable @F to be able to act component-wise on tensors
 */
template <typename FImpl>
struct RealFunction {
    RealFunction(double x_min_, double x_max_, FImpl impl_ = FImpl{})
            : impl{impl_}, x_min{x_min_}, x_max{x_max_} {}

  // This is impl itself
  double operator()(double x) const { return impl(x); }

  // Recipe to act on containers (Just works with std::vector really)
  template< template<class, class> typename F, typename T, typename Y >
  auto operator()(F<Y, T>& out, const F<Y, T>& in) const {
    return std::transform(std::begin(in), std::end(in), std::back_inserter(out),
                          [&me=impl](const Y& y) { return me(y); });
  }

  // template<template<class>typename EO, typename Derived1, typename Derived2 = Derived1, typename T = void>
  // auto operator()(EO< Derived1 >& out, EO< Derived2 >& in) {
  //     (out.colwise()).rowwise() = this->impl((in.colwise()).rowwise());
  // }c

  template<typename Derived1, typename Derived2>
  void operator()(Eigen::DenseBase<Derived1>& out, const Eigen::DenseBase<Derived2>& in)  {
    this->impl(out.colwise().rowwise()) = this->impl(out.colwise().rowwise());
  }
  
  // auto operator()(std::vector<double>& out, const std::vector<double>& in) const {
  //   using V = std::vector<double>;
  //   return std::transform(std::begin(in), std::end(in), std::back_inserter(out),
  //                         [&me=impl](const double& x){ return me(x); });
  // }

  auto operator()(std::vector<double>& out) const {
    return [&](const std::vector<double>& in) {
      return this->operator()(out, in);
    }; 
  }

    FImpl impl;
    double x_min = -10.0;
    double x_max = 10.0;
};

auto finite_domain_lambda = []<typename F>(double xmin, double xmax, F f) {
    return RealFunction(xmin, xmax, f);
};

template <typename F>
class DataCollector {
public:
    DataCollector(RealFunction<F> f) : FN{f} {}

    std::pair<double*, double*> pData_Training() {
        return std::make_pair(buf_trainx.data(), buf_trainy.data());
    }

    // TODO Check that data is still there
    std::pair<double*, double*> pData_Testing() { return std::make_pair(buf_testx.data(), buf_testy.data()); }

    std::pair<double*, double*> pData_Sample(const size_t N, const double var12) {
        generate_sample(N, var12);
        return std::make_pair(buf_samplex.data(), buf_sampley.data());
    }

    void InitTrainingData(const size_t n_train) {
        this->n_train = static_cast<Eigen::Index>(n_train);

        fill_domain_random(buf_trainx, n_train);
        FN(buf_trainy, buf_trainx);
    }

    void ResetTestData(const size_t n_tests) {
        fill_domain_random(buf_testx, n_tests);
        FN(buf_testy, buf_testx);
    }

    [[nodiscard]] double x_min() const { return FN.xmin; }
    [[nodiscard]] double x_max() const { return FN.xmax; }

private:
    std::random_device rd;
    std::mt19937 eng;

    RealFunction<F> FN;
    Eigen::Index n_train;
    std::vector<double> buf_trainx;
    std::vector<double> buf_trainy;
    std::vector<double> buf_samplex;
    std::vector<double> buf_sampley;
    std::vector<double> buf_testx;
    std::vector<double> buf_testy;
    bool initialized;

    std::vector<double>& fill_domain_random(std::vector<double>& inputs, const size_t N) {
        inputs.clear();

        std::uniform_real_distribution<> dist(FN.x_min, FN.x_max);
        std::generate_n(std::back_inserter(inputs), N, [&] { return dist(eng); });
        return inputs;
    }

    double add_gaussian_noise(double a, double var12 = 1.0) {
        std::normal_distribution<> dist{0, var12};
        return a + dist(eng);
    }

    template <typename Fun>
    auto add_gaussian_noise(Fun fun, double var12 = 1.0) {
        std::normal_distribution<> dist{0, var12};
        return [f = fun, n = dist(eng)](const auto& w) { return n + f(w); };
    }

    void add_gaussian_noise(std::vector<double>& inputs, const double var12) {
        std::normal_distribution<> dist(0, var12);
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto d) { return d + dist(eng); });
    }

    void generate_sample(const size_t N, double var12 = 1.0) {
        buf_samplex.clear();
        buf_sampley.clear();
        std::uniform_int_distribution<> dist(0, buf_trainx.size() - 1);
        std::generate_n(std::back_inserter(buf_samplex), N, [&] { return buf_trainx[dist(eng)]; });

        FN(buf_sampley, buf_samplex);
        add_gaussian_noise(buf_sampley, var12);
    }
};

// TODO: I need to group all those resources somewhere... this function can be
// called with different template parameters every time, so it will initialize a bunch
// of those static vectors anyway....
template <typename F>
class TestModel {
public:
    TestModel(DataCollector<F>* DC_, myANN* NN_) : DC{DC_}, NN{NN_} {}

    /**
     * Run @n_tests predictions with the current state of @nn and
     * return the average error.
     *
     * The sequence of predictions/actual is also available with a call to `results()`
     */
    bool run_test(const Eigen::Index n_tests) {
        // build_trainng will clear the other two
        // predictions_.clear();
        m_config = NN->get_config();
        this->n_tests = n_tests;

        DC->ResetTestData(n_tests);
        auto pdata = DC->pData_Testing();

        // Store pointer to data
        ptest_x = pdata.first;
        ptest_y = pdata.second;

        // Map the sequence of inputs/outputs as a matrix
        // NOTE: Compare with
        // Eigen::Map<const Eigen::MatrixXd> Y (ptest_y, OutSize, test_size);
        // which is the compile-time-fixed version
        Eigen::MatrixXd test_x = Eigen::Map<Eigen::MatrixXd>(ptest_x, m_config.InputSize, n_tests);
        Eigen::MatrixXd test_y = Eigen::Map<Eigen::MatrixXd>(ptest_y, m_config.OutputSize, n_tests);

        double error_sum = 0.0;
        bool okay = NN->Predict(test_x, test_y, buf_preds.data(), error_sum);

        if (not okay)
            std::cerr << "ERROR! The Predict method returned with error (most likely failed to map the space "
                         "of predictions)"
                      << std::endl;

        return okay;
    }

    [[nodiscard]] double average_loss() const { return error_sum / static_cast<double>(n_tests); }

    /**
     * Return the sequence of pairs prediction/actuals available after evaluating the model
     */
    [[nodiscard]] std::vector<std::pair<double, double>> results() const {
        std::vector<std::pair<double, double>> result;
        double* py = ptest_y;
        std::transform(ptest_y, ptest_y + n_tests, buf_preds.begin(), std::back_inserter(result),
                       [](const auto& y, const auto& p) { return std::make_pair(y, p); });
        return result;
    }

private:
    double* ptest_x;
    double* ptest_y;
    size_t n_tests;

    std::vector<double> buf_preds;
    double error_sum = 0.0;
    Config m_config;
    myANN* NN;
    DataCollector<F>* DC;
};

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    constexpr Eigen::Index tr_size = 5000;
    constexpr Eigen::Index n_epochs = 10;
    constexpr Eigen::Index minibatch_size = 100;
    using TrainT = Eigen::ArrayXd;
    using SampleT = Eigen::ArrayXd;
    // The standard deviation of the Gaussian noise injected
    // into the training samples by the data collector
    double var12 = 0.15;

    double xmin = -4 * 3.1456;
    double xmax = 4 * 3.1456;

    auto FDL = finite_domain_lambda(xmin, xmax, [](double x) { return std::cos(x); });

    ////////////////////////////////////////////////////////////
    // Build the training data
    ////////////////////////////////////////////////////////////
    // auto FN = Fn{ f };
    auto DC = DataCollector(FDL);
    DC.InitTrainingData(tr_size);

    constexpr Eigen::Index InputSize_ = 1;
    constexpr Eigen::Index OutputSize_ = 1;

    ////////////////////////////////////////////////////////////
    // Initialize the neural network
    ////////////////////////////////////////////////////////////
    Config config;
    myANN NN;

    config.InputSize = InputSize_;
    config.OutputSize = OutputSize_;
    config.HiddenLayers = {10, 10};
    config.LearningRate = 0.1;

    NN.setup(config);
    std::cout << "Set up network" << std::endl;

    ////////////////////////////////////////////////////////////
    // Record the evolution of the model
    ////////////////////////////////////////////////////////////
    auto Evaluator = TestModel{&DC, &NN};

    std::vector<std::vector<std::pair<double, double>>> results;
    std::vector<double> average_errors;

    ////////////////////////////////////////////////////////////
    // The objects whose data will be updated with the
    // training samples
    ////////////////////////////////////////////////////////////
    Eigen::MatrixXd sample_x;
    Eigen::MatrixXd sample_y;

    ////////////////////////////////////////////////////////////
    // MAIN LOOP
    ////////////////////////////////////////////////////////////
    for (Eigen::Index i = 0; i < n_epochs; ++i) {
        std::cout << "Epoch " << i << std::endl;

        ////////////////////////////////////////////////////////////
        // Collect trainng data
        ////////////////////////////////////////////////////////////

        auto [psample_x, psample_y] = DC.pData_Sample(minibatch_size, var12);

        // TODO A call to dc.pData_Sample might just be enough to update the Eigen::Map
        // view. This would only need to be created once
        new (&sample_x) Eigen::Map<Eigen::MatrixXd>(psample_x, InputSize_, minibatch_size);
        new (&sample_y) Eigen::Map<Eigen::MatrixXd>(psample_y, OutputSize_, minibatch_size);

        std::cout << "Constructed sample_x and sample_y, starting training." << std::endl;
        
        ////////////////////////////////////////////////////////////
        // Run a training step
        ////////////////////////////////////////////////////////////
        for (Eigen::Index i = 0; i < minibatch_size; ++i) {//minibatch_size - 1; ++i) {
              // TODO Implement Forward and Backward compatibility
            // with batch runs
            NN.Forward(sample_x.col(i));

            NN.Backward(sample_y.col(i));
        }

        std::cout << "\nTraining phase completed\n" << std::endl;
        
        /////////////////////////////////////////////////////////l///
        // Update the weights of the model according the that
        // last training step
        ////////////////////////////////////////////////////////////

        NN.UpdateWeights(minibatch_size);

        constexpr size_t n_tests = 50;

        Evaluator.run_test(n_tests);

        auto& avg_error = average_errors.emplace_back(Evaluator.average_loss());
        results.emplace_back(Evaluator.results());

        // template<Eigen::Index test_size, typename F, Eigen::Index InSize, Eigen::Index OutSize>
        // double TestModel(myANN* NN, const Fn<F>& fn, Eigen::Index size = test_size)

        std::cout << "\n*********\nAfter " << i << " epochs, "
                  << "\n  Average error is " << avg_error << std::endl;

    }  // for each epoch

    std::cout << "Done training\n\n***************************\n\n" << std::endl;

    for (auto i = 0; i < n_epochs; ++i) {
        std::cout << "RESULTS:\n"
                  << "Epoch: " << i << ", Average error: " << average_errors[i] << std::endl;

        std::cout << "\n\n The list of predictions made next to the expected output is:\n";

        for (auto i = 0; i < n_epochs; ++i) {
            std::cout << "\n************Epoch " << i << ":\n";
            for (const auto& pred_expec : results[i]) {
                std::cout << "    Expected: " << pred_expec.second << ", Predicted: " << pred_expec.first
                          << '\n';
            }
            std::cout << "average Error for epoch " << i << ": " << average_errors[i] << std::endl;
        }
    }

    std::cout << "\n\nByebye...  " << std::endl;

    return 0;
}
