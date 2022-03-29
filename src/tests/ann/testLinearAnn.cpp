#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL DEBUG
#endif

#include "TemplateAnn.h"
#include "Function.h"
#include "FunctionExampleData.h"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

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



template <typename F>
class Worker {
public:
    Worker(FunctionDataCollector<F>* DC_, ANN* NN_) : DC{DC_}, NN{NN_} {}

    /**
     * Run @n_tests predictions with the current state of @nn and
     * return the average error.
     *
     * The sequence of predictions/actual is also available with a call to `results()`
     */
    bool Evaluate(const Eigen::Index n_tests) {
        m_config = NN->get_config();
        this->n_tests = n_tests;

        const double minimum_precision = 0.00001f;
        DC->ResetTestData(n_tests, minimum_precision);  // Throws assert if any error is bigger
        auto pdata = DC->pData_Testing();

        // Store pointer to data (stored in std::vector's)
        ptest_x = pdata.first;
        ptest_y = pdata.second;

        assert(ptest_x != nullptr && ptest_y != nullptr && "ERROR in TestModel::run_test");

        double error_sum = 0.0;

        std::vector<double> outputs(n_tests * m_config.OutputSize, 0.0);

        std::cerr << "WARNING! GOING OVER UNIMPLEMENTED BIT" << std::endl;
        NN->ForwardBatch(outputs.data(), ptest_x, n_tests);

        std::cout << "Back in TestModel, the outputs vector is now\n\n";

        for (auto outs = outputs.begin(); outs < outputs.end(); outs += m_config.OutputSize) {
            std::copy(outs, outs + m_config.OutputSize, std::ostream_iterator<double>(std::cout, " "));
            std::cout << std::endl;
        }

        return true;
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
    ANN* NN;
    FunctionDataCollector<F>* DC;
};

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {

    SPDLOG_DEBUG("Entering main");

    constexpr Eigen::Index tr_size = 2000;
    constexpr Eigen::Index n_epochs = 15;
    constexpr Eigen::Index batch_size = 32;
    // The standard deviation of the Gaussian noise injected
    // into the training samples by the data collector
    double var12 = 0.15;

    // The domain of the unknown function
    double xmin = -4 * 3.1456;
    double xmax = 4 * 3.1456;

    // The function we will try to fit a simple network to
    auto FDL = finite_domain_lambda(xmin, xmax, [](double x) { return std::cos(x); });

    ////////////////////////////////////////////////////////////
    // Configure the network
    ////////////////////////////////////////////////////////////
    Config config;
    ANN NN;

    constexpr Eigen::Index InputSize_ = 1;
    constexpr Eigen::Index OutputSize_ = 1;
    config.InputSize = InputSize_;
    config.OutputSize = OutputSize_;
    config.HiddenLayers = {10, 10};
    config.LearningRate = 0.1;

    NN.setup(config);
    std::cout << "Set up network" << std::endl;

    ////////////////////////////////////////////////////////////
    // Build the training data
    ////////////////////////////////////////////////////////////
    // auto FN = Fn{ f };
    auto DC = DataCollector(FDL);
    DC.InitTrainingData(tr_size);

    ////////////////////////////////////////////////////////////
    // Record the evolution of the model
    ////////////////////////////////////////////////////////////
    auto Evaluator = Worker{&DC, &NN};

    std::vector<std::vector<std::pair<double, double>>> results;
    std::vector<double> average_errors;

    ////////////////////////////////////////////////////////////
    // WE OWN THE BATCH DATA, OFFER VIEWS
    ////////////////////////////////////////////////////////////
    std::vector<double> buffer_x(InputSize_ * batch_size, 0.0);
    std::vector<double> buffer_y(OutputSize_ * batch_size, 0.0);

    ////////////////////////////////////////////////////////////
    // **************  EPOCH LOOP  ****************************
    ////////////////////////////////////////////////////////////
    for (Eigen::Index i = 0; i < n_epochs; ++i) {

        std::cout << "Epoch " << i << std::endl;


        ////////////////////////////////////////////////////////////////
        // USE EIGEN::MAP TO MAKE DATA AVAILABLE
        ///////////////////////////////////////////////////////////////
        // stacks of column vectors for inputs
        // (we interpret the one-dimensional data as sitting in a
        // Row-Major matrix)
        //Eigen::Map<Eigen::MatrixXd> map_x(buffer_x.data(), batch_size, InputSize_);
        // stacks of row vectors for outputs
        //Eigen::Map<Eigen::MatrixXd> map_y(buffer_y.data(), OutputSize_, batch_size);

        // NOTE :This is how to trick the compiler into not making temporary copies,
        // the above leads to the same problem later
        Eigen::MatrixXd map_x =
            Eigen::Map<const Eigen::MatrixXd>(
                buffer_x.data(), // pointer to data
                batch_size,      // # Rows
                InputSize_);     // # Cols
        Eigen::MatrixXd map_y =
            Eigen::Map<Eigen::MatrixXd>(
                buffer_y.data(), // pointer to data
                InputSize_,      // # Rows
                batch_size);     // # Cols


        ////////////////////////////////////////////////////////////
        // Run a training step
        ////////////////////////////////////////////////////////////
        // for (Eigen::Index i = 0; i < batch_size; ++i) {  // batch_size - 1; ++i) {
        //     // TODO Implement Forward and Backward compatibility
        //     // with batch runs
        //     NN.Forward(sample_x.col(i));

        //     NN.Backward(sample_y.col(i));
        // }
        //
        // std::cout << "\nTraining phase completed\n" << std::endl;

        ////////////////////////////////////////////////////////////
        // Run the batch
        ////////////////////////////////////////////////////////////

        Eigen::MatrixXd outputs = Eigen::MatrixXd(batch_size, OutputSize_);

        NN.Forward(map_x, outputs);

        Eigen::MatrixXd Check(OutputSize_, batch_size);
        Check << outputs;

        /////////////////////////////////////////////////////////l///
        // Update the weights of the model according the that
        // last training step
        ////////////////////////////////////////////////////////////

        NN.UpdateWeights(batch_size);

        constexpr size_t n_tests = 50;

        Evaluator.Evaluate(n_tests);

        auto& avg_error = average_errors.emplace_back(Evaluator.average_loss());
        results.emplace_back(Evaluator.results());

        // template<Eigen::Index test_size, typename F, Eigen::Index InSize, Eigen::Index OutSize>
        // double TestModel(ANN* NN, const Fn<F>& fn, Eigen::Index size = test_size)

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
