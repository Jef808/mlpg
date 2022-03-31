
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include "TemplateAnn.h"
#include "FunctionExampleData.h"
#include "Function.h"

#include "Eigen/Core"
#include "spdlog/spdlog.h"
#include "spdlog/stopwatch.h"

#include <chrono>
#include <fstream>
#include <type_traits>

using namespace simple;


using FT = float;
using Matrix = std::conditional<std::is_same_v<FT, float>, Eigen::MatrixXf, Eigen::MatrixXd>::type;

constexpr auto n_train = 2048;
const auto FN = finite_domain_lambda<FT>(-6, 6, [](auto x) { return std::cos(x) + std::pow(std::cos(x), 2) - std::pow(std::sin(x), 2); });
//const auto FN = finite_domain_lambda(-6, 6, [](auto x) { return std::cos(x*7) + std::cos(x*11); });// + std::cos(x*5 + 0.5) + 0.3;  } );
constexpr auto batch_size = 32;
constexpr size_t n_batch = n_train / batch_size;
constexpr auto n_validation = 256;
constexpr auto InputSize = 1;
constexpr auto OutputSize = 1;
constexpr size_t n_epochs = 2500;
constexpr double learning_rate = 0.2;

int main() {
    spdlog::set_level(spdlog::level::debug);

    // Setup training data
    FunctionDataCollector DC{ FN };

    std::vector<FT> train_x;
    std::vector<FT> train_y;
    std::vector<FT> validation_x;
    std::vector<FT> validation_y;
    std::vector<FT> predictions(n_validation, 0.0);

    DC.Training(n_train, train_x, train_y);
    DC.Training(n_validation, validation_x, validation_y);

    SPDLOG_INFO("Generated {} training points", n_train);

    ANN<FT> NN;
    Config config;
    config.InputSize = InputSize;
    config.OutputSize = OutputSize;
    config.HiddenLayers = { 5, 16, 5 };
    config.batch_size = batch_size;
    config.LearningRate = learning_rate;
    NN.setup(config);
    SPDLOG_INFO("Set up the network");

    std::ofstream ofs { "predictions.csv" };
    for (auto i = 0; i < n_validation - 1; ++i)
        ofs << validation_x[i] << ',';
    ofs << validation_x[n_validation - 1] << '\n';
    for (auto i = 0; i < n_validation - 1; ++i)
        ofs << validation_y[i] << ',';
    ofs << validation_y[n_validation-1] << '\n';

    int streak = 0;
    using std::chrono::milliseconds;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    spdlog::stopwatch timer;
    decltype(duration_cast<milliseconds>(timer.elapsed())) time{};

    auto epoch = 0;
    for (; epoch < n_epochs; ++epoch)
    {
        timer.reset();
        DC.shuffle(train_x, train_y);

        // Train all batches except the last four
        for (auto i = 0; i < n_batch; ++i)
        {
            Eigen::Map<Matrix> batch_x(train_x.data() + static_cast<ptrdiff_t>(i * batch_size * InputSize),
                                                InputSize,
                                                batch_size);
            Eigen::Map<Matrix> batch_y(train_y.data() + static_cast<ptrdiff_t>(i * batch_size * OutputSize),
                                                OutputSize,
                                                batch_size);
            Matrix outputs(OutputSize, batch_size);
            Matrix errors(OutputSize, batch_size);

            NN.Forward(batch_x, outputs);

            NN.Error(batch_y, outputs, errors);

            NN.BackpropagateError(errors);

            NN.CalculateGradients();

            NN.UpdateWeights();
        }

        // Use the last four batches for validation
        double cost = 0.0;
        for (auto i = 0; i < n_validation / batch_size; ++i)
        {
            Eigen::Map<Matrix> test_x(validation_x.data() + static_cast<ptrdiff_t>(i * batch_size * InputSize),
                                      InputSize,
                                      batch_size);
            Eigen::Map<Matrix> test_y(validation_y.data() + static_cast<ptrdiff_t>(i * batch_size * OutputSize),
                                      OutputSize,
                                      batch_size);
            Eigen::Map<Matrix> outputs(predictions.data() + static_cast<ptrdiff_t>(i * batch_size * OutputSize),
                                       OutputSize,
                                       batch_size);
            NN.Forward(test_x, outputs);
            NN.CostSum(test_y, outputs, cost);
        }

        time += duration_cast<milliseconds>(timer.elapsed());

        cost /= (2*batch_size);
        SPDLOG_INFO("Average cost after epoch {}: {}", epoch, cost);

        for (auto i = 0; i < n_validation - 1; ++i) {
            ofs << predictions[i] << ',';
        }
        ofs << predictions[n_validation-1] << std::endl;

        // Break if we get five epochs in a row with low value of the Cost function
        if (cost > 0.005)
            streak = 0;
        else
            streak += 1;
        if (streak > 4)
            break;
    }

    SPDLOG_INFO("Averge time taken per epoch: {}ms", time.count() / epoch);
}
