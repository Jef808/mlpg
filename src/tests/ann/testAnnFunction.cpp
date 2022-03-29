#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include "TemplateAnn.h"
#include "FunctionExampleData.h"
#include "Function.h"

#include "Eigen/Core"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <fstream>

using namespace simple;


constexpr auto n_train = 1024;
const auto FN = finite_domain_lambda(-6, 6, [](auto x) { return std::cos(x); });
constexpr auto batch_size = 32;
constexpr auto n_test = 128;
constexpr size_t n_batch = n_train / batch_size;
constexpr auto InputSize = 1;
constexpr auto OutputSize = 1;
constexpr size_t n_epochs = 2500;
constexpr double learning_rate = 0.1;

int main() {
    spdlog::set_level(spdlog::level::debug);

    // Setup training data
    FunctionDataCollector DC{ FN };

    std::vector<double> train_x;
    std::vector<double> train_y;
    std::vector<double> validation_x;
    std::vector<double> validation_y;
    std::vector<double> predictions(n_test, 0.0);

    DC.Training(n_train, train_x, train_y);
    DC.Training(n_test, validation_x, validation_y);

    SPDLOG_INFO("Generated {} training points", n_train);

    ANN NN;
    Config config;
    config.InputSize = InputSize;
    config.OutputSize = OutputSize;
    config.HiddenLayers = { 10 };
    config.batch_size = batch_size;
    config.LearningRate = learning_rate;
    NN.setup(config);
    SPDLOG_INFO("Set up the network");
    NN.print(std::cout);

    std::ofstream ofs { "predictions.csv" };
    for (auto i = 0; i < n_test - 1; ++i)
        ofs << validation_x[i] << ',';
    ofs << validation_x[n_test - 1] << '\n';
    for (auto i = 0; i < n_test - 1; ++i)
        ofs << validation_y[i] << ',';
    ofs << validation_y[n_test-1] << '\n';

    for (auto epoch = 0; epoch < n_epochs; ++epoch)
    {
        SPDLOG_INFO("Epoch {}", epoch);

        DC.shuffle(train_x, train_y);
        SPDLOG_INFO("Shuffled training data");

        // Train each batch
        for (auto i = 0; i < n_batch; ++i)
        {
            Eigen::MatrixXd batch_x = Eigen::Map<Eigen::MatrixXd>(train_x.data() + i * batch_size * InputSize,
                                                                  InputSize,
                                                                  batch_size);
            Eigen::MatrixXd batch_y = Eigen::Map<Eigen::MatrixXd>(train_y.data() + i * batch_size * OutputSize,
                                                                  OutputSize,
                                                                  batch_size);

            Eigen::MatrixXd outputs(OutputSize, batch_size);
            outputs << Eigen::MatrixXd::Zero(OutputSize, batch_size);

            NN.Forward(batch_x, outputs);

            // SPDLOG_INFO("Outputs of Forward");
            // std::cout << outputs.eval() << std::endl;

            Eigen::MatrixXd errors(OutputSize, batch_size);
            errors << Eigen::MatrixXd::Zero(OutputSize, batch_size);

            NN.Error(batch_y, outputs, errors);

            // SPDLOG_INFO("Expected:");
            // std::cout << batch_y << std::endl;
            // SPDLOG_INFO("Errors:");
            // std::cout << errors << std::endl;

            NN.BackpropagateError(errors);

            NN.CalculateGradients();

            NN.UpdateWeights();
        }

        double cost = 0.0;

        for (auto i = 0; i < n_test / batch_size; ++i)
        {
            Eigen::MatrixXd test_x = Eigen::Map<Eigen::MatrixXd>(validation_x.data() + i * batch_size * InputSize,
                                                                 InputSize,
                                                                 batch_size);
            Eigen::MatrixXd test_y = Eigen::Map<Eigen::MatrixXd>(validation_y.data() + i * batch_size * OutputSize,
                                                                 OutputSize,
                                                                 batch_size);
            Eigen::Map<Eigen::MatrixXd> outputs(predictions.data() + i * batch_size * OutputSize,
                                                OutputSize,
                                                batch_size);
            NN.Forward(test_x, outputs);
            cost += ((test_y - outputs).colwise().squaredNorm()).rowwise().sum().value();
        }

        SPDLOG_INFO("Cost after epoch {}: {}", epoch, cost / (2*batch_size));

        for (auto i = 0; i < n_test - 1; ++i) {
            ofs << predictions[i] << ',';
        }
        ofs << predictions[n_test-1] << std::endl;

        if (cost < 0.05)
            break;
    }
}
