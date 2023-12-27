#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include "ann/TemplateAnn.h"
#include "ann/Config.h"

#include "Eigen/Dense"
#include "spdlog/spdlog.h"

#include <fstream>
#include <sstream>
#include <vector>

using namespace simple;
using namespace Catch::Matchers;

using FT = float;
using Matrix = Eigen::MatrixXf;

TEST_CASE("Basic single neuron network", "[Perceptron]") {
    spdlog::set_level(spdlog::level::debug);

    std::vector<std::vector<float>> weights = {{0.5, -0.5}};
    std::vector<std::vector<float>> bias = {{0.3}};

    ANN<FT> neuron;
    Config config;

    config.InputSize = 2;
    config.OutputSize = 1;
    config.batch_size = 4;

    neuron.setup(config);
    neuron.load_weights(weights.begin(), weights.end(), bias.begin(), bias.end());

    Matrix inputs(2, 4);
    inputs << 1.0, 0.0, 1.0, 0.0,
              0.0, 1.0, 1.0, 0.0;

    Matrix expected_outputs(1, 4);
    expected_outputs << 0.8F, -0.2F, 0.3F, 0.3F;

    Matrix output(1, 4);

    neuron.Forward(inputs, output);

    for (Eigen::Index i = 0; i < 4; ++i) {
        REQUIRE_THAT(output(0, i), WithinAbs(expected_outputs(0, i), 0.01));
    }
}

TEST_CASE("Single neuron network for AND operation", "[logical_operations]") {
    spdlog::set_level(spdlog::level::debug);

    std::vector<std::vector<float>> weights = {{1.0, 1.0}};
    std::vector<std::vector<float>> bias = {{-1.5}};

    ANN<FT> and_neuron;
    Config config;

    config.InputSize = 2;
    config.OutputSize = 1;
    config.batch_size = 4;

    and_neuron.setup(config);
    and_neuron.load_weights(weights.begin(), weights.end(), bias.begin(), bias.end());

    Eigen::MatrixXf inputs(2, 4);
    inputs << 0.0, 0.0, 1.0, 1.0,
              0.0, 1.0, 0.0, 1.0;

    std::vector<float> expected_outputs{0, 0, 0, 1};

    Eigen::MatrixXf outputs(1, 4);

    and_neuron.Forward(inputs, outputs);

    for (Eigen::Index i = 0; i < 4; ++i) {
        REQUIRE((outputs(0, i) > 0) == expected_outputs[static_cast<size_t>(i)]);
    }
}

TEST_CASE("Single neuron network for OR operation", "[logical_operations]") {
    spdlog::set_level(spdlog::level::debug);

    std::vector<std::vector<float>> weights = {{1.0, 1.0}};
    std::vector<std::vector<float>> bias = {{-0.5}};

    ANN<FT> or_neuron;
    Config config;

    config.InputSize = 2;
    config.OutputSize = 1;
    config.batch_size = 4;

    or_neuron.setup(config);
    or_neuron.load_weights(weights.begin(), weights.end(), bias.begin(), bias.end());

    Matrix inputs(2, 4);
    inputs << 0.0, 0.0, 1.0, 1.0,
              0.0, 1.0, 0.0, 1.0;

    std::vector<float> expected_outputs{0, 1, 1, 1};

    Matrix outputs(1, 4);

    or_neuron.Forward(inputs, outputs);

    for (Eigen::Index i = 0; i < 4; ++i) {
        REQUIRE((outputs(0, i) > 0) == expected_outputs[static_cast<size_t>(i)]);
    }
}

TEST_CASE("Multiple neuron network for XOR operation", "[logical_operations]") {
    spdlog::set_level(spdlog::level::debug);

    std::vector<std::vector<float>> weights = {{1.0, 1.0}};
    std::vector<std::vector<float>> bias = {{-0.5}};

    ANN<FT> or_neuron;
    Config config;

    config.InputSize = 2;
    config.OutputSize = 1;
    config.batch_size = 4;

    or_neuron.setup(config);
    or_neuron.load_weights(weights.begin(), weights.end(), bias.begin(), bias.end());

    Matrix inputs(2, 4);
    inputs << 0.0, 0.0, 1.0, 1.0,
              0.0, 1.0, 0.0, 1.0;

    std::vector<float> expected_outputs{0, 1, 1, 1};

    Matrix outputs(1, 4);

    or_neuron.Forward(inputs, outputs);

    for (Eigen::Index i = 0; i < 4; ++i) {
        REQUIRE((outputs(0, i) > 0) == expected_outputs[static_cast<size_t>(i)]);
    }
}
