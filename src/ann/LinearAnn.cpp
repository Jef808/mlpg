/**
 * Our sample network has shape as follows.
 *
 *   (4 INPUT NODES)    --->  (2 ACTIVATION NODES)    ---->   (2 ACTIVATION NODES)      ----->   (OUTPUT NODE)
 *
 * Because we put the bias in the weight matrices for convenience (except for the last node), we thus
 * get 3 weight matrices W1, W2, W3, of dimension (3 x 4), (3 x 2), (3 x 1) respectively.
 *
 * We encode the weight matrix W as a block
 *
 *  W  | b^T
 *  0  | 1
 *
 *  So that with the new W'... W' a = W a + b
 *
 */

#include "LinearAnn.h"
#include "ViewCmd.h"
#include "utils/stopwatch.h"

#include <iostream>
#include <vector>

#include <type_traits>

namespace simple {

utils::Stopwatch timer;

void
myANN::setup(Config config) {

    m_config = std::move(config);

    // Clear the state
    layout.clear();
    layers.clear();
    deltas.clear();
    weights.clear();

    n_layers = m_config.HiddenLayers.size() + 2;
    auto input_size = layout.emplace_back(m_config.InputSize);

    for (Index i = 0; i < n_layers - 1; ++i) {
        auto output_size = i < n_layers - 2 ? m_config.HiddenLayers[i]
                                            : m_config.OutputSize;

        layers.emplace_back(input_size + 1) << Eigen::VectorXd::Random(input_size), 1.0;
        cache.emplace_back(input_size + 1) << layers.back();
        deltas.emplace_back(input_size + 1) << Eigen::VectorXd::Zero(input_size + 1);
        if (i < n_layers - 2)
            weights.emplace_back(output_size + 1, input_size + 1) <<
                Eigen::MatrixXd::Random(output_size, input_size + 1),
                Eigen::MatrixXd::Zero(1, input_size), 1.0;
        else
            weights.emplace_back(output_size, input_size + 1) <<
                Eigen::MatrixXd::Random(output_size, input_size + 1);

        input_size = layout.emplace_back(output_size);
    }

    layers.emplace_back(layout.back()) << Eigen::VectorXd::Zero(layout.back());
}

void myANN::Forward(const Eigen::VectorXd& input) {
    // assign inputs to the first layer
    layers[0].head(input.size()) = input;

    for (Index i = 0; i < n_layers - 1; ++i) {
        cache[i+1] = layers[i+1];
        layers[i+1] = weights[i] * layers[i];
    }
}

void myANN::CalculateErrors(const Eigen::VectorXd& target) {

}

void myANN::Backward(const Eigen::VectorXd& target) {
    // assign outputs to the last layer
    layers.back() = target;
}

void myANN::print(std::ostream& out) {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    for (Index i = 0; i < n_layers - 1; ++i) {
        std::cout << "\n************* LAYER " << i << " ******************\n\n"
                  << "    input: " << layout[i] << "  ====>>   " << layout[i+1] << " :output"
                  << "\n\nWEIGHTS:\n"
                  << weights[i].format(CleanFmt)
                  << "\n\nInput values:\n" << layers[i].format(CleanFmt)
                  << "\n\ndeltas:\n" << deltas[i].format(CleanFmt)
                  << "\n\n****************************************\n" << std::endl;
    }
    show_compilation_timestamp(out);
}

void myANN::print_config(std::ostream& out) {
    std::cout << "\n    Input Size: "  << m_config.InputSize
              << "    Output Size: " << m_config.OutputSize
              << "    Hidden Layers: ";
    for (const auto& hl : m_config.HiddenLayers) std::cout << hl << ", ";

    show_compilation_timestamp(out);
    std::cout << std::endl;
}

}  // namespace simple
