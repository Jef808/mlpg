/**
 * Consider the example network
 *
 *   (4 INPUT NODES)    --->  (2 ACTIVATION NODES)    ---->   (2 ACTIVATION NODES)      ----->   (1 OUTPUT
 * NODE)
 *
 * We put the bias in the weight matrices for convenience (except for the last node): we thus
 * get 3 weight matrices W1, W2, W3, of dimension (3 x 5), (3 x 3), (3 x 1) respectively.
 *
 * Bias:
 * We encode the weight matrix W as a block W' equal to
 *
 *  W  | b
 *  0  | 1
 *
 * so that with the new W' and inputs a^T extended to (a | 1)^T, we get W' a = W a + b and we drop the b in
 * our equations.
 *
 * Cost:
 * For given training data \tau = (x_in, y_out), we consider the cost function given by (1/2) || y_out -
 * a^L(x_in) ||^2. On a finite sample of training data T inside of the space of all input/output in our data,
 * the (global) cost function is the corresponding MEAN LEAST SQUARE C = (1\2N) \sum_{\tau \in T} || y_out -
 * a^L(x_in) ||^2.
 *
 * In the above, a^L are the activation values of the output layer computed from the Forward pass.
 *
 * We define also delta^l = <partial derivative of C with respect to z^l> the `error` with respect to the
 * weighted inputs at layer l.
 *
 * Then at any fixed \tau = (x_in, y_out) with variable weights, we can write
 * (1) delta_j^l = f'(z_j^l) . [ ((W^(l+1)))^T delta^(l+1)) ]_j
 * (2) <partial C / partial W_jk^l> = a_k^(l-1) . delta_j^l since z_j^l = W_jk^l a_k^(l-1).
 *
 * Starting at the back with delta_j^L = f'(z_j^L) <partial C / partial a_j^L>, we use (2) to compute the
 * <partial C / partial w_jk^l>'s using backpropagation (1). This gives us the amount by which to modify the
 * weights in order to minimize the cost function: W'_jk^l = W_jk^l - a_k^(l-1) . delta_j^l
 */

#include "LinearAnn.h"
#include "ViewCmd.h"
#include "utils/stopwatch.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include <type_traits>

namespace simple {

utils::Stopwatch timer;

void myANN::setup(Config config) {
    m_config = std::move(config);

    // Clear the state
    layout.clear();
    layers.clear();
    deltas.clear();
    weights.clear();
    gradients.clear();

    n_layers = static_cast<Eigen::Index>(m_config.HiddenLayers.size()) + 2;
    learning_rate = m_config.LearningRate;

    auto input_size = layout.emplace_back(m_config.InputSize);
    layers.emplace_back(input_size + 1) << Eigen::VectorXd::Zero(input_size), 1.0;

    for (Index i = 0; i < n_layers - 1; ++i) {
        auto output_size = i < n_layers - 2 ? m_config.HiddenLayers[i] : m_config.OutputSize;

        if (i == n_layers - 2)  // all pairs of layers except the last
            layers.emplace_back(output_size) << Eigen::VectorXd::Zero(output_size);
        else
            layers.emplace_back(output_size + 1) << Eigen::VectorXd::Zero(output_size), 1.0;

        cache.emplace_back(output_size) << Eigen::VectorXd::Zero(output_size);
        deltas.emplace_back(output_size) << Eigen::VectorXd::Zero(output_size);
        weights.emplace_back(output_size, input_size + 1)
            << Eigen::MatrixXd::Random(output_size, input_size + 1);  //,
        gradients.emplace_back(output_size, input_size + 1)
            << Eigen::MatrixXd::Zero(output_size, input_size + 1);

        input_size = layout.emplace_back(output_size);
    }
}

double Sigmoid(const double i) { return 1.0 / (1.0 + std::exp(-i)); }

double DerSigmoid(const double i) { return Sigmoid(i) * (1.0 - Sigmoid(i)); }

void ActivationFunction(const Eigen::VectorXd& input, Eigen::VectorXd& output) {
    for (int i = 0; i < input.size(); ++i) {
        output[i] = Sigmoid(input[i]);
    }
}

void DerActivationFunction(const Eigen::VectorXd& input, Eigen::VectorXd& output) {
    for (int i = 0; i < input.size(); ++i) {
        output[i] = DerSigmoid(input[i]);
    }
}

void myANN::Forward(const Eigen::VectorXd& input) {
    // assign inputs to the first layer
    layers[0].head(input.size()) = input;

    for (Index i = 0; i < n_layers - 1; ++i) {
        // Cache the weighted input values
        cache[i] = weights[i] * layers[i];
        // Compute the activated neuron values
        ActivationFunction(cache[i], layers[i + 1]);
    }
}

void myANN::ErrorFunction(const Eigen::VectorXd& target, const Eigen::VectorXd& pred, double& error) const {
    error += (pred - target).squaredNorm();
}

void myANN::BackpropagateError(const Eigen::VectorXd& target) {
    // last error is given by difference between result of Forward and target,
    // proportional to the derivative of the output activation function.
    DerActivationFunction(cache.back(), cache.back());
    deltas.back() = cache.back().cwiseProduct(layers.back() - target);

    for (Index i = n_layers - 3; i >= 0; --i) {
        DerActivationFunction(cache[i], cache[i]);
        deltas[i] = (weights[i + 1].transpose() * deltas[i + 1]).head(layout[i + 1]).cwiseProduct(cache[i]);
    }
}

void myANN::Backward(const Eigen::VectorXd& target) {
    BackpropagateError(target);
    for (Index i = n_layers - 2; i >= 0; --i) {
        gradients[i] += (deltas[i] * layers[i].transpose());
    }
}

void myANN::UpdateWeights(Index minibatch_size) {
    static const double scale = learning_rate / static_cast<double>(minibatch_size);
    for (Index i = n_layers - 2; i >= 0; --i) weights[i] -= (scale * gradients[i]);
}

void myANN::Train(const Eigen::MatrixXd& train_x, const Eigen::MatrixXd& train_y) {
    Index minibatch_size = train_x.cols();

    std::vector<Eigen::MatrixXd> gradients;
    for (Eigen::Index i = 0; i < n_layers - 1; ++i)
        gradients.emplace_back(layout[i + 1], layout[i] + 1)
            << Eigen::MatrixXd::Zero(layout[i + 1], layout[i] + 1);

    for (Index t = 0; t < minibatch_size; ++t) {
        Forward(train_x.col(t));
        Backward(train_y.col(t));
    }

    UpdateWeights(minibatch_size);
}

bool myANN::ConnectToOutput(double* data) {
    new (&layers.back()) Eigen::Map<Eigen::VectorXd>(data, layout.back());

    // DBG
    if (std::addressof(layers.back().coeffRef(0)) == std::addressof(*data)) {
        return true;
    }
    std::cerr << "WARNING! address of layers.back()[0] not same as that of *data" << std::endl;
    return false;
}

bool myANN::Predict(const Eigen::MatrixXd& test_x, const Eigen::MatrixXd& test_y, double* ppredictions,
                    double& accumulated_error) {
    bool okay = ConnectToOutput(ppredictions);

    for (Index t = 0; t < test_x.cols(); ++t) {

        layers[0].head(test_x.rows()) = test_x.col(t);

        for (Eigen::Index i = 0; i < n_layers - 1; ++i) {
            Forward(test_x.col(t));

            double accumulated_error = 0.0;
            Eigen::VectorXd expected = test_y.col(i);
            Eigen::VectorXd output = layers.back();

            ErrorFunction(test_y.col(i), layers.back(), accumulated_error);
        }
    }

    return okay;
}

const Config& myANN::get_config() const { return m_config; }

void myANN::print(std::ostream& out) const {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << "number of layers: " << n_layers << std::endl;
    for (Index i = 0; i < n_layers - 1; ++i) {
        std::cout << "\n************* LAYERS (" << i << ", " << i + 1 << ") ******************\n\n"
                  << "    input: " << layout[i] << "  ====>>   " << layout[i + 1] << " :output"
                  << "\n\nWEIGHTS:\n"
                  << weights[i].format(CleanFmt) << "\n\nInput values:\n"
                  << layers[i].format(CleanFmt) << "\n\nOutput values:\n"
                  << layers[i + 1].format(CleanFmt) << "\n\ndeltas:\n"
                  << deltas[i].format(CleanFmt) << "\n\n****************************************\n"
                  << std::endl;
    }

    show_compilation_timestamp(out);
}

void myANN::print_config(std::ostream& out) const {
    std::cout << "\n    Input Size: " << m_config.InputSize << "    Output Size: " << m_config.OutputSize
              << "    Hidden Layers: ";
    for (const auto& hl : m_config.HiddenLayers) std::cout << hl << ", ";

    show_compilation_timestamp(out);
    std::cout << std::endl;
}

}  // namespace simple
