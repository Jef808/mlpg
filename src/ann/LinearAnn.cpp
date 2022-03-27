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
 *
 *
 * NOTES:
 *
 * The input and output layers are actually Eigen::Map objects.
 * This way, we can adapt the network for the different phases
 * an algorithm does go through.
 * See for example Forward where the input layer is fed and a
 * mutable layer is passed to receive the outputs.
 */

#include "LinearAnn.h"
#include "ViewCmd.h"
#include "Activations.h"

#include "Eigen/Core"

#include <cassert>
#include <iostream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <vector>

#include <type_traits>

namespace simple {


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

    for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
        auto output_size = m_config.HiddenLayers[i];

        layers.emplace_back(output_size + 1) << Eigen::VectorXd::Zero(output_size), 1.0;
        cache.emplace_back(output_size) << Eigen::VectorXd::Zero(output_size);
        deltas.emplace_back(output_size) << Eigen::VectorXd::Zero(output_size);
        weights.emplace_back(output_size, input_size + 1)
            << Eigen::MatrixXd::Random(output_size, input_size + 1);

        input_size = layout.emplace_back(output_size);
    }
}




// Cache the output vectors in class' memory until
// next step is called
// void myANN::Forward(const Eigen::VectorXd& input) {
//     // assign inputs to the first layer
//     layers[0].head(input.size()) = input;

//     for (Eigen::Index i = 0; i < n_layers - 1; ++i) {
//         // Cache the weighted input values
//         cache[i] = weights[i] * layers[i];
//         // Compute the activated neuron values
//         ActivationFunction(cache[i], layers[i + 1]);
//     }
// }


// stateless version, immediately outputs the last layer instead.
// can be piped!


void myANN::Forward(const Eigen::VectorXd& input, Eigen::VectorXd& output) {
    SPDLOG_DEBUG("Starting Forward pass");

    // assign inputs to the first layer
    layers[0].head(input.size()) = input;
    const Eigen::Index L = n_layers - 2;
    SPDLOG_DEBUG("Inserted `input` into layer 0");

    for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
        // Cache the weighted input values
        cache[i] = (weights[i] * layers[i]).eval();
        // Compute the activated neuron values
        ActivationFunction(cache[i], layers[i + 1]);
        SPDLOG_LOGGER_DEBUG("Processed layer {0}", i);
    }

    SPDLOG_LOGGER_DEBUG("Remain the output layer...");
    cache[L] = weights[L] * layers[L];
    ActivationFunction(cache[L], output);

    SPDLOG_LOGGER_DEBUG("Done!");
}

    // void myANN::Forward(const Eigen::MatrixXd& inputs, Eigen::MatrixXd& outputs) {

    //     // assign inputs to the first layer
    //     for (int i=0; i < inputs.cols(); ++i) {
    //         layers[0].head(layout[0] + 1);

    //         const Eigen::Index L = n_layers - 2;

    //         for (Eigen::Index i = 0; i < n_layers - 2; ++i) {
    //             // Cache the weighted input values
    //             cache[i] = (weights[i] * layers[i]).eval();
    //             // Compute the activated neuron values
    //             ActivationFunction(cache[i], layers[i + 1]);
    //         }

    //         cache[L] = weights[L] * layers[L];
    //         ActivationFunction(cache[L], outputs);
    //     }
    // }

/**
 * Inputs are now stacked row-wise: (they are ColVectors)
 * Instead of ((n+1) x 1) inputs, we
 * now consider inputs of shape ((n+1) x m)
 * where m is the size of the batch.
 *
 * By considering the map V -> V \otimes V ... \otimes V
 * taking acting on linear maps (the weights) simply by
 * w -> w \otimes w ... \otimes w.
 * (We repeat the weigth matrix m times to fit the new shape).
 *
 * Similarly, the extended output shape is now (m x n') instead
 * of one (1 x n') Row vector for only one input point.
 */
void myANN::ForwardBatch(double* data_out,
                         const double* data_in,
                         const Eigen::Index batch_size)
{
    Eigen::Index NRow0 = m_config.InputSize + 1; // inputs are stacked row-wise
    Eigen::Index NCol0 = batch_size;             // +1 for the bias node

    Eigen::Index NRowL = batch_size;             // the outputs are stacked row-wise
    Eigen::Index NColL = m_config.OutputSize;    // one output per row

    Eigen::Map<const Eigen::MatrixXd> inputs(data_in, NRow0, NCol0);
    Eigen::Map<const Eigen::MatrixXd> outputs(data_out, NRowL, NColL);

    std::cout << "The contents of MInputs is now\n\n" << inputs
              << "\n\n The contents of MOutputs is now\n\n" << outputs << std::endl;
}


void myANN::ErrorFunction(const Eigen::VectorXd& target, const Eigen::VectorXd& pred, double& error) const {
    error += (pred - target).squaredNorm();
}

void myANN::BackpropagateError(const Eigen::VectorXd& target) {
    // last error is given by difference between result of Forward and target,
    // proportional to the derivative of the output activation function.
    DerActivationFunction(cache.back(), cache.back());
    deltas.back() = cache.back().cwiseProduct(layers.back() - target);

    for (Eigen::Index i = n_layers - 3; i >= 0; --i) {
        DerActivationFunction(cache[i], cache[i]);
        deltas[i] = (weights[i + 1].transpose() * deltas[i + 1]).head(layout[i + 1]).cwiseProduct(cache[i]);
    }
}


void myANN::Backward(const Eigen::VectorXd& target, Eigen::MatrixXd& errors) {
    BackpropagateError(target);
    for (Eigen::Index i = n_layers - 2; i >= 0; --i) {
        gradients[i] += (deltas[i] * layers[i].transpose());
    }
}
void myANN::Backward(const Eigen::VectorXd& target) {
    BackpropagateError(target);
    for (Eigen::Index i = n_layers - 2; i >= 0; --i) {
        gradients[i] += (deltas[i] * layers[i].transpose());
    }
}

void myANN::UpdateWeights(Eigen::Index minibatch_size) {
    static const double scale = learning_rate / static_cast<double>(minibatch_size);
    for (Eigen::Index i = n_layers - 2; i >= 0; --i) weights[i] -= (scale * gradients[i]);
}


void myANN::Train(const Eigen::MatrixXd& train_x, const Eigen::MatrixXd& train_y) {
    Eigen::Index minibatch_size = train_x.cols();

    std::vector<Eigen::MatrixXd> gradients;
    for (Eigen::Index i = 0; i < n_layers - 1; ++i)
        gradients.emplace_back(layout[i + 1], layout[i] + 1)
            << Eigen::MatrixXd::Zero(layout[i + 1], layout[i] + 1);

    for (Eigen::Index t = 0; t < minibatch_size; ++t) {
        Forward(train_x.col(t));
        Backward(train_y.col(t));
    }

    UpdateWeights(minibatch_size);
}

    void myANN::Train(const Eigen::Index batch_size,
                      const double* train_x, const Eigen::Index Ntrain_x,
                      const double* train_y, const Eigen::Index Ntrain_y)
{
    Eigen::Index NRow0 = m_config.InputSize + 1; // +1 for the bias node
    Eigen::Index NCol0 = batch_size;             // inputs are stacked column-wise

    Eigen::Index NRowL = batch_size;             // the outputs are stacked row-wise
    Eigen::Index NColL = m_config.OutputSize;    // one output per row

    //Eigen::Map<const Eigen::MatrixXd> train_x(train_x, )
}


bool myANN::Predict(const Eigen::MatrixXd& test_x,
                        const Eigen::MatrixXd& test_y,
                        double* ppredictions,
                        double& accumulated_error)
    {
        bool okay = ppredictions != nullptr;
        return okay;
    }
//     // Create a temporary view over the provided
//     Eigen::Map<Eigen::VectorXd> tmp_last_layer(layers.back());
//     new (&layers.back()) Eigen::Map<Eigen::VectorXd>(data, layout.back());

//     for (Eigen::Index t = 0; t < test_x.cols(); ++t) {

//         // Recall, each non-output layers layer
//         // has an extra row for the bias nodes.
//         //layers.front().head(layout.front()) = test_x.col(t);

//         for (Eigen::Index i = 0; i < n_layers; ++i) {
//             Forward(test_x.col(t));

//             double accumulated_error = 0.0;
//             Eigen::VectorXd expected = test_y.col(i);
//             Eigen::VectorXd output = layers.back();

//             ErrorFunction(test_y.col(i), layers.back(), accumulated_error);
//         }
//     }

//     return okay;
// }

const Config& myANN::get_config() const { return m_config; }

void myANN::print(std::ostream& out) const {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << "number of layers: " << n_layers << std::endl;
    for (Eigen::Index i = 0; i < n_layers - 1; ++i) {
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
