#ifndef SERIALIZE_H_
#define SERIALIZE_H_

#include "ann/TemplateAnn.h"

#include "Eigen/Core"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <sstream>

namespace simple::Serialize {

struct ANNInfo {
  std::vector<std::pair<size_t, size_t>> shape;
  std::vector<std::vector<double>> weights;
};

/**
 * From the number of nodes per layer, deduce the shape of the weight matrices
 * and copy their data to the local \p weights vector.
 */
template <typename FT>
ANNInfo make_info_object(const ANN<FT>& network) {
  ANNInfo info;

  // The number of nodes per layer
  const auto& layout = network.get_layout();

  if (layout.size() < 2) {
    throw std::runtime_error("Failed assertion network.layout.size() >= 2 [trivial network]");
  }

  auto iter = layout.begin();

  while (std::distance(iter, layout.end()) > 1) {

    const auto layer = std::distance(layout.begin(), iter);

    // Read the shape and store it.
    // Note that we added an extra input at each layer to incorporate
    // affine transformations when applying the weight matrices.
    const auto input_dim = 1 + *iter++;
    const auto output_dim = *iter;
    info.shape.emplace_back(output_dim, input_dim);

    // Copy the weights.
    auto& wmat = info.weights.emplace_back();
    const size_t n_entries = input_dim * output_dim;
    wmat.reserve(n_entries);
    std::copy_n(network.get_weights()[layer],
                n_entries,
                std::back_inserter(wmat));
  }

  return info;
}

std::ostream& output_formatted(std::ostream& output, const ANNInfo& info) {
  const auto n_layers = info.shape.size();
  output << info.shape.size() << '\n';
  for (auto shape_it = info.shape.begin(); shape_it != info.shape.end(); ++shape_it) {
    auto [rows, cols] = *shape_it;
    const auto layer = static_cast<size_t>(std::distance(info.shape.begin(), shape_it));
    output << rows << ' ' << cols << '\n';
    std::copy(info.weights[layer].begin(), info.weights[layer].end(),
              std::ostream_iterator<double>{ output, " "});
    output << '\n';
  }
  return output;
}

template <typename FT>
void write_to_file(std::string_view filepath, const ANN<FT>& network) {
  std::ofstream ofs{ filepath.data() };
  if (!ofs) {
    std::stringstream ss;
    ss << "Failed to open file " << filepath.data() << " for writing";
    throw std::runtime_error(ss.str());
  }

  output_formatted(ofs, make_info_object(network)) << std::endl;
}

} // namespace simple::Serialize

#endif // SERIALIZE_H_
