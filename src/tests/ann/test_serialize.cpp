#include "ann/Config.h"
#include "ann/TemplateAnn.h"
#include "ann/Serialize.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include <limits>
#include <string>

using namespace simple;

using namespace Catch::Matchers;

TEST_CASE("Serializing a small network after configuration produces expected output", "[serialization]") {
  Config config;
  config.InputSize = 4;
  config.HiddenLayers = { 6 };
  config.OutputSize = 1;

  ANN<double> network;

  network.setup(config);

  auto info = Serialize::make_info_object(network);

  for (auto shp : info.shape) {
    std::cerr << "cols: " << shp.first << ", cols: "<< shp.second << std::endl;
  }
  Serialize::write_to_file("test_serialize_output.txt", network);

  std::ifstream file_in { "test_serialize_output.txt", std::ifstream::in };
  REQUIRE(file_in);

  std::vector<std::pair<size_t, size_t>> expected_shape = { {6, 5}, {1, 7} };
  std::vector<size_t> expected_weight_sizes = { 30, 7 };

  {
    int n_layers = -1;
    file_in >> n_layers;
    file_in.ignore();

    REQUIRE(n_layers == 2);
  }
  std::vector<double> weights;
  {
    size_t rows = std::numeric_limits<size_t>::max();
    size_t cols = std::numeric_limits<size_t>::max();
    file_in >> rows >> cols;
    file_in.ignore();

    REQUIRE(cols == expected_shape[0].second);
    REQUIRE(rows == expected_shape[0].first);

    std::string buf;
    std::getline(file_in, buf);
    std::stringstream stream{ buf };
    std::copy(std::istream_iterator<double>{stream},
              std::istream_iterator<double>{},
              std::back_inserter(weights));

    REQUIRE(weights.size() == expected_weight_sizes[0]);
    weights.clear();
  }
  {
    size_t rows = std::numeric_limits<size_t>::max();
    size_t cols = std::numeric_limits<size_t>::max();
    file_in >> rows >> cols;
    file_in.ignore();

    REQUIRE(cols == expected_shape[1].second);
    REQUIRE(rows == expected_shape[1].first);

    std::string buf;
    std::getline(file_in, buf);
    std::stringstream stream{ buf };
    std::copy(std::istream_iterator<double>{stream},
              std::istream_iterator<double>{},
              std::back_inserter(weights));

    REQUIRE(weights.size() == expected_weight_sizes[1]);
  }
}
