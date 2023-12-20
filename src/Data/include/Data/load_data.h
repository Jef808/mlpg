#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_

#include <exception>
#include <fstream>
#include <limits>
#include <iostream>
#include <string_view>
#include <vector>

namespace simple::Data {

/**
 * Simple method that reads a csv file
 * into a vector of vector of strings.
 *
 * The parameter  `n_rows` optionally sets the
 * maximum number of rows to read.
 */
std::vector<std::vector<std::string>>
read_csv(std::string_view fpath,
         size_t n_rows = std::numeric_limits<size_t>::max()) {
  std::vector<std::vector<std::string>> output;

  std::ifstream ifs{ fpath.data() };
  if (not ifs) {
    auto message = "Failed to open file " + std::string(fpath);
    throw std::runtime_error(message);
  }

  std::string buffer;

  try {
  while (output.size() < n_rows && std::getline(ifs, buffer)) {
    std::string_view row_view{buffer};
    auto& row_out = output.emplace_back();

    int cell_count = 0;
    while (not row_view.empty()) {
      try {
        auto idx = row_view.find(',');
        row_out.emplace_back(row_view.substr(idx));
        row_view.remove_prefix(std::min(idx + 1, row_view.size()));
      } catch (const std::exception& e) {
        std::cerr << "Failed reading csv file at/after cell number " << output.back().size() << std::endl;
      }
    }
  }
  } catch (const std::exception& e) {
    std::cerr << "Failed reading csv file at/after row number " << output.size() << std::endl;
  }

  return output;
}

}  // namespace simple::Data

#endif  // LOAD_DATA_H_
