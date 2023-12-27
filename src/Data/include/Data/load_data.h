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

  while (output.size() < n_rows && std::getline(ifs, buffer)) {
    auto& row_out = output.emplace_back();

    std::string_view row_view{buffer};

    bool is_row_consumed = row_view.length() > 0;

    while (not is_row_consumed) {

      // Commas indicate ends of words
      auto end_of_word = row_view.find(',');

      // Indicate time to go to next row when that was last word
      if (end_of_word == row_view.npos) {
        is_row_consumed = true;

        // Prevent out of bounds access and/or undefined behavior
        end_of_word = row_view.size();
      }

      try {

        // Store that word, iterate!
        row_out.emplace_back(row_view.substr(0, end_of_word));
        row_view.remove_prefix(end_of_word);

      } catch (const std::exception& e) {
        std::cerr << "Error at word " << row_out.size() << ": " << e.what() << std::endl;
      }
    }
  }

  return output;
}

}  // namespace simple::Data

#endif  // LOAD_DATA_H_
