#ifndef LOAD_CSV_H_
#define LOAD_CSV_H_

#include <cassert>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace simple::Data {


class load_csv_exception : public std::exception
{
public:
  load_csv_exception() noexcept = default;

  explicit load_csv_exception(std::string_view message) noexcept : m_message{ message } {}

  explicit load_csv_exception(unsigned long line_nb) noexcept
    : m_message{ "Csv file format error: Wrong number of fields at line " + std::to_string(line_nb) }
  {}

  [[nodiscard]] const char *what() const noexcept override { return m_message.c_str(); }

private:
  const std::string m_message;
};


/**
 * Parse and return the number of specified type found at the
 * beginning of the string. Remove it from the string, along with
 * the comma following it.
 *
 * Return a load_csv_exception if an error is thrown during parsing,
 * or if the next character is not a comma.
 *
 * \tparam T The type of the entry to parse.
 */
template<typename T> inline std::variant<T, load_csv_exception> parse_next(std::string_view &sv_buf) noexcept
{
  size_t idx = 0;
  T ret;
  try {
    if constexpr (std::is_integral_v<T>) {
      ret = static_cast<T>(std::stoi(sv_buf.data(), &idx));
    } else if constexpr (std::is_floating_point_v<T>) {
      ret = static_cast<T>(std::stod(sv_buf.data(), &idx));
    } else if constexpr (std::is_same_v<T, std::string>) {
      idx = sv_buf.find(',');
      ret = sv_buf.substr(idx);
    }
  } catch (const std::exception &e) {
    return load_csv_exception("Failed to parse " + std::string(sv_buf));
  }
  sv_buf.remove_prefix(std::min(idx + 1, sv_buf.size()));
  return ret;
}

/**
 * Parse and consume the label entry from \p sv_buf
 *
 * \tparam LabelT The type of the labels.
 * \tparam LabelsFront Set to true (resp. false) to indicate that the labels are at
 *         the beginning (resp. the end) of each lines.
 */
template<typename LabelT, bool LabelsFront = true>
std::variant<LabelT, load_csv_exception> parse_label(std::string_view &sv_buf) noexcept
{
  if constexpr (LabelsFront) {
    return parse_next<LabelT>(sv_buf);
  } else {
    size_t last_comma = sv_buf.find_last_of(',');
    std::string_view sv_back = sv_buf.substr(last_comma + 1);
    auto ret = parse_next<LabelT>(sv_back);
    sv_buf.remove_suffix(sv_buf.size() - last_comma);
    return ret;
  }
}


template<typename FT> struct OneHotEncoder
{
  explicit OneHotEncoder(ptrdiff_t NLabels) : m_NLabels{ NLabels } {}

  template<typename OutputIterator> void operator()(OutputIterator out, unsigned int label)
  {
    for (ptrdiff_t i = 0; i < m_NLabels; ++i) { out = static_cast<FT>(i == label ? 1 : 0); }
  }

private:
  ptrdiff_t m_NLabels;
};


/**
 * Parse the csv file into the output vectors.
 *
 * \tparam FirstLineHeader Whether the first line of the file is a header to discard.
 * \tparam LabelsFront Set to true (resp. false) to indicate that the labels are at
 *         the beginning (resp. the end) of each lines.
 * \tparam LabelT The type of the labels.
 * \tparam FT The scalar type.
 */
template<bool FirstLineHeader = false, bool LabelsFront = true, typename LabelT, typename FT>
std::optional<load_csv_exception> load_csv_labels(const std::filesystem::path& fpath,
  std::vector<FT> &inputs,
  unsigned int NLabels,
  std::vector<FT> &enc_labels,
  unsigned int n_data)
{
  std::ifstream ifs{ fpath };
  if (not ifs) {
    return std::make_optional(load_csv_exception("Failed to open file " + fpath.string()));
  }

  std::string buf;
  size_t line_count = 0;

  // Drop the first line if it is a header
  if constexpr (FirstLineHeader) {
    std::getline(ifs, buf);
    ++line_count;
  }

  ptrdiff_t counter = 0;
  size_t input_size = 0;
  OneHotEncoder<FT> oneHot{ NLabels };

  while (counter < n_data && std::getline(ifs, buf)) {
    ++counter;
    std::string_view buf_sv{ buf };

    // Parse label (removing it from the string view)
    std::variant<LabelT, load_csv_exception> var_label = parse_label<LabelT, LabelsFront>(buf_sv);

    // If successful, insert the corresponding one-hot encoded sequence.
    if (std::holds_alternative<LabelT>(var_label)) {
      oneHot(std::back_inserter(enc_labels), std::get<LabelT>(var_label));
    } else {
      return std::make_optional(std::get<load_csv_exception>(var_label));
    }

    size_t field_count = 0;

    while (not buf_sv.empty()) {
      auto var_input = parse_next<FT>(buf_sv);
      if (std::holds_alternative<FT>(var_input)) {
        inputs.push_back(std::get<FT>(var_input));
        ++field_count;
      } else {
        return std::make_optional(std::get<load_csv_exception>(var_input));
      }
    }

    // Store number of fields for format check
    if (input_size == 0) {
      input_size = field_count;
    }

    // Return exception if unequal number of fields
    else if (field_count != input_size) {
      return std::make_optional(load_csv_exception(line_count));
    }
  }

  return std::nullopt;
}


template<bool FirstLineHeader = true, typename FT>
std::optional<load_csv_exception> load_csv_nolabels(const std::filesystem::path& fpath, std::vector<FT> &inputs)
{
  std::ifstream ifs{ fpath };
  if (not ifs) {
    return std::make_optional(load_csv_exception("Failed to open file " + fpath.string()));
  }

  std::string buf;
  size_t line_count = 1;
  size_t input_size = 0;

  // Drop the first line if it is a header
  if constexpr (FirstLineHeader) {
    std::getline(ifs, buf);
    ++line_count;
  }

  while (std::getline(ifs, buf)) {
    std::string_view sv_buf{ buf };
    size_t field_count = 0;

    while (not sv_buf.empty()) {
      auto var_input = parse_next<FT>(sv_buf);
      if (std::holds_alternative<FT>(var_input)) {
        inputs.push_back(std::get<FT>(var_input));
        ++field_count;
      } else {
        return std::make_optional(std::get<load_csv_exception>(var_input));
      }
    }

    // Store number of fields for format check
    if (input_size == 0) {
      input_size = field_count;
    }

    // Return exception if unequal number of fields
    else if (field_count != input_size) {
      return load_csv_exception(line_count);
    }
  }

  return std::nullopt;
}

/**
 * Convenience function for calling load_csv_labels
 */
template<typename FT = std::string, typename LabelT = unsigned int>
std::optional<load_csv_exception> load_csv(std::string_view fpath,
  std::vector<FT> &inputs,
  std::vector<LabelT> &labels,
  size_t n_labels,
  size_t n_data = std::numeric_limits<size_t>::max(),
  bool labels_in_front = true,
  bool header = false)
{
  assert(n_labels > 0 && "Error: Number of labels not provided");

  return header
f           ? labels_in_front
               ? load_csv_labels<true, true, LabelT>(std::filesystem::path{ fpath }, inputs, n_labels, labels, n_data)
               : load_csv_labels<true, false, LabelT>(std::filesystem::path{ fpath }, inputs, n_labels, labels, n_data)
         : labels_in_front
           ? load_csv_labels<false, true, LabelT>(std::filesystem::path{ fpath }, inputs, n_labels, labels, n_data)
           : load_csv_labels<false, false, LabelT>(std::filesystem::path{ fpath }, inputs, n_labels, labels, n_data);
}

/**
 * Convenience function for calling load_csv_nolabels
 */
template<typename FT>
std::optional<load_csv_exception> load_csv(std::string_view fpath, std::vector<FT> &inputs, const bool header = false)
{
  return header ? load_csv_nolabels<true>(std::filesystem::path{ fpath }, inputs)
                : load_csv_nolabels<false>(std::filesystem::path{ fpath }, inputs);
}

template<typename FT>
std::optional<load_csv_exception> load_csv()

}// namespace simple::Data


#endif// LOAD_CSV_H_
