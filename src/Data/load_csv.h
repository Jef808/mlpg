#ifndef LOAD_CSV_H_
#define LOAD_CSV_H_

#include <cassert>
#include <exception>
#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <filesystem>
#include <system_error>
#include <type_traits>
#include <optional>

namespace simple::Data {


class load_csv_exception : public std::exception {
public:
    load_csv_exception() noexcept = default;

    explicit load_csv_exception(std::string_view message) noexcept
        : m_message{ message } {}

    explicit load_csv_exception(unsigned long line_nb) noexcept
        : m_message{ "Csv file format error: Wrong number of fields at line " + std::to_string(line_nb) } {}

    [[nodiscard]] const char* what() const noexcept override {
        return m_message.c_str();
    }

private:
    const std::string m_message;
};


namespace {

    /**
     * Parse and return the number of specified type found at the
     * beginning of the string. Remove it from the string, along with
     * the comma following it.
     *
     * Return a load_csv_exception if an error is thrown during parsing,
     * or if the next character is not a comma.
     */
    template<typename T>
    inline std::variant<T, load_csv_exception> parse_next(std::string_view& sv) noexcept
    {
        size_t idx = 0;
        T ret;
        try {
            if constexpr (std::is_integral_v<T>) {
                ret = static_cast<T>( std::stoi(sv.data(), &idx) );
            } else if constexpr (std::is_floating_point_v<T>) {
                ret = static_cast<T>( std::stod(sv.data(), &idx) );
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                idx = sv.find(',');
                ret = sv.substr(idx);
            }
        }
        catch (const std::exception& e) {
            return load_csv_exception("Failed to parse " + std::string(sv));
        }
        sv.remove_prefix(std::min(idx+1, sv.size()));
        return ret;
    }
}  // namespace


template<typename LabelT, bool LabelsFront = true>
std::variant<LabelT, load_csv_exception> parse_label(std::string_view& sv) noexcept
{
    if constexpr(LabelsFront)
                {
                    return parse_next<LabelT>(sv);
                }
    else
    {
        size_t last_comma = sv.find_last_of(',');
        std::string_view sv_back = sv.substr(last_comma + 1);
        auto ret = parse_next<LabelT>(sv_back);
        sv.remove_suffix(sv.size() - last_comma);
        return ret;
    }
}


template<typename FT>
struct OneHotEncoder {
    OneHotEncoder(ptrdiff_t NLabels) : m_NLabels{NLabels} { }

    template<typename OutputIterator>
    void operator()(OutputIterator out, unsigned int label) {
        for (ptrdiff_t i = 0; i < m_NLabels; ++i) {
            out = static_cast<FT>(i == label ? 1 : 0);
        }
    }

private:
    ptrdiff_t m_NLabels;
};


/**
 * Parse the csv file into the provided vectors.
 *
 * @FT The scalar type
 * @FirstLineHeader Whether the first line of the file is a header to discard
 * @LabelsFront Whether the labels are at the beginning or at the end of each lines
 * @LabelT The label types in the csv file
 */
template<bool FirstLineHeader=true,
         bool LabelsFront=true,
         typename LabelT,
         typename FT
         >
std::optional<load_csv_exception> load_csv_labels(
    std::filesystem::path fp,
    std::vector<FT>& inputs,
    unsigned int NLabels,
    std::vector<FT>& enc_labels,
    unsigned int n_data)
{
    std::ifstream ifs{ fp };
    if (not ifs)
        return std::make_optional(
            load_csv_exception("Failed to open file " + fp.string()));

    std::string buf;
    size_t line_count = 0;

    // Drop the first line if it is a header
    if constexpr (FirstLineHeader) {
        std::getline(ifs, buf);
        ++line_count;
    }

    ptrdiff_t counter = 0;
    size_t input_size = -1;
    OneHotEncoder<FT> oneHot{ NLabels };

    while (counter < n_data && std::getline(ifs, buf))
    {
        ++counter;
        std::string_view sv { buf };

        // Parse label (removing it from the string view)
        std::variant<LabelT, load_csv_exception> var_label = parse_label<LabelT, LabelsFront>(sv);
        // If successful, insert the corresponding one-hot encoded sequence.
        if (std::holds_alternative<LabelT>(var_label)) {
            oneHot(std::back_inserter(enc_labels), std::get<LabelT>(var_label));
        }
        else {
            return std::make_optional(std::get<load_csv_exception>(var_label));
        }

        size_t field_count = 0;

        while (not sv.empty())
        {
            auto var_input = parse_next<FT>(sv);
            if (std::holds_alternative<FT>(var_input)) {
                inputs.push_back(std::get<FT>(var_input));
                ++field_count;
            }
            else {
                return std::make_optional(std::get<load_csv_exception>(var_input));
            }
        }

        // Store number of fields for format check
        if (input_size == -1)
            input_size = field_count;

        // Return exception if unequal number of fields
        else if (field_count != input_size) {
            return std::make_optional(load_csv_exception(line_count));
        }
    }

    return std::nullopt;
}


template<bool FirstLineHeader=true,
         typename FT>
std::optional<load_csv_exception> load_csv_nolabels(
    std::filesystem::path fp,
    std::vector<FT>& inputs)
{
    std::ifstream ifs{ fp };
    if (not ifs)
        return std::make_optional(
            load_csv_exception("Failed to open file " + fp.string()));

    std::string buf;
    size_t line_count = 1;
    size_t input_size = -1;

    // Drop the first line if it is a header
    if constexpr (FirstLineHeader) {
        std::getline(ifs, buf);
        ++line_count;
    }
    
    while (std::getline(ifs, buf))
    {
        std::string_view sv { buf };
        size_t field_count = 0;

        while (not sv.empty())
        {
            auto var_input = parse_next<FT>(sv);
            if (std::holds_alternative<FT>(var_input)) {
                inputs.push_back(std::get<FT>(var_input));
                ++field_count;
            }
            else {
                return std::make_optional(std::get<load_csv_exception>(var_input));
            }
        }

        // Store number of fields for format check
        if (input_size == -1)
            input_size = field_count;

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
template<typename FT, typename LabelT=unsigned int>
std::optional<load_csv_exception> load_csv(std::string_view fp,
                                           std::vector<FT>& inputs,
                                           std::vector<FT>& labels,
                                           unsigned int n_labels,
                                           unsigned int n_data = std::numeric_limits<unsigned>::max(),
                                           const bool header=true,
                                           const bool labels_in_front=true)
{
    assert(n_labels > 0 &&
           "Error: Number of labels not provided");

    return header
        ? labels_in_front
        ? load_csv_labels<true, true, LabelT>(std::filesystem::path{fp},
                                              inputs,
                                              n_labels,
                                              labels,
                                              n_data)
        : load_csv_labels<true, false, LabelT>(std::filesystem::path{fp},
                                               inputs,
                                               n_labels,
                                               labels,
                                               n_data)
        : labels_in_front
        ? load_csv_labels<false, true, LabelT>(std::filesystem::path{fp},
                                               inputs,
                                               n_labels,
                                               labels,
                                               n_data)
        : load_csv_labels<false, false, LabelT>(std::filesystem::path{fp},
                                                inputs,
                                                n_labels,
                                                labels,
                                                n_data);
}

/**
 * Convenience function for calling load_csv_nolabels
 */
template<typename FT>
std::optional<load_csv_exception> load_csv(std::string_view fp,
                                           std::vector<FT>& inputs,
                                           const bool header=true)
{
    return header
        ? load_csv_nolabels<true>(std::filesystem::path{fp},
                                  inputs)
        : load_csv_nolabels<false>(std::filesystem::path{fp},
                                   inputs);
}


} // namespace simple::Data



#endif // LOAD_CSV_H_
