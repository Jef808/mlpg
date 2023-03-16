#ifndef MANIP_H_
#define MANIP_H_

#include <cassert>
#include <algorithm>
#include <random>


namespace simple::Data {

/**
 * Utility class to shuffle data in chunks of InputSize
 */
class Shuffler {
public:

    Shuffler() = default;

    template<typename IterX, typename IterY>
    void shuffle(IterX x_beg, IterY y_beg,
                 IterX x_end, IterY y_end,
                 const std::ptrdiff_t n_data) {
        using diff_t = std::vector<std::ptrdiff_t>::difference_type;
        using distr_t = std::uniform_int_distribution<diff_t>;
        using param_t = distr_t::param_type;

        const diff_t x_value_size = (x_end - x_beg) / n_data;
        const diff_t y_value_size = (y_end - y_beg) / n_data;

        // Indices from 0 to N-1
        zero_to_N.clear();
        std::generate_n(std::back_inserter(zero_to_N),
                        n_data, [n=0] () mutable { return n++; });
        distr_t D;

        // Shuffle the indices
        for (diff_t i = n_data - 1; i > 0; --i) {
            using std::swap;
            auto res = D(eng, param_t(0, i));
            swap(zero_to_N[i], zero_to_N[res]);
        }

        // Reorder the data according the the shuffled indices,
        // keeping the order within InputSize and Outputsize chunks
        for (diff_t i = 0; i < n_data; ++i) {
            using std::swap;
            diff_t shuffled_i = zero_to_N[i];
            for (diff_t j = 0; j < y_value_size; ++j)
                swap(*(y_beg + i*y_value_size + j), *(y_beg + shuffled_i*y_value_size + j));
            for (diff_t j = 0; j < x_value_size; ++j)
                swap(*(x_beg + i*x_value_size + j), *(x_beg + shuffled_i*x_value_size + j));
        }
    }
private:
    std::random_device rd;
    std::mt19937 eng{ rd() };
    std::vector<int> zero_to_N;
};


}  // namespace simple::Data

#endif // MANIP_H_
