#ifndef SHUFFLE_DATA_H_
#define SHUFFLE_DATA_H_

#include <algorithm>
#include <cassert>
#include <random>

namespace simple::Data {

/**
 * Utility class to shuffle blocks of data.
 *
 * Given the input data (\p x_beg to \p x_end) and
 * the expected output data (\p y_beg to \p y_end),
 * partition each datasets into \p n_data blocks and
 * shuffle those blocks while preserving the order
 * within them. The two collections of blocks coming
 * from the input and output datasets are shuffled by
 * applying the same permutation to both.
 * Typically, the blocks will be a training phase's batches
 * and this is used in order to avoid bias due to a potential
 * particular ordering of the batches.
 */
class Shuffler
{
public:
  Shuffler() = default;

  template <typename FT>
  void shuffle(FT* x_beg, FT* y_beg, FT* x_end, FT* y_end, std::ptrdiff_t nbatch) {
    using difference_type = std::ptrdiff_t;
    using discrete_uniform_dist_t = std::uniform_int_distribution<std::ptrdiff_t>;
    using param_type = typename discrete_uniform_dist_t::param_type;

    // The number of entries per batch in input and output data.
    const auto x_value_size = (x_end - x_beg) / nbatch;
    const auto y_value_size = (y_end - y_beg) / nbatch;

    // Indices from 0 to N-1
    zero_to_N.clear();
    std::generate_n(std::back_inserter(zero_to_N), nbatch, [count = 0]() mutable { return count++; });
    discrete_uniform_dist_t discrete_uniform_distribution;

    // Shuffle the indices
    for (difference_type i = nbatch - 1; i > 0; --i) {
      using std::swap;
      auto res = discrete_uniform_distribution(eng, param_type(0, i));
      swap(zero_to_N[i], zero_to_N[res]);
    }

    // keeping the order within InputSize and Outputsize chunks
    for (auto i = 0; i < nbatch; ++i) {
      using std::swap;
      auto shuffled_i = zero_to_N[i];
      for (auto j = 0; j < y_value_size; ++j) {
        swap(*(y_beg + i * y_value_size + j), *(y_beg + shuffled_i * y_value_size + j));
      }
      for (auto j = 0; j < x_value_size; ++j) {
        swap(*(x_beg + i * x_value_size + j), *(x_beg + shuffled_i * x_value_size + j));
      }
    }
  }

private:
  std::random_device rd{};
  std::mt19937 eng{ rd() };
  std::vector<int> zero_to_N;
};


}// namespace simple::Data

#endif// MANIP_H_
