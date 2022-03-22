#ifndef __RANDOMUTILS_H_
#define __RANDOMUTILS_H_

#include <array>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

namespace Rand {

template <typename Int_T>
class Util {
public:
    using Engine = typename std::
        conditional<sizeof(Int_T) <= 4, std::mt19937, std::mt19937_64>::type;

    Util() = default;
    Util(typename Engine::result_type seed)
        : gen(seed)
    {
    }
    using size_type = typename std::make_unsigned<Int_T>::type;

    /**
   * Returns a number from _min to _max (including  _max!)
   */
    Int_T get(Int_T _min, Int_T _max)
    {
        return std::uniform_int_distribution<Int_T>(_min, _max)(gen);
    }

    template <size_t N>
    auto gen_ordering(const Int_T _beg, const Int_T _end)
    {
        std::array<Int_T, N> ret {};
        std::iota(ret.begin(), ret.begin() + _end - _beg, _beg);

        for (int i = _beg; i < _end - _beg; ++i) {
            auto j = get(i, _end - _beg - 1);
            std::swap(ret[i], ret[j]);
        }

        return ret;
    }

    template <size_t N>
    void shuffle(std::array<Int_T, N>& arr, Int_T sz)
    {
        for (auto i = 0; i < sz; ++i) {
            auto j = get(i, sz - 1);
            std::swap(arr[i], arr[j]);
        }
    }

    /**
   * Choose a random element from a given container.
  */
    template <typename Container>
    typename Container::value_type choose(const Container& c)
    {
        return c[get(0, c.size() - 1)];
    }

    template<size_t N>
    struct Weighted_chooser
    {
        Weighted_chooser(const std::array<double, N>& w, Engine& _gen) :
            m_weights{ w }, m_dd(w.begin(), w.end()), r_gen(_gen)
        {
        }

        template<typename T>
        T operator()(const std::array<T, N>& arr)
        {
            auto ndx = m_dd(r_gen);
            return arr[ndx];
        }

        std::array<double, N> m_weights;
        std::discrete_distribution<int> m_dd;
        Engine& r_gen;
    };


    auto weighted_choose(const std::vector<double>& w)
    {
        std::discrete_distribution<int> dd(w.begin(), w.end());
    }

private:
    Engine gen { std::random_device {}() };
};

} // namespace Rand

#endif
