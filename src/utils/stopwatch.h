#ifndef __STOPWATCH_H_
#define __STOPWATCH_H_

#include <chrono>
#include <iostream>
#include <functional>
#include <vector>

namespace utils {


class Stopwatch {
public:
    using Clock = std::chrono::steady_clock;
    using Time_point = Clock::time_point;
    using Duration = std::chrono::milliseconds;
    using Discrete_duration = Clock::rep;

    Stopwatch() :
        m_start(Clock::now()),
        m_stored_times(1, m_start)
    {
    }
    Discrete_duration operator()() const
    {
        return ticks_count(m_start, Clock::now());
    }
    Duration get() const
    {
        return std::chrono::duration_cast<Duration>(Clock::now() - m_start);
    }
    template<typename Time>
    void show_elapsed(const Time& time) {
        std::cout << std::chrono::duration_cast<Duration>(Clock::now() - m_start).count() << std::endl;
    }
    template<typename Time>
    Discrete_duration get_elapsed(const Time& time) {
        return std::chrono::duration_cast<Duration>(Clock::now() - m_start).count();
    }
    std::vector<Discrete_duration> stored_times()
    {
        std::vector<Discrete_duration> ret;
        auto get_n_ticks = [start=m_stored_times.front()](const auto& tp){
            return ticks_count(start, tp);
        };
        std::transform(m_stored_times.begin(),
                       m_stored_times.end(),
                       std::back_inserter(ret),
                       get_n_ticks
                       );
        return ret;
    }
    void reset_start()
    {
        m_start = Clock::now();
    }

private:
    Time_point m_start;
    std::vector<Time_point> m_stored_times;

    static Discrete_duration ticks_count(
        const Time_point& start, const Time_point& tp)
    {
        using std::chrono::duration_cast;
        return duration_cast<Duration>(tp - start).count();
    }
};

} // namespace utils


#endif
