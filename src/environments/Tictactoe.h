#ifndef TICTACTOE_H_
#define TICTACTOE_H_

#include "types.h"

#include <algorithm>


class Tictactoe {
public:
    // The type to use for a unit.
    using IntT = std::size_t;

    /**
     * Implementation of the state.
     */
    class State {
    public:
        //! Number of components of the encoded state
        static constexpr inline std::size_t dimension = 18;
        //! The type used to represent a dummy state
        using Type = std::array<IntT, dimension>;

        /**
         * Construct a State instance.
         */
        State() = default;

        /**
         * Construct a State instance of given data
         */
        State(const Type& data) : m_data(data) { }

        //! Modify the internal representation of the state
        Type& data() { return m_data; }

        //! Encode the state into a single vector
        [[nodiscard]] const Type& encode() const { return m_data; }

    private:
        // Store the encoded state.
        Type m_data;
    };

    /**
     * Implementation of the action.
     */
    class Action {
    public:
        // The type used to represent actions
        using Type = IntT;
        // The size of the action space
        static constexpr inline std::size_t size = 9;
        // The value of the action
        Type action;
    };

    /**
     * Sampling of encoded State-actions.
     */
    bool Sample(const State& state,
                const Action& action,
                State& next_state) {
        // Copy the state
        next_state = state;
        // Turn on the bit of the player to move in target square
        next_state.data()[2*action.action + (n_steps_even_ ? 0 : 1)] = 1;

        ++n_steps_;
        n_steps_even_ = !n_steps_even_;

        return not is_terminal(next_state, action);
    };

    /**
     * The initial state.
     */
    [[nodiscard]] State initial_state() const {
        State ret;
        ret.data().fill(0);
        return ret;
    }

    /**
     * Function verifying if current state is terminal, depending on
     * knowledge of last action played.
     */
    [[nodiscard]] bool is_terminal(const State& state, const Action& action) const {
        using Line = std::array<std::size_t, 3>;
        static constexpr std::array<Line, 8> Lines
        {
            Line
            { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, { 0, 4, 8 },
            { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 }, { 2, 4, 6 }
        };

        static std::vector<Line> PotentialLines;
        PotentialLines.clear();
        std::copy_if(Lines.begin(), Lines.end(), std::back_inserter(PotentialLines), [&action](auto line)
            { return std::find(line.begin(), line.end(), action.action) != line.end(); });

        bool winner_found = std::any_of(Lines.begin(), Lines.end(), [&state](auto line)
        {
            // For each target square of a line, encoding is given by interpreting
            // x as the low bit and o as the high bit.
            auto encoded_line = std::array<std::size_t, 3>{};
            std::transform(line.begin(), line.end(), encoded_line.begin(),
                           [&state](auto n) {
                               return state.encode()[2*n]
                                   + 2*state.encode()[2*n + 1]; });
            auto [a, b, c] = encoded_line;
            return a > 0 && a == b && b == c;
        });

        return winner_found || (is_draw_ = (n_steps_ == 8));
    }

    /**
     * The number of steps played to date.
     */
    [[nodiscard]] std::size_t n_steps() const { return n_steps_; }

    /**
     * The parity of the number of steps played to date.
     */
    [[nodiscard]] std::size_t n_steps_even() const { return n_steps_even_; }

    /**
     * Return reward from the point of view of the
     * opponent of the finishing player.
     *
     * This results in 0.5 if game is drawn, else 0.0.
     */
    [[nodiscard]] double reward_terminal(const State& state) const {
        return 0.5 * is_draw_;
    }

private:
    //! The number of steps performed to date
    std::size_t n_steps_ = 0;
    //! The parity of n_steps to indicate turns
    bool n_steps_even_ = true;
    //! Switched to true if game is drawn
    mutable bool is_draw_ = false;
};

#endif // TICTACTOE_H_
