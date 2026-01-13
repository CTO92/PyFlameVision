#pragma once

/// @file random_transform.hpp
/// @brief Base class for transforms with random behavior
///
/// Provides thread-safe random number generation infrastructure for
/// data augmentation transforms like RandomHorizontalFlip, RandomRotation, etc.
///
/// @note Thread Safety: The RNG is protected by a mutex for thread-safe
///       random number generation across concurrent calls.

#include "pyflame_vision/transforms/transform_base.hpp"
#include "pyflame_vision/core/security.hpp"
#include <random>
#include <mutex>
#include <optional>

namespace pyflame_vision::transforms {

/// Base class for transforms with random behavior
///
/// Provides:
/// - Thread-safe random number generation via mutex-protected RNG
/// - Secure seed generation by default using generate_secure_seed()
/// - Explicit seeding for reproducible results
///
/// @note Thread Safety: All random number generation methods are thread-safe.
class RandomTransform : public Transform {
public:
    /// Set seed for reproducible results
    /// @param seed Random seed value
    void set_seed(uint64_t seed) {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        rng_.seed(seed);
        seed_ = seed;
        has_explicit_seed_ = true;
    }

    /// Get current seed (if explicitly set)
    /// @return Optional seed value, empty if using default secure seed
    /// @note Thread-safe via mutex
    std::optional<uint64_t> seed() const {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        if (has_explicit_seed_) {
            return seed_;
        }
        return std::nullopt;
    }

    /// Check if transform is deterministic (always false for random transforms)
    bool is_deterministic() const override { return false; }

protected:
    /// Generate random value in [0, 1)
    /// @note Thread-safe via mutex
    float random_uniform() const {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        return dist_(rng_);
    }

    /// Generate random value in [low, high)
    /// @note Thread-safe via mutex
    float random_uniform(float low, float high) const {
        return low + random_uniform() * (high - low);
    }

    /// Generate random boolean with given probability
    /// @param probability Probability of returning true (0.0 to 1.0)
    /// @note Thread-safe via mutex
    bool random_bool(float probability) const {
        return random_uniform() < probability;
    }

    /// Generate random integer in [low, high]
    /// @note Thread-safe via mutex
    int random_int(int low, int high) const {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        std::uniform_int_distribution<int> int_dist(low, high);
        return int_dist(rng_);
    }

    /// Initialize RNG with secure seed
    RandomTransform()
        : rng_(core::generate_secure_seed())
        , dist_(0.0f, 1.0f)
        , has_explicit_seed_(false)
    {}

    /// Virtual destructor
    virtual ~RandomTransform() = default;

private:
    mutable std::mutex rng_mutex_;
    mutable std::mt19937_64 rng_;
    mutable std::uniform_real_distribution<float> dist_;
    uint64_t seed_ = 0;
    bool has_explicit_seed_ = false;
};

}  // namespace pyflame_vision::transforms
