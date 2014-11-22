#ifndef WEIGHT_BP_HPP_
#define WEIGHT_BP_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < class Param >
class CWeightBP
    : public IWeight
{
public:

    CWeightBP(void)
        : IWeight()
    {
    }

    ~CWeightBP(void)
    {
    }

    void update(void)
    {
        // delta = - lr * dE / dWi
        // w <-- w + delta
        IWeight::update( - Param::learn_rate
                         * m_gradient_sum
                         / (double)m_pattern_num );
    }
};

} // namespace wwd

#endif // WEIGHT_BP_HPP_

