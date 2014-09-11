#ifndef WEIGHT_BP_HPP_
#define WEIGHT_BP_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < uint32 LearnRate = 50 >
class CWeightBP
    : public IWeight
{
public:

    CWeightBP(void)
        : IWeight()
        , m_learn_rate( (double)LearnRate / 1000.0 )
        , m_pattern_num(0)
    {
    }

    ~CWeightBP(void)
    {
    }

    const CWeightBP &operator>>( OUT CWeightBP &other ) const
    {
        IWeight::operator>>(other);

        other.m_pattern_num = 0;

        return *this;
    }

    CWeightBP &operator<<( IN const CWeightBP &other )
    {
        IWeight::operator<<(other);

        m_pattern_num += other.m_pattern_num;

        return *this;
    }

    void backward(void)
    {
        IWeight::backward();

        ++m_pattern_num;
    }

    inline void update(void)
    {
        double delta_weight = ( - m_learn_rate * m_gradient ) / (double)m_pattern_num;
        m_weight += delta_weight;

        m_gradient = 0.0;
        m_pattern_num = 0;
    }

private:

    CWeightBP( IN CWeightBP &other );
    inline CWeightBP &operator=( IN const CWeightBP &other );

private:

    const double m_learn_rate;

    uint32 m_pattern_num;
};

} // namespace wwd

#endif // WEIGHT_BP_HPP_

