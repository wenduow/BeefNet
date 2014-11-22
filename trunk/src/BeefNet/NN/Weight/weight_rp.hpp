#ifndef WEIGHT_RP_HPP_
#define WEIGHT_RP_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < class Param >
class CWeightRP
    : public IWeight
{
public:

    CWeightRP(void)
        : IWeight()
        , m_gradient_sum_prev(0.0)
        , m_delta( Param::update_init )
    {
    }

    ~CWeightRP(void)
    {
    }

    void update(void)
    {
        if ( m_gradient_sum_prev * m_gradient_sum > DOUBLE_EPSILON )
        {
            m_delta *= Param::fact_inc;
            m_delta_weight = - sign(m_gradient_sum) * m_delta;
            m_gradient_sum_prev = m_gradient_sum;
            IWeight::update(m_delta_weight);
        }
        else if ( m_gradient_sum_prev * m_gradient_sum < - DOUBLE_EPSILON )
        {
            m_delta *= Param::fact_dec;
            m_gradient_sum_prev = 0.0;
            IWeight::update( - m_delta_weight );
        }
        else
        {
            m_delta_weight = - sign(m_gradient_sum) * m_delta;
            m_gradient_sum_prev = m_gradient_sum;
            IWeight::update(m_delta_weight);
        }
    }

private:

    inline double sign( IN double val ) const
    {
        if ( val < - DOUBLE_EPSILON )
        {
            return -1.0;
        }
        else if ( val < DOUBLE_EPSILON )
        {
            return 0.0;
        }
        else
        {
            return 1.0;
        }
    }

private:

    double m_gradient_sum_prev;
    double m_delta;
    double m_delta_weight;
};

} // namespace wwd

#endif // WEIGHT_RP_HPP_

