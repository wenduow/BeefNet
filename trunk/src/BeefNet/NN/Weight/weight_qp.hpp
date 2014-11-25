#ifndef WEIGHT_QP_HPP_
#define WEIGHT_QP_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < class Param >
class CWeightQP
    : public IWeight
{
public:

    CWeightQP(void)
        : IWeight()
        , m_gradient_sum_prev(0.0)
        , m_delta_weight(0.0)
    {
    }

    ~CWeightQP(void)
    {
    }

    void update(void)
    {
        double rate = m_gradient_sum
                    / ( m_gradient_sum_prev - m_gradient_sum );

        if ( isfinite(rate) )
        {
            if ( rate < - Param::fact_max )
            {
                rate = - Param::fact_max;
            }
            else if ( rate > Param::fact_max )
            {
                rate = Param::fact_max;
            }
        }
        else
        {
            if ( m_gradient_sum >= 0.0
              && ( m_gradient_sum_prev - m_gradient_sum ) >= 0.0 )
            {
                rate = Param::fact_max;
            }
            else if ( m_gradient_sum >= 0.0
                   && ( m_gradient_sum_prev - m_gradient_sum ) < 0.0 )
            {
                rate = - Param::fact_max;
            }
            else if ( m_gradient_sum < 0.0
                   && ( m_gradient_sum_prev - m_gradient_sum ) >= 0.0 )
            {
                rate = - Param::fact_max;
            }
            else
            {
                rate = Param::fact_max;
            }
        }

        m_gradient_sum_prev = m_gradient_sum;

        if ( abs(m_delta_weight) < DOUBLE_EPSILON
          && abs(rate) > DOUBLE_EPSILON )
        {
            m_delta_weight = - Param::learn_rate
                             * m_gradient_sum
                             / (double)m_pattern_num;
            IWeight::update(m_delta_weight);
        }
        else
        {
            m_delta_weight *= rate;
            IWeight::update(m_delta_weight);
        }
    }

private:

    double m_gradient_sum_prev;
    double m_delta_weight;
};

} // namespace wwd

#endif // WEIGHT_QP_HPP_

