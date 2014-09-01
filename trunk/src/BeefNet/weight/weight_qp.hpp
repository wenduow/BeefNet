#ifndef WEIGHT_QP_HPP_
#define WEIGHT_QP_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < uint32 LearnRate = 50, uint32 FactMaxGrow = 1750 >
class CWeightQP
    : public IWeight
{
public:

    CWeightQP(void)
        : IWeight()
        , m_gradient_prev(0.0)
        , m_delta_weight(0.0)
        , m_learn_rate( (double)LearnRate / 1000.0 )
        , m_fact_max_grow( (double)FactMaxGrow / 1000.0 )
    {
    }

    ~CWeightQP(void)
    {
    }

    inline void update(void)
    {
        if ( m_delta_weight > - DOUBLE_EPSILON
          && m_delta_weight < DOUBLE_EPSILON )
        {
            m_delta_weight = ( - m_learn_rate * m_gradient ) / (double)m_pattern_num;
        }
        else
        {
            double max_grow = m_fact_max_grow
                            * ( m_delta_weight > 0.0 ? 1.0 : -1.0 )
                            * m_delta_weight;

            if ( m_gradient_prev - m_gradient < - DOUBLE_EPSILON
              || m_gradient_prev - m_gradient > DOUBLE_EPSILON )
            {
                m_delta_weight *= ( m_gradient
                                  / ( m_gradient_prev - m_gradient ) );
            }
            else
            {
                m_delta_weight = ( m_delta_weight > 0.0 ? 1.0 : -1.0 )
                               * max_grow;
            }

            if ( m_delta_weight > max_grow )
            {
                m_delta_weight = max_grow;
            }
            else if ( m_delta_weight < -max_grow )
            {
                m_delta_weight = -max_grow;
            }
        }

        m_weight += m_delta_weight;

        m_gradient_prev = m_gradient;
        m_gradient = 0.0;
        m_pattern_num = 0;
    }

private:

    CWeightQP( IN const CWeightQP &other );
    inline CWeightQP &operator=( IN const CWeightQP &other );

private:

    double m_gradient_prev;
    double m_delta_weight;

    const double m_learn_rate;
    const double m_fact_max_grow;
};

} // namespace wwd

#endif // WEIGHT_QP_HPP_

