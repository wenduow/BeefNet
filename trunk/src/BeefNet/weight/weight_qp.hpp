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
        , m_learn_rate( (double)LearnRate / 1000.0 )
        , m_fact_max_grow( (double)FactMaxGrow / 1000.0 )
        , m_gradient_prev(0.0)
        , m_delta_weight(0.0)
        , m_pattern_num(0)
    {
    }

    ~CWeightQP(void)
    {
    }

    const CWeightQP &operator>>( OUT CWeightQP &other ) const
    {
        IWeight::operator>>(other);

        other.m_pattern_num = 0;

        return *this;
    }

    CWeightQP &operator<<( IN const CWeightQP &other )
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
        if ( abs(m_gradient_prev) < DOUBLE_EPSILON
          && abs(m_gradient) > DOUBLE_EPSILON )
        {
            m_delta_weight = ( - m_learn_rate * m_gradient )
                           / (double)m_pattern_num;
        }
        else
        {
            double gradient_update = m_gradient - m_gradient_prev;

            if ( abs(gradient_update) > DOUBLE_EPSILON )
            {
                double abs_max_grow = abs(m_delta_weight);
                m_delta_weight *= ( - m_gradient / gradient_update );

                if ( abs(m_delta_weight) > abs_max_grow )
                {
                    if ( m_delta_weight >= 0.0 )
                    {
                        m_delta_weight = abs_max_grow;
                    }
                    else
                    {
                        m_delta_weight = - abs_max_grow;
                    }
                }
            }
            else
            {
                m_delta_weight *= m_fact_max_grow;
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

    const double m_learn_rate;
    const double m_fact_max_grow;

    double m_gradient_prev;
    double m_delta_weight;
    uint32 m_pattern_num;
};

} // namespace wwd

#endif // WEIGHT_QP_HPP_

