#ifndef WEIGHT_RP_HPP_
#define WEIGHT_RP_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < uint32 FactInc    = 1200,
           uint32 FactDec    = 500,
           uint32 UpdateInit = 100 >
class CWeightRP
    : public IWeight
{
private:

    typedef CWeightRP< FactInc, FactDec, UpdateInit > ThisType;

public:

    CWeightRP(void)
        : IWeight()
        , m_gradient_prev(0.0)
        , m_update( (double)UpdateInit / 1000.0 )
        , m_delta_weight(0.0)
        , m_fact_inc( (double)FactInc / 1000.0 )
        , m_fact_dec( (double)FactDec / 1000.0 )
    {
    }

    ~CWeightRP(void)
    {
    }

    inline void update(void)
    {
        if ( m_gradient_prev * m_gradient > DOUBLE_EPSILON )
        {
            m_update *= m_fact_inc;

            if ( m_gradient > DOUBLE_EPSILON )
            {
                m_delta_weight = - m_update;
            }
            else if ( m_gradient < -DOUBLE_EPSILON )
            {
                m_delta_weight = m_update;
            }
            else
            {
                m_delta_weight = 0.0;
            }
        }
        else if ( m_gradient_prev * m_gradient < -DOUBLE_EPSILON )
        {
            m_update *= m_fact_dec;
            m_gradient = 0.0;
            m_delta_weight = -m_delta_weight;
        }
        else
        {
            if ( m_gradient > DOUBLE_EPSILON )
            {
                m_delta_weight = - m_update;
            }
            else if ( m_gradient < -DOUBLE_EPSILON )
            {
                m_delta_weight = m_update;
            }
            else
            {
                m_delta_weight = 0.0;
            }
        }

        m_weight += m_delta_weight;
        m_gradient_prev = m_gradient;
        m_gradient = 0.0;
    }

private:

    CWeightRP( IN const CWeightRP &other );
    inline CWeightRP &operator=( IN const CWeightRP &other );

private:

    double m_gradient_prev;
    double m_update;
    double m_delta_weight;
    
    const double m_fact_inc;
    const double m_fact_dec;
};

} // namespace wwd

#endif // WEIGHT_RP_HPP_

