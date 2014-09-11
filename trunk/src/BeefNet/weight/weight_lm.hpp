#ifndef WEIGHT_LM_HPP_
#define WEIGHT_LM_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < uint32 Lambda = 500 >
class CWeightLM
    : public IWeight
{
public:

    CWeightLM(void)
        : IWeight()
        , m_delta_weight_den(0.0)
        , m_lambda( (double)Lambda / 1000.0 )
    {
    }

    ~CWeightLM(void)
    {
    }

    const CWeightLM &operator>>( OUT CWeightLM &other ) const
    {
        IWeight::operator>>(other);

        other.m_delta_weight_den = 0.0;

        return *this;
    }

    CWeightLM &operator<<( IN const CWeightLM &other )
    {
        IWeight::operator<<(other);

        m_delta_weight_den += other.m_delta_weight_den;

        return *this;
    }

    void backward(void)
    {
        IWeight::backward();

        if ( abs( m_output[0]->get_backward_input() ) > DOUBLE_EPSILON )
        {
            m_delta_weight_den += pow( m_backward_input / m_output[0]->get_backward_input() * m_forward_input, 2 );
        }
    }

    inline void update(void)
    {
        if ( abs(m_delta_weight_den) > DOUBLE_EPSILON )
        {
            double delta_weight = ( - m_gradient )
                                / ( ( 1.0 + m_lambda ) * m_delta_weight_den );
            m_weight += delta_weight;
        }

        m_gradient = 0.0;
        m_delta_weight_den = 0.0;
    }

private:

    CWeightLM( IN CWeightLM &other );
    inline CWeightLM &operator=( IN const CWeightLM &other );

private:

    double m_delta_weight_den;

    const double m_lambda;
};

} // namespace wwd

#endif // WEIGHT_LM_HPP_

