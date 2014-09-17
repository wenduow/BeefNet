#ifndef WEIGHT_LM_HPP_
#define WEIGHT_LM_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < uint32 Lambda = 10, uint32 Beta = 10 >
class CWeightLM
    : public IWeight
{
public:

    CWeightLM(void)
        : IWeight()
        , m_lambda( (double)Lambda / 1000.0 )
        , m_beta( (double)Beta )
        , m_delta_weight_den(0.0)
        , m_pattern_num(0)
        , m_err_prev(DOUBLE_MAX)
        , m_err(0.0)
    {
    }

    ~CWeightLM(void)
    {
    }

    const CWeightLM &operator>>( OUT CWeightLM &other ) const
    {
        IWeight::operator>>(other);

        other.m_delta_weight_den = 0.0;
        other.m_pattern_num = 0;
        other.m_err = 0.0;

        return *this;
    }

    CWeightLM &operator<<( IN const CWeightLM &other )
    {
        IWeight::operator<<(other);

        m_delta_weight_den += other.m_delta_weight_den;
        m_pattern_num += other.m_pattern_num;
        m_err += other.m_err;

        return *this;
    }

    void backward(void)
    {
        IWeight::backward();

        // Sum up ( f'(net) * x ) ^ 2
        m_delta_weight_den += pow( m_output[0]->get_backward_val()
                                 * m_forward_input, 2 );

        if ( m_delta_weight_den < DOUBLE_EPSILON )
        {
            m_delta_weight_den = DOUBLE_EPSILON;
        }

        ++m_pattern_num;
        m_err += pow( m_backward_input, 2 );
    }

    inline void update(void)
    {
        m_lambda *= ( ( m_err > m_err_prev ) ? m_beta : ( 1.0 / m_beta ) );
        
        double delta_weight = ( - m_gradient )
                            / ( ( 1.0 + m_lambda ) * m_delta_weight_den );

        m_weight += delta_weight;

        m_gradient = 0.0;
        m_delta_weight_den = 0.0;
        m_pattern_num = 0;
        m_err_prev = m_err;
        m_err = 0.0;
    }

private:

    CWeightLM( IN CWeightLM &other );
    inline CWeightLM &operator=( IN const CWeightLM &other );

private:

    double m_lambda;
    const double m_beta;

    double m_delta_weight_den;
    uint32 m_pattern_num;
    double m_err_prev;
    double m_err;
};

} // namespace wwd

#endif // WEIGHT_LM_HPP_

