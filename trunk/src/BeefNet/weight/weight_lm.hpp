#ifndef WEIGHT_LM_HPP_
#define WEIGHT_LM_HPP_

#include "weight_itf.hpp"

namespace wwd
{

template < uint32 Lambda = 500, uint32 LearnRate = 50 >
class CWeightLM
    : public IWeight
{
public:

    CWeightLM(void)
        : IWeight()
        , m_lambda( (double)Lambda / 1000.0 )
        , m_delta_weight_den(0.0)
        , m_learn_rate( (double)LearnRate / 1000.0 )
        , m_pattern_num(0)
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

        return *this;
    }

    CWeightLM &operator<<( IN const CWeightLM &other )
    {
        IWeight::operator<<(other);

        m_delta_weight_den += other.m_delta_weight_den;
        m_pattern_num += other.m_pattern_num;

        return *this;
    }

    void backward(void)
    {
        IWeight::backward();

        // Sum up ( f'(net) * x ) ^ 2
        m_delta_weight_den += pow( m_output[0]->get_backward_val()
                                 * m_forward_input, 2 );

        ++m_pattern_num;
    }

    inline void update(void)
    {
        double delta_weight;

        if ( abs(m_delta_weight_den) > DOUBLE_EPSILON )
        {
            delta_weight = ( - m_gradient )
                         / ( ( 1.0 + m_lambda ) * m_delta_weight_den );
        }
        else
        {
            delta_weight = ( - m_learn_rate * m_gradient )
                         / (double)m_pattern_num;
        }

        m_weight += delta_weight;

        m_gradient = 0.0;
        m_delta_weight_den = 0.0;
        m_pattern_num = 0;
    }

private:

    CWeightLM( IN CWeightLM &other );
    inline CWeightLM &operator=( IN const CWeightLM &other );

private:

    const double m_lambda;
    const double m_learn_rate;

    double m_delta_weight_den;
    uint32 m_pattern_num;
};

} // namespace wwd

#endif // WEIGHT_LM_HPP_

