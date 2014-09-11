#ifndef WEIGHT_ITF_HPP_
#define WEIGHT_ITF_HPP_

#include <cstdlib>
#include "../utility/input_itf.hpp"
#include "../utility/output_itf.hpp"

namespace wwd
{

class IWeight
    : public IInput<1>
    , public IOutput<1>
{
public:

    inline const IWeight &operator>>( OUT IWeight &other ) const
    {
        other.m_weight = m_weight;
        other.m_gradient = 0.0;

        return *this;
    }

    inline IWeight &operator<<( IN const IWeight &other )
    {
        m_gradient += other.m_gradient;

        return *this;
    }

    inline void forward(void)
    {
        m_forward_input = m_input[0]->get_forward_output();
        m_forward_output = m_weight * m_forward_input;
    }

    inline void backward(void)
    {
        m_backward_input = m_output[0]->get_backward_output();
        m_backward_output = m_weight * m_backward_input;

        // gradient = dE / dWi = - Sum( delta * f'(net) * Xi ),
        // where Sum through all input samples,
        // delta * f'(net) is calculated from next connected neuron.
        m_gradient -= ( m_backward_input * m_forward_input );
    }

    inline double get_weight(void) const
    {
        return m_weight;
    }

    inline double get_gradient(void) const
    {
        return m_gradient;
    }

protected:

    IWeight(void)
        : IInput<1>()
        , IOutput<1>()
        , m_weight( (double)rand() / (double)RAND_MAX )
        , m_gradient(1.0)
    {
    }

    ~IWeight(void)
    {
    }

private:

    IWeight( IN const IWeight &other );
    inline IWeight &operator=( IN const IWeight &other );

protected:

    double m_weight;
    double m_gradient;
};

} // namespace wwd

#endif // WEIGHT_ITF_HPP_

