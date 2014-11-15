#ifndef WEIGHT_HPP_
#define WEIGHT_HPP_

#ifdef _DEBUG
#include <iostream>
#endif // _DEBUG

#include <cstdlib>
#include "../../Utility/input_itf.hpp"
#include "../../Utility/output_itf.hpp"

namespace wwd
{

class CWeight
    : public IInput<1>
    , public IOutput<1>
{
public:

    CWeight(void)
        : IInput()
        , IOutput()
        , m_weight( (double)rand() / (double)RAND_MAX * 1.4 - 0.7 )
        , m_gradient(0.0)
    {
    }

    ~CWeight(void)
    {
    }

    inline void forward(void)
    {
        IForward::m_input_val = m_input_node[0]->get_output_val();
        IForward::m_output_val = m_weight * IForward::m_input_val;
    }

    inline void backward(void)
    {
        IBackward::m_input_val = m_output_node[0]->get_output_val();
        IBackward::m_output_val = m_weight * IBackward::m_input_val;

        // gradient = dE / dWi = - Sum( delta * f'(net) * Xi ),
        // where Sum is through all input samples.
        // get_backward_val here gets the f'(net) from next neuron.
        m_gradient -= ( IBackward::m_input_val
                      * m_output_node[0]->get_input_val()
                      * IForward::m_input_val );
    }

    inline void update( IN double delta_weight )
    {
        m_weight += delta_weight;
        m_gradient = 0.0;
    }

    inline double get_weight(void) const
    {
        return m_weight;
    }

    inline double get_gradient(void) const
    {
        return m_gradient;
    }

#ifdef _DEBUG
    void print_weight(void)
    {
        std::cout << m_weight << '\t';
    }
#endif // _DEBUG

private:

    double m_weight;
    double m_gradient;
};

} // namespace wwd

#endif // WEIGHT_HPP_

