#ifndef NEURON_HPP_
#define NEURON_HPP_

#include "input_itf.hpp"
#include "output_itf.hpp"

namespace wwd
{

template < uint32 InputNum, uint32 OutputNum, class Xfer >
class CNeuron
    : public IInput<OutputNum>
    , public IOutput<InputNum>
{
public:

    CNeuron(void)
        : IInput<OutputNum>()
        , IOutput<InputNum>()
    {
    }

    ~CNeuron(void)
    {
    }

    void forward(void)
    {
        m_forward_input = 0.0;

        for ( const auto &i : m_input )
        {
            m_forward_input += i->get_forward_output();
        }

        m_forward_output = m_xfer_fxn(m_forward_input);
    }

    void backward(void)
    {
        m_backward_input = 0.0;

        for ( const auto &i : m_output )
        {
            m_backward_input += i->get_backward_output();
        }

        m_backward_output = m_xfer_fxn.derivative(m_forward_input)
                          * m_backward_input;
    }

private:

    Xfer m_xfer_fxn;
};

} // namespace wwd

#endif // NEURON_HPP_

