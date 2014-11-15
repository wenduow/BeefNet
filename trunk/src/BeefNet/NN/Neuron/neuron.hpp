#ifndef NEURON_HPP_
#define NEURON_HPP_

#include "../../Utility/input_itf.hpp"
#include "../../Utility/output_itf.hpp"

namespace wwd
{

template < uint32 InputNum, uint32 OutputNum, class Xfer >
class CNeuron
    : public IInput<OutputNum>
    , public IOutput<InputNum>
{
public:

    CNeuron(void)
        : IInput()
        , IOutput()
        , m_xfer()
    {

    }

    ~CNeuron(void)
    {
    }

    void forward(void)
    {
        IForward::m_input_val = 0.0;

        for ( auto &i : m_input_node )
        {
            IForward::m_input_val += i->get_output_val();
        }

        IForward::m_output_val = m_xfer(IForward::m_input_val);
        IBackward::m_input_val = m_xfer.derivative(IForward::m_input_val);
    }

    void backward(void)
    {
        IBackward::m_output_val = 0.0;

        for ( auto &i : m_output_node )
        {
            IBackward::m_output_val += i->get_output_val();
        }
    }

private:

    CNeuron( IN const CNeuron &other );
    inline CNeuron &operator=( IN const CNeuron &other );

private:

    const Xfer m_xfer;
};

} // namespace wwd

#endif // NEURON_HPP_

