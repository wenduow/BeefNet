#ifndef NN_RECURRENT_HPP_
#define NN_RECURRENT_HPP_

#include "../layer/layer_input.hpp"
#include "../layer/layer.hpp"
#include "../layer/layer_target.hpp"

namespace wwd
{

template < class Update,
           uint32 InputNum,
           uint32 ForwardNum, class XferForward,
           uint32 BackwardNum, class XferBackward,
           uint32 OutputNum, class XferOutput >
class CNetRecurrent
{
public:

    CNetRecurrent(void)
    {
        connect_inner();
    }

    ~CNetRecurrent(void)
    {
    }

    void forward(void)
    {
        m_input.forward();

        m_bias_forward.forward();
        m_forward.forward();

        m_bias_output.forward();
        m_output.forward();

        m_bias_backward.forward();
        m_backward.forward();
    }

    void backward(void)
    {
        m_target.backward();
        m_backward.backward();
        m_output.backward();
        m_forward.backward();
    }

    void update(void)
    {
        m_forward.update();
        m_output.update();
        m_backward.update();
    }

    void set_input( IN const double *input )
    {
        m_input.set_input(input);
    }

    void set_target( IN const double *target )
    {
        m_target.set_target(target);
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        m_forward.print_weight();
        std::cout << std::endl;

        m_output.print_weight();
        std::cout << std::endl;

        m_backward.print_weight();
        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    void connect_inner(void)
    {
        const double bias[1] = { 1.0 };

        m_bias_forward.set_input(bias);
        m_forward.connect_input(m_bias_forward);
        m_forward.connect_input(m_input);
        m_forward.connect_input(m_backward);

        m_bias_output.set_input(bias);
        m_output.connect_input(m_bias_output);
        m_output.connect_input(m_forward);

        m_bias_backward.set_input(bias);
        m_backward.connect_input(m_bias_backward);
        m_backward.connect_input(m_forward);

        m_target.connect_input(m_output);
    }

private:

    CLayerInput< InputNum, ForwardNum > m_input;

    CLayerInput< 1, ForwardNum > m_bias_forward;
    CLayer< Update, 
            InputNum + BackwardNum + 1, 
            ForwardNum, XferForward, 
            OutputNum + BackwardNum > m_forward;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayer< Update, ForwardNum + 1, OutputNum, XferOutput, 1 > m_output;

    CLayerInput< 1, BackwardNum > m_bias_backward;
    CLayer< Update,
            ForwardNum + 1,
            BackwardNum, XferBackward,
            ForwardNum > m_backward;

    CLayerTarget< OutputNum > m_target;
};

} // namespace wwd

#endif // NN_RECURRENT_HPP_

