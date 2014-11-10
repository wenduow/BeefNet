#ifndef NN_1_LAYER_HPP_
#define NN_1_LAYER_HPP_

#include "../layer/layer_input.hpp"
#include "../layer/layer.hpp"
#include "../layer/layer_target.hpp"

namespace wwd
{

template < class Update,
           uint32 InputNum,
           uint32 HiddenNum, class Xfer,
           uint32 OutputNum, class XferOutput >
class CNet1Layer
{
public:

    CNet1Layer(void)
    {
        connect_inner();
    }

    ~CNet1Layer(void)
    {
    }

    void forward(void)
    {
        m_input.forward();

        m_bias.forward();
        m_hidden.forward();

        m_bias_output.forward();
        m_output.forward();
    }

    void backward(void)
    {
        m_target.backward();
        m_output.backward();
        m_hidden.backward();
    }

    void update(void)
    {
        m_hidden.update();
        m_output.update();
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
        m_hidden.print_weight();
        std::cout << std::endl;
        m_output.print_weight();
        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    void connect_inner(void)
    {
        const double bias[1] = { 1.0 };

        m_bias.set_input(bias);
        m_hidden.connect_input(m_bias);
        m_hidden.connect_input(m_input);

        m_bias_output.set_input(bias);
        m_output.connect_input(m_bias_output);
        m_output.connect_input(m_hidden);

        m_target.connect_input(m_output);
    }

private:

    CLayerInput< InputNum, HiddenNum > m_input;
    CLayerInput< 1, HiddenNum > m_bias;
    CLayer< Update, InputNum + 1, HiddenNum, Xfer, OutputNum > m_hidden;
    CLayerInput< 1, OutputNum > m_bias_output;
    CLayer< Update, HiddenNum + 1, OutputNum, XferOutput, 1 > m_output;
    CLayerTarget< OutputNum > m_target;
};

} // namespace wwd

#endif // NN_1_LAYER_HPP_

