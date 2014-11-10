#ifndef NN_3_LAYER_HPP_
#define NN_3_LAYER_HPP_

#include "../layer/layer_input.hpp"
#include "../layer/layer.hpp"
#include "../layer/layer_target.hpp"

namespace wwd
{

template < class Update,
           uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 HiddenNum2, class Xfer2,
           uint32 OutputNum, class XferOutput >
class CNet3Layer
{
public:

    CNet3Layer(void)
    {
        connect_inner();
    }

    ~CNet3Layer(void)
    {
    }

    void forward(void)
    {
        m_input.forward();

        m_bias_0.forward();
        m_hidden_0.forward();

        m_bias_1.forward();
        m_hidden_1.forward();

        m_bias_2.forward();
        m_hidden_2.forward();

        m_bias_output.forward();
        m_output.forward();
    }

    void backward(void)
    {
        m_target.backward();
        m_output.backward();
        m_hidden_2.backward();
        m_hidden_1.backward();
        m_hidden_0.backward();
    }

    void update(void)
    {
        m_hidden_0.update();
        m_hidden_1.update();
        m_hidden_2.update();
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
        m_hidden_0.print_weight();
        std::cout << std::endl;
        m_hidden_1.print_weight();
        std::cout << std::endl;
        m_hidden_2.print_weight();
        std::cout << std::endl;
        m_output.print_weight();
        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    void connect_inner(void)
    {
        const double bias[1] = { 1.0 };

        m_bias_0.set_input(bias);
        m_hidden_0.connect_input(m_bias_0);
        m_hidden_0.connect_input(m_input);

        m_bias_1.set_input(bias);
        m_hidden_1.connect_input(m_bias_1);
        m_hidden_1.connect_input(m_hidden_0);

        m_bias_2.set_input(bias);
        m_hidden_2.connect_input(m_bias_2);
        m_hidden_2.connect_input(m_hidden_1);

        m_bias_output.set_input(bias);
        m_output.connect_input(m_bias_output);
        m_output.connect_input(m_hidden_2);

        m_target.connect_input(m_output);
    }

private:

    CLayerInput< InputNum, HiddenNum0 > m_input;
    CLayerInput< 1, HiddenNum0 > m_bias_0;
    CLayer< Update, InputNum + 1, HiddenNum0, Xfer0, HiddenNum1 > m_hidden_0;
    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayer< Update, HiddenNum0 + 1, HiddenNum1, Xfer1, HiddenNum2 > m_hidden_1;
    CLayerInput< 1, HiddenNum2 > m_bias_2;
    CLayer< Update, HiddenNum1 + 1, HiddenNum2, Xfer1, OutputNum > m_hidden_2;
    CLayerInput< 1, OutputNum > m_bias_output;
    CLayer< Update, HiddenNum1 + 1, OutputNum, XferOutput, 1 > m_output;
    CLayerTarget< OutputNum > m_target;
};

} // namespace wwd

#endif // NN_3_LAYER_HPP_

