#ifndef NN_2_LAYER_HPP_
#define NN_2_LAYER_HPP_

#include "../Layer/layer_input.hpp"
#include "../Layer/layer.hpp"
#include "../Layer/layer_output.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 OutputNum, class XferOutput,
           template < uint32, class > class Weight,
           class Param >
class CNN2Layer
{
public:

    CNN2Layer(void)
    {
        connect_inner();
    }

    ~CNN2Layer(void)
    {
    }

    void forward(void)
    {
        m_input.forward();

        m_bias_0.forward();
        m_layer_0.forward();

        m_bias_1.forward();
        m_layer_1.forward();

        m_bias_output.forward();
        m_output.forward();
    }

    void backward(void)
    {
        m_output.backward();
        m_layer_1.backward();
        m_layer_0.backward();
    }

    void update(void)
    {
        m_layer_0.update();
        m_layer_1.update();
        m_output.update();
    }

    void set_input( IN const double *input )
    {
        m_input.set_input(input);
    }

    void set_target( IN const double *target )
    {
        m_output.set_target(target);
    }

#ifdef _DEBUG
    void print_weight(void)
    {
        m_layer_0.print_weight();
        m_layer_1.print_weight();
        m_output.print_weight();
    }
#endif // _DEBUG

private:

    void connect_inner(void)
    {
        double bias[1] = {1.0};

        m_bias_0.set_input(bias);
        m_layer_0.connect_input_layer(m_bias_0);
        m_layer_0.connect_input_layer(m_input);

        m_bias_1.set_input(bias);
        m_layer_1.connect_input_layer(m_bias_1);
        m_layer_1.connect_input_layer(m_layer_0);

        m_bias_output.set_input(bias);
        m_output.connect_input_layer(m_bias_output);
        m_output.connect_input_layer(m_layer_1);
    }

private:

    CLayerInput< InputNum, HiddenNum0 > m_input;

    CLayerInput< 1, HiddenNum0 > m_bias_0;
    CLayer< InputNum + 1, HiddenNum0, HiddenNum1, Xfer0,
            Weight, Param > m_layer_0;

    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayer< HiddenNum0 + 1, HiddenNum1, OutputNum, Xfer1,
            Weight, Param > m_layer_1;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayerOutput< HiddenNum1 + 1, OutputNum, XferOutput,
                  Weight, Param > m_output;
};

} // namespace wwd

#endif // NN_2_LAYER_HPP_

