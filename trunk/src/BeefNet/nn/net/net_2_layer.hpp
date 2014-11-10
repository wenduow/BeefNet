#ifndef NN_2_LAYER_HPP_
#define NN_2_LAYER_HPP_

#include "../layer/layer_input.hpp"
#include "../layer/layer_weight.hpp"
#include "../layer/layer.hpp"
#include "../layer/layer_target.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 OutputNum, class XferOutput,
           template < uint32, class > class WeightType,
           class WeightParam >
class CNet2Layer
{
public:

    CNet2Layer(void)
    {
        connect();
    }

    ~CNet2Layer(void)
    {
    }

    void forward(void)
    {
        m_input.forward();

        m_bias_0.forward();
        m_weight_bias_0.forward();
        m_weight_0.forward();
        m_hidden_0.forward();

        m_bias_1.forward();
        m_weight_bias_1.forward();
        m_weight_1.forward();
        m_hidden_1.forward();

        m_bias_output.forward();
        m_weight_bias_output.forward();
        m_weight_output.forward();
        m_output.forward();
    }

    void backward(void)
    {
        m_target.backward();

        m_output.backward();
        m_weight_bias_output.backward();
        m_weight_output.backward();
        
        m_hidden_1.backward();
        m_weight_bias_1.backward();
        m_weight_1.backward();

        m_hidden_0.backward();
        m_weight_bias_0.backward();
        m_weight_0.backward();
    }

    void update(void)
    {
        m_weight_bias_0.update();
        m_weight_0.update();

        m_weight_bias_1.update();
        m_weight_1.update();

        m_weight_bias_output.update();
        m_weight_output.update();
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
        m_weight_bias_0.print_weight();
        m_weight_0.print_weight();

        m_weight_bias_1.print_weight();
        m_weight_1.print_weight();

        m_weight_bias_output.print_weight();
        m_weight_output.print_weight();
    }
#endif // _DEBUG

private:

    void connect(void)
    {
        const double bias[1] = { 1.0 };

        m_bias_0.set_input(bias);
        m_weight_bias_0.connect_input(m_bias_0);
        m_weight_0.connect_input(m_input);
        m_hidden_0.connect_input(m_weight_bias_0);
        m_hidden_0.connect_input(m_weight_0);

        m_bias_1.set_input(bias);
        m_weight_bias_1.connect_input(m_bias_1);
        m_weight_1.connect_input(m_hidden_0);
        m_hidden_1.connect_input(m_weight_bias_1);
        m_hidden_1.connect_input(m_weight_1);

        m_bias_output.set_input(bias);
        m_weight_bias_output.connect_input(m_bias_output);
        m_weight_output.connect_input(m_hidden_1);
        m_output.connect_input(m_weight_bias_output);
        m_output.connect_input(m_weight_output);

        m_target.connect_input(m_output);
    }

private:

    CLayerInput< InputNum, HiddenNum0 > m_input;

    CLayerInput< 1, HiddenNum0 > m_bias_0;
    CLayerWeight< 1, HiddenNum0, WeightType, WeightParam > m_weight_bias_0;
    CLayerWeight< InputNum, HiddenNum0, WeightType, WeightParam > m_weight_0;
    CLayer< InputNum + 1, HiddenNum0, Xfer0, HiddenNum1 > m_hidden_0;

    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayerWeight< 1, HiddenNum1, WeightType, WeightParam > m_weight_bias_1;
    CLayerWeight< HiddenNum0, HiddenNum1, WeightType, WeightParam > m_weight_1;
    CLayer< HiddenNum0 + 1, HiddenNum1, Xfer1, OutputNum > m_hidden_1;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayerWeight< 1, OutputNum, WeightType, WeightParam > m_weight_bias_output;
    CLayerWeight< HiddenNum1,
                  OutputNum,
                  WeightType,
                  WeightParam > m_weight_output;
    CLayer< HiddenNum1 + 1, OutputNum, XferOutput, 1 > m_output;

    CLayerTarget< OutputNum > m_target;
};

} // namespace wwd

#endif // NN_2_LAYER_HPP_

