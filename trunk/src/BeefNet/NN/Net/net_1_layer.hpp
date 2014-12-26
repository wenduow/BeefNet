#ifndef NET_1_LAYER_HPP
#define NET_1_LAYER_HPP

#include <cmath>
#include "../Layer/layer.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 HiddenNum, class Xfer,
           uint32 OutputNum, class XferOutput,
           template <class> class WeightType, class Param >
class CNet1Layer
{
public:

    enum
    {
        input_num = InputNum,
        output_num = OutputNum,
    };

private:

    typedef CNet1Layer< InputNum,
                        HiddenNum, Xfer,
                        OutputNum, XferOutput,
                        WeightType, Param > ThisType;

public:

    CNet1Layer(void)
    {
        connect_inner();
    }

    ~CNet1Layer(void)
    {
    }

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        m_layer >> other.m_layer;
        m_layer_output >> other.m_layer_output;

        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        m_layer << other.m_layer;
        m_layer_output << other.m_layer_output;

        return *this;
    }

    void init(void)
    {
        m_layer.init();
        m_layer_output.init();
    }

    void forward(void)
    {
        m_input.forward();

        m_bias.forward();
        m_layer.forward();

        m_bias_output.forward();
        m_layer_output.forward();
    }

    void backward(void)
    {
        m_layer_output.backward();
        m_layer.backward();
    }

    void update(void)
    {
        m_layer.update();
        m_layer_output.update();
    }

    void set_input( IN const double *input )
    {
        m_input.set_input(input);
    }

    void set_target( IN const double *target )
    {
        m_layer_output.set_target(target);
    }

    void get_output( OUT double (&output)[OutputNum] ) const
    {
        m_layer_output.get_output(output);
    }

    double get_gradient(void) const
    {
        return ( m_layer.get_gradient_sum()
               + m_layer_output.get_gradient_sum() )
             / (double)( m_layer.get_gradient_num()
                       + m_layer_output.get_gradient_num() );
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        m_layer.save(stream);
        m_layer_output.save(stream);
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        m_layer.load(stream);
        m_layer_output.load(stream);
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        m_layer.print_weight();
        m_layer_output.print_weight();
    }
#endif // _DEBUG

private:

    void connect_inner(void)
    {
        double bias[1] = {1.0};

        m_bias.set_input(bias);
        m_layer.connect_input_layer(m_bias);
        m_layer.connect_input_layer(m_input);

        m_bias_output.set_input(bias);
        m_layer_output.connect_input_layer(m_bias_output);
        m_layer_output.connect_input_layer(m_layer);
    }

private:

    CLayerInput< InputNum, HiddenNum > m_input;

    CLayerInput< 1, HiddenNum > m_bias;
    CLayerHidden< InputNum + 1, HiddenNum, OutputNum, Xfer,
                  WeightType, Param > m_layer;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayerOutput< HiddenNum + 1, OutputNum, XferOutput,
                  WeightType, Param > m_layer_output;
};

} // namespace wwd

#endif // NET_1_LAYER_HPP

