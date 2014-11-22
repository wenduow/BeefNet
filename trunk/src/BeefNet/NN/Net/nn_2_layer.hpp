#ifndef NN_2_LAYER_HPP
#define NN_2_LAYER_HPP

#include "../Layer/layer_input.hpp"
#include "../Layer/layer.hpp"
#include "../Layer/layer_output.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 OutputNum, class XferOutput,
           template <class> class WeightType, class Param >
class CNN2Layer
{
private:

    typedef CNN2Layer< InputNum,
                       HiddenNum0, Xfer0,
                       HiddenNum1, Xfer1,
                       OutputNum, XferOutput,
                       WeightType, Param > ThisType;

public:

    CNN2Layer(void)
    {
        connect_inner();
    }

    ~CNN2Layer(void)
    {
    }

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        m_layer_0 >> other.m_layer_0;
        m_layer_1 >> other.m_layer_1;
        m_layer_output >> other.m_layer_output;

        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        m_layer_0 << other.m_layer_0;
        m_layer_1 << other.m_layer_1;
        m_layer_output << other.m_layer_output;

        return *this;
    }

    void forward(void)
    {
        m_input.forward();

        m_bias_0.forward();
        m_layer_0.forward();

        m_bias_1.forward();
        m_layer_1.forward();

        m_bias_output.forward();
        m_layer_output.forward();
    }

    void backward(void)
    {
        m_layer_output.backward();
        m_layer_1.backward();
        m_layer_0.backward();
    }

    void update(void)
    {
        m_layer_0.update();
        m_layer_1.update();
        m_layer_output.update();
    }

    void set_input( IN const double (&input)[InputNum] )
    {
        m_input.set_input(input);
    }

    void set_target( IN const double (&target)[OutputNum] )
    {
        m_layer_output.set_target(target);
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        m_layer_0.print_weight();
        m_layer_1.print_weight();
        m_layer_output.print_weight();
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
        m_layer_output.connect_input_layer(m_bias_output);
        m_layer_output.connect_input_layer(m_layer_1);
    }

private:

    CLayerInput< InputNum, HiddenNum0 > m_input;

    CLayerInput< 1, HiddenNum0 > m_bias_0;
    CLayer< InputNum + 1, HiddenNum0, HiddenNum1, Xfer0,
            WeightType, Param > m_layer_0;

    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayer< HiddenNum0 + 1, HiddenNum1, OutputNum, Xfer1,
            WeightType, Param > m_layer_1;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayerOutput< HiddenNum1 + 1, OutputNum, XferOutput,
                  WeightType, Param > m_layer_output;
};

template < uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 OutputNum, class XferOutput,
           class Param >
class CNN2Layer< InputNum,
                 HiddenNum0, Xfer0,
                 HiddenNum1, Xfer1,
                 OutputNum, XferOutput,
                 CWeightLM, Param >
{
private:

    typedef CNN2Layer< InputNum,
                       HiddenNum0, Xfer0,
                       HiddenNum1, Xfer1,
                       OutputNum, XferOutput,
                       CWeightLM, Param > ThisType;

public:

    CNN2Layer(void)
        : m_check(false)
        , m_se_prev(0.0)
        , m_se(0.0)
    {
        connect_inner();
    }

    ~CNN2Layer(void)
    {
    }

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        m_layer_0 >> other.m_layer_0;
        m_layer_1 >> other.m_layer_1;
        m_layer_output >> other.m_layer_output;

        if (m_check)
        {
            other.m_se = 0.0;
        }
        else
        {
            other.m_se_prev = 0.0;
        }

        other.m_check = m_check;
        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        m_layer_0 << other.m_layer_0;
        m_layer_1 << other.m_layer_1;
        m_layer_output << other.m_layer_output;

        if ( other.m_check )
        {
            m_se += other.m_se;
        }
        else
        {
            m_se_prev += other.m_se_prev;
        }

        return *this;
    }

    void forward(void)
    {
        m_input.forward();

        m_bias_0.forward();
        m_layer_0.forward();

        m_bias_1.forward();
        m_layer_1.forward();

        m_bias_output.forward();
        m_layer_output.forward();
    }

    void backward(void)
    {
        if (m_check)
        {
            for ( uint32 i = 0; i < OutputNum; ++i )
            {
                m_se += pow( m_layer_output.get_error(i), 2 );
            }
        }
        else
        {
            for ( uint32 i = 0; i < OutputNum; ++i )
            {
                double err = m_layer_output.get_error(i);

                m_layer_output.backward( err, i );
                m_layer_1.backward(err);
                m_layer_0.backward(err);

                m_se_prev += pow( err, 2 );
            }
        }
    }

    void update(void)
    {
        if (m_check)
        {
            if ( m_se < m_se_prev )
            {
                Param::lambda /= Param::beta;
                if ( Param::lambda < DOUBLE_EPSILON )
                {
                    Param::lambda = DOUBLE_EPSILON;
                }
            }
            else
            {
                revert();

                Param::lambda *= Param::beta;
                if ( Param::lambda > DOUBLE_MAX )
                {
                    Param::lambda = DOUBLE_MAX;
                }
            }

            m_se_prev = 0.0;
            m_check = false;
        }
        else
        {
            m_layer_0.update();
            m_layer_1.update();
            m_layer_output.update();

            m_se = 0.0;
            m_check = true;
        }
    }

    void set_input( IN const double (&input)[InputNum] )
    {
        m_input.set_input(input);
    }

    void set_target( IN const double (&target)[OutputNum] )
    {
        m_layer_output.set_target(target);
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        m_layer_0.print_weight();
        m_layer_1.print_weight();
        m_layer_output.print_weight();
    }
#endif // _DEBUG

private:

    void revert(void)
    {
        m_layer_0.revert();
        m_layer_1.revert();
        m_layer_output.revert();
    }

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
        m_layer_output.connect_input_layer(m_bias_output);
        m_layer_output.connect_input_layer(m_layer_1);
    }

private:

    CLayerInput< InputNum, HiddenNum0 > m_input;

    CLayerInput< 1, HiddenNum0 > m_bias_0;
    CLayer< InputNum + 1, HiddenNum0, HiddenNum1, Xfer0,
            CWeightLM, Param > m_layer_0;

    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayer< HiddenNum0 + 1, HiddenNum1, OutputNum, Xfer1,
            CWeightLM, Param > m_layer_1;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayerOutput< HiddenNum1 + 1, OutputNum, XferOutput,
                  CWeightLM, Param > m_layer_output;

    bool m_check;
    double m_se_prev;
    double m_se;
};

} // namespace wwd

#endif // NN_2_LAYER_HPP

