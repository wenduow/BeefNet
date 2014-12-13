#ifndef NET_2_LAYER_HPP
#define NET_2_LAYER_HPP

#include <cmath>
#include "../Layer/layer.hpp"

namespace wwd
{

template < uint32 InputNum,
           uint32 HiddenNum0, class Xfer0,
           uint32 HiddenNum1, class Xfer1,
           uint32 OutputNum, class XferOutput,
           template <class> class WeightType, class Param >
class CNet2Layer
{
public:

    enum
    {
        input_num = InputNum,
        output_num = OutputNum,
    };

private:

    typedef CNet2Layer< InputNum,
                        HiddenNum0, Xfer0,
                        HiddenNum1, Xfer1,
                        OutputNum, XferOutput,
                        WeightType, Param > ThisType;

public:

    CNet2Layer(void)
    {
        connect_inner();
    }

    ~CNet2Layer(void)
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

    friend std::istream &operator>>( INOUT std::istream &is,
                                     OUT ThisType &rhs )
    {
        is >> rhs.m_layer_0 >> rhs.m_layer_1 >> rhs.m_layer_output;

        return is;
    }

    friend std::ostream &operator<<( INOUT std::ostream &os,
                                     IN const ThisType &rhs )
    {
        os << rhs.m_layer_0 << rhs.m_layer_1 << rhs.m_layer_output;

        return os;
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
        return ( m_layer_0.get_gradient_sum()
               + m_layer_1.get_gradient_sum()
               + m_layer_output.get_gradient_sum() )
             / (double)( m_layer_0.get_gradient_num()
                       + m_layer_1.get_gradient_num()
                       + m_layer_output.get_gradient_num() );
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        m_layer_0.save(stream);
        m_layer_1.save(stream);
        m_layer_output.save(stream);
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        m_layer_0.load(stream);
        m_layer_1.load(stream);
        m_layer_output.load(stream);
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
    CLayerHidden< InputNum + 1, HiddenNum0, HiddenNum1, Xfer0,
                  WeightType, Param > m_layer_0;

    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayerHidden< HiddenNum0 + 1, HiddenNum1, OutputNum, Xfer1,
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
class CNet2Layer< InputNum,
                  HiddenNum0, Xfer0,
                  HiddenNum1, Xfer1,
                  OutputNum, XferOutput,
                  CWeightLM, Param >
{
public:

    enum
    {
        input_num = InputNum,
        output_num = OutputNum,
    };

private:

    typedef CNet2Layer< InputNum,
                        HiddenNum0, Xfer0,
                        HiddenNum1, Xfer1,
                        OutputNum, XferOutput,
                        CWeightLM, Param > ThisType;

public:

    CNet2Layer(void)
        : m_check(false)
        , m_se_prev(0.0)
        , m_se(0.0)
    {
        connect_inner();
    }

    ~CNet2Layer(void)
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

        if (m_check)
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
                m_se += std::pow( m_layer_output.get_error(i), 2 );
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

                m_se_prev += std::pow( err, 2 );
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
        return ( m_layer_0.get_gradient_sum()
               + m_layer_1.get_gradient_sum()
               + m_layer_output.get_gradient_sum() )
             / (double)( m_layer_0.get_gradient_num()
                       + m_layer_1.get_gradient_num()
                       + m_layer_output.get_gradient_num() );
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        m_layer_0.save(stream);
        m_layer_1.save(stream);
        m_layer_output.save(stream);
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        m_layer_0.load(stream);
        m_layer_1.load(stream);
        m_layer_output.load(stream);
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
    CLayerHidden< InputNum + 1, HiddenNum0, HiddenNum1, Xfer0,
                  CWeightLM, Param > m_layer_0;

    CLayerInput< 1, HiddenNum1 > m_bias_1;
    CLayerHidden< HiddenNum0 + 1, HiddenNum1, OutputNum, Xfer1,
                  CWeightLM, Param > m_layer_1;

    CLayerInput< 1, OutputNum > m_bias_output;
    CLayerOutput< HiddenNum1 + 1, OutputNum, XferOutput,
                  CWeightLM, Param > m_layer_output;

    bool m_check;
    double m_se_prev;
    double m_se;
};

} // namespace wwd

#endif // NET_2_LAYER_HPP

