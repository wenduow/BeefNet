#ifndef WEIGHT_VECTOR_HPP_
#define WEIGHT_VECTOR_HPP_

#include "weight.hpp"

namespace wwd
{

template < uint32 InputNum >
class CWeightVector
{
public:

    enum
    {
        input_num = InputNum
    };

public:

    CWeightVector(void)
        : m_output(NULL)
    {
    }

    ~CWeightVector(void)
    {
    }

    const CWeightVector &operator>>( OUT CWeightVector &other ) const
    {
        for ( uint32 i = 0; i < InputNum )
        {
            m_weight[i] >> other.m_weight[i];
        }

        return *this;
    }

    inline CWeightVector &operator>>( IN const CWeightVector &other )
    {
        for ( uint32 i = 0; i < InputNum )
        {
            m_weight[i] << other.m_weight[i];
        }

        return *this;
    }

    /** connect input neuron layer */
    template < class Input >
    void connect_input( INOUT Input &input )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].connect_input( input.get_neuron(i) );
        }
    }

    template < class Output >
    void set_output( IN const Output &output )
    {
        m_output = &output;
    }

    inline CWeight &get_weight( IN uint32 idx )
    {
        return m_weight[idx];
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( const auto &i : m_weight )
        {
            i.print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

protected:

    CWeight m_weight[InputNum];

    const IOutput *m_output;
};

}

#endif // WEIGHT_VECTOR_HPP_

