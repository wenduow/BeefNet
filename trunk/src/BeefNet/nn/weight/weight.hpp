#ifndef WEIGHT_ITF_HPP_
#define WEIGHT_ITF_HPP_

#ifdef _DEBUG
    #include <iostream>
#endif // _DEBUG

#include <cstdlib>
#include "../../utility/node.hpp"

namespace wwd
{

class CWeight
    : public CNode< 1, 1 >
{
public:

    CWeight(void)
        : CNode()
        , m_weight( (double)rand() / (double)RAND_MAX * 1.4 - 0.7 )
        , m_gradient(0.0)
    {
    }

    ~CWeight(void)
    {
    }

    inline const CWeight &operator>>( OUT CWeight &other ) const
    {
        other.m_weight = m_weight;
        other.m_gradient = 0.0;

        return *this;
    }

    inline CWeight &operator<<( IN const CWeight &other )
    {
        m_gradient += other.m_gradient;

        return *this;
    }

        inline void forward(void)
    {
        IForward::m_input = m_input_node[0]->get_output();
        IForward::m_output = m_weight * IForward::m_input;
    }

    inline void backward(void)
    {
        IBackward::m_input = m_output_node[0]->get_output();
        IBackward::m_output = m_weight * IBackward::m_input;

        // gradient = dE / dWi = - Sum( delta * f'(net) * Xi ),
        // where Sum is through all input samples.
        // get_backward_val here gets the f'(net) from next neuron.
        m_gradient -= ( IBackward::m_input
                    * m_output_node[0]->get_input()
                    * IForward::m_input );
    }

    inline void update( IN double delta_weight )
    {
        m_weight += delta_weight;
        m_gradient = 0.0;
    }

    inline double get_weight(void) const
    {
        return m_weight;
    }

    inline double get_gradient(void) const
    {
        return m_gradient;
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        std::cout << m_weight << '\t';
    }
#endif // _DEBUG

protected:

    double m_weight;
    double m_gradient;
};

} // namespace wwd

#endif // WEIGHT_ITF_HPP_

