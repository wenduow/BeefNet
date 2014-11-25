#ifndef WEIGHT_VECTOR_HPP_
#define WEIGHT_VECTOR_HPP_

#include "weight_lm.hpp"

namespace wwd
{

template < uint32 InputNum,
           template <class> class WeightType,
           class Param >
class CWeightVector
{
private:

    typedef CWeightVector< InputNum, WeightType, Param > ThisType;

public:

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i] >> other.m_weight[i];
        }

        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i] << other.m_weight[i];
        }

        return *this;
    }

    CWeightVector(void)
    {
    }

    ~CWeightVector(void)
    {
    }

    void forward(void)
    {
        for ( auto &i : m_weight )
        {
            i.forward();
        }
    }

    void backward(void)
    {
        for ( auto &i : m_weight )
        {
            i.backward();
        }
    }

    void update(void)
    {
        for ( auto &i : m_weight )
        {
            i.update();
        }
    }

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            if ( i.connect_input_node(neuron) )
            {
                neuron.connect_output_node(i);
                break;
            }
        }
    }

    template < class Neuron >
    void connect_output_neuron( IN Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            i.connect_output_node(neuron);
            neuron.connect_input_node(i);
        }
    }

    double get_gradient_sum(void) const
    {
        double gradient_sum = 0.0;

        for ( const auto &i : m_weight )
        {
            gradient_sum += i.get_gradient_sum();
        }

        return gradient_sum;
    }

    uint32 get_gradient_num(void) const
    {
        uint32 gradient_num = 0;

        for ( const auto &i : m_weight )
        {
            gradient_num += i.get_gradient_num();
        }

        return gradient_num;
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

private:

    WeightType<Param> m_weight[InputNum];
};

template < uint32 InputNum, class Param >
class CWeightVector< InputNum, CWeightLM, Param >
{
private:

    typedef CWeightVector< InputNum, CWeightLM, Param > ThisType;

public:

    CWeightVector(void)
        : m_vector_idx(0)
    {
    }

    ~CWeightVector(void)
    {
    }

    const ThisType &operator>>( OUT ThisType &other ) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i] >> other.m_weight[i];
        }

        other.m_vector_idx = 0;
        return *this;
    }

    ThisType &operator<<( IN const ThisType &other )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i] << other.m_weight[i];
        }

        for ( uint32 i = 0; i < other.m_vector_idx; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                m_jacobian[m_vector_idx][j] = other.m_jacobian[i][j];
            }

            m_err[m_vector_idx] = other.m_err[i];
            ++m_vector_idx;
        }

        return *this;
    }

    void forward(void)
    {
        for ( auto &i : m_weight )
        {
            i.forward();
        }
    }

    void backward( IN double err )
    {
        for ( auto &i : m_weight )
        {
            i.backward();
        }

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_jacobian[m_vector_idx][i]
                = - m_weight[i].IPathBackward::get_input_value()
                  * m_weight[i].IPathForward::get_input_value();
        }

        m_err[m_vector_idx] = err;

        ++m_vector_idx;
    }

    void update(void)
    {
        // calculate J'J
        double jacobian_transpose[InputNum][ Param::pattern_num
                                           * Param::output_num ];
        transpose( jacobian_transpose, m_jacobian );

        double hessian[InputNum][InputNum];
        multiply( hessian, jacobian_transpose, m_jacobian );

        // calculate J'J + lambda * diag(J'J)
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            hessian[i][i] += Param::lambda;
        }

        // calculate J'e
        double gradient[InputNum];
        multiply( gradient, jacobian_transpose, m_err );

        // calculate ( J'J + lambda * I ) ^ -1
        double hessian_inverse[InputNum][InputNum];
        invert( hessian_inverse, hessian );

        // calculate ( J'J + lambda * I ) ^ -1 * (J'e)
        multiply( m_weight_update, hessian_inverse, gradient );

        // update each weight
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].update( - m_weight_update[i] );
        }

        m_vector_idx = 0;
    }

    void revert(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].update( m_weight_update[i] );
        }

        m_vector_idx = 0;
    }

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            if ( i.connect_input_node(neuron) )
            {
                neuron.connect_output_node(i);
                break;
            }
        }
    }

    template < class Neuron >
    void connect_output_neuron( IN Neuron &neuron )
    {
        for ( auto &i : m_weight )
        {
            i.connect_output_node(neuron);
            neuron.connect_input_node(i);
        }
    }

    double get_gradient_sum(void) const
    {
        // calculate J'e
        double jacobian_transpose[InputNum][ Param::pattern_num
                                           * Param::output_num ];
        transpose( jacobian_transpose, m_jacobian );

        double gradient[InputNum];
        multiply( gradient, jacobian_transpose, m_err );

        double ret = 0.0;

        for ( const auto &i : gradient )
        {
            ret += i;
        }

        return ret;
    }

    inline uint32 get_gradient_num(void) const
    {
        return InputNum * m_vector_idx;
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( auto &i : m_weight )
        {
            i.print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    template < uint32 R, uint32 C >
    void transpose( OUT double (&trans)[C][R],
                    IN const double (&matrix)[R][C] ) const
    {
        for ( uint32 i = 0; i < R; ++i )
        {
            for ( uint32 j = 0; j < C; ++j )
            {
                trans[j][i] = matrix[i][j];
            }
        }
    }

    template < uint32 R1, uint32 C, uint32 R2 >
    void multiply( OUT double (&product)[R1][R2],
                   IN const double (&multiplier)[R1][C],
                   IN const double (&multiplicand)[C][R2] ) const
    {
        for ( uint32 i = 0; i < R1; ++i )
        {
            for ( uint32 j = 0; j < R2; ++j )
            {
                product[i][j] = 0.0;

                for ( uint32 k = 0; k < C; ++k )
                {
                    product[i][j] += multiplier[i][k] * multiplicand[k][j];
                }
            }
        }
    }

    template < uint32 R1, uint32 C >
    void multiply( OUT double (&product)[R1],
                   IN const double (&multiplier)[R1][C],
                   IN const double (&multiplicand)[C] ) const
    {
        for ( uint32 i = 0; i < R1; ++i )
        {
            product[i] = 0.0;

            for ( uint32 j = 0; j < C; ++j )
            {
                product[i] += multiplier[i][j] * multiplicand[j];
            }
        }
    }

    template < uint32 N >
    bool invert( OUT double (&inverse)[N][N],
                 IN const double (&matrix)[N][N] ) const
    {
        double copy[N][N];

        for ( uint32 i = 0; i < N; ++i )
        {
            for ( uint32 j = 0; j < N; ++j )
            {
                copy[i][j] = matrix[i][j];
                inverse[i][j] = ( i == j ) ? 1.0 : 0.0;
            }
        }

        for ( uint32 i = 0; i < N; ++i )
        {
            // regularize the i-th element in i-th row to 1.0
            double rate = copy[i][i];

            for ( uint32 j = 0; j < N; ++j )
            {
                copy[i][j] /= rate;
                inverse[i][j] /= rate;
            }

            // eliminate the i-th element in other rows to 0.0
            for ( uint32 j = 0; j < N; ++j )
            {
                if ( j != i )
                {
                    double rate = copy[j][i];

                    for ( uint32 k = 0; k < N; ++k )
                    {
                        copy[j][k] -= rate * copy[i][k];
                        inverse[j][k] -= rate * inverse[i][k];
                    }
                }
            }
        }

        return true;
    }

private:

    CWeightLM<Param> m_weight[InputNum];

    double m_jacobian[ Param::pattern_num * Param::output_num ][InputNum];
    double m_err[ Param::pattern_num * Param::output_num ];
    double m_weight_update[InputNum];
    uint32 m_vector_idx;
};

} // namespace wwd

#endif // WEIGHT_VECTOR_HPP_

