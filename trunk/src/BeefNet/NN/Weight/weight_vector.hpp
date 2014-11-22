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
            hessian[i][i] *= ( 1.0 + Param::lambda );
        }

        // calculate J'e
        double gradient[InputNum];
        multiply( gradient, jacobian_transpose, m_err );

        // calculate ( J'J + lambda * diag(J'J) ) ^ -1
        double hessian_inverse[InputNum][InputNum];
        invert( hessian_inverse, hessian );

        // calculate ( J'J + lambda * diag(J'J) ) ^ -1 * (J'e)
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

    inline double get_jacobian_value( IN uint32 pattern_idx,
                                      IN uint32 output_idx,
                                      IN uint32 input_idx ) const
    {
        return m_jacobian[ pattern_idx * Param::output_num + output_idx ]
                         [input_idx];
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
    void invert( OUT double (&inv)[N][N],
                 IN const double (&matrix)[N][N] ) const
    {
        double det = determinant(matrix);

        for ( uint32 i = 0; i < N; ++i )
        {
            for ( uint32 j = 0; j < N; ++j )
            {
                double rest[ N - 1 ][ N - 1 ];
                uint32 idx_i = 0;

                for ( uint32 k = 0; k < N; ++k )
                {
                    if ( k != i )
                    {
                        uint32 idx_j = 0;

                        for ( uint32 l = 0; l < N; ++l )
                        {
                            if ( l != j )
                            {
                                rest[idx_i][idx_j] = matrix[k][l];

                                ++idx_j;
                            }
                        }

                        ++idx_i;
                    }
                }

                if ( ( i + j ) % 2 == 0 )
                {
                    inv[i][j] = determinant(rest) / det;
                }
                else
                {
                    inv[i][j] = - determinant(rest) / det;
                }
            }
        }
    }

    template <>
    void invert( OUT double (&inv)[1][1],
                 IN const double (&matrix)[1][1] ) const
    {
        inv[0][0] = 1.0 / matrix[0][0];
    }

    template < uint32 N >
    double determinant( IN const double (&matrix)[N][N] ) const
    {
        double ret = 0.0;

        for ( uint32 i = 0; i < N; ++i )
        {
            double rest[ N - 1 ][ N - 1 ];
            uint32 idx = 0;

            for ( uint32 j = 0; j < N; ++j )
            {
                if ( i != j )
                {
                    for ( uint32 k = 0; k < N - 1; ++k )
                    {
                        rest[idx][k] = matrix[j][k];
                    }

                    ++idx;
                }
            }

            if ( ( i + N ) % 2 == 1 )
            {
                ret += matrix[i][ N - 1 ] * determinant(rest);
            }
            else
            {
                ret -= matrix[i][ N - 1 ] * determinant(rest);
            }
        }

        return ret;
    }

    template <>
    double determinant( IN const double (&matrix)[1][1] ) const
    {
        return matrix[0][0];
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

