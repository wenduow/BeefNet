#ifndef WEIGHT_VECTOR_LM_HPP_
#define WEIGHT_VECTOR_LM_HPP_

#include "weight_vector.hpp"
#include "weight_lm.hpp"

namespace wwd
{

template < uint32 InputNum, class Param >
class CWeightVector< InputNum, CWeightLM, Param >
{
private:

    typedef CWeightVector< InputNum, CWeightLM, Param > ThisType;

    typedef CWeightLM<Param> Weight;

public:

    CWeightVector(void)
        : m_vector_idx(0)
    {
        m_weight = new Weight[InputNum];

        m_jacobian = new double *[ Param::pattern_num * Param::output_num ];
        for ( uint32 i = 0; i < Param::pattern_num * Param::output_num; ++i )
        {
            m_jacobian[i] = new double[InputNum];
        }

        m_jacobian_transpose = new double*[InputNum];
        m_hessian = new double*[InputNum];
        m_hessian_inverse = new double*[InputNum];

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_jacobian_transpose[i] = new double[ Param::pattern_num
                                                * Param::output_num ];
            m_hessian[i] = new double[InputNum];
            m_hessian_inverse[i] = new double[InputNum];
        }

        m_err = new double[ Param::pattern_num * Param::output_num ];
        m_gradient = new double[InputNum];
        m_weight_update = new double[InputNum];
    }

    ~CWeightVector(void)
    {
        delete[] m_weight;
        m_weight = NULL;

        for ( uint32 i = 0; i < Param::pattern_num * Param::output_num; ++i )
        {
            delete[] m_jacobian[i];
            m_jacobian[i] = NULL;
        }
        delete[] m_jacobian;
        m_jacobian = NULL;

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            delete[] m_jacobian_transpose[i];
            m_jacobian_transpose[i] = NULL;
            delete[] m_hessian[i];
            m_hessian[i] = NULL;
            delete[] m_hessian_inverse[i];
            m_hessian_inverse[i] = NULL;
        }

        delete[] m_jacobian_transpose;
        m_jacobian_transpose = NULL;
        delete[] m_hessian;
        m_hessian = NULL;
        delete[] m_hessian_inverse;
        m_hessian_inverse = NULL;

        delete[] m_err;
        m_err = NULL;
        delete[] m_gradient;
        m_gradient = NULL;
        delete[] m_weight_update;
        m_weight_update = NULL;
    }

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

    void init(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].init();
        }

        m_vector_idx = 0;
    }

    void forward(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].forward();
        }
    }

    void backward( IN double err )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].backward();
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
        transpose< Param::pattern_num * Param::output_num, InputNum >
            ( m_jacobian_transpose, m_jacobian );

        multiply< InputNum, Param::pattern_num * Param::output_num, InputNum >
            ( m_hessian, m_jacobian_transpose, m_jacobian );

        // calculate J'J + lambda * diag(J'J)
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_hessian[i][i] += Param::lambda;
        }

        // calculate J'e
        multiply< InputNum, Param::pattern_num * Param::output_num >
            ( m_gradient, m_jacobian_transpose, m_err );

        // calculate ( J'J + lambda * I ) ^ -1
        invert<InputNum>( m_hessian_inverse, m_hessian );

        // calculate ( J'J + lambda * I ) ^ -1 * (J'e)
        multiply< InputNum, InputNum >
            ( m_weight_update, m_hessian_inverse, m_gradient );

        // update each weight
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].update( - m_weight_update[i] );
        }
    }

    void revert(void)
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].update( m_weight_update[i] );
        }
    }

    template < class Neuron >
    void connect_input_neuron( INOUT Neuron &neuron )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            if ( m_weight[i].connect_input_node(neuron) )
            {
                neuron.connect_output_node( m_weight[i] );
                break;
            }
        }
    }

    inline Weight &get_weight( IN uint32 idx )
    {
        return m_weight[idx];
    }

    double get_gradient_sum(void) const
    {
        // calculate J'e
        double ret = 0.0;

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            ret += m_gradient[i];
        }

        return ret;
    }

    inline uint32 get_gradient_num(void) const
    {
        return InputNum * m_vector_idx;
    }

    template < class STREAM >
    void save( OUT STREAM &stream ) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].save(stream);
        }
    }

    template < class STREAM >
    void load( INOUT STREAM &stream )
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].load(stream);
        }
    }

#ifdef _DEBUG
    void print_weight(void) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            m_weight[i].print_weight();
        }

        std::cout << std::endl;
    }
#endif // _DEBUG

private:

    template < uint32 R, uint32 C >
    void transpose( OUT double **trans, IN double **matrix ) const
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
    void multiply( OUT double **product,
                   IN double **multiplier,
                   IN double **multiplicand ) const
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

    template < uint32 R, uint32 C >
    void multiply( OUT double *product,
                   IN double **multiplier,
                   IN double *multiplicand ) const
    {
        for ( uint32 i = 0; i < R; ++i )
        {
            product[i] = 0.0;

            for ( uint32 j = 0; j < C; ++j )
            {
                product[i] += multiplier[i][j] * multiplicand[j];
            }
        }
    }

    template < uint32 N >
    void invert( OUT double **inverse, IN double **matrix ) const
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
    }

private:

    Weight *m_weight;

    double **m_jacobian;
    double **m_jacobian_transpose;
    double **m_hessian;
    double **m_hessian_inverse;

    double *m_err;
    double *m_gradient;
    double *m_weight_update;

    uint32 m_vector_idx;
};

} // namespace wwd

#endif // WEIGHT_VECTOR_LM_HPP_

