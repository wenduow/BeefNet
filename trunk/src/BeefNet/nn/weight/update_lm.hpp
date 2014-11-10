#ifndef UPDATE_LM_HPP_
#define UPDATE_LM_HPP_

#include <fstream>
#include <vector>
#include "weight_vector.hpp"

namespace wwd
{

template < uint32 InputNum, class Param >
class CUpdateLM
    : public CWeightVector<InputNum>
{
public:

    CUpdateLM(void)
        : m_lambda( (double)Param::lambda / 1000.0 )
        , m_square_err_prev(DOUBLE_MAX)
        , m_square_err(0.0)
    {
    }

    ~CUpdateLM(void)
    {
    }

    const CUpdateLM &operator>>( OUT CUpdateLM &other ) const
    {
        other.m_jacobian_matrix.clear();
        other.m_err.clear();

        other.m_square_err = 0.0;
        return *this;
    }

    CUpdateLM &operator<<( IN const CUpdateLM &other )
    {
        m_jacobian_matrix[i].insert( m_jacobian_matrix.end(),
                                     other.m_jacobian_matrix );
        m_err.insert( m_err.end(), other.m_err );

        m_square_err += other.m_square_err;
        return *this;
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
        std::vector<double> jacobian_vector;

        for ( auto &i : m_weight )
        {
            i.backward();
            jacobian_vector.push_back( m_output->get_input()
                                     * i.IForward::get_input() );
        }

        m_jacobian_matrix.push_back(jacobian_vector);
        m_err.push_back( m_output->get_output() );

        m_square_err += pow( m_output->get_output(), 2 );
    }

    void update(void)
    {
        // calculate J' * J
        double hessian[InputNum][InputNum];

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                hessian[i][j] = 0.0;

                for ( size_t k = 0; k < m_jacobian_matrix.size(); ++k )
                {
                    hessian[i][j] += m_jacobian_matrix[k][i]
                                   * m_jacobian_matrix[k][j];
                }
            }
        }

        // calculate J' * J + lambda * diag(J' * J)
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            hessian[i][i] *= ( 1.0 + m_lambda );
        }

        // calculate J' * delta
        double gradient[InputNum];

        for ( uint32 i = 0; i < InputNum; ++i )
        {
            gradient[i] = 0.0;

            for ( size_t j = 0; j < m_jacobian_matrix.size(); ++j )
            {
                gradient[i] += m_jacobian_matrix[j][i] * m_err[j];
            }
        }

        // calculate ( J' * J + lambda * diag(J' * J) ) ^ -1
        invert(hessian);

        // calculate ( J' * J + lambda * diag(J' * J) ) ^ -1 * ( J' * delta )
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            double delta_weight = 0.0;

            for ( uint32 j = 0; j < InputNum; ++j )
            {
                delta_weight += hessian[i][j] * gradient[j];
            }

            m_weight[i].update(delta_weight);
        }

        m_jacobian_matrix.clear();
        m_err.clear();

        m_lambda *= ( m_square_err > m_square_err_prev
                  ? (double)Param::beta 
                  : ( 1.0 / (double)Param::beta ) );
        m_square_err_prev = m_square_err;
        m_square_err = 0.0;
    }

private:

    CUpdateLM( IN CUpdateLM &other );
    inline CUpdateLM &operator=( IN const CUpdateLM &other );

    template < uint32 N >
    void invert( INOUT double (&matrix)[N][N] )
    {
        double det = determinant(matrix);
        double cofactor[N][N];

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
                    cofactor[i][j] = determinant(rest);
                }
                else
                {
                    cofactor[i][j] = - determinant(rest);
                }
            }
        }

        for ( uint32 i = 0; i < N; ++i )
        {
            for ( uint32 j = 0; j < N; ++j )
            {
                matrix[i][j] = cofactor[j][i] / det;
            }
        }
    }

    template <>
    void invert( INOUT double (&matrix)[1][1] )
    {
        matrix[0][0] = 1.0 / matrix[0][0];
    }

    template < uint32 N >
    double determinant( IN const double (&matrix)[N][N] )
    {
        double ret = 0.0;

        for ( uint32 i = 0; i < N; ++i )
        {
            double next[ N - 1 ][ N - 1 ];
            uint32 idx = 0;

            for ( uint32 j = 0; j < N; ++j )
            {
                if ( i != j )
                {
                    for ( uint32 k = 0; k < N - 1; ++k )
                    {
                        next[idx][k] = matrix[j][k];
                    }

                    ++idx;
                }
            }

            if ( ( i + N + 1 ) % 2 == 0 )
            {
                ret += matrix[i][ N - 1 ] * determinant(next);
            }
            else
            {
                ret -= matrix[i][ N - 1 ] * determinant(next);
            }
        }

        return ret;
    }

    template <>
    double determinant( IN const double (&matrix)[1][1] )
    {
        return matrix[0][0];
    }

private:

    std::vector< std::vector<double> > m_jacobian_matrix;
    std::vector<double> m_err;

    double m_square_err_prev;
    double m_square_err;

    double m_lambda;
};

} // namespace wwd

#endif // UPDATE_LM_HPP_

