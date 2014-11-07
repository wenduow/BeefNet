#ifndef NN_ITF_HPP_
#define NN_ITF_HPP_

#include <iostream>
#include "../utility/input.hpp"
#include "../utility/neuron.hpp"
#include "../utility/target.hpp"

namespace wwd
{

class INN
{
protected:

    INN(void)
    {
    }

    ~INN(void)
    {
    }

    template < class WeightBiasOther,
               class WeightBiasThis,
                class WeightInputOther,
               class WeightInputThis,
               uint32 InputNum,
               uint32 NeuronNum >
    void map_to( OUT WeightBiasOther
                     (&weight_bias_other)[NeuronNum],
                 IN const WeightBiasThis
                     (&weight_bias_this)[NeuronNum],
                 OUT WeightInputOther
                     (&weight_input_other)[NeuronNum][InputNum],
                 IN const WeightInputThis
                     (&weight_input_this)[NeuronNum][InputNum] ) const
    {
        // TODO: Explain why not use the overloaded function.
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input_this[i][j] >> weight_input_other[i][j];
            }

            weight_bias_this[i] >> weight_bias_other[i];
        }
    }

    template < class WeightOther,
               class WeightThis,
               uint32 InputNum,
               uint32 NeuronNum >
    void map_to( OUT WeightOther
                     (&weight_other)[NeuronNum][InputNum],
                 IN const WeightThis
                     (&weight_this)[NeuronNum][InputNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_this[i][j] >> weight_other[i][j];
            }
        }
    }

    template < class WeightBiasThis,
               class WeightBiasOther,
               class WeightInputThis,
               class WeightInputOther,
               uint32 InputNum,
               uint32 NeuronNum >
    void reduce_from( INOUT WeightBiasThis
                          (&weight_bias_this)[NeuronNum],
                      IN const WeightBiasOther
                          (&weight_bias_other)[NeuronNum],
                      INOUT WeightInputThis
                          (&weight_input_this)[NeuronNum][InputNum],
                      IN const WeightInputOther
                          (&weight_input_other)[NeuronNum][InputNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input_this[i][j] << weight_input_other[i][j];
            }

            weight_bias_this[i] << weight_bias_other[i];
        }
    }

    template < class WeightThis,
               class WeightOther,
               uint32 InputNum,
               uint32 NeuronNum >
    void reduce_from( INOUT WeightThis
                          (&weight_this)[NeuronNum][InputNum],
                      IN const WeightOther
                          (&weight_other)[NeuronNum][InputNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_this[i][j] << weight_other[i][j];
            }
        }
    }

    template < class Bias,
               class WeightBias,
               class Input,               
               class WeightInput,
               class Neuron,
               uint32 InputNum,
               uint32 NeuronNum >
    void connect( INOUT Bias        &bias,
                  INOUT WeightBias  (&weight_bias)[NeuronNum],
                  INOUT Input       (&input)[InputNum],
                  INOUT WeightInput (&weight_input)[NeuronNum][InputNum],
                  INOUT Neuron      (&neuron)[NeuronNum] ) const
    {
        bias = 1.0;

        // TODO: Explain why not use the overloaded function.
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input[i][j].connect_input( input[j] );
                neuron[i].connect_input( weight_input[i][j] );
            }

            weight_bias[i].connect_input(bias);
            neuron[i].connect_input( weight_bias[i] );
        }
    }

    template < class Input,
               class Weight,
               class Neuron,
               uint32 InputNum,
               uint32 NeuronNum >
    void connect( INOUT Input  (&input)[InputNum],
                  INOUT Weight (&weight)[NeuronNum][InputNum],
                  INOUT Neuron (&neuron)[NeuronNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight[i][j].connect_input( input[j] );
                neuron[i].connect_input( weight[i][j] );
            }
        }
    }

    template < class Output, class Target, uint32 OutputNum >
    void connect( INOUT Output (&output)[OutputNum],
                  INOUT Target (&target)[OutputNum] ) const
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            target[i].connect_input( output[i] );
        }
    }

    template < class Input >
    void forward( IN Input &input ) const
    {
        for ( auto &i : input )
        {
            i.forward();
        }
    }

    template < class Bias,
               class WeightBias ,
               class WeightInput,
               class Neuron,
               uint32 InputNum,
               uint32 NeuronNum >
    void forward( INOUT Bias        &bias,
                  INOUT WeightBias  (&weight_bias)[NeuronNum],
                  INOUT WeightInput (&weight_input)[NeuronNum][InputNum],
                  INOUT Neuron      (&neuron)[NeuronNum] ) const
    {
        bias.forward();

        // TODO: Explain forward order.
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            weight_bias[i].forward();

            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input[i][j].forward();
            }

            neuron[i].forward();
        }
    }

    template < class Weight, class Neuron, uint32 InputNum, uint32 NeuronNum >
    void forward( INOUT Weight (&weight)[NeuronNum][InputNum],
                  INOUT Neuron (&neuron)[NeuronNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight[i][j].forward();
            }

            neuron[i].forward();
        }
    }

    template < class Weight, uint32 InputNum, uint32 NeuronNum >
    void forward( INOUT Weight (&weight)[NeuronNum][InputNum] ) const
    {
        for ( auto &i : weight )
        {
            for ( auto &j : i )
            {
                j.forward();
            }
        }
    }

    template < class Target, uint32 TargetNum >
    void backward( IN Target (&target)[TargetNum] ) const
    {
        for ( auto &i : target )
        {
            i.backward();
        }
    }

    template < class WeightBias,
               class WeightInput,
               class Neuron,
               uint32 InputNum,
               uint32 NeuronNum >
    void backward( INOUT WeightBias  (&weight_bias)[NeuronNum],
                   INOUT WeightInput (&weight_input)[NeuronNum][InputNum],
                   INOUT Neuron      (&neuron)[NeuronNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            neuron[i].backward();

            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input[i][j].backward();
            }

            weight_bias[i].backward();
        }
    }

    template < class Weight,
               class Neuron,
               uint32 InputNum,
               uint32 NeuronNum >
    void backward( INOUT Weight (&weight_input)[NeuronNum][InputNum],
                   INOUT Neuron (&neuron)[NeuronNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            neuron[i].backward();

            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input[i][j].backward();
            }
        }
    }

    template < class Weight, uint32 InputNum, uint32 NeuronNum >
    void backward( INOUT Weight (&weight)[NeuronNum][InputNum] ) const
    {
        for ( auto &i : weight )
        {
            for ( auto &j : i )
            {
                j.backward();
            }
        }
    }

    template < class WeightBias,
               class WeightInput,
               uint32 InputNum,
               uint32 NeuronNum >
    void update( INOUT WeightBias  (&weight_bias)[NeuronNum],
                 INOUT WeightInput (&weight_input)[NeuronNum][InputNum] ) const
    {
        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            weight_bias[i].update();

            for ( uint32 j = 0; j < InputNum; ++j )
            {
                weight_input[i][j].update();
            }
        }
    }

    template < class Weight, uint32 InputNum, uint32 NeuronNum >
    void update( INOUT Weight (&weight)[NeuronNum][InputNum] ) const
    {
        for ( auto &i : weight )
        {
            for ( auto &j : i )
            {
                j.update();
            }
        }
    }

    template < class Input, uint32 InputNum >
    void set_input( OUT Input       (&input_node)[InputNum],
                    IN const double *input_val ) const
    {
        for ( uint32 i = 0; i < InputNum; ++i )
        {
            input_node[i] = input_val[i];
        }
    }

    template < class Target, uint32 TargetNum >
    void set_target( OUT Target      (&target_node)[TargetNum],
                     IN const double *target_val ) const
    {
        for ( uint32 i = 0; i < TargetNum; ++i )
        {
            target_node[i] = target_val[i];
        }
    }

    template < class Output, uint32 OutputNum >
    void get_output( OUT double (&output_val)[OutputNum],
                     IN const Output (&output_node)[OutputNum] ) const
    {
        for ( uint32 i = 0; i < OutputNum; ++i )
        {
            output_val[i] = output_node[i].get_forward_output();
        }
    }

    template < class WeightBias,
               class WeightInput,
               uint32 InputNum,
               uint32 NeuronNum >
    double get_gradient_abs_sum
        ( IN const WeightBias  (&weight_bias)[NeuronNum],
          IN const WeightInput (&weight_input)[NeuronNum][InputNum] ) const
    {
        double gradient = 0.0;

        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            gradient += abs( weight_bias[i].get_gradient() );

            for ( uint32 j = 0; j < InputNum; ++j )
            {
                gradient += abs( weight_input[i][j].get_gradient() );
            }
        }

        return gradient;
    }

    template < class Weight, uint32 InputNum, uint32 NeuronNum >
    double get_gradient_abs_sum
        ( IN const Weight (&weight)[NeuronNum][InputNum] ) const
    {
        double gradient = 0.0;

        for ( const auto &i : weight )
        {
            for ( const auto &j : i )
            {
                gradient += j.get_gradient();
            }
        }

        return gradient;
    }

    template < class WeightBias,
               class WeightInput,
               uint32 InputNum,
               uint32 NeuronNum >
    inline uint32 get_weight_num
        ( IN const WeightBias  (&weight_bias)[NeuronNum],
          IN const WeightInput (&weight_input)[NeuronNum][InputNum] ) const
    {
        (void)weight_bias;
        (void)weight_input;

        return NeuronNum + NeuronNum * InputNum;
    }

    template < class Weight, uint32 InputNum, uint32 NeuronNum >
    inline uint32 get_weight_num
        ( IN const Weight (&weight)[NeuronNum][InputNum] ) const
    {
        (void)weight;

        return NeuronNum * InputNum ;
    }

#if ( defined _DEBUG || defined PRINT_WEIGHT )

    template < class WeightBias,
               class WeightInput,
               class Neuron,
               uint32 InputNum,
               uint32 NeuronNum >
    void print( IN const WeightBias  (&weight_bias)[NeuronNum],
                IN const WeightInput (&weight_input)[NeuronNum][InputNum],
                IN const Neuron      (&neuron)[NeuronNum] ) const
    {
        std::cout << "bias and weight:" << std::endl;

        for ( uint32 i = 0; i < NeuronNum; ++i )
        {
            std::cout << weight_bias[i].get_weight() << ' ';

            for ( const auto &j : weight_input[i] )
            {
                std::cout << j.get_weight() << ' ';
            }

            std::cout << std::endl;
        }

        std::cout << "----------------" << std::endl;
        std::cout << "neuron:" << std::endl;

        for ( const auto &i : neuron )
        {
            std::cout << i.get_forward_output() << std::endl;
        }

        std::cout << "----------------" << std::endl;
    }

    template < class Weight, class Neuron, uint32 InputNum, uint32 NeuronNum >
    void print( IN const Weight (&weight)[NeuronNum][InputNum],
                IN const Neuron (&neuron)[NeuronNum] ) const
    {
        print(weight);

        std::cout << "neuron:" << std::endl;

        for ( const auto &i : neuron )
        {
            std::cout << i.get_forward_output() << std::endl;
        }

        std::cout << "----------------" << std::endl;
    }

    template < class Weight, uint32 InputNum, uint32 NeuronNum >
    void print( IN const Weight (&weight)[NeuronNum][InputNum] ) const
    {
        std::cout << "weight:" << std::endl;

        for ( const auto &i : weight )
        {
            for ( const auto &j : i )
            {
                std::cout << j.get_weight() << ' ';
            }

            std::cout << std::endl;
        }

        std::cout << "----------------" << std::endl;
    }

    template < class Input, uint32 InputNum >
    void print_input( IN const Input (&input)[InputNum] ) const
    {
        std::cout << "input:" << std::endl;

        for ( const auto &i : input )
        {
            std::cout << i.get_forward_input() << std::endl;
        }

        std::cout << "----------------" << std::endl;
    }

    template < class Target, uint32 TargetNum >
    void print_target( IN const Target (&target)[TargetNum] ) const
    {
        std::cout << "target:" << std::endl;

        for ( const auto &i : target )
        {
            std::cout << i.get_backward_input() << std::endl;
        }

        std::cout << "----------------" << std::endl;
    }

#endif // _DEBUG || PRINT_WEIGHT

    template < class NN >
    void save( IN const NN &nn, IN const char *path ) const
    {
        NN nn_disconnected;
        nn >> nn_disconnected;

        std::ofstream save_file( path, std::ios::binary );
        save_file.write( (const char*)&nn_disconnected,
                         sizeof(nn_disconnected) );
        save_file.close();
    }

    template < class NN >
    void load( OUT NN &nn, IN const char *path ) const
    {
        NN nn_disconnected;

        std::ifstream load_file( path, std::ios::binary );
        load_file.read( (char*)&nn_disconnected, sizeof(nn_disconnected) );
        load_file.close();

        nn_disconnected >> nn;
    }

private:

    INN( IN const INN &other );
    inline INN &operator=( IN const INN &other );
};

} // namespace wwd

#endif // NN_ITF_HPP_

