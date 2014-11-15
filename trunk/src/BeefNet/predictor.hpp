#ifndef PREDICTOR_HPP_
#define PREDICTOR_HPP_

#include "Utility/type.hpp"

namespace wwd
{

class CPredictor
{
public:

    CPredictor(void)
    {
    }

    ~CPredictor(void)
    {
    }

    template < template <uint32> class Reader, class NN >
    void predict( OUT double (&output)[ NN::output_num ],
                  INOUT NN &nn,
                  IN const char *input_path,
                  IN uint32 idx ) const
    {
        Reader< NN::input_num > input(input_path);

        nn.set_input( input.get_pattern(idx) );
        nn.forward();
        nn.get_output(output);

        input.close();
    }

private:

    CPredictor( IN const CPredictor &other );
    inline CPredictor &operator=( IN const CPredictor &other );
};

} // namespace wwd

#endif // PREDICTOR_HPP_

