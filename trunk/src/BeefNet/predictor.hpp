#ifndef PREDICTOR_HPP_
#define PREDICTOR_HPP_

#include "utility/type.hpp"

namespace wwd
{

template < class NN, class InputReader >
class CPredictor
{
private:

    typedef CPredictor< NN, InputReader > ThisType;
public:

    CPredictor(void)
    {
    }

    ~CPredictor(void)
    {
    }

    template < uint32 OutputNum >
    void predict( OUT double           (&output)[OutputNum],
                  INOUT NN             &nn,
                  IN const InputReader &input,
                  IN uint32            idx ) const
    {
        nn.set_input( input.get_pattern(idx) );
        nn.forward();
        nn.get_output(output);
    }

private:

    CPredictor( IN const CPredictor &other );
    inline CPredictor &operator=( IN const CPredictor &other );
};

} // namespace wwd

#endif // PREDICTOR_HPP_

