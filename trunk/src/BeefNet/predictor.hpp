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
        m_input.close();
    }

    template < uint32 OutputNum >
    void predict( OUT double (&output)[OutputNum],
                  INOUT NN &nn,
                  IN uint32 idx ) const
    {
        nn.set_input( m_input.get_pattern(idx) );
        nn.forward();
        nn.get_output(output);
    }

    void open_input( IN const char *path )
    {
        m_input.open(path);
    }

private:

    CPredictor( IN const CPredictor &other );
    inline CPredictor &operator=( IN const CPredictor &other );

private:

    InputReader m_input;
};

} // namespace wwd

#endif // PREDICTOR_HPP_

