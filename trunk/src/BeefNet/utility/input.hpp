#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "input_itf.hpp"

namespace wwd
{

template < uint32 OutputNum >
class CInput
    : public IInput<OutputNum>
{
private:

    typedef CInput<OutputNum> ThisType;

public:

    CInput(void)
        : IInput<OutputNum>()
    {
    }

    ~CInput(void)
    {
    }

    inline ThisType &operator=( IN double input )
    {
        m_forward_input = input;
        return *this;
    }

    inline void forward(void)
    {
        m_forward_output = m_forward_input;
    }
};

typedef CInput<0> CNullInput;

} // namespace wwd

#endif // INPUT_HPP_

