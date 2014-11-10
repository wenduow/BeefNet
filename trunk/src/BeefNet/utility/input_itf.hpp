#ifndef INPUT_ITF_HPP_
#define INPUT_ITF_HPP_

#include "forward_itf.hpp"

namespace wwd
{

class IInput
    : public IForward
{
protected:

    IInput(void)
        : IForward()
    {
    }

    ~IInput(void)
    {
    }
};

} // namespace wwd

#endif // INPUT_ITF_HPP_

