#ifndef OUTPUT_ITF_HPP_
#define OUTPUT_ITF_HPP_

#include "backward_itf.hpp"

namespace wwd
{

class IOutput
    : public IBackward
{
protected:

    IOutput(void)
        : IBackward()
    {
    }

    ~IOutput(void)
    {
    }
};

} // namespace wwd

#endif // OUTPUT_ITF_HPP_

