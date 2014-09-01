#ifndef TYPE_HPP_
#define TYPE_HPP_

#define IN
#define OUT
#define INOUT

#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ( (void*)0 )
#endif
#endif

#define INVALID_INT_64 ( (uint64)~0 )
#define INVALID_INT_32 ( (uint32)~0 )
#define INVALID_INT_16 ( (uint16)~0 )
#define INVALID_INT_8  ( (uint8)~0 )

#define DOUBLE_MAX     1e12
#define DOUBLE_MIN     -DOUBLE_MAX
#define DOUBLE_EPSILON 1e-12

namespace wwd
{

#ifdef __linux__

    typedef int64_t  int64;
    typedef int32_t  int32;
    typedef int16_t  int16;
    typedef int8_t   int8;

    typedef uint64_t uint64;
    typedef uint32_t uint32;
    typedef uint16_t uint16;
    typedef uint8_t  uint8;

#elif ( defined _WIN32 )

    typedef __int64          int64;
    typedef __int32          int32;
    typedef __int16          int16;
    typedef __int8           int8;

    typedef unsigned __int64 uint64;
    typedef unsigned __int32 uint32;
    typedef unsigned __int16 uint16;
    typedef unsigned __int8  uint8;

#else
    #error "Any type of OS should be specified!"
#endif // system compiler macro

} // namespace wwd

#endif // TYPE_HPP_

