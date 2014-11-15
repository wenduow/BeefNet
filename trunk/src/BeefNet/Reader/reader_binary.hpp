#ifndef READER_BINARY_HPP_
#define READER_BINARY_HPP_

#undef UNICODE

#include <sys/stat.h>
#include <Windows.h>
#include "../utility/type.hpp"

namespace wwd
{

template < uint32 FeatureNum >
class CReaderBinary
{
public:

    enum
    {
        feature_num = FeatureNum
    };

public:

    CReaderBinary(void)
        : m_file_buf(NULL)
        , m_pattern_num(0)
    {
    }

    CReaderBinary( IN const char *path )
        : m_file_buf(NULL)
        , m_pattern_num(0)
    {
        open(path);
    }

    ~CReaderBinary(void)
    {
    }

    void open( IN const char *path )
    {
        close();

        struct _stat64 file_stat;
        _stat64( path, &file_stat );
        m_pattern_num = (uint32)( file_stat.st_size / sizeof(double) / feature_num );

        HANDLE h_file = CreateFile( path,
                                    GENERIC_READ,
                                    FILE_SHARE_READ,
                                    NULL,
                                    OPEN_EXISTING,
                                    FILE_ATTRIBUTE_READONLY,
                                    NULL );

        HANDLE h_mmap = CreateFileMapping( h_file,
                                           NULL,
                                           PAGE_READONLY,
                                           0,
                                           0,
                                           NULL );

        m_file_buf = (const double*)MapViewOfFile( h_mmap, FILE_MAP_READ, 0, 0, 0 );

        CloseHandle(h_mmap);
        CloseHandle(h_file);
    }

    void close(void)
    {
        if (m_file_buf)
        {
            UnmapViewOfFile(m_file_buf);
            m_file_buf    = NULL;
            m_pattern_num = 0;
        }
    }

    inline const double *get_pattern( IN uint32 pattern_idx ) const
    {
        return m_file_buf + pattern_idx * feature_num;
    }

    inline uint32 get_pattern_num(void) const
    {
        return m_pattern_num;
    }

private:

    const double *m_file_buf;
    uint32 m_pattern_num;
};

} // namespace wwd

#endif // READER_BINARY_HPP_

