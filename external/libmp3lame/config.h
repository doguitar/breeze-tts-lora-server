/* Minimal build config for vendored libmp3lame 3.100 (encoder-only).
 * Based on upstream configMS.h, adapted for both MSVC and other toolchains. */
#ifndef LAME_CONFIG_H_INCLUDED
#define LAME_CONFIG_H_INCLUDED

#define SIZEOF_DOUBLE 8
#define SIZEOF_FLOAT 4
#define SIZEOF_INT 4
#define SIZEOF_LONG 4
#define SIZEOF_LONG_DOUBLE 8
#define SIZEOF_SHORT 2
#define SIZEOF_UNSIGNED_INT 4
#define SIZEOF_UNSIGNED_LONG 4
#define SIZEOF_UNSIGNED_SHORT 2
#define SIZEOF_LONG_LONG 8
#define SIZEOF_UNSIGNED_LONG_LONG 8

#define STDC_HEADERS 1
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_LIMITS_H 1
#define HAVE_STRCHR 1
#define HAVE_MEMCPY 1
#define HAVE_STRING_H 1
#define HAVE_STDLIB_H 1

#define PACKAGE "lame"
#define VERSION "3.100"
#define PROTOTYPES 1
#define USE_FAST_LOG 1
#define LAME_LIBRARY_BUILD 1

/* Encoder-only: do not pull in mpglib. */
#undef HAVE_MPGLIB

#if defined(_MSC_VER)
#  include <stdint.h>
#  define inline __inline
#elif defined(__GNUC__) || defined(__clang__)
#  include <stdint.h>
#else
#  include <stdint.h>
#endif

typedef long double ieee854_float80_t;
typedef double      ieee754_float64_t;
typedef float       ieee754_float32_t;

#endif /* LAME_CONFIG_H_INCLUDED */
