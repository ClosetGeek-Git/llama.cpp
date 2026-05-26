dnl $Id$
dnl config.m4 for extension swoole_llama

PHP_ARG_ENABLE(swoole_llama, swoole_llama support,
[  --enable-swoole-llama   Enable swoole_llama support], [no])

PHP_ARG_WITH(llama_dir, llama.cpp directory,
[  --with-llama-dir[=DIR]  Specify llama.cpp directory (must be built)], [no], [no])

if test "$PHP_SWOOLE_LLAMA" != "no"; then

    dnl Check for llama.cpp directory
    if test "$PHP_LLAMA_DIR" = "no"; then
        AC_MSG_ERROR([Please specify llama.cpp directory with --with-llama-dir])
    fi

    dnl Validate directory exists
    if test ! -d "$PHP_LLAMA_DIR"; then
        AC_MSG_ERROR([llama.cpp directory not found: $PHP_LLAMA_DIR])
    fi

    dnl Set up paths
    LLAMA_SRC_DIR="$PHP_LLAMA_DIR"
    LLAMA_BUILD_DIR="$PHP_LLAMA_DIR/build"

    dnl Check for required llama.cpp libraries
    if test ! -f "$LLAMA_BUILD_DIR/bin/libllama.so"; then
        AC_MSG_ERROR([libllama.so not found in $LLAMA_BUILD_DIR/bin - please build llama.cpp first])
    fi
    
    if test ! -f "$LLAMA_BUILD_DIR/common/libcommon.a"; then
        AC_MSG_ERROR([libcommon.a not found in $LLAMA_BUILD_DIR/common - please build llama.cpp first])
    fi

    dnl Check for Swoole extension headers (installed via pecl or make install)
    if test ! -d "$phpincludedir/ext/swoole"; then
        AC_MSG_ERROR([Swoole extension headers not found at $phpincludedir/ext/swoole - please install Swoole extension first])
    fi

    dnl Clang detection for C standard compatibility
    AC_MSG_CHECKING([for clang])
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],[[
        #ifndef __clang__
            not clang
        #endif
    ]])],[
        AC_MSG_RESULT([yes])
        CLANG_FLAGS="-std=gnu11"
    ],[
        AC_MSG_RESULT([no])
        CLANG_FLAGS=""
    ])

    AC_DEFINE(HAVE_SWOOLE_LLAMA, 1, [Have swoole_llama support])

    dnl Compiler flags
    dnl CRITICAL: Swoole extension is compiled with HAVE_CONFIG_H which affects class layouts
    dnl We MUST match this to avoid ABI mismatch
    EXTRA_CXXFLAGS="-DHAVE_CONFIG_H"
    EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -DCOMPILE_DL_SWOOLE_LLAMA"
    EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -DCPPHTTPLIB_OPENSSL_SUPPORT"
    EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -std=c++17"
    EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -fno-stack-protector"
    EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -Wall -Wno-unused-function -Wno-deprecated -Wno-deprecated-declarations"
    EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS $CLANG_FLAGS"

    dnl Include paths - llama.cpp
    PHP_ADD_INCLUDE($ext_srcdir)
    PHP_ADD_INCLUDE($LLAMA_SRC_DIR)
    PHP_ADD_INCLUDE($LLAMA_SRC_DIR/common)
    PHP_ADD_INCLUDE($LLAMA_SRC_DIR/include)
    PHP_ADD_INCLUDE($LLAMA_SRC_DIR/ggml/include)
    PHP_ADD_INCLUDE($LLAMA_SRC_DIR/vendor)
    PHP_ADD_INCLUDE($LLAMA_SRC_DIR/tools/mtmd)
    PHP_ADD_INCLUDE($LLAMA_BUILD_DIR)

    dnl Include paths - Swoole (installed extension headers)
    PHP_ADD_INCLUDE($phpincludedir/ext/swoole)
    PHP_ADD_INCLUDE($phpincludedir/ext/swoole/include)

    dnl Library paths
    PHP_ADD_LIBPATH($LLAMA_BUILD_DIR/bin, SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_ADD_LIBPATH($LLAMA_BUILD_DIR/common, SWOOLE_LLAMA_SHARED_LIBADD)

    dnl Link libraries - llama.cpp core only
    dnl GPU/CPU backends are loaded dynamically at runtime via ggml_backend_load_all()
    PHP_ADD_LIBRARY(llama, 1, SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_ADD_LIBRARY(ggml, 1, SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_ADD_LIBRARY(ggml-base, 1, SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_ADD_LIBRARY(mtmd, 1, SWOOLE_LLAMA_SHARED_LIBADD)

    dnl System libraries
    PHP_ADD_LIBRARY(ssl, 1, SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_ADD_LIBRARY(crypto, 1, SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_ADD_LIBRARY(pthread, 1, SWOOLE_LLAMA_SHARED_LIBADD)

    dnl Link static library for common (compiled with -fPIC in llama.cpp build)
    LDFLAGS="$LDFLAGS $LLAMA_BUILD_DIR/common/libcommon.a"

    dnl Source files - server-coro sources
    swoole_llama_sources="coro-extension.cpp"
    swoole_llama_sources="$swoole_llama_sources server-task.cpp"
    swoole_llama_sources="$swoole_llama_sources server-queue.cpp"
    swoole_llama_sources="$swoole_llama_sources server-common.cpp"
    swoole_llama_sources="$swoole_llama_sources server-context.cpp"

    PHP_SUBST(SWOOLE_LLAMA_SHARED_LIBADD)
    PHP_NEW_EXTENSION(swoole_llama, $swoole_llama_sources, $ext_shared,, $EXTRA_CXXFLAGS, cxx)
    
    dnl Declare runtime dependency on Swoole extension
    PHP_ADD_EXTENSION_DEP(swoole_llama, swoole)

    PHP_REQUIRE_CXX()
fi
