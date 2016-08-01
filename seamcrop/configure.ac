dnl required version of autoconf
AC_PREREQ([2.53])

dnl *********************************************************************** ADDED
AC_CONFIG_MACRO_DIR([m4])
dnl *********************************************************************** END

dnl 
AC_INIT([seamcrop],[1.0.0])

dnl required versions of gstreamer and plugins-base
GST_REQUIRED=1.0.0
GSTPB_REQUIRED=1.0.0

dnl *********************************************************************** ADDED
AC_CANONICAL_SYSTEM

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_LIBTOOL
AM_PROG_AS
AC_PROG_INSTALL
AC_PROG_LN_S
AC_LANG(C++)
AC_PATH_PROG(DOXYGEN, doxygen, "no")

dnl *********************************************************************** END

AC_CONFIG_SRCDIR([src/gstplugin.c])
dnl AC_CONFIG_HEADERS([config.h])

dnl required version of automake
AM_INIT_AUTOMAKE([1.10])

dnl *********************************************************************** ADD
AM_CONFIG_HEADER(src/seamcrop-module/include/MocaConfig.h)
dnl *********************************************************************** END

dnl enable mainainer mode by default
AM_MAINTAINER_MODE([enable])

dnl check for tools (compiler etc.)
AC_PROG_CC

dnl required version of libtool
LT_PREREQ([2.2.6])
LT_INIT

dnl give error and exit if we don't have pkgconfig
AC_CHECK_PROG(HAVE_PKGCONFIG, pkg-config, [ ], [
  AC_MSG_ERROR([You need to have pkg-config installed!])
])

dnl *********************************************************************** ADDED
# If cuda is requested to be enabled
AC_ARG_ENABLE(cuda, 
	AS_HELP_STRING([--enable-cuda=ARCH], [Enable cuda based modules for architecture ARCH (see nvcc option -arch).]),[

	# Search nvcc compiler
	AC_PATH_PROG(NVCC, nvcc, "no")
	AS_IF([test "x$NVCC" = "xno"],[
        	AC_MSG_ERROR([NVCC compiler not found!])
	])

        # Check nvcc version, should be 3.0
	AS_IF([nvcc --version | grep -q "release 3.0"],
        	[],
        	[AC_MSG_WARN([NVCC compiler version is NOT 3.0!])
	])

        # If $with_cuda is not empty, set to CUDA_ARCH to
        # supplied value, else set to value sm_11
	AS_IF([test "x$enableval" = "xyes"],[
	        CUDA_ARCH=" -arch=sm_11"
	],[
		CUDA_ARCH=" -arch=$enableval"
	])

        # Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
        # is substituted by "include".
        CUDA_CFLAGS=" -I${NVCC/'bin/nvcc'/include}"

        #Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
        #is substituted by "lib".
        CUDA_LIBS=" -L${NVCC/'bin/nvcc'/lib}"

        # If $build_cpu contains "_64", append "64" to CUDA_LIBS
	AS_IF([echo $build_cpu | grep -q "_64"],
		[CUDA_LIBS+="64"])

        # Append " -lcudart" to CUDA_LIBS
        CUDA_LIBS+=" -lcudart"

	# Symbolize that cuda is wanted
	with_cuda=$enableval

	# Make variables available in Makefile.am
	AC_SUBST(CUDA_CFLAGS)
	AC_SUBST(CUDA_LIBS)
	AC_SUBST(NVCC)
        AC_DEFINE([HAVE_CUDA], [1], [CUDA support enabled (manually set during configure).])
],AC_MSG_ERROR([CUDA arch (compute capability) must be specified: --enable-cuda=<arch> (e.g sm_20, sm_30 or similar).]))
# Set this conditional if cuda is wanted
AM_CONDITIONAL([WANT_CUDA], [test -n "$with_cuda"])

# Check whether to use device emulation mode for cuda (if no cuda capable gpu is available)
AC_ARG_ENABLE([emu],
	AS_HELP_STRING([--enable-emu], [Enable device emulation for cuda modules (nvcc version <= 3.0 only).]),
	[EMULATION=true],
	[EMULATION=false])

# Set up compilation flags for cuda compiler nvcc, if with_cuda is set
AS_IF([test -n "$with_cuda"],[

	# If debug flag is set apply debugging compilation flags, otherwise build compilation flags
	AS_IF([test "x$DEBUG" = "xtrue"],
		[NVCCFLAGS="-g"],
		[NVCCFLAGS="-O3"])
#[NVCCFLAGS="-g --compiler-options \"-fno-strict-aliasing -fno-inline -Wno-deprecated\""],
#[NVCCFLAGS="-O3 -use_fast_math --compiler-options \"-fno-strict-aliasing -fno-inline -Wno-deprecated\""])

	# Add architecture to flags
	NVCCFLAGS+=" $CUDA_ARCH"

	# If device emulation was set, add deviceemu flag
	AS_IF([test "x$EMULATION" = "xtrue"],
		[NVCCFLAGS+=" -deviceemu"])

])
# Make NVCCFLAGS available in Makefile.am
AC_SUBST(NVCCFLAGS)

# set flag to enable / disable target in top Makefile
AM_CONDITIONAL(MAKE_DOXYGEN, test x$DOXYGEN != xno)
AM_CONDITIONAL([BUILD_SHARED_LIB], [test x$enable_shared = xyes])

# Checks for header files.
AC_CHECK_HEADERS(limits.h fcntl.h stdlib.h sys/ioctl.h unistd.h)

# Check for CYGWIN
AC_MSG_CHECKING([for cygwin])
case $host_os in
  *cygwin* ) AC_MSG_RESULT([found])
	     AC_DEFINE(CYGWIN,true,[compiling with cygwin])
	     ;;
         * ) AC_MSG_RESULT([not found])
	     ;;
esac

# Check for mandatory libraries: gsl, opencv, ffmpeg, boost  
AC_CHECK_LIB([gslcblas], [cblas_sdsdot],,AC_MSG_ERROR([GSL CBLAS library not found.]))
AC_CHECK_LIB([gsl], [gsl_matrix_alloc],,AC_MSG_ERROR([GSL library not found.]))

AC_CHECK_HEADER(opencv2/opencv.hpp,,AC_MSG_ERROR([OpenCV library not found.]))

AC_CHECK_LIB([avcodec], [avcodec_register_all],,AC_MSG_ERROR([avcodec (ffmpeg) library not found.]))
AC_CHECK_LIB([avfilter], [avfilter_register_all],,AC_MSG_ERROR([avfilter (ffmpeg) library not found.]))
AC_CHECK_LIB([avdevice], [avdevice_register_all],,AC_MSG_ERROR([avdevice (ffmpeg) library not found.]))
AC_CHECK_LIB([avformat], [av_register_all],,AC_MSG_ERROR([avformat (ffmpeg) library not found.]))
AC_CHECK_LIB([avutil], [avutil_version],,AC_MSG_ERROR([avutil (ffmpeg) library not found.]))
AC_CHECK_LIB([postproc], [postproc_version],,AC_MSG_ERROR([avpostproc (ffmpeg) library not found.]))
AC_CHECK_LIB([swscale], [swscale_version],,AC_MSG_ERROR([avswscale (ffmpeg) library not found.]))


AC_MSG_CHECKING([for boost library])
AC_LANG_PUSH(C++)
	AC_COMPILE_IFELSE(
		[
		AC_LANG_PROGRAM([[#include <boost/version.hpp>]], [])
		],
		[
		AC_MSG_RESULT([found])
		AC_DEFINE(HAVE_BOOST,1,[Define to 1 if you have the `boost' library.])
		],
		[
		AC_MSG_RESULT([not found])
		AC_MSG_ERROR([BOOST library not found.])
		]
	)
AC_LANG_POP([C++])


# Check for other libraries
AC_CHECK_LIB([fltk], [main]) #fix
AC_CHECK_LIB([dc1394], [dc1394_new])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE


# check for platform specific stuff
AC_CHECK_SIZEOF(char, 1)
AC_CHECK_SIZEOF(unsigned char, 1)
AC_CHECK_SIZEOF(int, 4)
AC_CHECK_SIZEOF(unsigned int, 4)
AC_CHECK_SIZEOF(short int, 2)
AC_CHECK_SIZEOF(unsigned short int, 2)
AC_CHECK_SIZEOF(long int, 4)
AC_CHECK_SIZEOF(unsigned long int, 4)
AC_CHECK_SIZEOF(unsigned long long int, 8)
AC_CHECK_SIZEOF(long long int, 8)
AC_CHECK_SIZEOF(float, 4)
AC_CHECK_SIZEOF(double, 8)

dnl *********************************************************************** END


dnl Check for the required version of GStreamer core (and gst-plugins-base)
dnl This will export GST_CFLAGS and GST_LIBS variables for use in Makefile.am
dnl
dnl If you need libraries from gst-plugins-base here, also add:
dnl for libgstaudio-1.0: gstreamer-audio-1.0 >= $GST_REQUIRED
dnl for libgstvideo-1.0: gstreamer-video-1.0 >= $GST_REQUIRED
dnl for libgsttag-1.0: gstreamer-tag-1.0 >= $GST_REQUIRED
dnl for libgstpbutils-1.0: gstreamer-pbutils-1.0 >= $GST_REQUIRED
dnl for libgstfft-1.0: gstreamer-fft-1.0 >= $GST_REQUIRED
dnl for libgstinterfaces-1.0: gstreamer-interfaces-1.0 >= $GST_REQUIRED
dnl for libgstrtp-1.0: gstreamer-rtp-1.0 >= $GST_REQUIRED
dnl for libgstrtsp-1.0: gstreamer-rtsp-1.0 >= $GST_REQUIRED
dnl etc.
PKG_CHECK_MODULES(GST, [
  gstreamer-1.0 >= $GST_REQUIRED
  gstreamer-base-1.0 >= $GST_REQUIRED
  gstreamer-controller-1.0 >= $GST_REQUIRED
], [
  AC_SUBST(GST_CFLAGS)
  AC_SUBST(GST_LIBS)
], [
  AC_MSG_ERROR([
      You need to install or upgrade the GStreamer development
      packages on your system. On debian-based systems these are
      libgstreamer1.0-dev and libgstreamer-plugins-base1.0-dev.
      on RPM-based systems gstreamer1.0-devel, libgstreamer1.0-devel
      or similar. The minimum version required is $GST_REQUIRED.
  ])
])

dnl check if compiler understands -Wall (if yes, add -Wall to GST_CFLAGS)
AC_MSG_CHECKING([to see if compiler understands -Wall])
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -Wall"
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([ ], [ ])], [
  GST_CFLAGS="$GST_CFLAGS -Wall"
  AC_MSG_RESULT([yes])
], [
  AC_MSG_RESULT([no])
])

dnl set the plugindir where plugins should be installed (for src/Makefile.am)
if test "x${prefix}" = "x$HOME"; then
  plugindir="$HOME/.gstreamer-1.0/plugins"
else
  plugindir="\$(libdir)/gstreamer-1.0"
fi
AC_SUBST(plugindir)

dnl set proper LDFLAGS for plugins
GST_PLUGIN_LDFLAGS='-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\).*'
AC_SUBST(GST_PLUGIN_LDFLAGS)

AC_CONFIG_FILES([Makefile 
		src/Makefile
		src/seamcrop-module/Makefile])


AC_OUTPUT

