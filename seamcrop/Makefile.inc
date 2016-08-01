# Set C++ compiler flags and includes
AM_CPPFLAGS = -Wall -fno-strict-aliasing -g -I$(top_srcdir)/include

# Set linking flags and libs
#AM_LDFLAGS = -L$(top_builddir)/src/ -L$(top_builddir)/src/seamcrop-module/ -lboost_system-mt -lboost_thread-mt -lboost_filesystem-mt -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib
AM_LDFLAGS = -L$(top_builddir)/src/ -L$(top_builddir)/src/seamcrop-module/ -lboost_system -lboost_thread -lboost_filesystem -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_contrib -lopencv_calib3d -lopencv_objdetect -lopencv_video

#AM_CXXFLAGS = -msse

if WANT_CUDA
AM_CPPFLAGS += $(CUDA_CFLAGS)
AM_LDFLAGS += $(CUDA_LIBS)
#NVCCFLAGS += --compiler-bindir=/usr/local/GNU/bin # use older version of GCC for compatibility with NVCC
NVCCFLAGS += --pre-include $(top_srcdir)/src/seamcrop-module/include/undef_atomics_int128.h # required when using the CUDA 5.0 toolkit for Fedora 16 on Fedora 18 with GCC 4.7

.cu.o:
	$(NVCC) $(NVCCFLAGS) --compiler-options="-D__STRICT_ANSI__ $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -fno-strict-aliasing -fno-inline -D WANT_CUDA" -o $@ -c $< 
.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) $(NVCCFLAGS) --compiler-options=\"-D__STRICT_ANSI__ $(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -fno-strict-aliasing -fno-inline\" -c $<
endif

