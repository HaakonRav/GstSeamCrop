include $(top_srcdir)/Makefile.inc

AM_CPPFLAGS += -I/usr/include/ffmpeg/ -I/usr/include/gstreamer-1.0 -I/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include

# Necessary for the linker.
NVCCFLAGS += --shared

lib_LTLIBRARIES = libseamcrop.la

libseamcrop_la_SOURCES = BufferWriter.cpp seamCropWrapper.cpp seamCropCuda.cpp seamCropCuda_Kernels.cu 
libseamcrop_la_LIBADD = -lgstreamer-1.0 -lgstbase-1.0 -lgobject-2.0 -lglib-2.0 -lgstvideo-1.0 -L$(top_srcdir)/moca/.libs -lmoca -L/usr/local/cuda/lib64 -lcudart -lcudadevrt
libseamcrop_la_LDFLAGS = $(AM_LDFLAGS) -export-symbols-regex [_]*\(gst_\|Gst\|GST_\).*

