# Note: plugindir is set in configure
SUBDIRS = seamcrop-module

plugin_LTLIBRARIES = libgstseamcrop.la

# sources used to compile this plug-in
libgstseamcrop_la_SOURCES = gstseamcrop.c gstseamcrop.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstseamcrop_la_CFLAGS = $(GST_CFLAGS)
libgstseamcrop_la_LIBADD = -L$(top_srcdir)/src/seamcrop-module/.libs -lseamcrop -L$(top_srcdir)/moca/.libs $(GST_LIBS) -lgstvideo-1.0
libgstseamcrop_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstseamcrop_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = gstseamcrop.h
