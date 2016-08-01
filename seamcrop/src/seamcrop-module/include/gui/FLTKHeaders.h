#ifndef GUI_FLTK_HEADERS_H
#define GUI_FLTK_HEADERS_H

// MOCA headers 
#include "types/MocaTypes.h"

#ifdef HAVE_LIBFLTK

#ifdef WIN32
#pragma warning(push)               // stupid FLTK for windows...
#pragma warning(disable:4312 4311)  // disable the windows compiler warnings for FLTK only
#endif


#include <FL/fl_draw.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Double_Window.H>

#ifdef WIN32
#pragma warning(pop)
#endif

#endif // HAVE_LIBFLTK

#endif
