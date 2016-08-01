#ifndef GUI_GUIUTILS_H
#define GUI_GUIUTILS_H

#include "types/MocaTypes.h"
#include "types/Image8U.h"
#include "types/Rect.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <boost/shared_ptr.hpp>


namespace GUIutils
{
  void showAlert(std::string const msg);
  std::string showInput(std::string label, std::string defaultVal); // pops up an input window and returns the given input
  void showMessage(std::string const msg);
  void showFileChooser(std::string const title, std::string const pattern, char* fileName, sizeType fnBufsize); // the chosen file is stored in (which also acts as input for the default filename) fileName. If no file was chosen it is an empty string.
  void showFileChooser(std::string const title, std::string const pattern, std::string& fileName); // same as above
  uint32 showListView(std::string const title, std::vector<std::string> const& list); // allows the user to choose an element from a list
  uint32 showChoice(std::string const text, std::string const option0, std::string const option1, std::string const option2 = std::string()); // option2 may be empty (the button is then hidden)

  class SimpleImageWindow
  {
   public:
    SimpleImageWindow(std::string const& name); // a SimpleImageWindow's name must be unique
    ~SimpleImageWindow();
    
    void showImage(Image8U const& newImage);
    static int32 wait(int32 ms = 0);
    
   private:
    std::string name;
    IplImage* image;
  };

  // Template for showInput for convenience
  template <class T> T showInputTempl(std::string label, T defaultVal)
    {
      std::stringstream in, out;
      in << defaultVal;
      out << showInput(label, in.str());
      T result = defaultVal;
      if (!out.str().empty())
	out >> result;
      return result;
    }
};

#endif //GUI_GUIUTILS_H

