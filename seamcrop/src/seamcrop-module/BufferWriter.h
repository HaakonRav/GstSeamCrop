#ifndef IO_BUFFERWRITER_H
#define IO_BUFFERWRITER_H

#include <queue>
#include "include/types/MocaTypes.h"
#include "include/types/Image8U.h"
#include "seamCrop_commons.h"

extern "C" {
#define __STDC_CONSTANT_MACROS
#undef FF_API_FLAC_GLOBAL_OPTS
  #include <libavformat/avformat.h>
  #include <libavutil/imgutils.h>
  #include <libswscale/swscale.h>
  #include <gst/gst.h>
  #include <glib.h>
}


class BufferWriter
{
 public:
  /// Constructor.
  BufferWriter(GAsyncQueue *output_queue, sizeType const& width, sizeType const& height);
  /// Destructor. Calls stop().
  ~BufferWriter();
  
  /// Opens the video file and queries it's properties.
  void start();
  /// Closes the video file.
  void stop();
  // Adds a buffer to the queue.
  void addBuffer(GstBuffer *outbuf);
  // Inserts a converted frame into an output buffer.
  void putImage(Image8U const& image);
  // Sets the clock to obtain timestamps from.
  void setMeasurementClock(GstClock *cur_clock);
  // Writes measurements to terminal.
  void writeMeasurements();

  /// Returns the frame width.
  sizeType getImageWidth();
  /// Returns the frame height.
  sizeType getImageHeight();
  /// Returns currentFrame.
  int32 getCurrentFrame();


  // Queue to put finished buffers in.
  GAsyncQueue *outputQueue;
  // Queue to store output buffers.
  std::queue<GstBuffer*> outBuffers;
  
 private:
  sizeType width; /// Frame width.
  sizeType height; /// Frame height.
  int32 currentFrame; // current Frame.
  bool started;

  AVFrame *pFrameBGR, *pFrameYUV;
};

#endif
