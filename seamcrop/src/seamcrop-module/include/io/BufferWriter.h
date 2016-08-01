#ifndef IO_BUFFERWRITER_H
#define IO_BUFFERWRITER_H

/*
 * Medium quality bitrate
 * 
 * Codec   |     HD   |    PAL   | low res. |
 * mpeg1   | 4000000  | 1500000  | 1000000  |
 * mpeg2   | 4000000  | 1500000  | 1000000  |
 * mpeg4   | 3000000  | 1500000  | 1000000  |
 * FLV1    | 4500000  | 1500000  | 1000000  |
 * WMV1    | 3000000  | 1500000  | 1000000  |
 * WMV2    | 3500000  | 1500000  | 1000000  |
 */

/*
 * High quality bitrate
 * 
 * Codec   |    HD    |    PAL   | low res. |
 * mpeg1   | 11000000 | 3500000  | 2000000  |
 * mpeg2   | 11000000 | 3500000  | 2000000  |
 * mpeg4   | 10000000 | 3500000  | 2000000  |
 * FLV1    | 9000000  | 2500000  | 2000000  |
 * WMV1    | 9500000  | 3000000  | 2000000  |
 * WMV2    | 10000000 | 3000000  | 2000000  |
 */

/*
 * Losless Codecs
 * 
 * Codec   |    HD    |    PAL   | low res. |
 * FFV1    | >2000000 | >400000  | >300000  |
 * RAW     | >6500000 | >1000000 | >500000  |
 */

#include <queue>
#include "types/MocaTypes.h"
#include "types/Rect.h"
#include "types/Image8U.h"

extern "C" {
#define __STDC_CONSTANT_MACROS
#undef FF_API_FLAC_GLOBAL_OPTS
  #include <libavformat/avformat.h>
  #include <libavutil/imgutils.h>
  #include <libswscale/swscale.h>
  #include <gst/gst.h>
  #include <glib.h>
}

#define RGB24_ALIGNMENT   1
#define YUV420P_ALIGNMENT 4

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

  /// Returns the frame width.
  sizeType getImageWidth();
  /// Returns the frame height.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
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
