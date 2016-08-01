#ifndef IO_VIDEOWRITER_H
#define IO_VIDEOWRITER_H

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

#include <map>
#include "io/Writer.h"

extern "C" {
#define __STDC_CONSTANT_MACROS
  #include <libavcodec/version.h>
#undef FF_API_FLAC_GLOBAL_OPTS
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
  #include <libavutil/mathematics.h>
}

enum Codec {
  NONE,
  CODEC_MPEG1,
  CODEC_MPEG2,
  CODEC_MPEG4,
  CODEC_FLV1,
  CODEC_FFV1,
  CODEC_RAW,
  CODEC_WMV1,
  CODEC_WMV2
};


class VideoWriter : public Writer
{
 public:
  /// Constructor.
  VideoWriter(std::string const& fileName, sizeType const& width, sizeType const& height);
  /// Destructor. Calls stop().
  ~VideoWriter();
  
  /// Opens the video file and queries it's properties.
  void start();
  /// Closes the video file.
  void stop();

  void putImage(Image8U const& image);

  void setFilename(std::string const& newFileName);
  void setBitRate(int32 const& new_bit_rate);
  void setFrameRate(int32 const& new_frame_rate);
  void setCodec(Codec const& newCodec);

  /// Returns the frame width.
  sizeType getImageWidth();
  /// Returns the frame height.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
  /// Returns the number of channels of the image returned by getImage().
  /// Returns currentFrame.
  int32 getCurrentFrame();
  
 private:
  std::string fileName; /// Name of the video file.
  sizeType width; /// Frame width.
  sizeType height; /// Frame height.
  int32 currentFrame; // current Frame.
  bool started;

  AVOutputFormat *fmt;
  AVFormatContext *oc;
  AVStream *video_st;
  AVFrame *picture, *tmp_picture;
  uint8_t *video_outbuf;
  int32 video_outbuf_size, bit_rate, frame_rate;
  Codec codec;
  static std::map<int32, long int> codecMap;

  // Deprecated
  //AVStream *add_video_stream(enum CodecID codec_id);
  AVStream *add_video_stream(enum AVCodecID codec_id);
  AVFrame *alloc_picture(enum PixelFormat pix_fmt);
  void open_video();
  void initCodecMap();
};

#endif
