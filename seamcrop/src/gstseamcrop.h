/* GStreamer
 * Copyright (C) 2016 Haakon Wilhelm Ravik <haakonwr@student.matnat.uio.no>
 *
 * This file is a part of GstSeamCrop.
 * 
 * GstSeamCrop is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * GstSeamCrop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_SEAMCROP_H_
#define _GST_SEAMCROP_H_

#include <gst/gst.h>
#include <glib.h>

G_BEGIN_DECLS

#define GST_TYPE_SEAMCROP   (gst_seamcrop_get_type())
#define GST_SEAMCROP(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SEAMCROP,GstSeamCrop))
#define GST_SEAMCROP_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SEAMCROP,GstSeamCropClass))
#define GST_IS_SEAMCROP(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SEAMCROP))
#define GST_IS_SEAMCROP_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SEAMCROP))
#define GST_SEAMCROP_CAST(obj)  ((GstSeamCrop *)(obj))

/**
 *  GST_SEAMCROP_SINK_NAME:
 *  
 *  The name of the templates for the sink pad.
 */
#define GST_SEAMCROP_SINK_NAME  "sink"
/**
 *  GST_SEAMCROP_SRC_NAME:
 *  
 *  The name of the templates for the source pad.
 */
#define GST_SEAMCROP_SRC_NAME "src"

/**
 *  GST_SEAMCROP_SRC_PAD:
 *  @obj: seamcrop instance
 *  
 *  Gives the pointer to the source #GstPad object of the element.
 */
#define GST_SEAMCROP_SRC_PAD(obj)   (GST_SEAMCROP_CAST (obj)->srcpad)

/**
 *  GST_SEAMCROP_SINK_PAD:
 *  @obj: seamcrop instance
 *  
 *  Gives the pointer to the sink #GstPad object of the element.
 */
#define GST_SEAMCROP_SINK_PAD(obj)  (GST_SEAMCROP_CAST (obj)->sinkpad)


/**
 *  GST_SEAMCROP_FLOW_DROPPED:
 *  
 *  A #GstFlowReturn that can be returned from transform and transform_ip to
 *  indicate that no output buffer was generated.
 */
#define GST_SEAMCROP_BUFFER_PASSED   GST_FLOW_CUSTOM_SUCCESS

typedef struct _GstSeamCrop GstSeamCrop;
typedef struct _GstSeamCropClass GstSeamCropClass;

struct _GstSeamCrop
{
  GstElement base_seamcrop;

  /* Used for measurement. */
  GstClock *measure_clock;

  /*< protected >*/
  /* source and sink pads */
  GstPad        *sinkpad;
  GstPad        *srcpad;

  /* MT-protected (with STREAM_LOCK) */
  gboolean      have_segment;
  GstSegment    segment;

  /* Default submit_input_buffer places the buffer here,
   * for consumption by the generate_output method: */
  GstBuffer     *queued_buf;

  /* Queue to place output buffers into. */
  GAsyncQueue   *output_queue;

  /* Thread which will push data on the source pad. */
  GThread       *push_thread;

  /* The retval the dispatching thread exited on. */
  GstFlowReturn lastRetVal;

  /* The time at which to stop the stream. */
  guint64       stop_time;
  
  /* Input frame info */
  gint          input_width;
  gint          input_height;
  gint          prev_width;
  gint          prev_height;
  double        input_framerate;

  /* Output frame info. */
  gint          output_width;
  gint          output_frame_size;

  /* Factor by which the video frame width will be reduced. */
  float         retargeting_factor;

  /* Factor by which the borders during seam carving will be extended. */
  float         extend_border_factor;

  /* Amount of frames the retargeting module should process at a time. */
  gint          frame_window_size;

  /* To tell the dispatching thread that the stream has ended. */
  gboolean      stream_end;

  /* Whether seamcropcuda has been initialized or not.*/
  gboolean      started;

  /* Whether the pipeline is flushing. */
  gboolean      flushing;

  /* Whether we are performing measurements or not. */
  gboolean      measurement_mode;

  /* Whether a capability change requires reinitialization of the retargeting module. */
  gboolean      reinitialize_module;

  /* Added from _GstBaseTransformPrivate. Where should we put these? */

  GstCaps *cache_caps1;
  gsize cache_caps1_size;
  GstCaps *cache_caps2;
  gsize cache_caps2_size;
  gboolean have_same_caps;

  gboolean negotiated;

  /* QoS *//* with LOCK */
  gboolean qos_enabled;
  gdouble proportion;
  GstClockTime earliest_time;

  /* previous buffer had a discont */
  gboolean discont;

  GstPadMode pad_mode;

  gboolean gap_aware;
  // No need. We only operate in one mode.
  //gboolean prefer_passthrough;

  /* QoS stats */
  guint64 processed;
  guint64 dropped;

  GstClockTime position_out;

  GstBufferPool *pool;
  gboolean pool_active;
  GstAllocator *allocator;
  GstAllocationParams params;
  GstQuery *query;

  /*< private >*/
  gpointer       _gst_reserved[GST_PADDING_LARGE-1];
};

struct _GstSeamCropClass
{
  GstElementClass base_seamcrop_class;
};

GType   gst_seamcrop_get_type         (void);
void    gst_seamcrop_update_qos       (GstSeamCrop *seamcrop, gdouble proportion, GstClockTimeDiff diff, GstClockTime timestamp);
void    gst_seamcrop_set_qos_enabled  (GstSeamCrop *seamcrop, gboolean enabled);
gboolean  gst_seamcrop_is_qos_enabled (GstSeamCrop *seamcrop);


#ifdef G_DEFINE_AUTOPTR_CLEANUP_FUNC
G_DEFINE_AUTOPTR_CLEANUP_FUNC(GstSeamCrop, gst_object_unref)
#endif
G_END_DECLS

#endif
