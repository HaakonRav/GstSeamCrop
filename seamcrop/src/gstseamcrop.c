/* GStreamer
 * Copyright (C) 2016 Haakon Wilhelm Ravik <haakonwr@student.matnat.uio.no>
 *
 * This file is a part of GstSeamCrop.
 * 
 * GstSeamCrop is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GstSeamCrop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstseamcrop
 *
 * The seamcrop element receives raw I420 frames, retargets them using GPU acceleration and sends the frames onwards.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v filesrc location=<location> ! decodebin ! seamcrop ! queue ! xvimagesink
 * ]|
 * The pipeline processes video data from a source and produces a retargeted output.
 * </refsect2>
 */

/* TODO: Clean up. 
 *
 */

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

#include <stdlib.h>
#include <string.h>

#include <gst/gst.h>
#include <gst/video/video.h>
#include "gstseamcrop.h"
#include "seamcrop-module/seamCropWrapper.h"

GST_DEBUG_CATEGORY_STATIC (gst_seamcrop_debug);
#define GST_CAT_DEFAULT gst_seamcrop_debug

#define DEFAULT_PROP_QOS              FALSE
#define DEFAULT_PROP_MEASUREMENT      FALSE
#define DEFAULT_PROP_EXTEND           0.25f
#define DEFAULT_PROP_RETARGET         0.75f
#define DEFAULT_PROP_WINDOW_SIZE      100

enum
{
  PROP_0,
  PROP_QOS,
  PROP_EXTEND,
  PROP_RETARGET,
  PROP_WINDOW_SIZE,
  PROP_MEASUREMENT
};

/* prototypes */

static void read_video_props(GstSeamCrop *seamcrop, GstCaps *caps);
static void gst_seamcrop_push_output(GstSeamCrop *seamcrop);
static GstFlowReturn submit_input_buffer (GstSeamCrop * seamcrop, gboolean is_discont, GstBuffer * input);
static GstFlowReturn generate_output (GstSeamCrop * seamcrop, GstBuffer ** outbuf);

static void gst_seamcrop_class_init (GstSeamCropClass * klass);
static void gst_seamcrop_init (GstSeamCrop *seamcrop);
static void gst_seamcrop_finalize (GObject * object);
static void gst_seamcrop_set_property (GObject * object, guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_seamcrop_get_property (GObject * object, guint property_id, GValue * value, GParamSpec * pspec);
static gboolean gst_seamcrop_src_event (GstPad *pad, GstObject *parent, GstEvent *event);
static gboolean gst_seamcrop_sink_event (GstPad *pad, GstObject *parent, GstEvent *event);
static GstFlowReturn gst_seamcrop_sink_chain (GstPad *pad, GstObject *parent, GstBuffer *buffer);
static gboolean gst_seamcrop_query (GstPad * pad, GstObject * parent, GstQuery * query);
static gboolean gst_seamcrop_src_activate_mode (GstPad * pad, GstObject * parent, GstPadMode mode, gboolean active);
static gboolean gst_seamcrop_sink_activate_mode (GstPad * pad, GstObject * parent, GstPadMode mode, gboolean active);
static gboolean gst_seamcrop_decide_allocation (GstSeamCrop * seamcrop, GstQuery * query);
static gboolean gst_seamcrop_propose_allocation (GstSeamCrop * seamcrop, GstQuery * decide_query, GstQuery * query);
static GstCaps *gst_seamcrop_transform_caps (GstSeamCrop * seamcrop, GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_seamcrop_query_caps (GstSeamCrop * seamcrop, GstPad * pad, GstCaps * filter);
static gboolean gst_seamcrop_acceptcaps (GstSeamCrop * seamcrop, GstPadDirection direction, GstCaps * caps);
static gboolean gst_seamcrop_setcaps (GstSeamCrop * seamcrop, GstPad * pad, GstCaps * caps);
static GstCaps *gst_seamcrop_fixate_caps (GstSeamCrop * seamcrop, GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static GstFlowReturn prepare_output_buffer (GstSeamCrop * seamcrop, GstBuffer * inbuf, GstBuffer ** outbuf);
static gboolean copy_metadata (GstSeamCrop * seamcrop, GstBuffer * inbuf, GstBuffer * outbuf);
static gboolean gst_seamcrop_transform_meta (GstSeamCrop * seamcrop, GstBuffer * inbuf, GstMeta * meta, GstBuffer * outbuf);
static GstFlowReturn gst_seamcrop_transform (GstSeamCrop * seamcrop, GstBuffer * inbuf, GstBuffer * outbuf);


/** Parent class **/
static GstElementClass *parent_class = NULL;

/* Function to read video metadata from caps. */
static void read_video_props(GstSeamCrop *seamcrop, GstCaps *caps)
{
  gint denom, framerate;
  const GstStructure *str;

  seamcrop->prev_width = seamcrop->input_width;
  seamcrop->prev_height = seamcrop->input_height;

  str = gst_caps_get_structure (caps, 0);

  if (!gst_structure_get_int (str, "width", &seamcrop->input_width) ||
      !gst_structure_get_int (str, "height", &seamcrop->input_height) ||
      !gst_structure_get_fraction (str, "framerate", &framerate, &denom))
  {
    GST_LOG("read_video_props: No relevant properties available");
    return;
  }

  if((seamcrop->prev_width != seamcrop->input_width) || (seamcrop->prev_height != seamcrop->input_height))
    seamcrop->reinitialize_module = TRUE;

  /* Calculate the approximate framerate from the fraction in the caps.*/
  seamcrop->input_framerate = ((float)framerate / (float)denom);

  /* Compute output width. */
  seamcrop->output_width = seamcrop->input_width * seamcrop->retargeting_factor;
  if(seamcrop->output_width % 2)
    seamcrop->output_width -= 1;
}

/* pad templates */
static GstStaticPadTemplate gst_seamcrop_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (
      "video/x-raw, "
      "format = (string) I420, "
      "framerate = (fraction) [ 0/1, 2147483647/1 ], " 
      "width = (int) [ 1, 2147483647 ], " 
      "height = (int) [ 1, 2147483647 ]")
    );

static GstStaticPadTemplate gst_seamcrop_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, "
      "format = (string) I420, "
      "framerate = (fraction) [ 0/1, 2147483647/1 ], " 
      "width = (int) [ 1, 2147483647 ], " 
      "height = (int) [ 1, 2147483647 ]")
    );

/* class initialization */
G_DEFINE_TYPE_WITH_CODE (GstSeamCrop, gst_seamcrop, GST_TYPE_ELEMENT,
    GST_DEBUG_CATEGORY_INIT (gst_seamcrop_debug, "seamcrop", 0,
      "debug category for seamcrop element"));
  static void
gst_seamcrop_class_init (GstSeamCropClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_seamcrop_debug, "seamcrop", 0, "seamcrop element");

  GST_DEBUG ("gst_seamcrop_class_init");

  parent_class = g_type_class_peek_parent (klass);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_seamcrop_sink_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_seamcrop_src_template));

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "SeamCrop Plugin for GPU accelerated video retargeting.", "Generic", "Retargets videoframes "
      "using the SeamCrop algorithm.",
      "Haakon Wilhelm Ravik <haakonwr@student.matnat.uio.no>");

  gobject_class->set_property = gst_seamcrop_set_property;
  gobject_class->get_property = gst_seamcrop_get_property;
  gobject_class->finalize = gst_seamcrop_finalize;

  /* Properties */
  g_object_class_install_property(gobject_class, PROP_QOS,
      g_param_spec_boolean("qos","QoS", "Handle Quality-of-Service events",
        DEFAULT_PROP_QOS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_RETARGET,
      g_param_spec_float("retargeting-factor", "Retargeting Factor", "Factor for how "
        "much the video should be reduced in width.",
        0.10f, 0.99f, 0.75f, G_PARAM_READWRITE));

  g_object_class_install_property(gobject_class, PROP_EXTEND,
      g_param_spec_float("extend-border", "Extend Border Factor", "How much the border "
        "of the video should be extended with during seam carving.",
        0.10f, 0.99f, 0.25f, G_PARAM_READWRITE));

  g_object_class_install_property(gobject_class, PROP_WINDOW_SIZE,
      g_param_spec_int("frame-window-size", "Frame Window Size", "How many frames to retarget "
        "at a time from an incoming stream.",
        10, 500, 100, G_PARAM_READWRITE));

  g_object_class_install_property(gobject_class, PROP_MEASUREMENT,
      g_param_spec_boolean("measurement", "Measurement", "Whether to perform measurements ",
        FALSE, G_PARAM_READWRITE));
}

  static void
gst_seamcrop_init (GstSeamCrop *seamcrop)
{
  GST_DEBUG ("gst_seamcrop_init");

  /* Sinkpad functions */
  seamcrop->sinkpad = gst_pad_new_from_static_template (&gst_seamcrop_sink_template, "sink");

  gst_pad_use_fixed_caps(seamcrop->sinkpad);

  gst_pad_set_event_function(seamcrop->sinkpad, GST_DEBUG_FUNCPTR(gst_seamcrop_sink_event));
  gst_pad_set_chain_function(seamcrop->sinkpad, GST_DEBUG_FUNCPTR(gst_seamcrop_sink_chain));
  gst_pad_set_activatemode_function(seamcrop->sinkpad, GST_DEBUG_FUNCPTR(gst_seamcrop_sink_activate_mode));
  gst_pad_set_query_function(seamcrop->sinkpad, GST_DEBUG_FUNCPTR(gst_seamcrop_query));
  gst_element_add_pad(GST_ELEMENT(seamcrop), seamcrop->sinkpad);

  /* Sourcepad functions */
  seamcrop->srcpad = gst_pad_new_from_static_template (&gst_seamcrop_src_template, "src");

  gst_pad_use_fixed_caps(seamcrop->srcpad);

  gst_pad_set_event_function(seamcrop->srcpad, GST_DEBUG_FUNCPTR(gst_seamcrop_src_event));
  gst_pad_set_activatemode_function(seamcrop->srcpad, GST_DEBUG_FUNCPTR(gst_seamcrop_src_activate_mode));
  gst_pad_set_query_function(seamcrop->srcpad, GST_DEBUG_FUNCPTR(gst_seamcrop_query));
  gst_element_add_pad(GST_ELEMENT(seamcrop), seamcrop->srcpad);

  /* General value initializations */
  seamcrop->qos_enabled = DEFAULT_PROP_QOS;
  seamcrop->extend_border_factor = DEFAULT_PROP_EXTEND;
  seamcrop->retargeting_factor = DEFAULT_PROP_RETARGET;
  seamcrop->frame_window_size = DEFAULT_PROP_WINDOW_SIZE;
  seamcrop->measurement_mode = DEFAULT_PROP_MEASUREMENT;

  seamcrop->cache_caps1 = NULL;
  seamcrop->cache_caps2 = NULL;
  seamcrop->pad_mode = GST_PAD_MODE_NONE;
  seamcrop->gap_aware = FALSE;

  seamcrop->processed = 0;
  seamcrop->dropped = 0;
  seamcrop->prev_width = 0;
  seamcrop->prev_height = 0;
  seamcrop->input_width = 0;
  seamcrop->input_height = 0;
  seamcrop->reinitialize_module = FALSE;
  seamcrop->started = FALSE;

  /* Queue in which transformed frames will be put. */
  seamcrop->output_queue = g_async_queue_new();

  seamcrop->measure_clock = gst_system_clock_obtain();
}


/** FROM ORIGINAL GSTSEAMCROP **/
/* given @caps on the src or sink pad (given by @direction)
 * calculate the possible caps on the other pad.
 *
 * Returns new caps, unref after usage.
 */
  static GstCaps *
gst_seamcrop_transform_caps (GstSeamCrop * seamcrop,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCaps *ret;

  if (caps == NULL)
    return NULL;

  GST_DEBUG_OBJECT (seamcrop, "transform caps (direction = %d)", direction);

  ret = gst_caps_copy (caps);

  /* Copy other caps and modify as appropriate */
  /* This works for the simplest cases, where the transform modifies one
   * or more fields in the caps structure. 
   */
  if (direction == GST_PAD_SRC) {
    GST_DEBUG_OBJECT(seamcrop, "Going SRC: %" GST_PTR_FORMAT, ret);
    /* transform caps going upstream */
  } else {
    GST_DEBUG_OBJECT(seamcrop, "Going SINK: %" GST_PTR_FORMAT, ret);
    /* transform caps going downstream */
  }

  if (filter) {
    GstCaps *intersect;

    intersect = gst_caps_intersect (ret, filter);
    gst_caps_unref (ret);

    ret = intersect;
  }

  GST_DEBUG_OBJECT (seamcrop, "to: %" GST_PTR_FORMAT, ret);

  return ret;
}

  static gboolean
gst_seamcrop_transform_meta (GstSeamCrop * seamcrop,
    GstBuffer * inbuf, GstMeta * meta, GstBuffer * outbuf)
{
  const GstMetaInfo *info = meta->info;
  const gchar *const *tags;

  tags = gst_meta_api_type_get_tags (info->api);

  if (!tags)
    return TRUE;

  return FALSE;
}

/* get the caps that can be handled by @pad. We perform:
 *  
 * - take the caps of peer of otherpad,
 * - filter against the padtemplate of otherpad, 
 * - calculate all transforms of remaining caps
 * - filter against template of @pad
 * 
 * If there is no peer, we simply return the caps of the padtemplate of pad.
 */
/** FRA BASETRANSFORM. **/
  static GstCaps *
gst_seamcrop_query_caps (GstSeamCrop * seamcrop, GstPad * pad,
    GstCaps * filter)
{
  GstPad *otherpad;
  GstCaps *peercaps = NULL, *caps, *temp, *peerfilter = NULL;
  GstCaps *templ, *otempl;

  otherpad = (pad == seamcrop->srcpad) ? seamcrop->sinkpad : seamcrop->srcpad;

  templ = gst_pad_get_pad_template_caps (pad);
  otempl = gst_pad_get_pad_template_caps (otherpad);

  /* first prepare the filter to be send onwards. We need to filter and
   * transform it to valid caps for the otherpad. */
  if (filter) {
    GST_DEBUG_OBJECT (pad, "filter caps  %" GST_PTR_FORMAT, filter);

    /* filtered against our padtemplate of this pad */
    GST_DEBUG_OBJECT (pad, "our template  %" GST_PTR_FORMAT, templ);
    temp = gst_caps_intersect_full (filter, templ, GST_CAPS_INTERSECT_FIRST);
    GST_DEBUG_OBJECT (pad, "intersected %" GST_PTR_FORMAT, temp);

    /* then see what we can transform this to */
    peerfilter = gst_seamcrop_transform_caps (seamcrop,
        GST_PAD_DIRECTION (pad), temp, NULL);
    GST_DEBUG_OBJECT (pad, "transformed  %" GST_PTR_FORMAT, peerfilter);
    gst_caps_unref (temp);

    if (!gst_caps_is_empty (peerfilter)) {
      /* and filter against the template of the other pad */
      GST_DEBUG_OBJECT (pad, "our template  %" GST_PTR_FORMAT, otempl);
      /* We keep the caps sorted like the returned caps */
      temp =
        gst_caps_intersect_full (peerfilter, otempl,
            GST_CAPS_INTERSECT_FIRST);
      GST_DEBUG_OBJECT (pad, "intersected %" GST_PTR_FORMAT, temp);
      gst_caps_unref (peerfilter);
      peerfilter = temp;
    }
  }

  GST_DEBUG_OBJECT (pad, "peer filter caps %" GST_PTR_FORMAT, peerfilter);

  if (peerfilter && gst_caps_is_empty (peerfilter)) {
    GST_DEBUG_OBJECT (pad, "peer filter caps are empty");
    caps = peerfilter;
    peerfilter = NULL;
    goto done;
  }

  /* query the peer with the transformed filter */
  peercaps = gst_pad_peer_query_caps (otherpad, peerfilter);

  if (peerfilter)
    gst_caps_unref (peerfilter);

  if (peercaps) {
    GST_DEBUG_OBJECT (pad, "peer caps  %" GST_PTR_FORMAT, peercaps);

    /* filtered against our padtemplate on the other side */
    GST_DEBUG_OBJECT (pad, "our template  %" GST_PTR_FORMAT, otempl);
    temp = gst_caps_intersect_full (peercaps, otempl, GST_CAPS_INTERSECT_FIRST);
    GST_DEBUG_OBJECT (pad, "intersected %" GST_PTR_FORMAT, temp);
  } else {
    temp = gst_caps_ref (otempl);
  }

  /* then see what we can transform this to */
  caps = gst_seamcrop_transform_caps (seamcrop,
      GST_PAD_DIRECTION (otherpad), temp, filter);
  GST_DEBUG_OBJECT (pad, "transformed  %" GST_PTR_FORMAT, caps);
  gst_caps_unref (temp);
  if (caps == NULL || gst_caps_is_empty (caps))
    goto done;

  if (peercaps) {
    /* and filter against the template of this pad */
    GST_DEBUG_OBJECT (pad, "our template  %" GST_PTR_FORMAT, templ);
    /* We keep the caps sorted like the returned caps */
    temp = gst_caps_intersect_full (caps, templ, GST_CAPS_INTERSECT_FIRST);
    GST_DEBUG_OBJECT (pad, "intersected %" GST_PTR_FORMAT, temp);
    gst_caps_unref (caps);
    caps = temp;

  } else {
    gst_caps_unref (caps);
    /* no peer or the peer can do anything, our padtemplate is enough then */
    if (filter) {
      caps = gst_caps_intersect_full (filter, templ, GST_CAPS_INTERSECT_FIRST);
    } else {
      caps = gst_caps_ref (templ);
    }
  }

done:
  GST_DEBUG_OBJECT (seamcrop, "returning  %" GST_PTR_FORMAT, caps);

  if (peercaps)
    gst_caps_unref (peercaps);

  gst_caps_unref (templ);
  gst_caps_unref (otempl);

  return caps;
}

/* takes ownership of the pool, allocator and query */
  static gboolean
gst_seamcrop_set_allocation (GstSeamCrop * seamcrop,
    GstBufferPool * pool, GstAllocator * allocator,
    GstAllocationParams * params, GstQuery * query)
{
  GstAllocator *oldalloc;
  GstBufferPool *oldpool;
  GstQuery *oldquery;

  GST_OBJECT_LOCK (seamcrop);
  oldpool = seamcrop->pool;
  seamcrop->pool = pool;
  seamcrop->pool_active = FALSE;

  oldalloc = seamcrop->allocator;
  seamcrop->allocator = allocator;

  oldquery = seamcrop->query;
  seamcrop->query = query;

  if (params)
    seamcrop->params = *params;
  else
    gst_allocation_params_init (&seamcrop->params);
  GST_OBJECT_UNLOCK (seamcrop);

  if (oldpool) {
    GST_DEBUG_OBJECT (seamcrop, "deactivating old pool %p", oldpool);
    gst_buffer_pool_set_active (oldpool, FALSE);
    gst_object_unref (oldpool);
  }
  if (oldalloc) {
    gst_object_unref (oldalloc);
  }
  if (oldquery) {
    gst_query_unref (oldquery);
  }

  return TRUE;
}

/** FRA BASETRANSFORM. **/
  static gboolean
gst_seamcrop_decide_allocation (GstSeamCrop * seamcrop,
    GstQuery * query)
{
  guint i, n_metas;
  GstCaps *outcaps;
  GstBufferPool *pool;
  guint size, min, max;
  GstAllocator *allocator;
  GstAllocationParams params;
  GstStructure *config;
  gboolean update_allocator;

  n_metas = gst_query_get_n_allocation_metas (query);
  for (i = 0; i < n_metas; i++) {
    GType api;
    const GstStructure *params;
    gboolean remove;

    api = gst_query_parse_nth_allocation_meta (query, i, &params);

    /* by default we remove all metadata */
    if (gst_meta_api_type_has_tag (api, _gst_meta_tag_memory)) {
      /* remove all memory dependent metadata because we are going to have to
       * allocate different memory for input and output. */
      GST_LOG_OBJECT (seamcrop, "removing memory specific metadata %s",
          g_type_name (api));
      remove = TRUE;
    } else {
      GST_LOG_OBJECT (seamcrop, "removing metadata %s", g_type_name (api));
      remove = TRUE;
    }

    if (remove) {
      gst_query_remove_nth_allocation_meta (query, i);
      i--;
      n_metas--;
    }
  }

  gst_query_parse_allocation (query, &outcaps, NULL);

  /* we got configuration from our peer or the decide_allocation method,
   * parse them */
  if (gst_query_get_n_allocation_params (query) > 0) {
    /* try the allocator */
    gst_query_parse_nth_allocation_param (query, 0, &allocator, &params);
    update_allocator = TRUE;
  } else {
    allocator = NULL;
    gst_allocation_params_init (&params);
    update_allocator = FALSE;
  }

  if (gst_query_get_n_allocation_pools (query) > 0) {
    /* Size is gotten here. */
    gst_query_parse_nth_allocation_pool (query, 0, &pool, &size, &min, &max);

    if (pool == NULL) {
      /* no pool, we can make our own */
      GST_DEBUG_OBJECT (seamcrop, "no pool, making new pool");
      pool = gst_buffer_pool_new ();
    }
  } else {
    /* rather create pool here? */
    pool = NULL;
    size = min = max = 0;
  }

  /* now configure */
  if (pool) {
    config = gst_buffer_pool_get_config (pool);

    gst_buffer_pool_config_set_params (config, outcaps, size, min, max);
    gst_buffer_pool_config_set_allocator (config, allocator, &params);

    /* buffer pool may have to do some changes */
    if (!gst_buffer_pool_set_config (pool, config)) {
      config = gst_buffer_pool_get_config (pool);

      /* If changes are not acceptable, fallback to generic pool */
      if (!gst_buffer_pool_config_validate_params (config, outcaps, size, min, max)) {
        GST_DEBUG_OBJECT (seamcrop, "unsupported pool, making new pool");

        gst_object_unref (pool);
        pool = gst_buffer_pool_new ();
        gst_buffer_pool_config_set_params (config, outcaps, size, min, max);
        gst_buffer_pool_config_set_allocator (config, allocator, &params);
      }

      if (!gst_buffer_pool_set_config (pool, config))
        goto config_failed;
    }
  }

  if (update_allocator)
    gst_query_set_nth_allocation_param (query, 0, allocator, &params);
  else
    gst_query_add_allocation_param (query, allocator, &params);
  if (allocator)
    gst_object_unref (allocator);

  if (pool) {
    gst_query_set_nth_allocation_pool (query, 0, pool, size, min, max);
    gst_object_unref (pool);
  }

  return TRUE;

config_failed:
  GST_ELEMENT_ERROR (seamcrop, RESOURCE, SETTINGS,
      ("Failed to configure the buffer pool"),
      ("Configuration is most likely invalid."));
  return FALSE;
}

  static gboolean
gst_seamcrop_do_bufferpool (GstSeamCrop * seamcrop, GstCaps * outcaps)
{
  GstQuery *query;
  gboolean result = TRUE;
  GstBufferPool *pool = NULL;
  GstAllocator *allocator;
  GstAllocationParams params;

  /* find a pool for the negotiated caps now */
  GST_DEBUG_OBJECT (seamcrop, "doing allocation query");
  query = gst_query_new_allocation (outcaps, TRUE);
  if (!gst_pad_peer_query (seamcrop->srcpad, query)) {
    /* not a problem, just debug a little */
    GST_DEBUG_OBJECT (seamcrop, "peer ALLOCATION query failed");
  }

  GST_DEBUG_OBJECT (seamcrop, "calling decide_allocation");
  result = gst_seamcrop_decide_allocation(seamcrop, query);

  GST_DEBUG_OBJECT (seamcrop, "ALLOCATION (%d) params: %" GST_PTR_FORMAT, result, query);

  if (!result)
    goto no_decide_allocation;

  /* we got configuration from our peer or the decide_allocation method,
   * parse them */
  if (gst_query_get_n_allocation_params (query) > 0) {
    gst_query_parse_nth_allocation_param (query, 0, &allocator, &params);
  } else {
    allocator = NULL;
    gst_allocation_params_init (&params);
  }

  if (gst_query_get_n_allocation_pools (query) > 0)
    gst_query_parse_nth_allocation_pool (query, 0, &pool, NULL, NULL, NULL);

  /* now store */
  result = gst_seamcrop_set_allocation (seamcrop, pool, allocator, &params, query);

  return result;

  /* Errors */
no_decide_allocation:
  {
    GST_WARNING_OBJECT (seamcrop, "Failed to decide allocation");
    gst_query_unref (query);

    return result;
  }
}

/* function triggered when the in and out caps are negotiated and need
 * to be configured in the retargeting module. */
  static gboolean
gst_seamcrop_configure_caps (GstSeamCrop * seamcrop, GstCaps * in,
    GstCaps * out)
{
  gboolean ret = TRUE;

  GST_DEBUG_OBJECT (seamcrop, "in caps:  %" GST_PTR_FORMAT, in);
  GST_DEBUG_OBJECT (seamcrop, "out caps: %" GST_PTR_FORMAT, out);

  /* clear the cache */
  gst_caps_replace (&seamcrop->cache_caps1, NULL);
  gst_caps_replace (&seamcrop->cache_caps2, NULL);

  // Set new width on output.
  GValue newWidth = G_VALUE_INIT;
  g_value_init(&newWidth, G_TYPE_INT);
  g_value_set_int(&newWidth, seamcrop->output_width);

  gst_caps_set_value(out,"width", &newWidth);

  GST_DEBUG_OBJECT (seamcrop, "out caps POST WIDTH: %" GST_PTR_FORMAT, out);

  /* figure out same caps state */
  seamcrop->have_same_caps = gst_caps_is_equal (in, out);
  GST_DEBUG_OBJECT (seamcrop, "have_same_caps: %d", seamcrop->have_same_caps);

  GST_DEBUG_OBJECT (seamcrop, "Reinitializing retargeting environment.");
  // Reinitialize the retargeting environment for the new format.

  return ret;
}

  static GstCaps *
gst_seamcrop_fixate_caps (GstSeamCrop * seamcrop,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  read_video_props(seamcrop, caps);
  othercaps = gst_caps_fixate (othercaps);
  GST_DEBUG_OBJECT (seamcrop, "fixated to %" GST_PTR_FORMAT, othercaps);

  return othercaps;
}


/* given a fixed @caps on @pad, create the best possible caps for the
 * other pad.
 * @caps must be fixed when calling this function.
 * 
 * This function calls the transform caps vmethod of the seamcrop to figure
 * out the possible target formats. It then tries to select the best format from
 * this list by:
 * 
 * - attempt passthrough if the target caps is a superset of the input caps
 * - fixating by using peer caps
 * - fixating with transform fixate function
 * - fixating with pad fixate functions.
 * 
 * this function returns a caps that can be transformed into and is accepted by
 * the peer element.
 */
  static GstCaps *
gst_seamcrop_find_transform (GstSeamCrop * seamcrop, GstPad * pad,
    GstCaps * caps)
{
  GstPad *otherpad, *otherpeer;
  GstCaps *othercaps;
  gboolean is_fixed;

  /* caps must be fixed here, this is a programming error if it's not */
  g_return_val_if_fail (gst_caps_is_fixed (caps), NULL);

  otherpad = (pad == seamcrop->srcpad) ? seamcrop->sinkpad : seamcrop->srcpad;
  otherpeer = gst_pad_get_peer (otherpad);

  /* see how we can transform the input caps. We need to do this even for
   * passthrough because it might be possible that this element cannot support
   * passthrough at all. */
  othercaps = gst_seamcrop_transform_caps (seamcrop,
      GST_PAD_DIRECTION (pad), caps, NULL);

  /* The caps we can actually output is the intersection of the transformed
   * caps with the pad template for the pad */
  if (othercaps && !gst_caps_is_empty (othercaps)) {
    GstCaps *intersect, *templ_caps;

    templ_caps = gst_pad_get_pad_template_caps (otherpad);
    GST_DEBUG_OBJECT (seamcrop, "intersecting against padtemplate %" GST_PTR_FORMAT, templ_caps);

    intersect = gst_caps_intersect_full (othercaps, templ_caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (othercaps);
    gst_caps_unref (templ_caps);
    othercaps = intersect;
  }

  /* check if transform is empty */
  if (!othercaps || gst_caps_is_empty (othercaps))
    goto no_transform;

  /* if the othercaps are not fixed, we need to fixate them, first attempt
   * is by attempting passthrough if the othercaps are a superset of caps. */
  /*  maybe the caps is not fixed because it has multiple structures of
   * fixed caps */
  is_fixed = gst_caps_is_fixed (othercaps);
  if (!is_fixed) {
    GST_DEBUG_OBJECT (seamcrop, "transform returned non fixed  %" GST_PTR_FORMAT, othercaps);

    /* Now let's see what the peer suggests based on our transformed caps */
    if (otherpeer) {
      GstCaps *peercaps, *intersection, *templ_caps;

      GST_DEBUG_OBJECT (seamcrop, "Checking peer caps with filter %" GST_PTR_FORMAT, othercaps);

      peercaps = gst_pad_query_caps (otherpeer, othercaps);
      GST_DEBUG_OBJECT (seamcrop, "Resulted in %" GST_PTR_FORMAT, peercaps);
      if (!gst_caps_is_empty (peercaps)) {
        templ_caps = gst_pad_get_pad_template_caps (otherpad);

        GST_DEBUG_OBJECT (seamcrop, "Intersecting with template caps %" GST_PTR_FORMAT, templ_caps);

        intersection = gst_caps_intersect_full (peercaps, templ_caps, GST_CAPS_INTERSECT_FIRST);
        GST_DEBUG_OBJECT (seamcrop, "Intersection: %" GST_PTR_FORMAT, intersection);
        gst_caps_unref (peercaps);
        gst_caps_unref (templ_caps);
        peercaps = intersection;

        GST_DEBUG_OBJECT (seamcrop, "Intersecting with transformed caps %" GST_PTR_FORMAT, othercaps);
        intersection = gst_caps_intersect_full (peercaps, othercaps, GST_CAPS_INTERSECT_FIRST);

        GST_DEBUG_OBJECT (seamcrop, "Intersection: %" GST_PTR_FORMAT, intersection);
        gst_caps_unref (peercaps);
        gst_caps_unref (othercaps);
        othercaps = intersection;
      } else {
        gst_caps_unref (othercaps);
        othercaps = peercaps;
      }

      is_fixed = gst_caps_is_fixed (othercaps);
    } 
    /*
       else {
       GST_DEBUG_OBJECT (seamcrop, "no peer, doing passthrough");
       gst_caps_unref (othercaps);
       othercaps = gst_caps_ref (caps);
       is_fixed = TRUE;
       }
       */
  }
  if (gst_caps_is_empty (othercaps))
    goto no_transform_possible;

  GST_DEBUG ("have %sfixed caps %" GST_PTR_FORMAT, (is_fixed ? "" : "non-"), othercaps);

  /* second attempt at fixation, call the fixate vmethod */
  /* caps could be fixed but the subclass may want to add fields */
  GST_DEBUG_OBJECT (seamcrop, "calling fixate_caps for %" GST_PTR_FORMAT
      " using caps %" GST_PTR_FORMAT " on pad %s:%s", othercaps, caps,
      GST_DEBUG_PAD_NAME (otherpad));
  /* note that we pass the complete array of structures to the fixate
   * function, it needs to truncate itself */
  othercaps = gst_seamcrop_fixate_caps (seamcrop, GST_PAD_DIRECTION (pad), caps, othercaps);
  is_fixed = gst_caps_is_fixed (othercaps);
  GST_DEBUG_OBJECT (seamcrop, "after fixating %" GST_PTR_FORMAT, othercaps);

  /* caps should be fixed now, if not we have to fail. */
  if (!is_fixed)
    goto could_not_fixate;

  /* and peer should accept */
  if (otherpeer && !gst_pad_query_accept_caps (otherpeer, othercaps))
    goto peer_no_accept;

  GST_DEBUG_OBJECT (seamcrop, "Input caps were %" GST_PTR_FORMAT
      ", and got final caps %" GST_PTR_FORMAT, caps, othercaps);

  if (otherpeer)
    gst_object_unref (otherpeer);

  return othercaps;

  /* ERRORS */
no_transform:
  {
    GST_DEBUG_OBJECT (seamcrop,
        "transform returned useless  %" GST_PTR_FORMAT, othercaps);
    goto error_cleanup;
  }
no_transform_possible:
  {
    GST_DEBUG_OBJECT (seamcrop,
        "transform could not transform %" GST_PTR_FORMAT
        " in anything we support", caps);
    goto error_cleanup;
  }
could_not_fixate:
  {
    GST_DEBUG_OBJECT (seamcrop, "FAILED to fixate %" GST_PTR_FORMAT, othercaps);
    goto error_cleanup;
  }
peer_no_accept:
  {
    GST_DEBUG_OBJECT (seamcrop, "FAILED to get peer of %" GST_PTR_FORMAT
        " to accept %" GST_PTR_FORMAT, otherpad, othercaps);
    goto error_cleanup;
  }
error_cleanup:
  {
    if (otherpeer)
      gst_object_unref (otherpeer);
    if (othercaps)
      gst_caps_unref (othercaps);
    return NULL;
  }
}

  static gboolean
gst_seamcrop_acceptcaps (GstSeamCrop * seamcrop,
    GstPadDirection direction, GstCaps * caps)
{
  GstPad *pad, *otherpad;
  GstCaps *templ, *otempl, *ocaps = NULL;
  gboolean ret = TRUE;

  pad = (direction == GST_PAD_SINK) ? GST_SEAMCROP_SINK_PAD (seamcrop) : GST_SEAMCROP_SRC_PAD (seamcrop);
  otherpad = (direction == GST_PAD_SINK) ? GST_SEAMCROP_SRC_PAD (seamcrop) : GST_SEAMCROP_SINK_PAD (seamcrop);

  GST_DEBUG_OBJECT (seamcrop, "accept caps %" GST_PTR_FORMAT, caps);

  templ = gst_pad_get_pad_template_caps (pad);
  otempl = gst_pad_get_pad_template_caps (otherpad);

  /* get all the formats we can handle on this pad */
  GST_DEBUG_OBJECT (seamcrop, "intersect with pad template: %" GST_PTR_FORMAT, templ);
  if (!gst_caps_can_intersect (caps, templ))
    goto reject_caps;

  GST_DEBUG_OBJECT (seamcrop, "trying to transform with filter: %" GST_PTR_FORMAT " (the other pad template)", otempl);
  ocaps = gst_seamcrop_transform_caps (seamcrop, direction, caps, otempl);
  if (!ocaps || gst_caps_is_empty (ocaps))
    goto no_transform_possible;

done:
  GST_DEBUG_OBJECT (seamcrop, "accept-caps result: %d", ret);
  if (ocaps)
    gst_caps_unref (ocaps);
  gst_caps_unref (templ);
  gst_caps_unref (otempl);
  return ret;

  /* ERRORS */
reject_caps:
  {
    GST_DEBUG_OBJECT (seamcrop, "caps can't intersect with the template");
    ret = FALSE;
    goto done;
  }
no_transform_possible:
  {
    GST_DEBUG_OBJECT (seamcrop, "transform could not transform %" GST_PTR_FORMAT " in anything we support", caps);
    ret = FALSE;
    goto done;
  }
}

/* called when new caps arrive on the sink pad,
 * We try to find the best caps for the other side using our _find_transform()
 * function. If there are caps, we configure the transform for this new
 * transformation.
 */
  static gboolean
gst_seamcrop_setcaps (GstSeamCrop * seamcrop, GstPad * pad,
    GstCaps * incaps)
{
  //GstSeamCropPrivate *priv = seamcrop->priv;
  GstCaps *outcaps, *prev_incaps = NULL, *prev_outcaps = NULL;
  gboolean ret = TRUE;

  GST_DEBUG_OBJECT (pad, "have new caps %p %" GST_PTR_FORMAT, incaps, incaps);

  /* find best possible caps for the other pad */
  outcaps = gst_seamcrop_find_transform (seamcrop, pad, incaps);
  if (!outcaps || gst_caps_is_empty (outcaps))
    goto no_transform_possible;

  /* configure the element now */

  prev_incaps = gst_pad_get_current_caps (seamcrop->sinkpad);
  prev_outcaps = gst_pad_get_current_caps (seamcrop->srcpad);
  if (prev_incaps && prev_outcaps && gst_caps_is_equal (prev_incaps, incaps) && gst_caps_is_equal (prev_outcaps, outcaps)) {
    g_print("EQUALITY BRUV\n");
    GST_DEBUG_OBJECT (seamcrop, "New caps equal to old ones: %" GST_PTR_FORMAT " -> %" GST_PTR_FORMAT, incaps, outcaps);
    ret = TRUE;
  } else {
    /* call configure now */
    if (!(ret = gst_seamcrop_configure_caps (seamcrop, incaps, outcaps)))
      goto failed_configure;

    if (!prev_outcaps || !gst_caps_is_equal (outcaps, prev_outcaps))
    {
      /* let downstream know about our caps */
      ret = gst_pad_set_caps (seamcrop->srcpad, outcaps);
    }
  }

  if (ret) {
    /* try to get a pool when needed */
    ret = gst_seamcrop_do_bufferpool (seamcrop, outcaps);
  }

done:
  if (outcaps)
    gst_caps_unref (outcaps);
  if (prev_incaps)
    gst_caps_unref (prev_incaps);
  if (prev_outcaps)
    gst_caps_unref (prev_outcaps);

  GST_OBJECT_LOCK (seamcrop);
  seamcrop->negotiated = ret;
  GST_OBJECT_UNLOCK (seamcrop);

  return ret;

  /* ERRORS */
no_transform_possible:
  {
    GST_WARNING_OBJECT (seamcrop,
        "transform could not transform %" GST_PTR_FORMAT
        " in anything we support", incaps);
    ret = FALSE;
    goto done;
  }
failed_configure:
  {
    GST_WARNING_OBJECT (seamcrop, "FAILED to configure incaps %" GST_PTR_FORMAT
        " and outcaps %" GST_PTR_FORMAT, incaps, outcaps);
    ret = FALSE;
    goto done;
  }
}


  static gboolean
gst_seamcrop_propose_allocation (GstSeamCrop * seamcrop,
    GstQuery * decide_query, GstQuery * query)
{
  gboolean ret;

  guint i, n_metas;
  /* copy all metadata, decide_query does not contain the
   * metadata anymore that depends on the buffer memory */
  n_metas = gst_query_get_n_allocation_metas (decide_query);
  for (i = 0; i < n_metas; i++) {
    GType api;
    const GstStructure *params;

    api = gst_query_parse_nth_allocation_meta (decide_query, i, &params);
    GST_DEBUG_OBJECT (seamcrop, "proposing metadata %s", g_type_name (api));
    gst_query_add_allocation_meta (query, api, params);
    ret = TRUE;
  }
  return ret;
}


  static gboolean
gst_seamcrop_reconfigure (GstSeamCrop * seamcrop)
{
  gboolean reconfigure, ret = TRUE;

  reconfigure = gst_pad_check_reconfigure (seamcrop->srcpad);

  if (G_UNLIKELY (reconfigure)) {
    GstCaps *incaps;

    GST_DEBUG_OBJECT (seamcrop, "we had a pending reconfigure");

    incaps = gst_pad_get_current_caps (seamcrop->sinkpad);
    if (incaps == NULL)
      goto done;

    /* if we need to reconfigure we pretend new caps arrived. This
     * will reconfigure the transform with the new output format. */
    if (!gst_seamcrop_setcaps (seamcrop, seamcrop->sinkpad, incaps)) {
      GST_ELEMENT_WARNING (seamcrop, STREAM, FORMAT, ("not negotiated"), ("not negotiated"));
      ret = FALSE;
    }
    gst_caps_unref (incaps);
  }
done:
  return ret;
}


  static gboolean
gst_seamcrop_query (GstPad * inPad, GstObject * parent, GstQuery * query)
{
  GstSeamCrop *seamcrop = GST_SEAMCROP(parent);
  GstPadDirection direction = GST_PAD_DIRECTION(inPad);
  gboolean ret = FALSE;
  GstPad *pad, *otherpad;

  if (direction == GST_PAD_SRC) {
    pad = seamcrop->srcpad;
    otherpad = seamcrop->sinkpad;
  } else {
    pad = seamcrop->sinkpad;
    otherpad = seamcrop->srcpad;
  }

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_ALLOCATION:
      {
        /* We received a reply to our allocation query. */
        GstQuery *decide_query = NULL;

        /* can only be done on the sinkpad */
        if (direction != GST_PAD_SINK)
          goto done;

        /* See if reconfiguration is necessary.*/
        ret = gst_seamcrop_reconfigure (seamcrop);
        if (G_UNLIKELY (!ret))
          goto done;

        GST_OBJECT_LOCK (seamcrop);

        if (!seamcrop->negotiated) {
          GST_DEBUG_OBJECT (seamcrop, "not negotiated yet but need negotiation, can't answer ALLOCATION query");
          GST_OBJECT_UNLOCK (seamcrop);
          goto done;
        }

        decide_query = seamcrop->query;
        seamcrop->query = NULL;
        GST_OBJECT_UNLOCK (seamcrop);

        GST_DEBUG_OBJECT (seamcrop, "calling propose allocation with query %" GST_PTR_FORMAT, decide_query);

        /* Copy allocation metas from the received query. */
        ret = gst_seamcrop_propose_allocation (seamcrop, decide_query, query);

        if (decide_query) {
          GST_OBJECT_LOCK (seamcrop);

          if (seamcrop->query == NULL)
            seamcrop->query = decide_query;
          else
            gst_query_unref (decide_query);

          GST_OBJECT_UNLOCK (seamcrop);
        }

        GST_DEBUG_OBJECT (seamcrop, "ALLOCATION ret %d, %" GST_PTR_FORMAT, ret, query);
        break;
      }
    case GST_QUERY_POSITION:
      {
        GstFormat format;

        gst_query_parse_position (query, &format, NULL);
        if (format == GST_FORMAT_TIME && seamcrop->segment.format == GST_FORMAT_TIME) {
          gint64 pos;
          ret = TRUE;

          if ((direction == GST_PAD_SINK) || (seamcrop->position_out == GST_CLOCK_TIME_NONE)) {
            pos = gst_segment_to_stream_time (&seamcrop->segment, GST_FORMAT_TIME, seamcrop->segment.position);
          } else {
            pos = gst_segment_to_stream_time (&seamcrop->segment, GST_FORMAT_TIME, seamcrop->position_out);
          }
          gst_query_set_position (query, format, pos);
        } else {
          ret = gst_pad_peer_query (otherpad, query);
        }
        break;
      }
    case GST_QUERY_ACCEPT_CAPS:
      {
        GstCaps *caps;

        gst_query_parse_accept_caps (query, &caps);
        ret = gst_seamcrop_acceptcaps (seamcrop, direction, caps);
        gst_query_set_accept_caps_result (query, ret);

        /* return TRUE, we answered the query */
        ret = TRUE;
        break;
      }
    case GST_QUERY_CAPS:
      {
        GstCaps *filter, *caps;

        gst_query_parse_caps (query, &filter);
        caps = gst_seamcrop_query_caps (seamcrop, pad, filter);
        gst_query_set_caps_result (query, caps);
        gst_caps_unref (caps);
        ret = TRUE;
        break;
      }
    default:
      ret = gst_pad_peer_query (otherpad, query);
      break;
  }

done:
  return ret;
}


/* this function either returns the input buffer without incrementing the
 * refcount or it allocates a new (writable) buffer */
  static GstFlowReturn 
prepare_output_buffer (GstSeamCrop * seamcrop, GstBuffer * inbuf, GstBuffer ** outbuf)
{
  GstFlowReturn ret;
  GstCaps *outcaps;
  gsize outsize;

  /* we can't reuse the input buffer, try to get pool. */
  if (seamcrop->pool) {
    if (!seamcrop->pool_active) {
      GST_DEBUG_OBJECT (seamcrop, "setting pool %p active", seamcrop->pool);
      if (!gst_buffer_pool_set_active (seamcrop->pool, TRUE))
        goto activate_failed;
      seamcrop->pool_active = TRUE;
    }
    GST_DEBUG_OBJECT (seamcrop, "using pool alloc");
    ret = gst_buffer_pool_acquire_buffer (seamcrop->pool, outbuf, NULL);
    if (ret != GST_FLOW_OK)
      goto alloc_failed;

    goto copy_meta;
  }
  /* If a pool didnt exist, allocate custom output buffer. */

  /* srcpad might be flushing already if we're being shut down */
  outcaps = gst_pad_get_current_caps (seamcrop->srcpad);

  if (outcaps == NULL)
    goto no_outcaps;

  gst_caps_unref (outcaps);

  /* use the output size we got when initializing seamcrop. */
  outsize = seamcrop->output_frame_size;

  if(outsize == 0)
    goto unknown_size;

  GST_DEBUG_OBJECT (seamcrop, "doing alloc of size %" G_GSIZE_FORMAT, outsize);
  *outbuf = gst_buffer_new_allocate (seamcrop->allocator, outsize, &seamcrop->params);

  if (!*outbuf) {
    ret = GST_FLOW_ERROR;
    goto alloc_failed;
  }

copy_meta:
  /* copy the metadata */
  if (!copy_metadata (seamcrop, inbuf, *outbuf)) {
    /* something failed, post a warning */
    GST_ELEMENT_WARNING (seamcrop, STREAM, NOT_IMPLEMENTED, ("could not copy metadata"), (NULL));
  }
  return GST_FLOW_OK;

  /* ERRORS */
activate_failed:
  {
    GST_ELEMENT_ERROR (seamcrop, RESOURCE, SETTINGS,
        ("failed to activate bufferpool"), ("failed to activate bufferpool"));
    return GST_FLOW_ERROR;
  }
unknown_size:
  {
    GST_ERROR_OBJECT (seamcrop, "unknown output size");
    return GST_FLOW_ERROR;
  }
alloc_failed:
  {
    GST_DEBUG_OBJECT (seamcrop, "could not allocate buffer from pool");
    return ret;
  }
no_outcaps:
  {
    GST_DEBUG_OBJECT (seamcrop, "no output caps, source pad has been deactivated");
    return GST_FLOW_FLUSHING;
  }
}


typedef struct
{
  GstSeamCrop *seamcrop;
  GstBuffer *outbuf;
} CopyMetaData;

  static gboolean
foreach_metadata (GstBuffer * inbuf, GstMeta ** meta, gpointer user_data)
{
  CopyMetaData *data = user_data;
  GstSeamCrop *seamcrop = data->seamcrop;
  const GstMetaInfo *info = (*meta)->info;
  GstBuffer *outbuf = data->outbuf;
  gboolean do_copy = FALSE;

  if (gst_meta_api_type_has_tag (info->api, _gst_meta_tag_memory)) {
    /* never call the transform_meta with memory specific metadata */
    GST_DEBUG_OBJECT (seamcrop, "not copying memory specific metadata %s",
        g_type_name (info->api));
    do_copy = FALSE;
  } else {
    do_copy = gst_seamcrop_transform_meta (seamcrop, outbuf, *meta, inbuf);
    GST_DEBUG_OBJECT (seamcrop, "transformed metadata %s: copy: %d", g_type_name (info->api), do_copy);
  }

  /* we only copy metadata when the subclass implemented a transform_meta
   * function and when it returns %TRUE */
  if (do_copy) {
    GstMetaTransformCopy copy_data = { FALSE, 0, -1 };
    GST_DEBUG_OBJECT (seamcrop, "copy metadata %s", g_type_name (info->api));
    /* simply copy then */
    info->transform_func (outbuf, *meta, inbuf, _gst_meta_transform_copy, &copy_data);
  }
  return TRUE;
}

  static gboolean 
copy_metadata (GstSeamCrop * seamcrop, GstBuffer * inbuf, GstBuffer * outbuf)
{
  CopyMetaData data;

  /* now copy the metadata */
  GST_DEBUG_OBJECT (seamcrop, "copying metadata");

  /* this should not happen, buffers allocated from a pool or with
   * new_allocate should always be writable. */
  if (!gst_buffer_is_writable (outbuf))
    goto not_writable;

  /* when we get here, the metadata should be writable */
  gst_buffer_copy_into (outbuf, inbuf,
      GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS, 0, -1);

  /* clear the GAP flag when the subclass does not understand it */
  if (!seamcrop->gap_aware)
    GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_GAP);

  data.seamcrop = seamcrop;
  data.outbuf = outbuf;

  gst_buffer_foreach_meta (inbuf, foreach_metadata, &data);

  return TRUE;

  /* ERRORS */
not_writable:
  {
    GST_WARNING_OBJECT (seamcrop, "buffer %p not writable", outbuf);
    return FALSE;
  }
}

/** Replaces base_transform_sink_eventfunc **/
  static gboolean
gst_seamcrop_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  gboolean ret = TRUE, forward = TRUE;

  GstSeamCrop * seamcrop;
  seamcrop = GST_SEAMCROP(parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_START:
      g_print("Flush starting.\n");
      if(seamcrop->started)
      {
        /* Set element to flushing internally. */
        GST_OBJECT_LOCK (seamcrop);
        seamcrop->flushing = TRUE;
        seamcrop->started = FALSE;
        GST_OBJECT_UNLOCK (seamcrop);

        /* Flush the retargeting module. */
        flushCurrentInstance();
        GstBuffer *outbuf;

        /* Force dispatch thread to quit in case it has not. */
        gst_buffer_pool_acquire_buffer (seamcrop->pool, &outbuf, NULL);
        g_async_queue_push(seamcrop->output_queue, outbuf);
        g_thread_join(seamcrop->push_thread);
        gst_buffer_unref(outbuf);

        /* Clear thread and queue. */
        g_thread_unref(seamcrop->push_thread);
        g_async_queue_unref(seamcrop->output_queue);
        seamcrop->output_queue = g_async_queue_new();

      }
      break;
    case GST_EVENT_FLUSH_STOP:

      g_print("Flush stopping.\n");

      /* reset QoS parameters */
      GST_OBJECT_LOCK (seamcrop);
      seamcrop->proportion = 1.0;
      seamcrop->earliest_time = -1;
      seamcrop->discont = FALSE;
      seamcrop->processed = 0;
      seamcrop->dropped = 0;
      seamcrop->flushing = FALSE;
      GST_OBJECT_UNLOCK (seamcrop);

      seamcrop->have_segment = FALSE;
      gst_segment_init (&seamcrop->segment, GST_FORMAT_UNDEFINED);
      seamcrop->position_out = GST_CLOCK_TIME_NONE;
      break;
    case GST_EVENT_EOS:
      {
        signalEndOfStream();
        break;
      }
    case GST_EVENT_TAG:
      break;
    case GST_EVENT_CAPS:
      {
        GstCaps *caps;

        GST_INFO_OBJECT(seamcrop, "received EVENT_CAPS %" GST_PTR_FORMAT,
            &event);
        gst_event_parse_caps (event, &caps);
        /* clear any pending reconfigure flag */
        gst_pad_check_reconfigure (seamcrop->srcpad);
        ret = gst_seamcrop_setcaps (seamcrop, seamcrop->sinkpad, caps);

        read_video_props(seamcrop,caps);

        // No need to reinitialize if size is unchanged.
        if(seamcrop->reinitialize_module)
        {
          if(seamcrop->started)
          {
            if(seamcrop->lastRetVal == GST_FLOW_FLUSHING)
            {
              // We need to adapt to the new caps configuration after a flush.
              flushCurrentInstance();

              GST_OBJECT_LOCK(seamcrop);
              seamcrop->started = FALSE;
              seamcrop->lastRetVal = GST_FLOW_OK;
              g_async_queue_unref(seamcrop->output_queue);
              seamcrop->output_queue = g_async_queue_new();
              GST_OBJECT_UNLOCK(seamcrop);
            } else {
              // Caps are being changed without flushing.
              signalEndOfStream();
              //flushCurrentInstance();
              seamcrop->output_frame_size =
                initSeamCrop(seamcrop->output_queue, seamcrop->input_width, 
                    seamcrop->input_height, seamcrop->input_framerate,
                    seamcrop->retargeting_factor, seamcrop->extend_border_factor,
                    seamcrop->frame_window_size);
              seamcrop->reinitialize_module = FALSE;
            }
          }
        }

        forward = FALSE;
        break;
      }
    case GST_EVENT_SEGMENT:
      {
        gst_event_copy_segment (event, &seamcrop->segment);
        seamcrop->have_segment = TRUE;

        GST_DEBUG_OBJECT (seamcrop, "received SEGMENT %" GST_SEGMENT_FORMAT,
            &seamcrop->segment);
        GST_DEBUG_OBJECT (seamcrop, "Initializing SeamCropCuda.");

        if(seamcrop->started) // In case we're already running. Reset.
          signalEndOfStream();
        //flushCurrentInstance();

        if(seamcrop->measurement_mode) {
          g_print("Initializing: New segment\n");
          setMeasurementClock(seamcrop->measure_clock);
        }

        /* Initialize seamCropCuda. */
        seamcrop->output_frame_size =
          initSeamCrop(seamcrop->output_queue, seamcrop->input_width, 
              seamcrop->input_height, seamcrop->input_framerate,
              seamcrop->retargeting_factor, seamcrop->extend_border_factor,
              seamcrop->frame_window_size);

        GST_OBJECT_LOCK (seamcrop);
        seamcrop->started = TRUE;
        seamcrop->flushing = FALSE;
        seamcrop->reinitialize_module = FALSE;
        GST_OBJECT_UNLOCK (seamcrop);

        /* Start thread for dispatching output. */
        seamcrop->push_thread = g_thread_new("Dispatcher", (GThreadFunc)gst_seamcrop_push_output, seamcrop);
        break;
      }
    default:
      break;
  }

  if (ret && forward)
    ret = gst_pad_push_event (seamcrop->srcpad, event);
  else
    gst_event_unref (event);

  return ret;
}

  static gboolean
gst_seamcrop_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  gboolean ret;
  GstSeamCrop * seamcrop;
  seamcrop = GST_SEAMCROP(parent);


  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
      break;
    case GST_EVENT_NAVIGATION:
      break;
    case GST_EVENT_QOS:
      {
        gdouble proportion;
        GstClockTimeDiff diff;
        GstClockTime timestamp;

        gst_event_parse_qos (event, NULL, &proportion, &diff, &timestamp);
        gst_seamcrop_update_qos (seamcrop, proportion, diff, timestamp);
        break;
      }
    default:
      break;
  }

  ret = gst_pad_push_event (seamcrop->sinkpad, event);

  return ret;
}


/* Takes the input buffer */
  static GstFlowReturn
submit_input_buffer (GstSeamCrop * seamcrop, gboolean is_discont,
    GstBuffer * inbuf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstClockTime running_time;
  GstClockTime timestamp;

  if (G_UNLIKELY (!gst_seamcrop_reconfigure (seamcrop)))
    goto not_negotiated;

  if (GST_BUFFER_OFFSET_IS_VALID (inbuf))
    GST_DEBUG_OBJECT (seamcrop,
        "handling buffer %p of size %" G_GSIZE_FORMAT ", PTS %" GST_TIME_FORMAT
        " and offset %" G_GUINT64_FORMAT, inbuf, gst_buffer_get_size (inbuf),
        GST_TIME_ARGS (GST_BUFFER_PTS (inbuf)), GST_BUFFER_OFFSET (inbuf));
  else
    GST_DEBUG_OBJECT (seamcrop,
        "handling buffer %p of size %" G_GSIZE_FORMAT ", PTS %" GST_TIME_FORMAT
        " and offset NONE", inbuf, gst_buffer_get_size (inbuf),
        GST_TIME_ARGS (GST_BUFFER_PTS (inbuf)));

  /* Don't allow buffer handling before negotiation */
  if (!seamcrop->negotiated)
    goto not_negotiated;

  /* can only do QoS if the segment is in TIME */
  if (seamcrop->segment.format != GST_FORMAT_TIME)
    goto no_qos;

  /* QOS is done on the running time of the buffer, get it now */
  timestamp = GST_BUFFER_TIMESTAMP (inbuf);
  running_time = gst_segment_to_running_time (&seamcrop->segment, GST_FORMAT_TIME, timestamp);

  if (running_time != -1) {
    gboolean need_skip;
    GstClockTime earliest_time;
    gdouble proportion;
    /* lock for getting the QoS parameters that are set (in a different thread)
     * with the QOS events */
    GST_OBJECT_LOCK (seamcrop);
    earliest_time = seamcrop->earliest_time;
    proportion = seamcrop->proportion;
    /* check for QoS, don't perform conversion for buffers
     * that are known to be late. */
    need_skip = seamcrop->qos_enabled && earliest_time != -1 && running_time <= earliest_time;
    GST_OBJECT_UNLOCK (seamcrop);

    if (need_skip) {
      GstMessage *qos_msg;
      GstClockTime duration;
      guint64 stream_time;
      gint64 jitter;

      GST_DEBUG_OBJECT (seamcrop, "skipping transform");
      //  Gives compiler error due to no GST_CAT_QOS (defined in gst_private.h).
      /*
         GST_CAT_DEBUG_OBJECT (GST_CAT_QOS, seamcrop, "skipping transform: qostime %"
         GST_TIME_FORMAT " <= %" GST_TIME_FORMAT,
         GST_TIME_ARGS (running_time), GST_TIME_ARGS (earliest_time));
         */
      seamcrop->dropped++;

      duration = GST_BUFFER_DURATION (inbuf);
      stream_time = gst_segment_to_stream_time (&seamcrop->segment, GST_FORMAT_TIME, timestamp);
      jitter = GST_CLOCK_DIFF (running_time, earliest_time);

      qos_msg = gst_message_new_qos (GST_OBJECT_CAST (seamcrop), FALSE, running_time, stream_time, timestamp, duration);
      gst_message_set_qos_values (qos_msg, jitter, proportion, 1000000);
      gst_message_set_qos_stats (qos_msg, GST_FORMAT_BUFFERS, seamcrop->processed, seamcrop->dropped);
      gst_element_post_message (GST_ELEMENT_CAST (seamcrop), qos_msg);

      //g_print("Need skip.\n");

      /* mark discont for next buffer */
      seamcrop->discont = TRUE;
      ret = GST_SEAMCROP_BUFFER_PASSED;
      goto skip;
    }
  }

no_qos:
  /* Stash input buffer where the default generate_output
   * function can find it */
  if (seamcrop->queued_buf)
    gst_buffer_unref (seamcrop->queued_buf);
  seamcrop->queued_buf = inbuf;
  return ret;
skip:
  gst_buffer_unref (inbuf);
  return ret;

not_negotiated:
  {
    gst_buffer_unref (inbuf);
    if (GST_PAD_IS_FLUSHING (seamcrop->srcpad))
      return GST_FLOW_FLUSHING;
    return GST_FLOW_NOT_NEGOTIATED;
  }
}


  static GstFlowReturn
generate_output (GstSeamCrop * seamcrop, GstBuffer ** outbuf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *inbuf;

  /* Retrieve stashed input buffer, if the default submit_input_buffer
   * was run. Takes ownership back from there */
  inbuf = seamcrop->queued_buf;
  seamcrop->queued_buf = NULL;

  /* This default processing method needs one input buffer to feed to
   * the transform functions, we can't do anything without it */
  if (inbuf == NULL)
    return GST_FLOW_OK;

  /* first try to allocate an output buffer based on the currently negotiated
   * format. outbuf will contain a buffer suitable for doing the configured
   * transform after this function. */
  GST_DEBUG_OBJECT (seamcrop, "calling prepare buffer");
  ret = prepare_output_buffer (seamcrop, inbuf, outbuf);

  if (ret != GST_FLOW_OK || *outbuf == NULL)
    goto no_buffer;

  GST_DEBUG_OBJECT (seamcrop, "using allocated buffer in %p, out %p", inbuf, *outbuf);

  /* now perform the needed transform */
  GST_DEBUG_OBJECT (seamcrop, "performing retargeting");

  ret = gst_seamcrop_transform (seamcrop, inbuf, *outbuf);

  /* only unref input buffer if we allocated a new outbuf buffer. If we reused
   * the input buffer, no refcount is changed to keep the input buffer writable
   * when needed. */
  if (*outbuf != inbuf)
    gst_buffer_unref (inbuf);

  return ret;

  /* ERRORS */
no_buffer:
  {
    gst_buffer_unref (inbuf);
    *outbuf = NULL;
    GST_WARNING_OBJECT (seamcrop, "could not get buffer from pool: %s",
        gst_flow_get_name (ret));
    return ret;
  }
}


/* The flow of the chain function is the reverse of the
 * getrange() function - we iterate, pushing buffers it generates until it either
 * wants more data or returns an error */
  static GstFlowReturn
gst_seamcrop_sink_chain (GstPad * pad, GstObject * parent, GstBuffer * buffer)
{
  GstSeamCrop *seamcrop = GST_SEAMCROP (parent);
  GstFlowReturn ret;
  GstBuffer *outbuf = NULL;

  if(seamcrop->measurement_mode)
    g_print("RECV %lu\n",GST_TIME_AS_MSECONDS(gst_clock_get_internal_time(seamcrop->measure_clock)));

  /* Takes ownership of input buffer */
  ret = submit_input_buffer (seamcrop, seamcrop->discont, buffer);

  if (ret != GST_FLOW_OK)
    goto done;

  ret = generate_output (seamcrop, &outbuf);

done:
  // convert internal flow to OK and mark discont for the next buffer. 
  if (ret == GST_SEAMCROP_BUFFER_PASSED) {
    GST_DEBUG_OBJECT (seamcrop, "Buffer passed to SeamCrop");
    seamcrop->discont = TRUE;
    ret = GST_FLOW_OK;
  }

  return ret;
}

  static void
gst_seamcrop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSeamCrop *seamcrop;
  seamcrop = GST_SEAMCROP (object);

  GST_DEBUG_OBJECT (seamcrop, "set_property");

  switch (prop_id) {
    case PROP_QOS:
      gst_seamcrop_set_qos_enabled (seamcrop, g_value_get_boolean (value));
      break;
    case PROP_EXTEND:
      seamcrop->extend_border_factor = g_value_get_float(value);
      break;
    case PROP_RETARGET:
      seamcrop->retargeting_factor = g_value_get_float(value);
      break;
    case PROP_WINDOW_SIZE:
      seamcrop->frame_window_size = g_value_get_int(value);
      break;
    case PROP_MEASUREMENT:
      seamcrop->measurement_mode = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

  static void
gst_seamcrop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstSeamCrop *seamcrop;
  seamcrop = GST_SEAMCROP (object);

  GST_DEBUG_OBJECT (seamcrop, "get_property");

  switch (prop_id) {
    case PROP_QOS:
      g_value_set_boolean(value, gst_seamcrop_is_qos_enabled (seamcrop));
      break;
    case PROP_EXTEND:
      g_value_set_float(value, seamcrop->extend_border_factor);
      break;
    case PROP_RETARGET:
      g_value_set_float(value, seamcrop->retargeting_factor);
      break;
    case PROP_WINDOW_SIZE:
      g_value_set_int(value, seamcrop->frame_window_size);
      break;
    case PROP_MEASUREMENT:
      g_value_set_boolean(value, seamcrop->measurement_mode);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* not a vmethod of anything, just an internal method */
  static gboolean
gst_seamcrop_activate (GstSeamCrop * seamcrop, gboolean active)
{

  gboolean result = TRUE;

  GST_DEBUG_OBJECT (seamcrop, "activate");
  /**** STARTUP ****/
  if (active) {
    GstCaps *incaps, *outcaps;

    incaps = gst_pad_get_current_caps (seamcrop->sinkpad);
    outcaps = gst_pad_get_current_caps (seamcrop->srcpad);

    GST_OBJECT_LOCK (seamcrop);

    GST_DEBUG_OBJECT (seamcrop, "activate - incaps: %" GST_PTR_FORMAT, incaps);
    GST_DEBUG_OBJECT (seamcrop, "activate - outcaps: %" GST_PTR_FORMAT, outcaps);

    if (incaps && outcaps)
      seamcrop->have_same_caps = gst_caps_is_equal (incaps, outcaps);

    GST_DEBUG_OBJECT (seamcrop, "have_same_caps %d", seamcrop->have_same_caps);
    seamcrop->negotiated = FALSE;
    seamcrop->have_segment = FALSE;
    gst_segment_init (&seamcrop->segment, GST_FORMAT_UNDEFINED);
    seamcrop->position_out = GST_CLOCK_TIME_NONE;
    seamcrop->proportion = 1.0;
    seamcrop->earliest_time = -1;
    seamcrop->discont = FALSE;
    seamcrop->processed = 0;
    seamcrop->dropped = 0;
    GST_OBJECT_UNLOCK (seamcrop);

    if (incaps)
      gst_caps_unref (incaps);
    if (outcaps)
      gst_caps_unref (outcaps);
    /**** SHUTDOWN ****/
  } else {
    /* We must make sure streaming has finished before resetting things
     * and calling the ::stop vfunc */
    GST_PAD_STREAM_LOCK (seamcrop->sinkpad);
    GST_PAD_STREAM_UNLOCK (seamcrop->sinkpad);

    seamcrop->have_same_caps = FALSE;

    gst_caps_replace (&seamcrop->cache_caps1, NULL);
    gst_caps_replace (&seamcrop->cache_caps2, NULL);

    /* Make sure any left over buffer is freed */
    gst_buffer_replace (&seamcrop->queued_buf, NULL);

    gst_seamcrop_set_allocation (seamcrop, NULL, NULL, NULL, NULL);
  }

  return result;
}

  static gboolean
gst_seamcrop_sink_activate_mode (GstPad * pad, GstObject * parent,
    GstPadMode mode, gboolean active)
{
  gboolean result = FALSE;
  GstSeamCrop *seamcrop;

  seamcrop = GST_SEAMCROP (parent);

  switch (mode) {
    case GST_PAD_MODE_PUSH:
      {
        result = gst_seamcrop_activate (seamcrop, active);

        if (result)
          seamcrop->pad_mode = active ? GST_PAD_MODE_PUSH : GST_PAD_MODE_NONE;
        break;
      }
    default:
      result = TRUE;
      break;
  }
  return result;
}

  static gboolean
gst_seamcrop_src_activate_mode (GstPad * pad, GstObject * parent,
    GstPadMode mode, gboolean active)
{
  gboolean result = FALSE;
  GstSeamCrop *seamcrop;

  seamcrop = GST_SEAMCROP (parent);

  switch (mode) {
    case GST_PAD_MODE_PULL:
      {
        result = gst_pad_activate_mode (seamcrop->sinkpad, GST_PAD_MODE_PULL, active);

        if (result)
          result &= gst_seamcrop_activate (seamcrop, active);

        if (result)
          seamcrop->pad_mode = active ? mode : GST_PAD_MODE_NONE;
        break;
      }
    default:
      result = TRUE;
      break;
  }

  return result;
}

/**
 *  gst_seamcrop_update_qos:
 *  @seamcrop: a #GstSeamCrop
 *  @proportion: the proportion
 *  @diff: the diff against the clock
 *  @timestamp: the timestamp of the buffer generating the QoS expressed in
 *  running_time.
 *  
 *  Set the QoS parameters in the transform. This function is called internally
 *  when a QOS event is received but subclasses can provide custom information
 *  when needed.
 *  
 *  MT safe.
 */
  void
gst_seamcrop_update_qos (GstSeamCrop * seamcrop,
    gdouble proportion, GstClockTimeDiff diff, GstClockTime timestamp)
{
  g_return_if_fail (GST_IS_SEAMCROP (seamcrop));

  GST_DEBUG_OBJECT (seamcrop, "updating qos");
  //  Gives compiler error due to no GST_CAT_QOS (defined in gst_private.h).
  /*
     GST_CAT_DEBUG_OBJECT (GST_CAT_QOS, seamcrop,
     "qos: proportion: %lf, diff %" G_GINT64_FORMAT ", timestamp %"
     GST_TIME_FORMAT, proportion, diff, GST_TIME_ARGS (timestamp));
     */
  GST_OBJECT_LOCK (seamcrop);
  seamcrop->proportion = proportion;
  seamcrop->earliest_time = timestamp + diff;
  GST_OBJECT_UNLOCK (seamcrop);
}

/**
 *  gst_seamcrop_set_qos_enabled:
 *  @seamcrop: a #GstSeamCrop
 *  @enabled: new state
 *  
 *  Enable or disable QoS handling in the transform.
 *  
 *  MT safe.
 */
  void
gst_seamcrop_set_qos_enabled (GstSeamCrop * seamcrop, gboolean enabled)
{
  g_return_if_fail (GST_IS_SEAMCROP (seamcrop));

  GST_DEBUG_OBJECT (seamcrop, "enabling qos");
  // Gives compiler error due to no GST_CAT_QOS (defined in gst_private.h).
  //GST_CAT_DEBUG_OBJECT (GST_CAT_QOS, seamcrop, "enabled: %d", enabled);

  GST_OBJECT_LOCK (seamcrop);
  seamcrop->qos_enabled = enabled;
  GST_OBJECT_UNLOCK (seamcrop);
}

/**
 *  gst_seamcrop_is_qos_enabled:
 *  @seamcrop: a #GstSeamCrop
 *  
 *  Queries if the transform will handle QoS.
 *  
 *  Returns: %TRUE if QoS is enabled.
 *  
 *  MT safe.
 */
  gboolean
gst_seamcrop_is_qos_enabled (GstSeamCrop * seamcrop)
{
  gboolean result;

  g_return_val_if_fail (GST_IS_SEAMCROP (seamcrop), FALSE);

  GST_OBJECT_LOCK (seamcrop);
  result = seamcrop->qos_enabled;
  GST_OBJECT_UNLOCK (seamcrop);

  return result;
}

  void
gst_seamcrop_finalize (GObject * object)
{
  GstSeamCrop *seamcrop = GST_SEAMCROP (object);

  GST_DEBUG_OBJECT (seamcrop, "finalize");

  //seamcrop->measure_clock

  // In case of a sudden quit.
  signalEndOfStream();
  //flushCurrentInstance();
  /* clean up object here */

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static GstFlowReturn gst_seamcrop_transform (GstSeamCrop * seamcrop,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstClockTime added;
  // Pass buffers to seamCropWrapper to begin the transformation.
  //passBuffers(inbuf, outbuf);
  added = passBuffers(inbuf, outbuf);

  if(seamcrop->measurement_mode)
    g_print("ADD %lu\n",GST_TIME_AS_MSECONDS(added));

  return GST_FLOW_OK;
}

/* Initialized when a new segment is received from an upstream element.
 * Continually pushes output buffers as they become available from the seamcrop module
 * or until flow return value differs from GST_FLOW_OK.
 */
static void gst_seamcrop_push_output (GstSeamCrop * seamcrop)
{
  GstBuffer *outbuf;

  GstFlowReturn ret;
  GstClockTime position = GST_CLOCK_TIME_NONE;
  GstClockTime timestamp, duration;

  ret = GST_FLOW_OK;

  do {
    outbuf = NULL;

    /* Retrieve an item from the queue. Blocks until one arrives from the seamcrop module. */
    outbuf = g_async_queue_pop(seamcrop->output_queue);

    /* Are we flushing? */
    if(seamcrop->flushing)
    {
      gst_buffer_unref(outbuf);
      break;
    }

    ret = GST_FLOW_OK;

    timestamp = GST_BUFFER_TIMESTAMP (outbuf);
    duration = GST_BUFFER_DURATION (outbuf);

    /* calculate end position of the outgoing buffer */
    if (timestamp != GST_CLOCK_TIME_NONE) {
      if (duration != GST_CLOCK_TIME_NONE)
        position = timestamp + duration;
      else
        position = timestamp;
    }

    /* Set discont flag so we can mark the outgoing buffer */
    if (GST_BUFFER_IS_DISCONT (outbuf)) {
      GST_DEBUG_OBJECT (seamcrop, "got DISCONT buffer %p", outbuf);
      seamcrop->discont = TRUE;
    }

    GST_DEBUG_OBJECT (seamcrop, "push_output: outbuf %" GST_PTR_FORMAT, outbuf);

    /* sanity check for a null-buffer. */
    if (outbuf != NULL) {
      if (ret == GST_FLOW_OK) {
        GstClockTime position_out = GST_CLOCK_TIME_NONE;

        // Remember last stop position 
        if (position != GST_CLOCK_TIME_NONE && seamcrop->segment.format == GST_FORMAT_TIME)
          seamcrop->segment.position = position;

        if (GST_BUFFER_TIMESTAMP_IS_VALID (outbuf)) {
          position_out = GST_BUFFER_TIMESTAMP (outbuf);
          if (GST_BUFFER_DURATION_IS_VALID (outbuf))
            position_out += GST_BUFFER_DURATION (outbuf);
        } else if (position != GST_CLOCK_TIME_NONE) {
          position_out = position;
        }
        if (position_out != GST_CLOCK_TIME_NONE && seamcrop->segment.format == GST_FORMAT_TIME)
          seamcrop->position_out = position_out;

        // apply DISCONT flag if the buffer is not yet marked as such
        if (seamcrop->discont) {
          GST_DEBUG_OBJECT (seamcrop, "we have a pending DISCONT");
          if (!GST_BUFFER_IS_DISCONT (outbuf)) {
            GST_DEBUG_OBJECT (seamcrop, "marking DISCONT on output buffer");
            outbuf = gst_buffer_make_writable (outbuf);
            GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DISCONT);
          }
          seamcrop->discont = FALSE;
        }
        seamcrop->processed++;

        GST_DEBUG_OBJECT (seamcrop, "push_output: PUSHING %" GST_PTR_FORMAT, outbuf);

        if(seamcrop->measurement_mode)
          g_print("SEND %lu\n",GST_TIME_AS_MSECONDS(gst_clock_get_internal_time(seamcrop->measure_clock)));
        ret = gst_pad_push (seamcrop->srcpad, outbuf);
      } else {

        GST_DEBUG_OBJECT (seamcrop, "We got return %s", gst_flow_get_name (ret));
        gst_buffer_unref (outbuf);
      }
    }
  } while ((ret == GST_FLOW_OK && outbuf != NULL) || 
      (ret == GST_FLOW_EOS && outbuf != NULL && g_async_queue_length(seamcrop->output_queue) > 0));
  GST_DEBUG_OBJECT (seamcrop, "OUTPUT THREAD EXIT %" GST_PTR_FORMAT, outbuf);
  //g_print("Dispatch thread exiting. Retval: %s\n",gst_flow_get_name(ret));

  seamcrop->lastRetVal = ret;
}

  static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "seamcrop", GST_RANK_NONE,
      GST_TYPE_SEAMCROP);
}

#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "seamcrop"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "gstseamcrop-plugin"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://Dreggs@bitbucket.org/Dreggs/gstreamer-paraseamcrop.git"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    seamcrop,
    "A GPU accelerated video retargeting plugin using the SeamCrop algorithm.",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

