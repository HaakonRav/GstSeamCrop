#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

/* Compile with:
 * libtool --mode=link gcc -ggdb `pkg-config --cflags --libs gstreamer-1.0 glib-2.0` -o evaluationseamcrop evaluationseamcrop.c
 */
static GstElement *pipeline, *source, *decoder, *que, *que2, *seam, *sink, *scale, *capfilt;

static int file_variations[3];
static char *scale_variations[8];
static int current_file;
static int num_changes;
static int runs;
static gboolean initial_latency;

enum
{
  WANT_FILE,
  WANT_URI,
  WANT_DISPLAY,
  NO_DISPLAY
};

#define DEFAULT_SINK        "xvimagesink"
#define DEFAULT_FILE        "/home/dreggs/Master/src/big_buck_bunny_360p.mp4"
#define DEFAULT_DECODER     "decodebin"
#define DEFAULT_SCALE       "video/x-raw,width=512,height=288"
#define DEFAULT_FREQUENCY   90
#define DEFAULT_WINDOW_SIZE 100
#define DEFAULT_RET_FACTOR  0.75f
#define DEFAULT_EXT_FACTOR  0.25f
#define DEFAULT_MEASUREMENT FALSE
#define DEFAULT_DYNAMIC     FALSE
#define DEFAULT_SYNC        TRUE

#define HD_QUALITY          720
#define SD_QUALITY          480
#define MIN_QUALITY         360

#define SCALE_HD            "video/x-raw,width=1280,height=720"
#define SCALE_SD            "video/x-raw,width=854,height=480"
#define SCALE_MIN           "video/x-raw,width=640,height=360"
#define MEASUREMENT_RUNS    2

GstElement *video;

/* Function used to dynamically alter the input to the pipeline in this application. */
static gboolean change_input (gpointer data)
{
  if(num_changes == runs)
  {
    // Close the pipeline.
    GMainLoop *loop = (GMainLoop *) data;
    GstMessage *msg;

    gst_element_send_event(pipeline, gst_event_new_eos());
    msg = gst_bus_timed_pop_filtered(GST_ELEMENT_BUS(pipeline), GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS | GST_MESSAGE_ERROR);

    gst_message_unref(msg);

    g_main_loop_quit(loop);

    return FALSE;
  }

  gint ret;
  char filename[100];
  
  gst_element_set_state(pipeline,GST_STATE_READY);

  if(initial_latency) {
    if(num_changes % 10 == 0 && num_changes != 0) {
      g_print("Changing file\n");
      current_file = current_file + 1; 
      sprintf(filename,"/home/dreggs/Master/src/big_buck_bunny_%dp.mp4",file_variations[current_file]);
      g_object_set(G_OBJECT(source), "location", filename, NULL);
    }
  } else {
    // Setting new source.
    sprintf(filename,"/home/dreggs/Master/src/big_buck_bunny_%dp.mp4",file_variations[current_file]);
    current_file = (current_file == 2) ? 0 : current_file + 1; 
    g_object_set(G_OBJECT(source), "location", filename, NULL);
  }

  gst_element_set_state(pipeline,GST_STATE_PLAYING);

  num_changes += 1;

  return TRUE;
}

static gboolean force_resolution(gpointer data)
{
  if(num_changes == runs)
  {
    // Close the pipeline.
    GMainLoop *loop = (GMainLoop *) data;

    gst_element_set_state(pipeline,GST_STATE_READY);
    gst_element_set_state(pipeline,GST_STATE_NULL);
    g_main_loop_quit(loop);

    return FALSE;
  }

  gst_element_set_state(pipeline,GST_STATE_PAUSED);

  g_object_set(G_OBJECT(capfilt), "caps", gst_caps_from_string(scale_variations[num_changes]), NULL);

  num_changes += 1;
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  return TRUE;
}

static gboolean close_pipeline(gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    GstMessage *msg;

    gst_element_send_event(pipeline, gst_event_new_eos());
    msg = gst_bus_timed_pop_filtered(GST_ELEMENT_BUS(pipeline), GST_CLOCK_TIME_NONE, GST_MESSAGE_EOS | GST_MESSAGE_ERROR);

    gst_message_unref(msg);

    g_main_loop_quit(loop);
    
    return FALSE;
}

/*
static gboolean set_bandwidth ()
{
  g_print("BANDWIDTH: %d\n", bandwidth[current_bandwidth]); 
  char command[100];

  sprintf(command, "wondershaper enp8s0 %d %d", bandwidth[current_bandwidth], (bandwidth[current_bandwidth] / 2));
  system(command);

  current_bandwidth = (current_bandwidth == 2) ? 0 : current_bandwidth + 1;
}
*/
static gboolean bus_call (GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;

  switch (GST_MESSAGE_TYPE (msg)) {

    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;

    case GST_MESSAGE_ERROR: 
      {
        gchar  *debug;
        GError *error;

        gst_message_parse_error (msg, &error, &debug);
        g_free (debug);

        g_printerr ("Error: %s\n", error->message);
        g_error_free (error);

        g_main_loop_quit (loop);
        break;
      }
    default:
      break;
  }

  return TRUE;
}


static void on_pad_added (GstElement *decodebin, GstPad *pad, gpointer data)
{
  GstCaps *caps;
  GstStructure *str;
  GstPad *videopad;

  /* only link once */
  videopad = gst_element_get_static_pad (video, "sink");
  if (GST_PAD_IS_LINKED (videopad)) {
    g_object_unref (videopad);
    return;
  }

  /* check media type */
  caps = gst_pad_query_caps (pad, NULL);
  str = gst_caps_get_structure (caps, 0);
  if (!g_strrstr (gst_structure_get_name (str), "video")) {
    gst_caps_unref (caps);
    gst_object_unref (videopad);
    return;
  }
  gst_caps_unref (caps);

  /* link'n'play */
  gst_pad_link (pad, videopad);

  g_object_unref (videopad); 
}



int main (int argc, char *argv[])
{

  GMainLoop *loop;
  GstPad *videopad;
  GstBus *bus;
  guint bus_watch_id;

  file_variations[0]  = MIN_QUALITY;
  file_variations[1]  = SD_QUALITY;
  file_variations[2]  = HD_QUALITY;
  
  scale_variations[0] = SCALE_SD;
  scale_variations[1] = SCALE_HD;
  scale_variations[2] = SCALE_SD;
  scale_variations[3] = SCALE_HD;
  scale_variations[4] = SCALE_MIN;
  scale_variations[5] = SCALE_HD;
  scale_variations[6] = SCALE_SD;
  scale_variations[7] = SCALE_MIN;

  initial_latency     = FALSE;
  current_file        = 1;
  num_changes         = 0;
  runs                = MEASUREMENT_RUNS;
  int opt_index       = 0, c = 0;
  char *scale_str     = DEFAULT_SCALE;
  char *decoder_name  = DEFAULT_DECODER;
  char *sink_name     = DEFAULT_SINK;
  char *filename      = DEFAULT_FILE;
  int display         = WANT_DISPLAY;
  int decoder_option  = WANT_FILE;
  int window_size     = DEFAULT_WINDOW_SIZE;
  int frequency       = DEFAULT_FREQUENCY;
  float ret_factor    = DEFAULT_RET_FACTOR, 
        ext_factor    = DEFAULT_EXT_FACTOR;
  gboolean measure    = DEFAULT_MEASUREMENT;
  gboolean sync       = DEFAULT_SYNC;
  gboolean dynamic    = DEFAULT_DYNAMIC;

  /* Initialisation */
  gst_init (&argc, &argv);

  loop = g_main_loop_new (NULL, FALSE);

  /* Parse options. */
  while(c != -1)
  {
    static struct option long_options[] =
    {
      {"file",      required_argument, 0, 'f'},
      {"uri",       required_argument, 0, 'u'},
      {"retarget",  required_argument, 0, 'r'},
      {"extend",    required_argument, 0, 'e'},
      {"size",      required_argument, 0, 's'},
      {"scale",     required_argument, 0, 'c'},
      {"frequency", required_argument, 0, 'd'},
      {"runs",      required_argument, 0, 'n'},
      {"measure",   no_argument,       0, 'm'},
      {"no-sync",   no_argument,       0, 'p'},
      {"initial",   no_argument,       0, 'i'},
      {"dynamic",   no_argument,       0, 't'}
    };

    c = getopt_long(argc, argv, "f:r:e:s:c:m:p:t", long_options, &opt_index);

    if(c == -1)
      break;

    switch(c)
    {
      case 'f':
        filename = optarg;
        break;
      case 'u':
        filename = optarg;
        decoder_name = "uridecodebin";
        decoder_option = WANT_URI;
        break;
      case 'r':
        ret_factor = strtof(optarg, NULL);
        break;
      case 'e':
        ext_factor = strtof(optarg, NULL);
        break;
      case 's':
        window_size = atoi(optarg);
        break;
      case 'c':
        scale = gst_element_factory_make ("videoscale", "scale");
        capfilt = gst_element_factory_make ("capsfilter","capsfilt");
        if(!scale || !capfilt)
        {
          g_printerr ("One element could not be created. Exiting.\n");
          return -1;
        }
        scale_str = optarg;
        GstCaps *scale_caps =  gst_caps_from_string(scale_str);
        g_assert(scale_caps);
        g_object_set (G_OBJECT (capfilt), "caps", scale_caps, NULL);
        gst_caps_unref(scale_caps);
        break;
      case 'm':
        measure = TRUE;
        break;
      case 'd':
        frequency = atoi(optarg);
        break;
      case 'n':
        runs = atoi(optarg) - 1;
        break;
      case 'p':
        sync = FALSE;
        break;
      case 't':
        dynamic = TRUE;
        break;
      case 'i':
        current_file = 0;
        initial_latency = TRUE;
        break;
      default:
        break;
    }
  }

  /* Create gstreamer elements. */
  pipeline  = gst_pipeline_new ("pipeline");
  que       = gst_element_factory_make ("queue",     "que");
  que2      = gst_element_factory_make ("queue",    "que2");
  seam      = gst_element_factory_make ("seamcrop",  "seamcrop");
  decoder   = gst_element_factory_make (decoder_name, "decoder");
  sink      = gst_element_factory_make (sink_name,   "video-output");

  if (!pipeline || !que || !que2 || !seam || !decoder || !sink) 
  {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Set seamcrop element properties. */
  g_object_set (G_OBJECT (seam), "retargeting-factor", ret_factor, NULL);
  g_object_set (G_OBJECT (seam), "extend-border", ext_factor, NULL);
  g_object_set (G_OBJECT (seam), "frame-window-size", window_size, NULL);
  g_object_set (G_OBJECT (seam), "measurement", measure, NULL);

  /* Disregard stream synchronization if requested. */
  if(sync == FALSE) 
    g_object_set (G_OBJECT (sink), "sync", sync, NULL); 

  /* Options for the decoder. Sets up pipeline and links accordingly. */
  switch (decoder_option)
  {
    case WANT_FILE:
      source = gst_element_factory_make ("filesrc", "file-source");
      if(!source)
      { 
        g_printerr("Filesrc could not be created. Exiting"); 
        return -1;
      }
      g_object_set (G_OBJECT (source), "location", filename, NULL);

      gst_bin_add_many(GST_BIN(pipeline), source, decoder, NULL);
      gst_element_link (source, decoder);
      break;
    case WANT_URI:
      g_object_set (G_OBJECT (decoder), "uri", filename, NULL); 
      gst_bin_add(GST_BIN(pipeline), decoder);
      break;
    default:
      break;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* set up pad callback connector. */
  g_signal_connect (decoder, "pad-added", G_CALLBACK (on_pad_added), NULL);

  /* set up videobin. */
  video = gst_bin_new("videobin");

  if(scale) {
    videopad = gst_element_get_static_pad(scale,"sink");
    gst_bin_add_many(GST_BIN(video), scale, capfilt, que, seam, que2, sink, NULL);

    // Link elements.
    gst_element_link(scale,capfilt);
    gst_element_link(capfilt,que);
    gst_element_link(que,seam);
    gst_element_link(seam,que2);
    gst_element_link(que2,sink);

  } else {
    videopad = gst_element_get_static_pad(que,"sink");
    gst_bin_add_many(GST_BIN(video), que, seam, que2, sink, NULL);

    // Link elements.
    gst_element_link(que,seam);
    gst_element_link(seam,que2);
    gst_element_link(que2,sink);
  }

  // Add a ghost pad to the video bin so that it can be used as an element.
  gst_element_add_pad(video,
      gst_ghost_pad_new("sink",videopad));

  gst_object_unref(videopad);

  // Add 'video' as element in 'pipeline'.
  gst_bin_add(GST_BIN(pipeline),video);

  /* Set the pipeline to "playing" state*/
  g_print ("Now playing.\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Timeout to change input resolution */
  if(dynamic)
    g_timeout_add_seconds (frequency, change_input, loop);
  else if(decoder_option == WANT_URI) 
    g_timeout_add_seconds (frequency, force_resolution, loop);

  /* Iterate */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);

  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  return 0;
}

