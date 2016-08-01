#ifndef TOOLS_THREAD_H
#define TOOLS_THREAD_H

#include <boost/thread/thread.hpp>


class Thread
{
  // only make thread-safe methods/attributes public in subclasses!
 public:
  Thread();
  virtual ~Thread();

  // starts a thread
  void start();

  // interrupts the thread and waits for it to terminate
  // will interrupt blocking on a BlockingQueue
  void stop();

  // main thread function. Don't call directly.
  // has to be public to be called by boost::thread
  void operator()();

 protected:
  // doOnce() is called once every time the thread is started
  virtual void doOnce();
 
  // doStuff() is called repeatedly in a loop until interrupted
  virtual void doStuff() = 0;

  // cleanup() is called once when interrupted
  virtual void cleanup();

 private:
  boost::thread thread; // boost implementation of a thread
  bool isRunning;
};


#endif
