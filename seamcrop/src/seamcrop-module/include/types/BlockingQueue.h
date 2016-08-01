#ifndef TYPES_BLOCKING_QUEUE_H
#define TYPES_BLOCKING_QUEUE_H

#include <queue>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/condition_variable.hpp>


template<class T>
class BlockingQueue
{
 private:
  typedef boost::shared_ptr<T> Ptr;
  typedef boost::posix_time::milliseconds Milliseconds;

 public:
  BlockingQueue(){}

  // push a shared pointer of an object into the queue
  void push(Ptr obj)
  {
    // lock to be thread-safe
    boost::unique_lock<boost::mutex> lock(mutex);
    queue.push(obj);
    // notify a thread blocking on a pop()
    cond.notify_one();
  }

  // ms < 0: block until queue becomes non-empty
  // ms == 0: poll without waiting. Null pointer is returned if empty
  // ms > 0: block for 'ms' milliseconds before returning null if empty
  Ptr pop(int ms = -1)
  {
    // acquire lock on our mutex for thread-safety
    boost::unique_lock<boost::mutex> lock(mutex);
    
    // thread may be awakened spuriously (without push())
    // always check the queue again after waking up
    while (queue.empty())
      {
        if (ms < 0) // block indefinitely?
          cond.wait(lock); // wait for notify_one()
        else
          // wait for ms milliseconds.
          // timed_wait() returns false if not awakened by notify_one()
          if (!cond.timed_wait<Milliseconds>(lock, Milliseconds(ms)))
            return Ptr();
      }
    assert(!queue.empty());
    Ptr obj = queue.front();
    queue.pop();
    return obj;
  }
  
  bool empty()
  {
    boost::unique_lock<boost::mutex> lock(mutex);
    return queue.empty();
  }
  
 private:
  std::queue<Ptr> queue;
  boost::mutex mutex; 
  boost::condition_variable cond; // signaling between push() and pop()
};


#endif
