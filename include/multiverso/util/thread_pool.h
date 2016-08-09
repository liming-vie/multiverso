#ifndef MULTIVERSO_UTIL_THREAD_POOL_H_
#define MULTIVERSO_UTIL_THREAD_POOL_H_

#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

#include "mt_queue.h"

namespace multiverso {

template <typename Task, typename TaskArg>
class ThreadPool {
private:
  template <typename Task, typename TaskArg>
  class ThreadInstance {
  public:
    ThreadInstance(ThreadPool<Task, TaskArg>* pool) :
      waiter_(nullptr), working_(false), pool_(pool) { }

    ~ThreadInstance() {
      Join();
    }

    void Run() {
      thread_.reset(new std::thread(&ThreadInstance::Main, this));
      while (!working_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

    void Join() {
      if (working_) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          working_ = false;
          cv_.notify_one();
        }
        thread_->join();
      }
    }

    void Main() {
      working_ = true;
      ThreadInstance<Task, TaskArg> *tmp = this;
      std::unique_lock<std::mutex> lock(mutex_);
      while (true) {
        cv_.wait(lock);

        if (!working_) {
          break;
        }
        task_(task_args_);
        if (waiter_ != nullptr) {
          waiter_->Notify();
          waiter_ = nullptr;
        }

        pool_->free_.Push(tmp);
      }
    }
    void SetTask(Task& task, TaskArg& args, Waiter* waiter = nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      task_args_ = std::move(args);
      task_ = task;
      waiter_ = waiter;
      cv_.notify_one();
    }
  private:
    Task task_;
    TaskArg task_args_;
    Waiter* waiter_;

    bool working_;
    std::unique_ptr<std::thread> thread_;
    ThreadPool<Task, TaskArg>* pool_;

    std::mutex mutex_;
    std::condition_variable cv_;
  }; // class ThreadInstance

public:
  ThreadPool(int num_thread) {
    for (int i = 0; i < num_thread; ++i) {
      auto instance = new ThreadInstance<Task, TaskArg>(this);
      instances_.push_back(instance);
      free_.Push(instance);
    }
  }
  ~ThreadPool() {
    Stop();
    for (auto instance : instances_) {
      delete instance;
    }
  }
  void Start() {
    for (auto instance : instances_) {
      instance->Run();
    }
  }
  void Stop() {
    for (auto instance : instances_) {
      instance->Join();
   }
  }
  void RunTask(Task& task, TaskArg& args, Waiter* waiter = nullptr) {
    ThreadInstance<Task, TaskArg>* instance;
    free_.Pop(instance);
    instance->SetTask(task, args, waiter);
  }
private:
  std::vector<ThreadInstance<Task, TaskArg>*> instances_;
  MtQueue<ThreadInstance<Task, TaskArg>*> free_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_UTIL_THREAD_POOL_H_