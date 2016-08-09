#ifndef MULTIVERSO_WAITER_H_
#define MULTIVERSO_WAITER_H_

#include <mutex>
#include <condition_variable>

namespace multiverso {

class Waiter {
public:
  explicit Waiter(int num_wait = 1) : num_wait_(num_wait) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (num_wait_ > 0) cv_.wait(lock);
  }

  void Notify() {
    std::lock_guard<std::mutex> lock(mutex_);
    --num_wait_;
    cv_.notify_all();
  }

  void Reset(int num_wait) {
    std::lock_guard<std::mutex> lock(mutex_);
    num_wait_ = num_wait;
  }

  void AddNumWait(int delta) {
    std::lock_guard<std::mutex> lock(mutex_);
    num_wait_ += delta;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int num_wait_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_WAITER_H_
