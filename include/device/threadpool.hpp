#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tnn {
// Thread pool for CPU async operations
class ThreadPool {
private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  std::atomic<bool> stop_;

public:
  explicit ThreadPool(size_t num_threads = 1) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_.load() || !tasks_.empty(); });

            if (stop_.load() && tasks_.empty()) {
              return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
          }
          try {
            task();
          } catch (...) {
            // Silently catch exceptions to prevent thread termination
          }
        }
      });
    }
  }

  ~ThreadPool() { shutdown(); }

  void shutdown() {
    stop_.store(true);
    condition_.notify_all();
    for (std::thread &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  template <typename F> void enqueue(F &&f) {
    if (stop_.load()) {
      return; // Don't accept new tasks if shutting down
    }
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;
};
} // namespace tnn