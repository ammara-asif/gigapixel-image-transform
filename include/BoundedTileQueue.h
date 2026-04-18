#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <optional>

class BoundedTileQueue
{
private:
    std::queue<Tile> queue;
    std::mutex mtx;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;
    size_t maxSize;
    std::atomic<bool> finished{false};

public:
    explicit BoundedTileQueue(size_t size) : maxSize(size) {}

    // Called by Producer
    void push(Tile tile)
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Wait if queue is full
        cv_push.wait(lock, [this]()
                     { return queue.size() < maxSize; });

        queue.push(std::move(tile));

        lock.unlock();
        cv_pop.notify_one(); // Wake up a sleeping consumer
    }

    // Called by Consumers
    bool pop(Tile &tile)
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Wait until there is a tile OR the producer is completely finished
        cv_pop.wait(lock, [this]()
                    { return !queue.empty() || finished; });

        if (queue.empty() && finished)
        {
            return false; // Queue is empty and no more tiles are coming
        }

        tile = std::move(queue.front());
        queue.pop();

        lock.unlock();
        cv_push.notify_one(); // Wake up the producer if it was blocked
        return true;
    }

    // Signals consumers that no more tiles will be added
    void setFinished()
    {
        finished = true;
        cv_pop.notify_all(); // Wake all consumers to let them exit gracefully
    }
};