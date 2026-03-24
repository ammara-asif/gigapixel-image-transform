#pragma once
#include "Tile.h"
#include <string>
#include <queue>
#include <tiffio.h>
#include <omp.h>

class TiledOutputWriter {
public:
    // full_width / full_height : dimensions of the complete output image
    // channels                 : 1 (grayscale), 3 (RGB), 4 (RGBA) — match reader
    // total_tiles              : exactly how many submit_tile() calls to expect
    // big_tiff                 : true → open as BigTIFF (needed for images > 4 GB)
    TiledOutputWriter(const std::string& filename,
                      int full_width,
                      int full_height,
                      int channels,
                      int total_tiles,
                      bool big_tiff = true);

    ~TiledOutputWriter();

    // Thread-safe. Called by the processor thread(s) once a tile is ready.
    // The Tile's data vector is copied internally — caller can reuse/free theirs.
    void submit_tile(const Tile& tile);

    // Push a sentinel and wait until every tile has been written to disk.
    // Call this AFTER all submit_tile() calls are done.
    void finalize();

    // The writer loop — run this in a dedicated OMP thread.
    // Returns only after the sentinel is dequeued (i.e. after finalize() signals it).
    void run();

private:
    // ---- config ----
    std::string filename_;
    int         full_width_;
    int         full_height_;
    int         channels_;
    int         total_tiles_;
    bool        big_tiff_;

    // ---- libtiff handle ----
    TIFF*       tif_;

    // ---- producer-consumer queue ----
    // Processor threads push Tile*, writer thread pops them.
    // nullptr is the sentinel.
    std::queue<Tile*>  queue_;
    omp_lock_t         queue_lock_;
    volatile bool      done_;       // set true by run() when sentinel is dequeued

    // ---- internal helpers ----
    void open_tiff();

    // Deep-copies a Tile onto the heap so the caller can reuse its vector.
    // Writer thread deletes the copy after writing.
    Tile* clone_tile(const Tile& src);

    // Strips overlap, then calls write_stripped_tile().
    void process_and_write(Tile* tile);

    // Writes the stripped pixel data into the correct rows of the TIFF.
    // stripped_data : contiguous rows, NO overlap, width = logical_w, height = logical_h
    // pixel_x/y     : top-left corner in the full output image
    void write_stripped_tile(const uint8_t* stripped_data, int logical_w, int logical_h, 
                             int pixel_x,int pixel_y);

    // Blocks until at least one item is in the queue, then pops and returns it.
    Tile* dequeue_blocking();
};