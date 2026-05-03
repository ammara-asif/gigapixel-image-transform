#include "OutputWriter.h"
#include <cstring>

#include <tiffio.h>
#define TIF reinterpret_cast<TIFF *>(tif_) // memcpy
#include <cstdio>                          // printf
#include <stdexcept>                       // runtime_error
#include <algorithm>                       // std::min
#include <omp.h>

// Constructor
TiledOutputWriter::TiledOutputWriter(
    const std::string &filename,
    int full_width,
    int full_height,
    int channels,
    int total_tiles,
    int logical_tile_size,
    bool big_tiff) : filename_(filename),
                     full_width_(full_width),
                     full_height_(full_height),
                     channels_(channels),
                     total_tiles_(total_tiles),
                     logical_tile_size_(logical_tile_size),
                     big_tiff_(big_tiff),
                     tif_(nullptr),
                     done_(false)
{
    omp_init_lock(&queue_lock_);
    open_tiff(); // opens the TIFF file and writes all required tags
}

// Destructor
TiledOutputWriter::~TiledOutputWriter()
{
    if (tif_)
    {
        TIFFClose(TIF);
        tif_ = nullptr;
    }
    omp_destroy_lock(&queue_lock_);
}

// submit_tile  (called by processor thread — thread-safe)
// Deep-copies the Tile onto the heap so the processor can
// immediately reuse/overwrite its own buffer.  The writer
// thread will delete the copy after writing.
void TiledOutputWriter::submit_tile(const Tile &tile)
{
    Tile *copy = clone_tile(tile);

    omp_set_lock(&queue_lock_);
    queue_.push(copy);
    omp_unset_lock(&queue_lock_);
}

// finalize  (called by processor thread after last tile)
// Pushes nullptr (sentinel) so the writer loop knows to stop,
// then spin-waits until done_ == true.
void TiledOutputWriter::finalize()
{
    // Push the sentinel
    omp_set_lock(&queue_lock_);
    queue_.push(nullptr); // nullptr == sentinel
    omp_unset_lock(&queue_lock_);

    // Wait for the writer thread to drain the queue and set done_
    while (!done_)
    {
#pragma omp taskyield // yield CPU so the writer thread can run
    }

    // Close the TIFF only after the writer loop has exited
    if (tif_)
    {
        TIFFClose(TIF);
        tif_ = nullptr;
    }
    printf("[Writer] All %d tiles written and file closed.\n", total_tiles_);
}

// run  (writer thread loop)
// Called from the dedicated writer OMP thread.
// Loops: dequeue → strip overlap → write to TIFF → free copy.
// Exits when the sentinel (nullptr) is dequeued.
void TiledOutputWriter::run()
{
    int tiles_written = 0;

    while (true)
    {
        Tile *tile = dequeue_blocking(); // blocks until something is in the queue

        if (tile == nullptr)
        {
            // Sentinel received — all tiles have been processed
            break;
        }

        process_and_write(tile);
        tiles_written++;

        printf("[Writer] Wrote tile at (%d,%d)  [%d / %d]\n",
               tile->x, tile->y, tiles_written, total_tiles_);

        delete tile; // free the heap copy made in clone_tile()
    }

    done_ = true; // signal finalize() that we are done
}

// open_tiff  (private)
// Opens the output TIFF/BigTIFF and sets all required tags.
// We use SCANLINE organisation here (not tiled storage) because
// scanlines let us write individual rows via TIFFWriteScanline,
// which is the only way to do random-access writes in standard
// libtiff without pre-allocating a tile buffer for every TIFF tile.
void TiledOutputWriter::open_tiff()
{
    // "w8" = BigTIFF (supports files > 4 GB)
    // "w"  = classic TIFF (max ~4 GB)
    const char *mode = big_tiff_ ? "w8" : "w";

    tif_ = reinterpret_cast<tiff *>(TIFFOpen(filename_.c_str(), mode));
    if (!tif_)
        throw std::runtime_error("[Writer] Cannot open output TIFF: " + filename_);

    // Basic image dimensions
    TIFFSetField(TIF, TIFFTAG_IMAGEWIDTH, (uint32_t)full_width_);
    TIFFSetField(TIF, TIFFTAG_IMAGELENGTH, (uint32_t)full_height_);
    TIFFSetField(TIF, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)channels_);
    TIFFSetField(TIF, TIFFTAG_BITSPERSAMPLE, (uint16_t)8);

    if (channels_ == 1)
        TIFFSetField(TIF, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    else
        TIFFSetField(TIF, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    TIFFSetField(TIF, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(TIF, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    uint32_t tile_dim = (logical_tile_size_ + 15) & ~15;

    TIFFSetField(TIF, TIFFTAG_TILEWIDTH, tile_dim);
    TIFFSetField(TIF, TIFFTAG_TILELENGTH, tile_dim);

    printf("[Writer] Opened output: %s  (%dx%d  %dch  %s)\n",
           filename_.c_str(), full_width_, full_height_, channels_,
           big_tiff_ ? "BigTIFF" : "TIFF");
}

// clone_tile  (private)
// Deep-copies a Tile from the processor onto the heap.
// The processor's std::vector<uint8_t> is NOT referenced after
// this call — the processor can immediately free or reuse it.
Tile *TiledOutputWriter::clone_tile(const Tile &src)
{
    Tile *t = new Tile();
    t->x = src.x;
    t->y = src.y;
    t->width = src.width;   // buffer width  INCLUDING overlap
    t->height = src.height; // buffer height INCLUDING overlap
    t->overlap = src.overlap;
    t->data = src.data; // std::vector copy (deep copy of pixel bytes)
    return t;
}

// process_and_write  (private)
void TiledOutputWriter::process_and_write(Tile *tile)
{
    int ov = tile->overlap;

    // 1. Calculate EXACTLY how much overlap exists on the left and top.
    // Edge tiles at coordinate 0 will yield 0 overlap here.
    int actual_left_ov = tile->x - std::max(0, tile->x - ov);
    int actual_top_ov = tile->y - std::max(0, tile->y - ov);

    // 2. Calculate the expected logical output dimensions
    // (logical_tile_size_ was added to your class constructor in the last step)
    int logical_w = std::min(logical_tile_size_, full_width_ - tile->x);
    int logical_h = std::min(logical_tile_size_, full_height_ - tile->y);

    // This absolutely guarantees memcpy cannot read past the tile buffer,
    // even if logical_tile_size_ is wrong.
    logical_w = std::min(logical_w, tile->width - actual_left_ov);
    logical_h = std::min(logical_h, tile->height - actual_top_ov);

    if (logical_w <= 0 || logical_h <= 0)
        return;

    // ---- Strip the actual overlap border ----
    int src_row_stride = tile->width * channels_; // buffer width
    int dst_row_stride = logical_w * channels_;

    std::vector<uint8_t> stripped((size_t)logical_h * dst_row_stride);

    for (int row = 0; row < logical_h; ++row)
    {

        // Offset the pointer using the ACTUAL overlap, not the theoretical overlap
        const uint8_t *src_row = tile->getRawPtr() +
        (size_t)(actual_top_ov + row) * src_row_stride +
        (size_t)actual_left_ov * channels_;
        uint8_t *dst_row = stripped.data() + (size_t)row * dst_row_stride;

        memcpy(dst_row, src_row, (size_t)dst_row_stride);
    }

    // ---- Write stripped rows to the TIFF ----
    write_stripped_tile(stripped.data(), logical_w, logical_h, tile->x, tile->y);
}

// write_stripped_tile  (private)
void TiledOutputWriter::write_stripped_tile(const uint8_t *stripped_data, int logical_w,
                                            int logical_h, int pixel_x, int pixel_y)
{

    // Get the exact dimensions libtiff expects for a tile buffer
    uint32_t tw, th;
    TIFFGetField(TIF, TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(TIF, TIFFTAG_TILELENGTH, &th);

    // libtiff requires the buffer to be exactly TILEWIDTH * TILELENGTH,
    // even for edge tiles that are smaller. We allocate and zero-pad.
    std::vector<uint8_t> padded_tile(tw * th * channels_, 0);

    int src_row_stride = logical_w * channels_;
    int dst_row_stride = tw * channels_;

    // Copy the stripped data into the top-left of the padded buffer
    for (int row = 0; row < logical_h; ++row)
    {
        memcpy(padded_tile.data() + (size_t)row * dst_row_stride,
               stripped_data + (size_t)row * src_row_stride,
               (size_t)src_row_stride);
    }

    // Write the tile. This is completely safe to do out-of-order!
    if (TIFFWriteTile(TIF, padded_tile.data(), (uint32_t)pixel_x, (uint32_t)pixel_y, 0, 0) < 0)
    {
        throw std::runtime_error(
            "[Writer] TIFFWriteTile failed at (" +
            std::to_string(pixel_x) + "," + std::to_string(pixel_y) + ")");
    }
}

// dequeue_blocking  (private)
// Busy-waits until the queue is non-empty, then pops and
// returns the front element.  Uses #pragma omp taskyield
// so the OMP scheduler can run other tasks while we spin —
// exactly the same pattern as the lab PDF's dequeue().
Tile *TiledOutputWriter::dequeue_blocking()
{
    Tile *item = nullptr;
    bool got = false;

    while (!got)
    {
        omp_set_lock(&queue_lock_);
        if (!queue_.empty())
        {
            item = queue_.front();
            queue_.pop();
            got = true;
        }
        omp_unset_lock(&queue_lock_);

        if (!got)
        {
#pragma omp taskyield // give other threads a chance to run
        }
    }
    return item;
}
