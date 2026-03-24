#include "OutputWriter.h"
#include <cstring>

#define TIF reinterpret_cast<TIFF*>(tif_)      // memcpy
#include <cstdio>       // printf
#include <stdexcept>    // runtime_error
#include <algorithm>    // std::min
#include <omp.h>

// Constructor
TiledOutputWriter::TiledOutputWriter(const std::string& filename, int full_width, 
                                     int full_height, int channels, int total_tiles,
                                     bool big_tiff) : filename_(filename),
                                    full_width_(full_width), full_height_(full_height),
                                    channels_(channels), total_tiles_(total_tiles),
                                    big_tiff_(big_tiff), tif_(nullptr), done_(false){
    omp_init_lock(&queue_lock_);
    open_tiff();   // opens the TIFF file and writes all required tags
}

// Destructor 
TiledOutputWriter::~TiledOutputWriter(){
    if (tif_) {
        TIFFClose(TIF);
        tif_ = nullptr;
    }
    omp_destroy_lock(&queue_lock_);
}

// submit_tile  (called by processor thread — thread-safe)
// Deep-copies the Tile onto the heap so the processor can
// immediately reuse/overwrite its own buffer.  The writer
// thread will delete the copy after writing.
void TiledOutputWriter::submit_tile(const Tile& tile){
    Tile* copy = clone_tile(tile);

    omp_set_lock(&queue_lock_);
    queue_.push(copy);
    omp_unset_lock(&queue_lock_);
}

// finalize  (called by processor thread after last tile)
// Pushes nullptr (sentinel) so the writer loop knows to stop,
// then spin-waits until done_ == true.
void TiledOutputWriter::finalize(){
    // Push the sentinel
    omp_set_lock(&queue_lock_);
    queue_.push(nullptr);   // nullptr == sentinel
    omp_unset_lock(&queue_lock_);

    // Wait for the writer thread to drain the queue and set done_
    while (!done_) {
        #pragma omp taskyield   // yield CPU so the writer thread can run
    }

    // Close the TIFF only after the writer loop has exited
    if (tif_) {
        TIFFClose(TIF);
        tif_ = nullptr;
    }
    printf("[Writer] All %d tiles written and file closed.\n", total_tiles_);
}

// run  (writer thread loop)
// Called from the dedicated writer OMP thread.
// Loops: dequeue → strip overlap → write to TIFF → free copy.
// Exits when the sentinel (nullptr) is dequeued.
void TiledOutputWriter::run(){
    int tiles_written = 0;

    while (true) {
        Tile* tile = dequeue_blocking();   // blocks until something is in the queue

        if (tile == nullptr) {
            // Sentinel received — all tiles have been processed
            break;
        }

        process_and_write(tile);
        tiles_written++;

        printf("[Writer] Wrote tile at (%d,%d)  [%d / %d]\n",
               tile->x, tile->y, tiles_written, total_tiles_);

        delete tile;   // free the heap copy made in clone_tile()
    }

    done_ = true;   // signal finalize() that we are done
}

// open_tiff  (private)
// Opens the output TIFF/BigTIFF and sets all required tags.
// We use SCANLINE organisation here (not tiled storage) because
// scanlines let us write individual rows via TIFFWriteScanline,
// which is the only way to do random-access writes in standard
// libtiff without pre-allocating a tile buffer for every TIFF tile.
void TiledOutputWriter::open_tiff(){
    // "w8" = BigTIFF (supports files > 4 GB)
    // "w"  = classic TIFF (max ~4 GB)
    const char* mode = big_tiff_ ? "w8" : "w";

    tif_ = reinterpret_cast<tiff*>(TIFFOpen(filename_.c_str(), mode));
    if (!tif_)
        throw std::runtime_error("[Writer] Cannot open output TIFF: " + filename_);

    // Basic image dimensions
    TIFFSetField(TIF, TIFFTAG_IMAGEWIDTH,      (uint32_t)full_width_);
    TIFFSetField(TIF, TIFFTAG_IMAGELENGTH,     (uint32_t)full_height_);
    TIFFSetField(TIF, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)channels_);
    TIFFSetField(TIF, TIFFTAG_BITSPERSAMPLE,   (uint16_t)8);

    if (channels_ == 1)
        TIFFSetField(TIF, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    else
        TIFFSetField(TIF, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    TIFFSetField(TIF, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(TIF, TIFFTAG_ROWSPERSTRIP, (uint32_t)1);
    TIFFSetField(TIF, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    printf("[Writer] Opened output: %s  (%dx%d  %dch  %s)\n",
           filename_.c_str(), full_width_, full_height_, channels_,
           big_tiff_ ? "BigTIFF" : "TIFF");
}

// clone_tile  (private)
// Deep-copies a Tile from the processor onto the heap.
// The processor's std::vector<uint8_t> is NOT referenced after
// this call — the processor can immediately free or reuse it.
Tile* TiledOutputWriter::clone_tile(const Tile& src){
    Tile* t  = new Tile();
    t->x      = src.x;
    t->y      = src.y;
    t->width  = src.width;    // buffer width  INCLUDING overlap
    t->height = src.height;   // buffer height INCLUDING overlap
    t->overlap = src.overlap;
    t->data   = src.data;     // std::vector copy (deep copy of pixel bytes)
    return t;
}

// process_and_write  (private)
void TiledOutputWriter::process_and_write(Tile* tile){
    int ov       = tile->overlap;

    // Logical (output) dimensions — strip overlap from both sides
    int logical_w = tile->width  - 2 * ov;
    int logical_h = tile->height - 2 * ov;

    // Clamp to image boundary (edge tiles may be smaller)
    logical_w = std::min(logical_w, full_width_  - tile->x);
    logical_h = std::min(logical_h, full_height_ - tile->y);

    if (logical_w <= 0 || logical_h <= 0) {
        printf("[Writer] Skipping zero-size tile at (%d,%d)\n", tile->x, tile->y);
        return;
    }

    // ---- Strip the overlap border ----
    // Source buffer row stride (includes left+right overlap)
    int src_row_stride = tile->width * channels_;
    // Destination row stride (no overlap)
    int dst_row_stride = logical_w * channels_;

    std::vector<uint8_t> stripped((size_t)logical_h * dst_row_stride);

    for (int row = 0; row < logical_h; ++row) {
        // In source buffer, row (ov + row) is the first non-overlap row.
        // Within that row, skip (ov * channels_) bytes on the left.
        const uint8_t* src_row = tile->data.data()
            + (size_t)(ov + row) * src_row_stride
            + (size_t)ov * channels_;

        uint8_t* dst_row = stripped.data() + (size_t)row * dst_row_stride;

        memcpy(dst_row, src_row, (size_t)dst_row_stride);
    }

    // ---- Write stripped rows to the TIFF ----
    write_stripped_tile(stripped.data(),
                        logical_w, logical_h,
                        tile->x, tile->y);
}

// write_stripped_tile  (private)
void TiledOutputWriter::write_stripped_tile(const uint8_t* stripped_data, int logical_w,
                                            int logical_h, int pixel_x, int pixel_y){
    // Persistent scanline buffer: one full row of the output image
    std::vector<uint8_t> scanline((size_t)full_width_ * channels_, 0);

    int tile_row_stride = logical_w * channels_;    // bytes per tile row
    int full_row_stride = full_width_ * channels_;  // bytes per full scanline

    for (int row = 0; row < logical_h; ++row) {
        int global_row = pixel_y + row;

        // ---- Step 1: Read the existing full scanline ----
        // This preserves columns already written by other tiles in the same row.
        // TIFFReadScanline returns -1 on error.
        int read_result = TIFFReadScanline(TIF,
                                           scanline.data(),
                                           (uint32_t)global_row,
                                           0);
        if (read_result < 0) {
            // On a brand-new file, unwritten rows read as 0 — that is fine.
            // Fill with zeros as a safe fallback.
            std::fill(scanline.begin(), scanline.end(), 0);
        }

        // ---- Step 2: Overwrite our tile's columns ----
        uint8_t* dst = scanline.data() + (size_t)pixel_x * channels_;
        const uint8_t* src = stripped_data + (size_t)row * tile_row_stride;
        memcpy(dst, src, (size_t)tile_row_stride);

        // ---- Step 3: Write the updated full scanline back ----
        if (TIFFWriteScanline(TIF,
                              scanline.data(),
                              (uint32_t)global_row,
                              0) < 0) {
            throw std::runtime_error(
                "[Writer] TIFFWriteScanline failed at row " +
                std::to_string(global_row));
        }
    }
}

// dequeue_blocking  (private)
// Busy-waits until the queue is non-empty, then pops and
// returns the front element.  Uses #pragma omp taskyield
// so the OMP scheduler can run other tasks while we spin —
// exactly the same pattern as the lab PDF's dequeue().
Tile* TiledOutputWriter::dequeue_blocking(){
    Tile* item = nullptr;
    bool  got  = false;

    while (!got) {
        omp_set_lock(&queue_lock_);
        if (!queue_.empty()) {
            item = queue_.front();
            queue_.pop();
            got = true;
        }
        omp_unset_lock(&queue_lock_);

        if (!got) {
            #pragma omp taskyield   // give other threads a chance to run
        }
    }
    return item;
}
