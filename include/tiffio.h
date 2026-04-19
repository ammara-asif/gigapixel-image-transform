// Mock TIFF library for compilation testing
// This provides minimal TIFF types and functions to allow compilation
// without requiring the actual libtiff library

#pragma once

#include <cstdint>
#include <cstdio>

// Basic TIFF types
typedef uint32_t ttag_t;
typedef uint16_t tdir_t;
typedef uint16_t tsample_t;
typedef uint32_t tstrip_t;
typedef uint32_t ttile_t;
typedef int32_t tsize_t;

// Mock TIFF structure
typedef struct tiff TIFF;

// Mock functions - these are stubs that do nothing
TIFF* TIFFOpen(const char* name, const char* mode);
void TIFFClose(TIFF* tif);
int TIFFGetField(TIFF* tif, ttag_t tag, ...);
int TIFFSetField(TIFF* tif, ttag_t tag, ...);
int TIFFReadScanline(TIFF* tif, void* buf, uint32_t row, tsample_t sample = 0);
int TIFFWriteScanline(TIFF* tif, void* buf, uint32_t row, tsample_t sample = 0);
int TIFFReadTile(TIFF* tif, void* buf, uint32_t x, uint32_t y, uint32_t z, tsample_t sample = 0);
int TIFFWriteTile(TIFF* tif, void* buf, uint32_t x, uint32_t y, uint32_t z, tsample_t sample = 0);
uint32_t TIFFNumberOfTiles(TIFF* tif);
void TIFFGetFieldDefaulted(TIFF* tif, ttag_t tag, ...);
int TIFFIsTiled(TIFF* tif);
uint32_t TIFFTileSize(TIFF* tif);
uint32_t TIFFScanlineSize(TIFF* tif);
uint32_t TIFFStripSize(TIFF* tif);
uint32_t TIFFNumberOfStrips(TIFF* tif);
int TIFFReadEncodedStrip(TIFF* tif, tstrip_t strip, void* buf, tsize_t size);
int TIFFReadEncodedTile(TIFF* tif, ttile_t tile, void* buf, tsize_t size);

// TIFF tag constants
#define TIFFTAG_IMAGEWIDTH 256
#define TIFFTAG_IMAGELENGTH 257
#define TIFFTAG_BITSPERSAMPLE 258
#define TIFFTAG_COMPRESSION 259
#define TIFFTAG_PHOTOMETRIC 262
#define TIFFTAG_STRIPOFFSETS 273
#define TIFFTAG_ORIENTATION 274
#define TIFFTAG_SAMPLESPERPIXEL 277
#define TIFFTAG_ROWSPERSTRIP 278
#define TIFFTAG_STRIPBYTECOUNTS 279
#define TIFFTAG_XRESOLUTION 282
#define TIFFTAG_YRESOLUTION 283
#define TIFFTAG_PLANARCONFIG 284
#define TIFFTAG_RESOLUTIONUNIT 296
#define TIFFTAG_TILEWIDTH 322
#define TIFFTAG_TILELENGTH 323
#define TIFFTAG_TILEOFFSETS 324
#define TIFFTAG_TILEBYTECOUNTS 325
#define TIFFTAG_SAMPLEFORMAT 339

// Photometric interpretation
#define PHOTOMETRIC_MINISWHITE 0
#define PHOTOMETRIC_MINISBLACK 1
#define PHOTOMETRIC_RGB 2
#define PHOTOMETRIC_PALETTE 3
#define PHOTOMETRIC_MASK 4
#define PHOTOMETRIC_SEPARATED 5
#define PHOTOMETRIC_YCBCR 6
#define PHOTOMETRIC_CIELAB 8
#define PHOTOMETRIC_ICCLAB 9
#define PHOTOMETRIC_ITULAB 10

// Compression types
#define COMPRESSION_NONE 1
#define COMPRESSION_LZW 5
#define COMPRESSION_JPEG 6
#define COMPRESSION_PACKBITS 32773

// Sample formats
#define SAMPLEFORMAT_UINT 1
#define SAMPLEFORMAT_INT 2
#define SAMPLEFORMAT_IEEEFP 3
#define SAMPLEFORMAT_VOID 4

// Planar configuration
#define PLANARCONFIG_CONTIG 1
#define PLANARCONFIG_SEPARATE 2