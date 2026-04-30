// Mock TIFF library implementations
#include "tiffio.h"

// Mock implementations - these are stubs that do nothing
TIFF* TIFFOpen(const char* name, const char* mode) { return nullptr; }
void TIFFClose(TIFF* tif) {}
int TIFFGetField(TIFF* tif, ttag_t tag, ...) { return 0; }
int TIFFSetField(TIFF* tif, ttag_t tag, ...) { return 1; }
int TIFFReadScanline(TIFF* tif, void* buf, uint32_t row, tsample_t sample) { return 0; }
int TIFFWriteScanline(TIFF* tif, void* buf, uint32_t row, tsample_t sample) { return 0; }
int TIFFReadTile(TIFF* tif, void* buf, uint32_t x, uint32_t y, uint32_t z, tsample_t sample) { return 0; }
int TIFFWriteTile(TIFF* tif, void* buf, uint32_t x, uint32_t y, uint32_t z, tsample_t sample) { return 0; }
uint32_t TIFFNumberOfTiles(TIFF* tif) { return 0; }
void TIFFGetFieldDefaulted(TIFF* tif, ttag_t tag, ...) {}
int TIFFIsTiled(TIFF* tif) { return 0; }
uint32_t TIFFTileSize(TIFF* tif) { return 0; }
uint32_t TIFFScanlineSize(TIFF* tif) { return 0; }
uint32_t TIFFStripSize(TIFF* tif) { return 0; }
uint32_t TIFFNumberOfStrips(TIFF* tif) { return 0; }
int TIFFReadEncodedStrip(TIFF* tif, tstrip_t strip, void* buf, tsize_t size) { return 0; }
int TIFFReadEncodedTile(TIFF* tif, ttile_t tile, void* buf, tsize_t size) { return 0; }