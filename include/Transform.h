#pragma once

#include "Tile.h"

// Applies a transformation algorithm to a single image tile.
// Modifies the tile.data buffer in-place.
void processTile(Tile &tile);