// #include <opencv2/opencv.hpp>
// #include <iostream>

// struct TileMeta
// {
//     int row_offset;
//     int col_offset;
//     int full_image_height;
//     int full_image_width;
//     int overlap;
// };

// // ROTATE
// cv::Mat rotate_tile(const cv::Mat &tile, double angle, const TileMeta &meta)
// {
//     double global_cx = meta.full_image_width / 2.0;
//     double global_cy = meta.full_image_height / 2.0;

//     cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(global_cx, global_cy), angle, 1.0);

//     M.at<double>(0, 2) -= meta.col_offset;
//     M.at<double>(1, 2) -= meta.row_offset;

//     cv::Mat result;
//     cv::warpAffine(tile, result, M, tile.size(),
//                    cv::INTER_LINEAR, cv::BORDER_REFLECT);
//     return result;
// }

// // RESIZE
// cv::Mat resize_tile(const cv::Mat &tile, double scale_factor)
// {
//     int new_w = (int)(tile.cols * scale_factor);
//     int new_h = (int)(tile.rows * scale_factor);
//     cv::Mat result;
//     cv::resize(tile, result, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
//     return result;
// }

// // CROP
// cv::Mat crop_tile(const cv::Mat &tile, int crop_x, int crop_y, int crop_w, int crop_h, const TileMeta &meta)
// {
//     int inter_x1 = std::max(meta.col_offset, crop_x);
//     int inter_y1 = std::max(meta.row_offset, crop_y);
//     int inter_x2 = std::min(meta.col_offset + tile.cols, crop_x + crop_w);
//     int inter_y2 = std::min(meta.row_offset + tile.rows, crop_y + crop_h);

//     if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2)
//     {
//         return cv::Mat();
//     }

//     int lx1 = inter_x1 - meta.col_offset;
//     int ly1 = inter_y1 - meta.row_offset;
//     int lx2 = inter_x2 - meta.col_offset;
//     int ly2 = inter_y2 - meta.row_offset;

//     return tile(cv::Rect(lx1, ly1, lx2 - lx1, ly2 - ly1)).clone();
// }

// // STRIP OVERLAP
// cv::Mat strip_overlap(const cv::Mat &tile, int overlap)
// {
//     if (overlap == 0)
//         return tile;
//     return tile(cv::Rect(overlap, overlap,
//                          tile.cols - 2 * overlap,
//                          tile.rows - 2 * overlap))
//         .clone();
// }

// // MAIN INTERFACE
// cv::Mat apply_transform(const cv::Mat &tile, const std::string &transform_type,
//                         double param, const TileMeta &meta,
//                         int crop_x = 0, int crop_y = 0, int crop_w = 0, int crop_h = 0)
// {
//     cv::Mat result;

//     if (transform_type == "rotate")
//     {
//         result = rotate_tile(tile, param, meta);
//     }
//     else if (transform_type == "resize")
//     {
//         result = resize_tile(tile, param);
//     }
//     else if (transform_type == "crop")
//     {
//         result = crop_tile(tile, crop_x, crop_y, crop_w, crop_h, meta);
//         if (result.empty())
//             return cv::Mat();
//     }
//     else
//     {
//         std::cerr << "Unknown transform: " << transform_type << std::endl;
//         return cv::Mat();
//     }

//     return strip_overlap(result, meta.overlap);
// }

// // // TEST
// // int main() {
// //     cv::Mat tile(100, 100, CV_8UC3);
// //     cv::randu(tile, 0, 255);

// //     TileMeta meta;
// //     meta.row_offset = 512;
// //     meta.col_offset = 512;
// //     meta.full_image_height = 2000;
// //     meta.full_image_width = 2000;
// //     meta.overlap = 10;

// //     // Test rotate
// //     cv::Mat r1 = apply_transform(tile, "rotate", 45.0, meta);
// //     std::cout << "Rotate output shape: " << r1.rows << "x" << r1.cols << std::endl;

// //     // Test resize
// //     cv::Mat r2 = apply_transform(tile, "resize", 0.5, meta);
// //     std::cout << "Resize output shape: " << r2.rows << "x" << r2.cols << std::endl;

// //     // Test crop
// //     cv::Mat r3 = apply_transform(tile, "crop", 0, meta, 500, 500, 200, 200);
// //     if (r3.empty())
// //         std::cout << "Crop output: None (tile outside crop)" << std::endl;
// //     else
// //         std::cout << "Crop output shape: " << r3.rows << "x" << r3.cols << std::endl;

// //     return 0;
// // }

#include "Transform.h"
#include <cstdint>
#include <cstddef>

// Greyscale transformation
void processTile(Tile &tile)
{
    // Dynamically determine if the image is RGB (3) or RGBA (4)
    int channels = tile.data.size() / (tile.width * tile.height);
    size_t numPixels = tile.width * tile.height;

    for (size_t i = 0; i < numPixels; ++i)
    {
        size_t idx = i * channels;

        // Extract the pixel's RGB values
        uint8_t r = tile.data[idx];
        uint8_t g = tile.data[idx + 1];
        uint8_t b = tile.data[idx + 2];

        // Apply the standard luminance formula
        uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);

        // Write the new values back into the tile
        tile.data[idx] = gray;
        tile.data[idx + 1] = gray;
        tile.data[idx + 2] = gray;

        // If the image is RGBA (4 channels), tile.data[idx + 3] is the Alpha channel.
        // We leave it completely untouched so the image doesn't lose transparency!
    }
}