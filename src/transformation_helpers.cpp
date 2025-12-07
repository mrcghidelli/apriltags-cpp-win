// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transformation_helpers.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <sstream>
#include <cmath>
#include <climits>
#include <cstdint>

#include <vector>

struct color_point_t
{
    int16_t xyz[3];
    uint8_t rgb[3];
};

void tranformation_helpers_write_point_cloud(const k4a_image_t point_cloud_image,
                                             const k4a_image_t color_image,
                                             const k4a_float3_t* marker_corners_3d,
                                             int marker_corners_count,
                                             const char *file_name)
{
    std::vector<color_point_t> points;

    int width = k4a_image_get_width_pixels(point_cloud_image);
    int height = k4a_image_get_height_pixels(color_image);

    int16_t *point_cloud_image_data = (int16_t *)(void *)k4a_image_get_buffer(point_cloud_image);
    uint8_t *color_image_data = k4a_image_get_buffer(color_image);


    for (int i = 0; i < width * height; i++)
    {
        color_point_t point;
        point.xyz[0] = point_cloud_image_data[3 * i + 0];
        point.xyz[1] = point_cloud_image_data[3 * i + 1];
        point.xyz[2] = point_cloud_image_data[3 * i + 2];
        if (point.xyz[2] == 0)
        {
            continue;
        }

        // Keep original colored points; we'll append synthetic red spheres around marker corners later.
        point.rgb[0] = color_image_data[4 * i + 0];
        point.rgb[1] = color_image_data[4 * i + 1];
        point.rgb[2] = color_image_data[4 * i + 2];
        uint8_t alpha = color_image_data[4 * i + 3];

        if (point.rgb[0] == 0 && point.rgb[1] == 0 && point.rgb[2] == 0 && alpha == 0)
        {
            continue;
        }
        points.push_back(point);
    }

#define PLY_START_HEADER "ply"
#define PLY_END_HEADER "end_header"
#define PLY_ASCII "format ascii 1.0"
#define PLY_ELEMENT_VERTEX "element vertex"

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << PLY_START_HEADER << std::endl;
    ofs << PLY_ASCII << std::endl;
    ofs << PLY_ELEMENT_VERTEX << " " << points.size() << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "property uchar red" << std::endl;
    ofs << "property uchar green" << std::endl;
    ofs << "property uchar blue" << std::endl;
    ofs << PLY_END_HEADER << std::endl;
    ofs.close();

    std::stringstream ss;
    for (size_t i = 0; i < points.size(); ++i)
    {
        // image data is BGR
        ss << (float)points[i].xyz[0] << " " << (float)points[i].xyz[1] << " " << (float)points[i].xyz[2];
        ss << " " << (float)points[i].rgb[2] << " " << (float)points[i].rgb[1] << " " << (float)points[i].rgb[0];
        ss << std::endl;
    }
    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());

    // Add synthetic red spheres around each provided marker corner.
    const int sphere_radius_mm = 10; // 10 mm
    const int step_mm = 5; // spacing between synthetic points
    for (int m = 0; m < marker_corners_count; ++m)
    {
        float mx = marker_corners_3d[m].xyz.x;
        float my = marker_corners_3d[m].xyz.y;
        float mz = marker_corners_3d[m].xyz.z;
        if (mx == 0.0f && my == 0.0f && mz == 0.0f) continue;

        // iterate a small cubic grid around the corner and keep points inside the sphere
        for (int dz = -sphere_radius_mm; dz <= sphere_radius_mm; dz += step_mm) {
            for (int dy = -sphere_radius_mm; dy <= sphere_radius_mm; dy += step_mm) {
                for (int dx = -sphere_radius_mm; dx <= sphere_radius_mm; dx += step_mm) {
                    int dist_sq = dx*dx + dy*dy + dz*dz;
                    if (dist_sq > sphere_radius_mm * sphere_radius_mm) continue;

                    color_point_t sp;
                    int sx = static_cast<int>(std::lround(mx)) + dx;
                    int sy = static_cast<int>(std::lround(my)) + dy;
                    int sz = static_cast<int>(std::lround(mz)) + dz;
                    // clamp to int16 range
                    if (sx < INT16_MIN) sx = INT16_MIN;
                    if (sx > INT16_MAX) sx = INT16_MAX;
                    if (sy < INT16_MIN) sy = INT16_MIN;
                    if (sy > INT16_MAX) sy = INT16_MAX;
                    if (sz < INT16_MIN) sz = INT16_MIN;
                    if (sz > INT16_MAX) sz = INT16_MAX;
                    sp.xyz[0] = static_cast<int16_t>(sx);
                    sp.xyz[1] = static_cast<int16_t>(sy);
                    sp.xyz[2] = static_cast<int16_t>(sz);
                    sp.rgb[0] = 0;   // B
                    sp.rgb[1] = 0;   // G
                    sp.rgb[2] = 255; // R
                    points.push_back(sp);
                }
            }
        }
    }

    // Append the synthetic sphere points to the PLY file as well
    std::stringstream ss2;
    for (size_t i = 0; i < points.size(); ++i)
    {
        ss2 << (float)points[i].xyz[0] << " " << (float)points[i].xyz[1] << " " << (float)points[i].xyz[2];
        ss2 << " " << (float)points[i].rgb[2] << " " << (float)points[i].rgb[1] << " " << (float)points[i].rgb[0];
        ss2 << std::endl;
    }
    // rewrite header + points (replace file previously written)
    std::ofstream ofs3(file_name);
    ofs3 << PLY_START_HEADER << std::endl;
    ofs3 << PLY_ASCII << std::endl;
    ofs3 << PLY_ELEMENT_VERTEX << " " << points.size() << std::endl;
    ofs3 << "property float x" << std::endl;
    ofs3 << "property float y" << std::endl;
    ofs3 << "property float z" << std::endl;
    ofs3 << "property uchar red" << std::endl;
    ofs3 << "property uchar green" << std::endl;
    ofs3 << "property uchar blue" << std::endl;
    ofs3 << PLY_END_HEADER << std::endl;
    ofs3.write(ss2.str().c_str(), (std::streamsize)ss2.str().length());
    ofs3.close();

}

k4a_image_t downscale_image_2x2_binning(const k4a_image_t color_image)
{
    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    int color_image_downscaled_width_pixels = color_image_width_pixels / 2;
    int color_image_downscaled_height_pixels = color_image_height_pixels / 2;
    k4a_image_t color_image_downscaled = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                 color_image_downscaled_width_pixels,
                                                 color_image_downscaled_height_pixels,
                                                 color_image_downscaled_width_pixels * 4 * (int)sizeof(uint8_t),
                                                 &color_image_downscaled))
    {
        printf("Failed to create downscaled color image\n");
        return color_image_downscaled;
    }

    uint8_t *color_image_data = k4a_image_get_buffer(color_image);
    uint8_t *color_image_downscaled_data = k4a_image_get_buffer(color_image_downscaled);
    for (int j = 0; j < color_image_downscaled_height_pixels; j++)
    {
        for (int i = 0; i < color_image_downscaled_width_pixels; i++)
        {
            int index_downscaled = j * color_image_downscaled_width_pixels + i;
            int index_tl = (j * 2 + 0) * color_image_width_pixels + i * 2 + 0;
            int index_tr = (j * 2 + 0) * color_image_width_pixels + i * 2 + 1;
            int index_bl = (j * 2 + 1) * color_image_width_pixels + i * 2 + 0;
            int index_br = (j * 2 + 1) * color_image_width_pixels + i * 2 + 1;

            color_image_downscaled_data[4 * index_downscaled + 0] = (uint8_t)(
                (color_image_data[4 * index_tl + 0] + color_image_data[4 * index_tr + 0] +
                 color_image_data[4 * index_bl + 0] + color_image_data[4 * index_br + 0]) /
                4.0f);
            color_image_downscaled_data[4 * index_downscaled + 1] = (uint8_t)(
                (color_image_data[4 * index_tl + 1] + color_image_data[4 * index_tr + 1] +
                 color_image_data[4 * index_bl + 1] + color_image_data[4 * index_br + 1]) /
                4.0f);
            color_image_downscaled_data[4 * index_downscaled + 2] = (uint8_t)(
                (color_image_data[4 * index_tl + 2] + color_image_data[4 * index_tr + 2] +
                 color_image_data[4 * index_bl + 2] + color_image_data[4 * index_br + 2]) /
                4.0f);
            color_image_downscaled_data[4 * index_downscaled + 3] = (uint8_t)(
                (color_image_data[4 * index_tl + 3] + color_image_data[4 * index_tr + 3] +
                 color_image_data[4 * index_bl + 3] + color_image_data[4 * index_br + 3]) /
                4.0f);
        }
    }

    return color_image_downscaled;
}
