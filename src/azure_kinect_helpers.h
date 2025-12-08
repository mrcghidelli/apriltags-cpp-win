// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef KINECT_AZURE_LIBS
#include <k4a/k4a.hpp>
#include <k4a/k4a.h>
#include <opencv2/opencv.hpp>

void tranformation_helpers_write_point_cloud(const k4a_image_t point_cloud_image,
                                             const k4a_image_t color_image,
                                             const k4a_float3_t* marker_corners_3d,
                                             int marker_corners_count,
                                             const char *file_name);

k4a_image_t downscale_image_2x2_binning(const k4a_image_t color_image);

bool k4a_color_image_to_cv_mat(k4a::image k4a_image, cv::Mat& output_mat);

bool point_cloud_depth_to_color(k4a_transformation_t transformation_handle,
                                       const k4a_image_t depth_image,
                                       const k4a_image_t color_image,
                                       const k4a_float3_t* marker_corners_3d,
                                       int marker_corners_count,
                                       std::string file_name);

#endif