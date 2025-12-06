/*********************************************************************
 * This file is distributed as part of the C++ port of the APRIL tags
 * library. The code is licensed under GPLv2.
 *
 * Original author: Edwin Olson <ebolson@umich.edu>
 * C++ port and modifications: Matt Zucker <mzucker1@swarthmore.edu>
 ********************************************************************/

#include "TagDetector.h"
#ifdef __linux__
#include <sys/time.h>
#elif _WIN32
#include <time.h>
#endif
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <vector>
#include <cstdint>
#include <string>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/videoio.hpp>

#include "CameraUtil.h"

#define DEFAULT_TAG_FAMILY "Tag36h11"

#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include "transformation_helpers.h"

typedef struct CamTestOptions {
  CamTestOptions() :
      params(),
      family_str(DEFAULT_TAG_FAMILY),
      error_fraction(1),
      device_num(0),
      focal_length(500),
      tag_size(0.22),
      frame_width(0),
      frame_height(0),
      mirror_display(false)
  {
  }
  TagDetectorParams params;
  std::string family_str;
  double error_fraction;
  int device_num;
  double focal_length;
  double tag_size;
  int frame_width;
  int frame_height;
  bool mirror_display;
} CamTestOptions;


void print_usage(const char* tool_name, FILE* output=stderr) {

  TagDetectorParams p;
  CamTestOptions o;

  fprintf(output, "\
Usage: %s [OPTIONS]\n\
Run a tool to test tag detection. Options:\n\
 -h              Show this help message.\n\
 -D              Use decimation for segmentation stage.\n\
 -S SIGMA        Set the original image sigma value (default %.2f).\n\
 -s SEGSIGMA     Set the segmentation sigma value (default %.2f).\n\
 -a THETATHRESH  Set the theta threshold for clustering (default %.1f).\n\
 -m MAGTHRESH    Set the magnitude threshold for clustering (default %.1f).\n\
 -V VALUE        Set adaptive threshold value for new quad algo (default %f).\n\
 -N RADIUS       Set adaptive threshold radius for new quad algo (default %d).\n\
 -b              Refine bad quads using template tracker.\n\
 -r              Refine all quads using template tracker.\n\
 -n              Use the new quad detection algorithm.\n\
 -f FAMILY       Look for the given tag family (default \"%s\")\n\
 -e FRACTION     Set error detection fraction (default %f)\n\
 -d DEVICE       Set camera device number (default %d)\n\
 -F FLENGTH      Set the camera's focal length in pixels (default %f)\n\
 -z SIZE         Set the tag size in meters (default %f)\n\
 -W WIDTH        Set the camera image width in pixels\n\
 -H HEIGHT       Set the camera image height in pixels\n\
 -M              Toggle display mirroring\n",
          tool_name,
          p.sigma,
          p.segSigma,
          p.thetaThresh,
          p.magThresh,
          p.adaptiveThresholdValue,
          p.adaptiveThresholdRadius,
          DEFAULT_TAG_FAMILY,
          o.error_fraction,
          o.device_num,
          o.focal_length,
          o.tag_size);


  fprintf(output, "Known tag families:");
  TagFamily::StringArray known = TagFamily::families();
  for (size_t i = 0; i < known.size(); ++i) {
    fprintf(output, " %s", known[i].c_str());
  }
  fprintf(output, "\n");
}

bool k4a_color_image_to_cv_mat(k4a::image k4a_image, cv::Mat& output_mat) {
    
    if (!k4a_image.is_valid()) {
      std::cout << "\033[1;31mInvalid k4a::image provided\033[0m" << std::endl;
      return false;
    }
    
    int rows = k4a_image.get_height_pixels();
    int cols = k4a_image.get_width_pixels();
    
    // Check buffer size
    size_t buffer_size = k4a_image.get_size();
    uint8_t *buffer = k4a_image.get_buffer();
    
    // Calculate expected buffer size
    size_t expected_size = rows * cols * 4; // 4 bytes per pixel for BGRA
    
    // Get the stride (bytes per row) from the image
    size_t stride = k4a_image.get_stride_bytes();
    
    // Check image format
    k4a_image_format_t format = k4a_image.get_format();
    
    if (stride == 0 || buffer_size < expected_size) {
      // Handle compressed image
      std::vector<uint8_t> compressed_data(buffer, buffer + buffer_size);
      
      // Decode the compressed image
      cv::Mat decoded_image = cv::imdecode(compressed_data, cv::IMREAD_COLOR);
      
      if (decoded_image.empty()) {
        std::cout << "\033[1;31mFailed to decode compressed image!\033[0m" << std::endl;
        return false;
      }

      // Convert to BGR if necessary (OpenCV imdecode usually returns BGR)
      if (decoded_image.channels() == 3) {
        output_mat = decoded_image.clone();
      } else {
        cv::cvtColor(decoded_image, output_mat, cv::COLOR_BGRA2BGR);
      }
    } else {
      // Handle raw BGRA data
      try {
        cv::Mat temp_mat = cv::Mat(rows, cols, CV_8UC4, (void *)buffer, stride);
        cv::cvtColor(temp_mat, output_mat, cv::COLOR_BGRA2BGR);
      } catch (const cv::Exception& e) {
        std::cout << "\033[1;31mColor conversion failed: " << e.what() << "\033[0m" << std::endl;
        return false;
      } catch (const std::exception& e) {
        std::cout << "\033[1;31mException creating Mat: " << e.what() << "\033[0m" << std::endl;
        return false;
      }
    }
    
    // Final check if Mat was created successfully
    if (output_mat.empty()) {
      std::cout << "\033[1;31mFailed to create cv::Mat from k4a::image!\033[0m" << std::endl;
      return false;
    }  

    return true;
  }
/*
static bool point_3d_depth_to_color(k4a_transformation_t transformation_handle,
                                       const k4a_image_t depth_image,
                                       const k4a_image_t color_image,
                                       const k4a_float2_t source_point_2d_color,
                                       const k4a_float3_t* target_point_3d)
{
    // transform color image into depth camera geometry
    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    k4a_image_t transformed_depth_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * (int)sizeof(uint16_t),
                                                 &transformed_depth_image))
    {
        printf("Failed to create transformed depth image\n");
        return false;
    }

    k4a_image_t point_cloud_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                 &point_cloud_image))
    {
        printf("Failed to create point cloud image\n");
        return false;
    }

    
    if (K4A_RESULT_SUCCEEDED !=
        k4a_transformation_depth_image_to_color_camera(transformation_handle, depth_image, transformed_depth_image))
    {
        printf("Failed to compute transformed depth image\n");
        return false;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation_handle,
                                                                              transformed_depth_image,
                                                                              K4A_CALIBRATION_TYPE_COLOR,
                                                                              point_cloud_image))
    {
        printf("Failed to compute point cloud\n");
        return false;
    }

    k4a_image_release(transformed_depth_image);
    k4a_image_release(point_cloud_image);

    return true;
}

*/
static bool point_cloud_depth_to_color(k4a_transformation_t transformation_handle,
                                       const k4a_image_t depth_image,
                                       const k4a_image_t color_image,
                                       const k4a_float3_t* marker_corners_3d,
                                       std::string file_name)
{
    // transform color image into depth camera geometry
    int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
    int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
    k4a_image_t transformed_depth_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * (int)sizeof(uint16_t),
                                                 &transformed_depth_image))
    {
        printf("Failed to create transformed depth image\n");
        return false;
    }

    k4a_image_t point_cloud_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                 color_image_width_pixels,
                                                 color_image_height_pixels,
                                                 color_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                 &point_cloud_image))
    {
        printf("Failed to create point cloud image\n");
        return false;
    }

    
    if (K4A_RESULT_SUCCEEDED !=
        k4a_transformation_depth_image_to_color_camera(transformation_handle, depth_image, transformed_depth_image))
    {
        printf("Failed to compute transformed depth image\n");
        return false;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation_handle,
                                                                              transformed_depth_image,
                                                                              K4A_CALIBRATION_TYPE_COLOR,
                                                                              point_cloud_image))
    {
        printf("Failed to compute point cloud\n");
        return false;
    }

    tranformation_helpers_write_point_cloud(point_cloud_image, color_image, marker_corners_3d, file_name.c_str());

    k4a_image_release(transformed_depth_image);
    k4a_image_release(point_cloud_image);

    return true;
}

// Render a simple top-down projection of a PLY (x,y) into a 2D image for quick visualization.
static cv::Mat render_ply_projection(const std::string& ply_path, int width = 900, int height = 900) {
  std::ifstream in(ply_path);
  if (!in.is_open()) {
    std::cerr << "Cannot open PLY: " << ply_path << "\n";
    return cv::Mat();
  }

  std::string line;
  bool header_done = false;
  std::vector<float> xs, ys;
  std::vector<cv::Vec3b> cols;
  float minx = std::numeric_limits<float>::max();
  float maxx = std::numeric_limits<float>::lowest();
  float miny = std::numeric_limits<float>::max();
  float maxy = std::numeric_limits<float>::lowest();

  while (std::getline(in, line)) {
    if (!header_done) {
      if (line == "end_header") {
        header_done = true;
      }
      continue;
    }

    float x, y, z, r, g, b;
    std::istringstream iss(line);
    if (!(iss >> x >> y >> z >> r >> g >> b)) {
      continue;
    }
    xs.push_back(x);
    ys.push_back(y);
    cols.emplace_back(static_cast<unsigned char>(b), static_cast<unsigned char>(g), static_cast<unsigned char>(r));
    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (y < miny) miny = y;
    if (y > maxy) maxy = y;
  }

  if (xs.empty()) {
    std::cerr << "No points parsed from PLY: " << ply_path << "\n";
    return cv::Mat();
  }

  float range_x = std::max(1e-6f, maxx - minx);
  float range_y = std::max(1e-6f, maxy - miny);

  cv::Mat img(height, width, CV_8UC3, cv::Scalar::all(0));
  for (size_t i = 0; i < xs.size(); ++i) {
    int px = static_cast<int>(((xs[i] - minx) / range_x) * (width - 1));
    int py = static_cast<int>(((maxy - ys[i]) / range_y) * (height - 1));
    if (px < 0 || px >= width || py < 0 || py >= height) continue;
    img.at<cv::Vec3b>(py, px) = cols[i];
  }

  // mirror vertically
  cv::flip(img, img, 0);
  return img;
}

CamTestOptions parse_options(int argc, char** argv) {
  CamTestOptions opts;
  const char* options_str = "hDS:s:a:m:V:N:brnf:e:d:F:z:W:H:M";
  int c;
  while ((c = getopt(argc, argv, options_str)) != -1) {
    switch (c) {
      // Reminder: add new options to 'options_str' above and print_usage()!
      case 'h': print_usage(argv[0], stdout); exit(0); break;
      case 'D': opts.params.segDecimate = true; break;
      case 'S': opts.params.sigma = atof(optarg); break;
      case 's': opts.params.segSigma = atof(optarg); break;
      case 'a': opts.params.thetaThresh = atof(optarg); break;
      case 'm': opts.params.magThresh = atof(optarg); break;
      case 'V': opts.params.adaptiveThresholdValue = atof(optarg); break;
      case 'N': opts.params.adaptiveThresholdRadius = atoi(optarg); break;
      case 'b': opts.params.refineBad = true; break;
      case 'r': opts.params.refineQuads = true; break;
      case 'n': opts.params.newQuadAlgorithm = true; break;
      case 'f': opts.family_str = optarg; break;
      case 'e': opts.error_fraction = atof(optarg); break;
      case 'd': opts.device_num = atoi(optarg); break;
      case 'F': opts.focal_length = atof(optarg); break;
      case 'z': opts.tag_size = atof(optarg); break;
      case 'W': opts.frame_width = atoi(optarg); break;
      case 'H': opts.frame_height = atoi(optarg); break;
      case 'M': opts.mirror_display = !opts.mirror_display; break;
      default:
        fprintf(stderr, "\n");
        print_usage(argv[0], stderr);
        exit(1);
    }
  }
  opts.params.adaptiveThresholdRadius += (opts.params.adaptiveThresholdRadius+1) % 2;
  return opts;
}

int main(int argc, char** argv) {

  CamTestOptions opts = parse_options(argc, argv);

  TagFamily family(opts.family_str);

  if (opts.error_fraction >= 0 && opts.error_fraction <= 1) {
    family.setErrorRecoveryFraction(opts.error_fraction);
  }

  std::cout << "family.minimumHammingDistance = " << family.minimumHammingDistance << "\n";
  std::cout << "family.errorRecoveryBits = " << family.errorRecoveryBits << "\n";
  

  // Setup azure kinect and open camera
  
  k4a_device_configuration_t device_config; /**< the Kinect Azure device configuration */
  device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;  
  device_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;

  k4a::device kinect_device; /**< the Kinect Azure device */
  kinect_device = k4a::device::open(0);
  kinect_device.start_cameras(&device_config);

  //camera_sn = kinect_device.get_serialnum();
  k4a::calibration kinect_calibration; /**< the Kinect Azure calibration object */
  kinect_calibration = kinect_device.get_calibration(device_config.depth_mode, device_config.color_resolution);

  k4a::transformation kinect_transform = k4a::transformation(kinect_calibration); /**< the Kinect Azure transformation object */
  
  k4a::capture k4a_rgbd_capture; /**< the Kinect Azure capture object */
  kinect_device.get_capture(&k4a_rgbd_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE));

  k4a::image k4a_color_image; /**< the Kinect Azure color image */
  k4a_color_image = k4a_rgbd_capture.get_color_image();

  k4a::image k4a_depth_image; /**< the Kinect Azure color image */
  k4a_depth_image = k4a_rgbd_capture.get_depth_image();

  /*cv::VideoCapture vc;
  vc.open(opts.device_num);

  if (opts.frame_width && opts.frame_height) {

    // Use uvcdynctrl to figure this out dynamically at some point?
    vc.set(cv::CAP_PROP_FRAME_WIDTH, opts.frame_width);
    vc.set(cv::CAP_PROP_FRAME_HEIGHT, opts.frame_height);
    

  }

  std::cout << "Set camera to resolution: "
            << vc.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << vc.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";

  cv::Mat frame;
  cv::Point2d opticalCenter;

  vc >> frame;
  if (frame.empty()) {
    std::cerr << "no frames!\n";
    exit(1);
  }*/

  cv::Mat frame;
  cv::Point2d opticalCenter;

  while (!k4a_color_image_to_cv_mat(k4a_color_image, frame)) {
    std::cout << "\033[1;31mFailed to convert k4a::image to cv::Mat, retrying...\033[0m" << std::endl;
    kinect_device.get_capture(&k4a_rgbd_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE));
    k4a_color_image = k4a_rgbd_capture.get_color_image();
    k4a_depth_image = k4a_rgbd_capture.get_depth_image();
  }

  opticalCenter.x = frame.cols * 0.5;
  opticalCenter.y = frame.rows * 0.5;

  std::string win = "AprilTag tracker - " + opts.family_str;

  TagDetectorParams& params = opts.params;
  TagDetector detector(family, params);
  
  TagDetectionArray detections;

  k4a_float3_t marker_corners_3d[16];
  int marker_count = 0;

  while (1) {

    //vc >> frame;
    kinect_device.get_capture(&k4a_rgbd_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE));
    k4a_color_image = k4a_rgbd_capture.get_color_image();
    k4a_depth_image = k4a_rgbd_capture.get_depth_image();    
    if (!k4a_color_image_to_cv_mat(k4a_color_image, frame)) {
      std::cout << "\033[1;31mFailed to convert k4a::image to cv::Mat!\033[0m" << std::endl;
      continue;
    }
    if (frame.empty()) { break; }
    // reset marker corners array for this frame
    marker_count = 0;
    for (int mi = 0; mi < 16; ++mi) {
      marker_corners_3d[mi].xyz.x = 0.0f;
      marker_corners_3d[mi].xyz.y = 0.0f;
      marker_corners_3d[mi].xyz.z = 0.0f;
    }

    detector.process(frame, opticalCenter, detections);

    cv::Mat show;
    if (detections.empty()) {

      show = frame;

    } else {

      //show = family.superimposeDetections(frame, detections);
      show = frame;
      
      for (size_t i=0; i<detections.size(); ++i) {
        const TagDetection &d = detections[i];

        // Draw a small filled circle on each corner of the detected tag
        cv::Scalar cornerColor(0, 255, 0); // green (B,G,R)
        const int cornerRadius = 4;
        const int cornerThickness = -1; // filled
        for (int j = 0; j < 4; ++j) {
          cv::Point pt(cvRound(d.p[j].x), cvRound(d.p[j].y));
          cv::circle(show, pt, cornerRadius, cornerColor, cornerThickness, cv::LINE_AA);
        }

        // Prepare ID text and draw it centered at the tag's center
        std::string idText = std::to_string(d.id);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int textThickness = 2;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(idText, fontFace, fontScale, textThickness, &baseline);
        cv::Point centerPt(cvRound(d.cxy.x), cvRound(d.cxy.y));
        cv::Point textOrg(centerPt.x - textSize.width/2, centerPt.y + textSize.height/2);

        // Draw black outline for readability, then white text
        cv::putText(show, idText, textOrg, fontFace, fontScale, cv::Scalar(0,0,0), textThickness + 2, cv::LINE_AA);
        cv::putText(show, idText, textOrg, fontFace, fontScale, cv::Scalar(255,255,255), textThickness, cv::LINE_AA);

        for (int corner_idx = 0; corner_idx < 4; ++corner_idx) {
          // Get the corner points on azure depth image
          k4a_float2_t color_corner;
          color_corner.xy.x = static_cast<float>(d.p[corner_idx].x);
          color_corner.xy.y = static_cast<float>(d.p[corner_idx].y);

          k4a_float2_t depth_corner;
          kinect_calibration.convert_color_2d_to_depth_2d(color_corner, k4a_depth_image, &depth_corner);
          //std::cout << "Tag ID: " << d.id
          //          << " Color Corner (x,y): (" << color_corner.xy.x << ", " << color_corner.xy.y << ")"
          //          << " Depth Corner (x,y): (" << depth_corner.xy.x << ", " << depth_corner.xy.y << ")\n";

          uint16_t depth_value = 0;
          if (k4a_depth_image.get_size() > 0) {
            int depth_x = static_cast<int>(depth_corner.xy.x);
            int depth_y = static_cast<int>(depth_corner.xy.y);
            if (depth_x >= 0 && depth_x < k4a_depth_image.get_width_pixels() &&
                depth_y >= 0 && depth_y < k4a_depth_image.get_height_pixels()) {
              uint16_t* depth_buffer = reinterpret_cast<uint16_t*>(k4a_depth_image.get_buffer());
              depth_value = depth_buffer[depth_y * k4a_depth_image.get_width_pixels() + depth_x];
            }
          }
          k4a_float2_t depth_point;
          depth_point.xy.x = depth_corner.xy.x;
          depth_point.xy.y = depth_corner.xy.y;
          k4a_float3_t this_corner_3d;
          kinect_calibration.convert_2d_to_3d(depth_point, static_cast<float>(depth_value), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &this_corner_3d);
          
          // store into marker array if space
          if (marker_count < 16) {
            marker_corners_3d[marker_count] = this_corner_3d;
            ++marker_count;
          }
          
        }

      }
                                                          

    }

    if (opts.mirror_display) {
      cv::flip(show, show, 1);
    }

    cv::imshow(win, show);
    int k = cv::waitKey(5);
    if (k % 256 == 's') {
      // Save color image
      cv::imwrite("./images/azure/frame.png", frame);
      std::cout << "wrote frame.png\n";

      // Save a colored PLY point cloud aligned to the color image and add marker corner points
      try {
        k4a_transformation_t kinect_transform_handle = k4a_transformation_create(&kinect_calibration);
        k4a_image_t k4a_depth_image_handle = k4a_depth_image.handle();
        k4a_image_t k4a_color_image_handle = k4a_color_image.handle();
        bool pc_success = point_cloud_depth_to_color(kinect_transform_handle,
                       k4a_depth_image_handle,
                       k4a_color_image_handle,
                       marker_corners_3d,
                       "./images/azure/point_cloud.ply");
        if (pc_success) {
          std::cout << "wrote point_cloud.ply\n";
        } else {
          std::cerr << "Failed to write point_cloud.ply\n";
        }
        k4a_transformation_destroy(kinect_transform_handle);
        break;

      } catch (const std::exception &e) {
        std::cerr << "Exception saving PLY: " << e.what() << "\n";
        break;
      }

    }

  }    

  // show point cloud with markers
  try {
    cv::Mat pc_vis = render_ply_projection("./images/azure/point_cloud.ply", 900, 900);
    if (!pc_vis.empty()) {
      cv::imshow("Saved point cloud", pc_vis);
      cv::waitKey(0);
    } else {
      std::cerr << "Unable to render point_cloud.ply for visualization\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Exception while visualizing point cloud: " << e.what() << "\n";
  }

  detector.reportTimers();

  kinect_device.stop_cameras();
  kinect_device.close();

  return 0;


}
