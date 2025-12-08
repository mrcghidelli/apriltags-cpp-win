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

#define DEFAULT_CAMERA_DEVICE "opencv"
#define DEFAULT_TAG_FAMILY "Tag36h11"
#define DEFAULT_TAG_SIZE_M 0.22

#ifdef KINECT_AZURE_LIBS
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4arecord/playback.h>
#include "azure_kinect_helpers.h"
#endif

typedef struct CameraOptions {
  CameraOptions() :
        params(),
        family_str(DEFAULT_TAG_FAMILY),
        camera_type(DEFAULT_CAMERA_DEVICE),
        error_fraction(1),
        device_num(0),
        focal_length(500),
        tag_size(DEFAULT_TAG_SIZE_M),
        frame_width(0),
        frame_height(0),
        mirror_display(false),
        file_path("")
  {
  }
  TagDetectorParams params;
  std::string family_str;
  std::string camera_type;
  double error_fraction;
  int device_num;
  double focal_length;
  double tag_size;
  int frame_width;
  int frame_height;
  bool mirror_display;
  std::string file_path;
} CameraOptions;


void print_usage(const char* tool_name, FILE* output=stderr) {

  TagDetectorParams p;
  CameraOptions o;

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
 -c CAMERA       Use the given camera device, chose between \"azure\" or \"opencv\" (default \"%s\")\n\
 -e FRACTION     Set error detection fraction (default %f)\n\
 -d DEVICE       Set camera device number (default %d)\n\
 -F FLENGTH      Set the camera's focal length in pixels (default %f)\n\
 -z SIZE         Set the tag size in meters (default %f)\n\
 -W WIDTH        Set the camera image width in pixels\n\
 -H HEIGHT       Set the camera image height in pixels\n\
 -M              Toggle display mirroring\n\
 -i FILEPATH     Set recording file path\n",
 
          tool_name,
          p.sigma,
          p.segSigma,
          p.thetaThresh,
          p.magThresh,
          p.adaptiveThresholdValue,
          p.adaptiveThresholdRadius,
          DEFAULT_TAG_FAMILY,
          DEFAULT_CAMERA_DEVICE,
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

CameraOptions parse_options(int argc, char** argv) {
  CameraOptions opts;
  const char* options_str = "hDS:s:a:m:V:N:brnf:c:e:d:F:z:W:H:Mi:";
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
        case 'c': opts.camera_type = optarg; break;
        case 'e': opts.error_fraction = atof(optarg); break;
        case 'd': opts.device_num = atoi(optarg); break;
        case 'F': opts.focal_length = atof(optarg); break;
        case 'z': opts.tag_size = atof(optarg); break;
        case 'W': opts.frame_width = atoi(optarg); break;
        case 'H': opts.frame_height = atoi(optarg); break;
        case 'M': opts.mirror_display = !opts.mirror_display; break;
        case 'i': opts.file_path = optarg; break;
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

    CameraOptions opts = parse_options(argc, argv);

    TagFamily family(opts.family_str);
    std::string win = "AprilTag tracker - " + opts.family_str;

    if (opts.error_fraction >= 0 && opts.error_fraction <= 1) {
    family.setErrorRecoveryFraction(opts.error_fraction);
    }

    TagDetectorParams& params = opts.params;
    TagDetector detector(family, params);
    
    TagDetectionArray detections;

    std::cout << "Using camera type: " << opts.camera_type << std::endl;
    std::cout << "Using tag family: " << opts.family_str << std::endl;
    std::cout << "Tag size: " << opts.tag_size << " meters" << std::endl;

    // Set up camera
    cv::Mat frame;
    cv::Point2d opticalCenter;
    cv::VideoCapture vc;

    // frame to save when 's' is pressed
    cv::Mat saved_frame;

#ifdef KINECT_AZURE_LIBS
    // Kinect Azure variables  
    k4a_device_configuration_t device_config; /**< the Kinect Azure device configuration */
    k4a::device kinect_device; /**< the Kinect Azure device */
    k4a::calibration kinect_calibration; /**< the Kinect Azure calibration object */
    k4a::capture k4a_rgbd_capture; /**< the Kinect Azure capture object */
    k4a::image k4a_color_image; /**< the Kinect Azure color image */
    k4a::image k4a_depth_image; /**< the Kinect Azure color image */
    k4a_playback_t kinect_mkv_playback_handle; /**< the Kinect Azure playback handle from MKV playback */
    k4a_capture_t kinect_mkv_capture_handle; /**< the Kinect Azure capture object from MKV playback */

    k4a_float3_t marker_corners_3d[16];

    // frames to save when 's' is pressed
    k4a::image saved_k4a_depth_image;
    k4a::image saved_k4a_color_image;
    k4a_float3_t saved_marker_corners_3d[16];
    int saved_marker_corners_count = 0;
#endif

    if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
        if (!opts.file_path.empty()) {
            std::cout << "\n[MKV Recording Mode]" << std::endl;
            std::cout << "Loading Azure Kinect MKV file: " << opts.file_path << std::endl;
            std::cout << std::endl;

            // check if file exists 
            std::ifstream mkv_file(opts.file_path);
            if (!mkv_file.good()) {
                std::cerr << "Error: MKV file does not exist: " << opts.file_path << std::endl;
                return -1;
            }

            // A C++ Wrapper for the k4a_playback_t does not exist, so we use the C API directly
            if (k4a_playback_open(opts.file_path.c_str(), &kinect_mkv_playback_handle) != K4A_RESULT_SUCCEEDED) {
                std::cerr << "\033[1;31mFailed to open mkv file for playback\033[0m" << std::endl;
                return -1;
            }
            
            // Check recording length and configuration
            uint64_t recording_length = k4a_playback_get_recording_length_usec(kinect_mkv_playback_handle);
            if (recording_length == K4A_RESULT_SUCCEEDED) {
                std::cout << "Recording length: " << recording_length / 1000000.0 << " seconds" << std::endl;
            }
            
            if (k4a_playback_get_calibration(kinect_mkv_playback_handle, &kinect_calibration) != K4A_RESULT_SUCCEEDED) {
                std::cerr << "\033[1;31mFailed to get calibration from mkv file\033[0m" << std::endl;
                return -1;
            }

            std::cout << "Camera resolution: "
                        << kinect_calibration.color_camera_calibration.resolution_width << "x"
                        << kinect_calibration.color_camera_calibration.resolution_height << "\n";

        }
        else {
            std::cout << "\n[Live Camera Mode]" << std::endl;
            std::cout << "Using Azure Kinect device: " << opts.device_num << std::endl;
            std::cout << std::endl;

            // Setup azure kinect and open camera
            device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
            device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;  
            device_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
            device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;

            kinect_device = k4a::device::open(0);
            kinect_device.start_cameras(&device_config);

            //camera_sn = kinect_device.get_serialnum();
            kinect_calibration = kinect_device.get_calibration(device_config.depth_mode, device_config.color_resolution);

            std::cout << "Camera resolution: "
                        << kinect_calibration.color_camera_calibration.resolution_width << "x"
                        << kinect_calibration.color_camera_calibration.resolution_height << "\n";

        }

        while (!k4a_color_image_to_cv_mat(k4a_color_image, frame)){
            if (!opts.file_path.empty()) {
                // Try to get the next capture
                if (k4a_playback_get_next_capture(kinect_mkv_playback_handle, &kinect_mkv_capture_handle) != K4A_RESULT_SUCCEEDED)
                    std::cout << "\033[1;31mFailed to get next capture from dummy file during setup\033[0m" << std::endl;

                // Wrap the C handle in a C++ object
                k4a_rgbd_capture = k4a::capture(kinect_mkv_capture_handle);
            }
            else{
                std::cout << "\033[1;31mFailed to convert k4a::image to cv::Mat, retrying...\033[0m" << std::endl;
                kinect_device.get_capture(&k4a_rgbd_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE));  
            }

            k4a_color_image = k4a_rgbd_capture.get_color_image();
            k4a_depth_image = k4a_rgbd_capture.get_depth_image();

        }

#endif
    } else if (opts.camera_type == "opencv") {

        if (!opts.file_path.empty()) {
            std::cout << "\n[Recording Mode]" << std::endl;
            std::cout << "Loading recorded file: " << opts.file_path << std::endl;
            std::cout << std::endl;

            if (!std::ifstream(opts.file_path).good()) {
                std::cerr << "Error: video file does not exist: " << opts.file_path << std::endl;
                return -1;
            }

            vc.open(opts.file_path);

            std::cout << "Camera resolution: "
                        << vc.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                        << vc.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
            
        }
        else{
            std::cout << "\n[Live Camera Mode]" << std::endl;
            std::cout << "Using camera device: " << opts.device_num << std::endl;
            std::cout << std::endl;

            vc.open(opts.device_num);

            if (opts.frame_width && opts.frame_height) {
                // Use uvcdynctrl to figure this out dynamically at some point?
                vc.set(cv::CAP_PROP_FRAME_WIDTH, opts.frame_width);
                vc.set(cv::CAP_PROP_FRAME_HEIGHT, opts.frame_height);
            }

            std::cout << "Camera resolution: "
                        << vc.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                        << vc.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
            
        }

        vc >> frame;
        if (frame.empty()) {
            std::cerr << "No frames!\n";
            return -1;
        }
        
    } else {
        std::cerr << "Unknown camera type: " << opts.camera_type << std::endl;
        return -1;
    }
    
    opticalCenter.x = frame.cols * 0.5;
    opticalCenter.y = frame.rows * 0.5;

    // initialize marker corners array
    int marker_corners_count = 0;

    std::cout << "Starting main loop. Press 's' to save the current frame image (and point cloud if available)." << std::endl;

    // acquire frames
    while (true) {
        if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
            if (!opts.file_path.empty()) {
                if (k4a_playback_get_next_capture(kinect_mkv_playback_handle, &kinect_mkv_capture_handle) != K4A_RESULT_SUCCEEDED){
                    std::cout << "\033[1;31mFailed to get next capture from mkv file during setup\033[0m" << std::endl;
                    continue;
                }
                k4a_rgbd_capture = k4a::capture(kinect_mkv_capture_handle);
            }
            else{
                kinect_device.get_capture(&k4a_rgbd_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE));  
            }

            k4a_color_image = k4a_rgbd_capture.get_color_image();
            k4a_depth_image = k4a_rgbd_capture.get_depth_image();   

            if (!k4a_color_image_to_cv_mat(k4a_color_image, frame)) {
                std::cout << "\033[1;31mFailed to convert k4a::image to cv::Mat!\033[0m" << std::endl;
                continue;
            }   
#endif
        } else if (opts.camera_type == "opencv") {
           vc >> frame;
        } else {
            std::cerr << "Unknown camera type: " << opts.camera_type << std::endl;
            return -1;
        }

        if (frame.empty()) { 
            std::cerr << "Empty frame, breaking loop." << std::endl;
            break; 
        }

        // process frames
        detector.process(frame, opticalCenter, detections);

        cv::Mat show = frame;
        
        if (!detections.empty()) {
            for (size_t i=0; i<detections.size(); ++i) {
                const TagDetection &d = detections[i];

                // for each corner of the detected tag compute 3D location
                for (int corner_idx = 0; corner_idx < 4; ++corner_idx) {

                    if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
                        try
                        {
                            // Get the corner points on azure depth image
                            k4a_float2_t color_corner;
                            color_corner.xy.x = static_cast<float>(d.p[corner_idx].x);
                            color_corner.xy.y = static_cast<float>(d.p[corner_idx].y);

                            k4a_float2_t depth_corner;
                            kinect_calibration.convert_color_2d_to_depth_2d(color_corner, k4a_depth_image, &depth_corner);

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
                            marker_corners_3d[marker_corners_count] = this_corner_3d;
                            ++marker_corners_count;
                            
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << "Exception in corner 3D computation: " << e.what() << '\n';
                            continue;                                    
                        }
#endif
                    }
                    else if (opts.camera_type == "opencv") {
                        // For opencv camera, we do not have depth information, so we skip 3D corner computation
                    }
                
                    
                
                }

                // log results

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

            }
        }

        if (opts.mirror_display) {
        cv::flip(show, show, 1);
        }

        cv::imshow(win, show);
        int k = cv::waitKey(5);
        if (k % 256 == 's') {
            
            if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
                saved_k4a_depth_image = k4a_depth_image;
                saved_k4a_color_image = k4a_color_image;
                std::copy(std::begin(marker_corners_3d), std::end(marker_corners_3d), std::begin(saved_marker_corners_3d));
                saved_marker_corners_count = marker_corners_count;
#endif
            }
            else if (opts.camera_type == "opencv") {
                // For opencv camera, we do not have depth information, so we skip point cloud saving
                saved_frame = frame.clone();
            }
            std::cout << "Loaded the current frame image (and point cloud if available) to save." << std::endl;
            std::cout << "The data will be saved in memory when the program exits." << std::endl;
            std::cout << "Press 's' again to overwrite frame to save." << std::endl;
        }
        else if (k % 256 == 27) { // ESC key
            std::cout << "ESC pressed. Exiting main loop." << std::endl;
            break;
        }
       
    }  

    // store / display results
    // Save color image
    cv::imwrite("./images/azure/frame.png", frame);
    std::cout << "wrote frame.png\n";

    if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
        // Save a colored PLY point cloud aligned to the color image and add marker corner points
        try {
            k4a_transformation_t kinect_transform_handle = k4a_transformation_create(&kinect_calibration);
            k4a_image_t k4a_depth_image_handle = saved_k4a_depth_image.handle();
            k4a_image_t k4a_color_image_handle = saved_k4a_color_image.handle();

            bool pc_success = point_cloud_depth_to_color(kinect_transform_handle,
                            k4a_depth_image_handle,
                            k4a_color_image_handle,
                            saved_marker_corners_3d,
                            saved_marker_corners_count,
                            "./images/azure/point_cloud.ply");
                            
            if (pc_success) {
                std::cout << "wrote point_cloud.ply\n";
            } else {
                std::cerr << "Failed to write point_cloud.ply\n";
            }
            
            k4a_transformation_destroy(kinect_transform_handle);

        } catch (const std::exception &e) {
            std::cerr << "Exception saving PLY: " << e.what() << "\n";
        }
#endif
    }
    else if (opts.camera_type == "opencv") {
        // For opencv camera, we do not have depth information, so we skip point cloud saving
        std::cout << "No point cloud saved for opencv camera type." << std::endl;
    }
}