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
#include <array>
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
#include <algorithm>
#include <filesystem>

#include <nlohmann/json.hpp>

#include "CameraUtil.h"


#define DEFAULT_CAMERA_DEVICE "opencv"
#define DEFAULT_TAG_FAMILY "Tag36h11"
#define DEFAULT_TAG_SIZE_M 0.22
#define DEFAULT_MIN_ID 0
#define DEFAULT_MAX_ID 20
#define DEFAULT_OUTPUT_DIR "images"

#ifdef KINECT_AZURE_LIBS
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4arecord/playback.h>
#include "azure_kinect_helpers.h"
#endif

using json = nlohmann::json;

typedef struct CameraOptions {
    CameraOptions() :
        params(),
        family_str(DEFAULT_TAG_FAMILY),
        min_id(DEFAULT_MIN_ID),
        max_id(DEFAULT_MAX_ID),
        camera_type(DEFAULT_CAMERA_DEVICE),
        error_fraction(1),
        device_num(0),
        focal_length(500),
        tag_size(DEFAULT_TAG_SIZE_M),
        frame_width(0),
        frame_height(0),
        mirror_display(false),
        input_file_path(""),
        output_file_path(DEFAULT_OUTPUT_DIR)
    {
    }
    TagDetectorParams params;
    std::string family_str;
    int min_id;
    int max_id;
    std::string camera_type;
    double error_fraction;
    int device_num;
    double focal_length;
    double tag_size;
    int frame_width;
    int frame_height;
    bool mirror_display;
    std::string input_file_path;
    std::string output_file_path;
} CameraOptions;

// Struttura per memorizzare i corner di un tag nel frame
struct TagCorners {
    int id;
    std::vector<cv::Point3f> corners; // 4 corner points
};

// Mappa per accumulare dati corner: frame_number -> vector<TagCorners>
std::map<int, std::vector<TagCorners>> frames_corners_data;

// Function to save corners data to JSON file
void save_corners_to_json(const std::string& filename) {
    json output = json::array();
    
    // Iterate through each frame (std::map keeps frames sorted by frame_number)
    for (const auto& frame_pair : frames_corners_data) {
        int frame_num = frame_pair.first;
        const auto& tags_in_frame = frame_pair.second;
        
        json frame_obj;
        frame_obj["frame_number"] = frame_num;
        
        json tags_array = json::array();
        
        // Iterate through each tag detected in this frame
        for (const auto& tag : tags_in_frame) {
            json tag_entry;
            tag_entry["id"] = tag.id;
            
            // Add the 4 corner points for this tag
            json corners_array = json::array();
            for (const auto& corner : tag.corners) {
                json corner_obj;
                corner_obj["x"] = corner.x;
                corner_obj["y"] = corner.y;
                corner_obj["z"] = corner.z;
                corners_array.push_back(corner_obj);
            }
            
            tag_entry["corners"] = corners_array;
            tags_array.push_back(tag_entry);
        }
        
        frame_obj["tags"] = tags_array;
        output.push_back(frame_obj);
    }
    
    // Write to file
    std::ofstream json_file(filename);
    if (json_file.is_open()) {
        json_file << output.dump(4); // Pretty print with indentation
        json_file.close();
        std::cout << "Saved corner data to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open JSON file for writing: " << filename << std::endl;
    }
}

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
 -t MIN_ID       Set the minimum tag ID to detect (default %d)\n\
 -T MAX_ID       Set the maximum tag ID to detect (default %d)\n\
 -c CAMERA       Use the given camera device, chose between \"azure\" or \"opencv\" (default \"%s\")\n\
 -e FRACTION     Set error detection fraction (default %f)\n\
 -d DEVICE       Set camera device number (default %d)\n\
 -F FLENGTH      Set the camera's focal length in pixels (default %f)\n\
 -z SIZE         Set the tag size in meters (default %f)\n\
 -W WIDTH        Set the camera image width in pixels\n\
 -H HEIGHT       Set the camera image height in pixels\n\
 -M              Toggle display mirroring\n\
 -i FILEPATH     Set recording file path\n\
 -o FILEPATH     Set output directory for saved images (default \"%s\" or input file directory if not specified)\n",
 
            tool_name,
            p.sigma,
            p.segSigma,
            p.thetaThresh,
            p.magThresh,
            p.adaptiveThresholdValue,
            p.adaptiveThresholdRadius,
            DEFAULT_TAG_FAMILY,
            DEFAULT_MIN_ID,
            DEFAULT_MAX_ID,
            DEFAULT_CAMERA_DEVICE,
            o.error_fraction,
            o.device_num,
            o.focal_length,
            o.tag_size,
            DEFAULT_OUTPUT_DIR);

  fprintf(output, "Known tag families:");
  TagFamily::StringArray known = TagFamily::families();
  for (size_t i = 0; i < known.size(); ++i) {
    fprintf(output, " %s", known[i].c_str());
  }
  fprintf(output, "\n");
}

CameraOptions parse_options(int argc, char** argv) {
  CameraOptions opts;
  const char* options_str = "hDS:s:a:m:V:N:brnf:t:T:c:e:d:F:z:W:H:Mi:o:";
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
        case 't': opts.min_id = atoi(optarg); break;
        case 'T': opts.max_id = atoi(optarg); break;
        case 'c': opts.camera_type = optarg; break;
        case 'e': opts.error_fraction = atof(optarg); break;
        case 'd': opts.device_num = atoi(optarg); break;
        case 'F': opts.focal_length = atof(optarg); break;
        case 'z': opts.tag_size = atof(optarg); break;
        case 'W': opts.frame_width = atoi(optarg); break;
        case 'H': opts.frame_height = atoi(optarg); break;
        case 'M': opts.mirror_display = !opts.mirror_display; break;
        case 'i': opts.input_file_path = optarg; break;
        case 'o': opts.output_file_path = optarg; break;
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

    int max_num_corners_marker = (opts.max_id - opts.min_id + 1)*4;

    // Frame counter for JSON data
    int frame_number = 0;

    // check options if valid
    // opts.camera_type should be either "azure" or "opencv"
    // if "azure" is selected, the program will try to use Kinect Azure SDK
    // and if not found, it will exit with error
    // if "opencv" is selected, it will use OpenCV VideoCapture
    if (opts.camera_type == "azure"){
#ifndef KINECT_AZURE_LIBS
        std::cerr << "\033[1;31mError: Azure Kinect SDK not found. Please install the SDK or use \"opencv\" camera option.\033[0m" << std::endl;
        return -1;
#endif
    }
    
    std::cout << "Using camera type: " << opts.camera_type << std::endl;
    std::cout << "Using tag family: " << opts.family_str << std::endl;
    std::cout << "Tag size: " << opts.tag_size << " meters" << std::endl;

    if (opts.input_file_path != "") {
        std::cout << "Recording file path: " << opts.input_file_path << std::endl;
    }
    if (opts.output_file_path != DEFAULT_OUTPUT_DIR) {
        std::cout << "Output directory for saved images: " << opts.output_file_path << std::endl;
    }

    // Set up camera
    cv::Mat frame;
    cv::Point2d opticalCenter;
    cv::VideoCapture vc;

    // prepare variables for 3D projection ("opencv" camera type)
    double s = opts.tag_size;
    double ss = 0.5*s;

    enum { npoints = 4 };

    cv::Point3d src[npoints] = {
        cv::Point3d(-ss, -ss, 0),
        cv::Point3d( ss, -ss, 0),
        cv::Point3d( ss,  ss, 0),
        cv::Point3d(-ss,  ss, 0),
    };

    double f = opts.focal_length;

    double K[9] = {
        f, 0, opticalCenter.x,
        0, f, opticalCenter.y,
        0, 0, 1
    };

    cv::Mat_<cv::Point3d> srcmat(npoints, 1, src);

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

    std::vector<k4a_float3_t> marker_corners_3d(max_num_corners_marker);

    // frames to save when 's' is pressed
    k4a::image saved_k4a_depth_image;
    k4a::image saved_k4a_color_image;
    std::vector<k4a_float3_t> saved_marker_corners_3d(max_num_corners_marker);
    int saved_marker_corners_count = 0;
#endif

    if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
        if (!opts.input_file_path.empty()) {
            std::cout << "\n[MKV Recording Mode]" << std::endl;
            std::cout << "Loading Azure Kinect MKV file: " << opts.input_file_path << std::endl;
            std::cout << std::endl;

            // check if file exists 
            std::ifstream mkv_file(opts.input_file_path);
            if (!mkv_file.good()) {
                std::cerr << "Error: MKV file does not exist: " << opts.input_file_path << std::endl;
                return -1;
            }

            // A C++ Wrapper for the k4a_playback_t does not exist, so we use the C API directly
            if (k4a_playback_open(opts.input_file_path.c_str(), &kinect_mkv_playback_handle) != K4A_RESULT_SUCCEEDED) {
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

            try {
                kinect_device = k4a::device::open(opts.device_num);
            } catch (const std::exception& e) {
                std::cerr << "\033[1;31mFailed to open Azure Kinect device. Error: " << e.what() << "\033[0m" << std::endl;
                return -1;
            }

            kinect_device.start_cameras(&device_config);

            //camera_sn = kinect_device.get_serialnum();
            kinect_calibration = kinect_device.get_calibration(device_config.depth_mode, device_config.color_resolution);

            std::cout << "Camera resolution: "
                        << kinect_calibration.color_camera_calibration.resolution_width << "x"
                        << kinect_calibration.color_camera_calibration.resolution_height << "\n";

            std::cout << "Intrinsics: " << std::endl;
            std::cout << kinect_calibration.color_camera_calibration.intrinsics.parameters.param.fx << " "
                      << kinect_calibration.color_camera_calibration.intrinsics.parameters.param.fy << " "
                      << kinect_calibration.color_camera_calibration.intrinsics.parameters.param.cx << " "
                      << kinect_calibration.color_camera_calibration.intrinsics.parameters.param.cy << std::endl;
            



        }

        while (!k4a_color_image_to_cv_mat(k4a_color_image, frame)){
            if (!opts.input_file_path.empty()) {
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

        if (!opts.input_file_path.empty()) {
            std::cout << "\n[Recording Mode]" << std::endl;
            std::cout << "Loading recorded file: " << opts.input_file_path << std::endl;
            std::cout << std::endl;

            if (!std::ifstream(opts.input_file_path).good()) {
                std::cerr << "Error: video file does not exist: " << opts.input_file_path << std::endl;
                return -1;
            }

            vc.open(opts.input_file_path);

            std::cout << "Camera resolution: "
                        << vc.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                        << vc.get(cv::CAP_PROP_FRAME_HEIGHT) << "\n";
            
        }
        else{
            std::cout << "\n[Live Camera Mode]" << std::endl;
            std::cout << "Using camera device: " << opts.device_num << std::endl;
            std::cout << std::endl;

            vc.open(opts.device_num);

            if (!vc.isOpened()) {
                std::cerr << "Error: could not open camera device: " << opts.device_num << std::endl;
                return -1;
            }

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
        marker_corners_count = 0; // reset per frame to avoid stale or overflowed data
        if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
            if (!opts.input_file_path.empty()) {
                
                k4a_stream_result_t get_capture_result = k4a_playback_get_next_capture(kinect_mkv_playback_handle, &kinect_mkv_capture_handle);
                while (get_capture_result == K4A_STREAM_RESULT_FAILED){
                    get_capture_result = k4a_playback_get_next_capture(kinect_mkv_playback_handle, &kinect_mkv_capture_handle);
                    std::cout << "\033[1;31mFailed to get next capture from mkv file during setup\033[0m" << std::endl;
                }

                if (get_capture_result == K4A_STREAM_RESULT_EOF) {
                    std::cout << "End of MKV file reached" << std::endl;
                    break; 
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

                // Filter by ID range
                if (d.id < opts.min_id || d.id > opts.max_id) {
                    continue; // Skip this detection
                }
                
                TagCorners tag_corners;
                tag_corners.id = d.id;
                
                if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
                    // for each corner of the detected tag compute 3D location
                    for (int corner_idx = 0; corner_idx < 4; ++corner_idx) {
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
                            if (marker_corners_count < max_num_corners_marker) {
                                marker_corners_3d[marker_corners_count] = this_corner_3d;

                                // also store into tag corners for JSON export
                                tag_corners.corners.push_back(cv::Point3f(this_corner_3d.xyz.x, this_corner_3d.xyz.y, this_corner_3d.xyz.z) * 0.001f); // convert mm to meters
                                ++marker_corners_count;
                            }
                            
                        }
                        catch(const std::exception& e)
                        {
                            std::cerr << "Exception in corner 3D computation: " << e.what() << '\n';
                            continue;                                    
                        }
                    }
#endif
                }
                else if (opts.camera_type == "opencv") {
                    
                    // Get the 4x4 transformation matrix (tag frame -> camera frame)
                    cv::Mat_<double> M = CameraUtil::homographyToPose(f, f, s, detections[i].homography, false);

                    // Extract rotation (3x3) and translation (3x1) from the 4x4 matrix M
                    cv::Mat_<double> R = M.rowRange(0, 3).colRange(0, 3);
                    cv::Mat_<double> t = M.rowRange(0, 3).col(3);

                    // Transform each corner from tag frame to camera frame: p_cam = R * p_tag + t
                    for (int corner_idx = 0; corner_idx < npoints; ++corner_idx) {
                        cv::Point3d p_tag = src[corner_idx];
                        cv::Mat_<double> p_tag_mat = (cv::Mat_<double>(3,1) << p_tag.x, p_tag.y, p_tag.z);
                        cv::Mat_<double> p_cam_mat = R * p_tag_mat + t;
                        
                        cv::Point3f corner_3d(
                            static_cast<float>(p_cam_mat(0, 0)),
                            static_cast<float>(p_cam_mat(1, 0)),
                            static_cast<float>(p_cam_mat(2, 0))
                        );
                        tag_corners.corners.push_back(corner_3d);
                    }

                }

                // Accumulate corner data for JSON export                
                frames_corners_data[frame_number].push_back(tag_corners);
             

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
        
        // Increment frame counter at end of processing
        ++frame_number;
        if (k % 256 == 's') {
            
            if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
                saved_k4a_depth_image = k4a_depth_image;
                saved_k4a_color_image = k4a_color_image;
                std::copy(marker_corners_3d.begin(), marker_corners_3d.end(), saved_marker_corners_3d.begin());
                saved_marker_corners_count = marker_corners_count;
#endif
            }
            else if (opts.camera_type == "opencv") {
                // For opencv camera, we do not have depth information, so we skip point cloud saving
                saved_frame = frame.clone();
            }
            std::cout << "Loaded the current frame image (and point cloud if available) to save." << std::endl;
            std::cout << "The data will be saved in memory when the program exits." << std::endl;
            std::cout << "Press 's' again to overwrite the frame to save." << std::endl;
        }
        else if (k % 256 == 27) { // ESC key
            std::cout << "ESC pressed. Exiting main loop." << std::endl;
            break;
        }
       
    }  

    // Prepare filename and directory
    std::string filename_output;
    std::string base_filename;
    
    // Determine base filename
    if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
        base_filename = opts.input_file_path.empty() 
            ? "SN" + kinect_device.get_serialnum()
            : opts.input_file_path.substr(opts.input_file_path.find_last_of("/\\") + 1, opts.input_file_path.find_last_of(".") - opts.input_file_path.find_last_of("/\\") - 1);
#endif
    }
    else if (opts.camera_type == "opencv") {
        base_filename = opts.input_file_path.empty() 
            ? "camera"
            : opts.input_file_path.substr(opts.input_file_path.find_last_of("/\\") + 1, opts.input_file_path.find_last_of(".") - opts.input_file_path.find_last_of("/\\") - 1);
    }
    
    // Combine with output directory
    if (opts.output_file_path != DEFAULT_OUTPUT_DIR) {
        try {
            std::filesystem::create_directories(opts.output_file_path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating output directory: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Output directory: " << opts.output_file_path << std::endl;
    filename_output = opts.output_file_path + "/" + base_filename;

    // Save color image
    cv::imwrite(filename_output + "_color.png", frame);
    std::cout << "wrote " << filename_output + "_color.png" << "\n";

    // Save corner data to JSON
    save_corners_to_json(filename_output + "_corners.json");
    if (opts.camera_type == "azure") {
#ifdef KINECT_AZURE_LIBS
        // Save a colored PLY point cloud aligned to the color image and add marker corner points
        try {
            if (!saved_k4a_depth_image.is_valid() || !saved_k4a_color_image.is_valid()) {
                std::cerr << "No saved Azure Kinect frame/point cloud to write (press 's' during capture). Skipping PLY export." << std::endl;
                return 0;
            }
            k4a_transformation_t kinect_transform_handle = k4a_transformation_create(&kinect_calibration);
            k4a_image_t k4a_depth_image_handle = saved_k4a_depth_image.handle();
            k4a_image_t k4a_color_image_handle = saved_k4a_color_image.handle();

            bool pc_success = point_cloud_depth_to_color(kinect_transform_handle,
                            k4a_depth_image_handle,
                            k4a_color_image_handle,
                            saved_marker_corners_3d.data(),
                            saved_marker_corners_count,
                            filename_output + "_point_cloud.ply");
                            
            if (pc_success) {
                std::cout << "wrote " << filename_output + "_point_cloud.ply" << "\n";
            } else {
                std::cerr << "Failed to write " << filename_output + "_point_cloud.ply" << "\n";
            }
            
            k4a_transformation_destroy(kinect_transform_handle);

        } catch (const std::exception &e) {
            std::cerr << "Exception saving PLY: " << e.what() << "\n";
        }

        kinect_device.stop_cameras();
        kinect_device.close();
#endif
    }
    else if (opts.camera_type == "opencv") {
        // For opencv camera, we do not have depth information, so we skip point cloud saving
        std::cout << "No point cloud saved for opencv camera type." << std::endl;

        vc.release();
    }

    detector.reportTimers();
}