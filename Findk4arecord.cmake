# Findk4arecord.cmake - Find k4arecord library
#
# This module defines:
#  k4arecord_FOUND - True if k4arecord is found
#  k4arecord_INCLUDE_DIRS - Include directories for k4arecord
#  k4arecord_LIBRARIES - Libraries to link against k4arecord
#  k4a::k4arecord - Imported target for k4arecord

# Find the header file
find_path(k4arecord_INCLUDE_DIR
    NAMES k4arecord/record.h
    PATHS
        "C:/Program Files/Azure Kinect SDK v1.4.1/sdk/include"
        "C:/Program Files/Azure Kinect SDK v1.4.0/sdk/include"
        "$ENV{KINECT_AZURE_SDK}/sdk/include"
        "$ENV{ProgramFiles}/Azure Kinect SDK v1.4.1/sdk/include"
        "$ENV{ProgramFiles}/Azure Kinect SDK v1.4.0/sdk/include"
        /usr/include
        /usr/local/include
    DOC "Path to k4arecord include directory"
)

# Find the library
find_library(k4arecord_LIBRARY
    NAMES k4arecord
    PATHS
        "C:/Program Files/Azure Kinect SDK v1.4.1/sdk/windows-desktop/amd64/release/lib"
        "C:/Program Files/Azure Kinect SDK v1.4.0/sdk/windows-desktop/amd64/release/lib"
        "$ENV{KINECT_AZURE_SDK}/sdk/windows-desktop/amd64/release/lib"
        "$ENV{ProgramFiles}/Azure Kinect SDK v1.4.1/sdk/windows-desktop/amd64/release/lib"
        "$ENV{ProgramFiles}/Azure Kinect SDK v1.4.0/sdk/windows-desktop/amd64/release/lib"
        /usr/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
    DOC "Path to k4arecord library"
)

# Find the DLL on Windows
if(WIN32)
    find_file(k4arecord_DLL
        NAMES k4arecord.dll
        PATHS
            "C:/Program Files/Azure Kinect SDK v1.4.1/sdk/windows-desktop/amd64/release/bin"
            "C:/Program Files/Azure Kinect SDK v1.4.0/sdk/windows-desktop/amd64/release/bin"
            "$ENV{KINECT_AZURE_SDK}/sdk/windows-desktop/amd64/release/bin"
            "$ENV{ProgramFiles}/Azure Kinect SDK v1.4.1/sdk/windows-desktop/amd64/release/bin"
            "$ENV{ProgramFiles}/Azure Kinect SDK v1.4.0/sdk/windows-desktop/amd64/release/bin"
        DOC "Path to k4arecord DLL"
    )
endif()

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(k4arecord
    FOUND_VAR k4arecord_FOUND
    REQUIRED_VARS k4arecord_LIBRARY k4arecord_INCLUDE_DIR
)

# Set the output variables
if(k4arecord_FOUND)
    set(k4arecord_INCLUDE_DIRS ${k4arecord_INCLUDE_DIR})
    set(k4arecord_LIBRARIES ${k4arecord_LIBRARY})
    
    # Create imported target
    if(NOT TARGET k4a::k4arecord)
        add_library(k4a::k4arecord SHARED IMPORTED)
        set_target_properties(k4a::k4arecord PROPERTIES
            IMPORTED_LOCATION "${k4arecord_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${k4arecord_INCLUDE_DIR}"
        )
        
        # Set the DLL location on Windows
        if(WIN32 AND k4arecord_DLL)
            set_target_properties(k4a::k4arecord PROPERTIES
                IMPORTED_IMPLIB "${k4arecord_LIBRARY}"
                IMPORTED_LOCATION "${k4arecord_DLL}"
            )
        endif()
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(
    k4arecord_INCLUDE_DIR
    k4arecord_LIBRARY
    k4arecord_DLL
)
