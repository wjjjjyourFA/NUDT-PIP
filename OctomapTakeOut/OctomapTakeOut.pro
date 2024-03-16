TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt
TARGET = OctomapTakeOut
# DESTDIR += ./../../../bin/
DESTDIR += ./bin/

QMAKE_CXXFLAGS += -DFH_DEBUG
QMAKE_CFLAGS_ISYSTEM = -I

CONFIG(debug, debug|release) {
    TARGET = octomap_Debug
    OBJECTS_DIR = ./objects
    LIBS += -L/usr/local/lib/opencv_debug
}
CONFIG(release, debug|release) {
    TARGET = octomap_Release
    OBJECTS_DIR = ./objects
    LIBS += -L/usr/local/lib/opencv_release
    QMAKE_CXXFLAGS_RELEASE += -O3 -D NDEBUG -D BOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG #full optimization
}

SOURCES += \
    analysis_bt.cpp \
    colors.cpp
    
HEADERS += \
    colors.hpp \
    point.h

INCLUDEPATH += \
    ./../../../thirdpart/octomap-1.9.0/dynamicEDT3D/include \
    ./../../../thirdpart/octomap-1.9.0/octomap/include \
    ./../../../thirdpart/octomap-1.9.0/octovis/include \
    /usr/include/boost \
    /usr/include/eigen3 \
    /usr/include/pcl-1.10 \
#    /usr/include/pcl-1.11 \
    /usr/include/vtk-7.1 \
    /usr/include

LIBS += \
    -L/usr/lib/x86_64-linux-gnu \
    -L./../../../thirdpart/octomap-1.9.0/lib \
    -lpcl_common -lpcl_features -lpcl_filters -lpcl_io -lpcl_io_ply -lpcl_kdtree -lpcl_keypoints \
    -lpcl_octree -lpcl_outofcore -lpcl_people -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search \
    -lpcl_segmentation -lpcl_surface -lpcl_tracking -lpcl_visualization \
    -lboost_system -lboost_filesystem -lboost_thread \
    -lvtkCommonCore-7.1 -lvtkFiltersCore-7.1 -lvtksys-7.1 -lvtkRenderingCore-7.1 -lvtkFiltersHybrid-7.1 -lvtkCommonDataModel-7.1 -lvtkCommonMath-7.1 \
    -ldynamicedt3d -loctomap -loctomath
