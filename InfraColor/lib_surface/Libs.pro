TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += plugin
TEMPLATE = lib
TARGET = Getxyz
DESTDIR += ./bin/
QT -= gui

CONFIG(debug, debug|release) {
    OBJECTS_DIR = ./tmp_Debug
}
CONFIG(release, debug|release) {
    OBJECTS_DIR = ./tmp_Release
}

SOURCES += \
    main.cpp

HEADERS += \
    Getxyz.h \
    generator.h \
    Merge.h

DISTFILES += \
    Getxyz.cu \
    generator.cu \
    Merge.cu

INCLUDEPATH += \
    /usr/local/include \
    /usr/include/opencv \
    /usr/include/opencv2 \
    /usr/include/opencv4 \
    /usr/local/cuda/include \

LIBS += `pkg-config opencv4 --cflags --libs`

LIBS += \
    -L/usr/local \
#    /usr/lib/utils.so \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcublas -lcudnn

LIBS += \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcublas -lcurand -lnvrtc \
    -lcudart -lcublas -lnvcaffe_parser -lnvinfer -lnvinfer_plugin -lnvparsers

