TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

MOC_DIR         = ./moc
RCC_DIR         = ./rcc
UI_DIR          = ./qui
OBJECTS_DIR     = ./obj

win32 {
    LIBS += \
        -L'D:\TensorRT-7.2.2.3\lib' nvinfer.lib nvinfer_plugin.lib nvonnxparser.lib\
        -L'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64' cuda.lib cublas.lib cublasLt.lib cudart.lib cudnn.lib
        CONFIG(debug,debug|release){
            LIBS += -L'D:\Opencv\build\install\x64\vc16\lib' opencv_core3411d.lib opencv_dnn3411d.lib opencv_imgcodecs3411d.lib opencv_imgproc3411d.lib
        }
        CONFIG(release,debug|release){
            LIBS += -L'D:\Opencv\build\install\x64\vc16\lib' opencv_core3411.lib opencv_dnn3411.lib opencv_imgcodecs3411.lib opencv_imgproc3411.lib
        }

}

win32 {
    INCLUDEPATH += \
        'D:\TensorRT-7.2.2.3\include' \
        'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include' \
        'D:\Opencv\build\install\include'
}



SOURCES += \
        UpsamplePlugin.cpp \
        calibrator.cpp \
        main.cpp \
        jsoncpp.cpp \
        trt.cpp \
        utils.cpp

HEADERS += \
    utils.h \
    UpsamplePlugin.h \
    UpsmapleKernel.h \
    calibrator.h \
    json.h \
    trt.h \
    yololayer.h \
    hardswish.h


CUDA_SOURCES += \
                UpsampleKernel.cu \
                yololayer.cu \
                hardswish.cu

win32 {
        SYSTEM_NAME = x64
        SYSTEM_TYPE = 64
        CUDA_ARCH = compute_35
        CUDA_CODE = sm_35
        CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
        MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
        MSVCRT_LINK_FLAG_RELEASE   = "/MD"
        # Configuration of the Cuda compiler
        CONFIG(debug, debug|release) {
                # Debug mode
                cuda.input = CUDA_SOURCES
                cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
                cuda.commands = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/nvcc.exe -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
                QMAKE_EXTRA_COMPILERS += cuda
        } else {
                # Release mode
                cuda.input = CUDA_SOURCES
                cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
                cuda.commands = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin/nvcc.exe -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
                QMAKE_EXTRA_COMPILERS += cuda
        }
}

