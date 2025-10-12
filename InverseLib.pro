QT = core

CONFIG += c++17 cmdline

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += include/GA
INCLUDEPATH += include/MCMC
INCLUDEPATH += Utilities

DEFINES += GSL

SOURCES += \
        Utilities/Matrix.cpp \
        Utilities/Matrix_arma.cpp \
        Utilities/Vector.cpp \
        Utilities/Vector_arma.cpp \
        Utilities/QuickSort.cpp \
        Utilities/Utilities.cpp \
        levenbergmarquardt.hpp \
        main.cpp \
        observation.cpp \
        parameter.cpp \
        parameter_set.cpp \
        polynomialmodel.cpp \
        src/GA/Binary.cpp \
        src/GA/Distribution.cpp \
        src/GA/DistributionNUnif.cpp \
        src/GA/Individual.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    Utilities/Matrix.h \
    Utilities/Matrix_arma.h \
    Utilities/QuickSort.h \
    Utilities/TimeSeries.h \
    Utilities/TimeSeries.hpp \
    Utilities/TimeSeriesSet.h \
    Utilities/TimeSeriesSet.hpp \
    Utilities/Utilities.h \
    include/GA/Binary.h \
    include/GA/Distribution.h \
    include/GA/DistributionNUnif.h \
    include/GA/GA.h \
    include/GA/GA.hpp \
    include/GA/Individual.h \
    include/MCMC/MCMC.h \
    include/MCMC/MCMC.hpp \
    levenbergmarquardt.h \
    observation.h \
    parameter.h \
    parameter_set.h \
    polynomialmodel.h


QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -fopenmp

linux {
    #sudo apt-get install libblas-dev liblapack-dev
     DEFINES += ARMA_USE_LAPACK ARMA_USE_BLAS
     LIBS += -larmadillo -llapack -lblas -lgsl -lopenblas

}
