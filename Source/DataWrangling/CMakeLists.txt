SET(DATA_WRANGLING_SOURCES
  Kaggle/ParseDigitRecognizer.cpp
  Kaggle/TransformDigitRecognizerData.cpp)

ADD_LIBRARY(DataWrangling
  ${DATA_WRANGLING_SOURCES}
  )