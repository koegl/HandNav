cmake_minimum_required(VERSION 3.13.4)

project(XNav)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/koegl/HandNav")
set(EXTENSION_CATEGORY "Informatics")
set(EXTENSION_CONTRIBUTORS "Fryderyk Kögl (TUM, BWH)")
set(EXTENSION_DESCRIPTION "HandNav is a Slicer extension to extract the data hierarchy of cases")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/koegl/SlicerMRUSLandmarking/main/misc/icon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/koegl/SlicerMRUSLandmarking/main/misc/GUIpreview.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(HandNav)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
