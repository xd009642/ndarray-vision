# Change Log

## [Unreleased]
### Added
- HoG feature extractor

### Changed
- Fixed full sobel angle calculation to use atan2 instead of atan

### Removed

## [0.2.0] 2020-06-06
### Added
* Padding strategies (`NoPadding`, `ConstantPadding`, `ZeroPadding`)
* Threshold module with Otsu and Mean threshold algorithms
* Image transformations and functions to create affine transform matrices
* Type alias `Image` for `ImageBase<OwnedRepr<T>, _>` replicated old `Image` type
* Type alias `ImageView` for `ImageBase<ViewRepr<&'a T>, _>`
* Morphology module with dilation, erosion, union and intersection of binary images

### Changed
* Integrated Padding strategies into convolutions
* Updated `ndarray-stats` to 0.2.0 adding `noisy_float` for median change
* [INTERNAL] Disabled code coverage due to issues with tarpaulin and native libraries
* Renamed `Image` to `ImageBase` which can take any implementor of the ndaray `Data` trait
* Made images have `NoPadding` by default
* No pad behaviour now keeps pixels near the edges the same as source value instead of making them black
* Various performance enhancements in convolution and canny functions

### Removed 

## [0.1.1] - 2019-07-31
### Added

### Changed
* Applied zero padding by default in convolutions

### Removed 

## [0.1.0] - 2019-03-24
### Added
* Image type
* Colour Models (RGB, Gray, HSV, CIEXYZ, Channel-less)
* Histogram equalisation
* Image convolutions
* `PixelBound` type to aid in rescaling images
* Canny edge detector
* `KernelBuilder` and `FixedDimensionKernelBuilder` to create kernels
* Builder implementations for Sobel, Gaussian, Box Linear filter, Laplace
* Median filter
* Sobel Operator
* PPM encoding and decoding for images
