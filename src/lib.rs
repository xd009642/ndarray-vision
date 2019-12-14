//! This crate is a computer vision and image analysis crate built on `ndarray`.
//!
//! By using `ndarray`, this project aims to make full use of other crates in
//! the ecosystem like `ndarray_stats`. This should also allow users to easily
//! integrate other `ndarray` crates with `ndarray-vision` and avoid this crate
//! becoming a monolith by fulfilling every potential usecase in a rather large
//! field. Instead the main focus of this crate will be as follows:
//!
//! * An `Image` type which makes use of Rust to ensure proper and safe use
//! * Conversions between different colour models
//! * Encoding and decoding images
//! * Image processing intrinsics like convolution and a selection of common filters
//! * Common image enhancement algorithms
//! * Geometric image transformations
//! * Intrinsics required for feature detection and matching
//! * Camera Calibration
//! * Frequency domain image processing
//!
//! This may seem like a lot but is still a lot less than OpenCV offers. Also,
//! where possible algorithms will be used from other crates in the ecosystem
//! when those operations aren't Computer Vision specific. For example,
//! `ndarray-stats` has histogram calculation.
//!
//! This crate is a work in progress and as such most of these features aren't
//! yet present and those that are may not be stable. Although, there will be
//! some effort to ensure things don't break.

/// The core of `ndarray-vision` contains the `Image` type and colour models
pub mod core;
/// Image enhancement intrinsics and algorithms
#[cfg(feature="enhancement")]
pub mod enhancement;
/// Feature extraction
pub mod features;
/// Image formats - encoding and decoding images from bytes for saving and
/// loading
#[cfg(feature="format")]
pub mod format;
/// Operations relating to morphological image processing
#[cfg(feature="morphology")]
pub mod morphology;
/// Image processing intrinsics and common filters/algorithms.
#[cfg(feature="processing")]
pub mod processing;
/// Image transforms and warping
#[cfg(feature="transform")]
pub mod transform;
