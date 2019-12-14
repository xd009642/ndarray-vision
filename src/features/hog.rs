use crate::core::*;
use crate::processing::conv::ConvolutionExt;
use ndarray::{prelude::*, s, DataMut};
use std::ops::Fn;

#[derive(Debug, Clone)]
pub struct Gradient {
    /// Magnitude of the gradient
    magnitude: Array2<f64>,
    /// Angle of the gradient
    angle: Array2<f64>,
}

pub enum GradientType {
    /// Horizontal gradients only [-1, 0, 1]
    Horizontal,
    /// Vertical gradients only [-1, 0, 1]^T
    Vertical,
    /// Vertical and Horizontal gradients f(Image, Horizontal) + f(Image, Vertical)
    Full,
    /// Specify a single convolution kernel to apply to the image
    Custom(Box<dyn Fn(ArrayView<f64, Ix3>) -> Gradient>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HogType {
    /// Use a rectangular region to work out the histogram
    Rectangular,
    /// Use a radial region to work out the histogram
    Radial,
}

pub struct HistogramOfGradientsBuilder {
    /// Method used to calculate the image gradients
    gradient: Option<GradientType>,
    /// Binning method for the histograms
    hog_type: Option<HogType>,
    /// Number of orientations in the historgram
    orientations: Option<usize>,
    /// Number of pixels in a cell
    cell_width: Option<usize>,
    /// Width of an NxN block of cells
    block_width: Option<usize>,
}

pub struct HistogramOfGradientsExtractor {
    gradient: GradientType,
    hog_type: HogType,
    orientations: usize,
    cell_width: usize,
    block_width: usize,
}

impl GradientType {
    /// Calculate the gradient for the image data
    pub fn run<T>(&self, data: &ArrayBase<T, Ix3>) -> Gradient
    where
        T: DataMut<Elem = f64>,
    {
        match self {
            Self::Horizontal => {
                let w = vec![-1.0, 0.0, 1.0];
                let hor: Array3<f64> = Array3::from_shape_vec((1, 3, 1), w).unwrap();
                let mag = data
                    .conv2d(hor.view())
                    .unwrap()
                    .slice(s![.., .., 0])
                    .to_owned();
                let dim = mag.dim();
                Gradient {
                    magnitude: mag,
                    angle: Array::zeros(dim),
                }
            }
            Self::Vertical => {
                let w = vec![-1.0, 0.0, 1.0];
                let ver: Array3<f64> = Array3::from_shape_vec((3, 1, 1), w).unwrap();
                let mag = data
                    .conv2d(ver.view())
                    .unwrap()
                    .slice(s![.., .., 0])
                    .to_owned();
                let dim = mag.dim();
                Gradient {
                    magnitude: mag,
                    angle: Array::zeros(dim),
                }
            }
            Self::Full => {
                let w = vec![-1.0, 0.0, 1.0];
                let hor: Array3<f64> = Array3::from_shape_vec((1, 3, 1), w.clone()).unwrap();
                let ver: Array3<f64> = Array3::from_shape_vec((3, 1, 1), w).unwrap();
                let mut magnitude = hor.slice(s![.., .., 0]).mapv(|x| x.powi(2))
                    + ver.slice(s![.., .., 0]).mapv(|x| x.powi(2));
                magnitude.mapv_inplace(|x| x.sqrt());

                let mut angle = (ver / hor).slice(s![.., .., 0]).to_owned();
                angle.mapv_inplace(|x| x.atan());
                Gradient { magnitude, angle }
            }
            Self::Custom(f) => f(data.view()),
        }
    }
}

impl HistogramOfGradientsBuilder {
    pub fn build(self) -> HistogramOfGradientsExtractor {
        let gradient = match self.gradient {
            Some(g) => g,
            None => GradientType::Full,
        };
        let hog_type = match self.hog_type {
            Some(h) => h,
            None => HogType::Rectangular,
        };
        let orientations = match self.orientations {
            Some(o) => o,
            None => 9,
        };
        let cell_width = match self.cell_width {
            Some(c) => c,
            None => 8,
        };
        let block_width = match self.block_width {
            Some(b) => b,
            None => 2,
        };

        HistogramOfGradientsExtractor {
            gradient,
            hog_type,
            orientations,
            cell_width,
            block_width,
        }
    }

    pub fn gradient(mut self, g: GradientType) -> Self {
        self.gradient = Some(g);
        self
    }

    pub fn hog_type(mut self, h: HogType) -> Self {
        self.hog_type = Some(h);
        self
    }

    pub fn orientations(mut self, o: usize) -> Self {
        self.orientations = Some(o);
        self
    }

    pub fn cell_width(mut self, c: usize) -> Self {
        self.cell_width = Some(c);
        self
    }

    pub fn block_width(mut self, b: usize) -> Self {
        self.block_width = Some(b);
        self
    }
}

impl HistogramOfGradientsExtractor {
    pub fn get_features<T>(&self, image: &ImageBase<T, Gray>) -> Array2<f64>
    where
        T: DataMut<Elem = f64>,
    {
        let grad = self.gradient.run(&image.data);
        // Binning

        // descriptor blocks

        // block normalisation

        unimplemented!()
    }
}
