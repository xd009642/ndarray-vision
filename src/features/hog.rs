use crate::core::*;
use crate::processing::conv::ConvolutionExt;
use ndarray::{prelude::*, s, DataMut};
use std::f64::consts::PI;
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

pub struct HistogramOfGradientsBuilder {
    /// Method used to calculate the image gradients
    gradient: Option<GradientType>,
    /// Number of orientations in the historgram
    orientations: Option<usize>,
    /// Number of pixels in a cell
    cell_width: Option<usize>,
    /// Width of an NxN block of cells
    block_width: Option<usize>,
}

pub struct HistogramOfGradientsExtractor {
    gradient: GradientType,
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
            orientations,
            cell_width,
            block_width,
        }
    }

    pub fn gradient(mut self, g: GradientType) -> Self {
        self.gradient = Some(g);
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
    pub fn get_features<T>(&self, image: &ImageBase<T, Gray>) -> Array1<f64>
    where
        T: DataMut<Elem = f64>,
    {
        let grad = self.gradient.run(&image.data);
        let histograms = self.create_histograms(image.rows(), image.cols(), &grad);
        // descriptor blocks and normalisation
        let mut feature_vector = Vec::new();
        for window in histograms.windows((self.block_width, self.block_width, self.orientations)) {
            // turn window into block and normalise
            // L2 norm by default
            let norm = window.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let mut block = window.iter().map(|v| v / norm).collect::<Vec<f64>>();
            feature_vector.append(&mut block);
        }

        feature_vector.into()
    }

    fn create_histograms(&self, rows: usize, cols: usize, grad: &Gradient) -> Array3<f64> {
        let h_cells = cols / self.cell_width;
        let v_cells = rows / self.cell_width;
        let mut result = Array3::zeros((v_cells, h_cells, self.orientations));

        let delta = (2.0 * PI) / (self.orientations as f64);
        // Binning
        for r in 0..h_cells {
            for c in 0..v_cells {
                let r_start = r * self.cell_width;
                let r_end = r_start + self.cell_width;
                let c_start = c * self.cell_width;
                let c_end = c_start + self.cell_width;
                for (a, m) in grad
                    .angle
                    .slice(s![r_start..r_end, c_start..c_end])
                    .iter()
                    .zip(
                        grad.magnitude
                            .slice(s![r_start..r_end, c_start..c_end])
                            .iter(),
                    )
                {
                    let bucket = ((a + PI) / delta).floor() as usize;
                    if let Some(v) = result.get_mut((r, c, bucket)) {
                        *v += m;
                    }
                }
            }
        }
        result
    }
}
