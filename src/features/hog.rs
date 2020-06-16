use crate::core::*;
use crate::processing::conv::ConvolutionExt;
use ndarray::{prelude::*, s, DataMut};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Gradient {
    /// Magnitude of the gradient
    magnitude: Array2<f64>,
    /// Angle of the gradient
    angle: Array2<f64>,
}

#[derive(Default)]
pub struct HistogramOfGradientsBuilder {
    /// Number of orientations in the historgram
    orientations: Option<usize>,
    /// Number of pixels in a cell
    cell_width: Option<usize>,
    /// Width of an NxN block of cells
    block_width: Option<usize>,
}

pub struct HistogramOfGradientsExtractor {
    orientations: usize,
    cell_width: usize,
    block_width: usize,
}

fn get_image_gradients<T>(data: &ArrayBase<T, Ix3>) -> Gradient
where
    T: DataMut<Elem = f64>,
{
    let w = vec![-1.0, 0.0, 1.0];
    let hor: ArrayView3<f64> = ArrayView3::from_shape((1, 3, 1), &w).unwrap();
    let mut h_mag = data
        .conv2d(hor)
        .unwrap()
        .slice_mut(s![.., .., 0])
        .to_owned();
    h_mag.mapv_inplace(|x| x.powi(2));

    let ver: ArrayView3<f64> = ArrayView3::from_shape((3, 1, 1), &w).unwrap();
    let mut v_mag = data
        .conv2d(ver)
        .unwrap()
        .slice_mut(s![.., .., 0])
        .to_owned();
    v_mag.mapv_inplace(|x| x.powi(2));

    let magnitude = &h_mag + &v_mag;
    let mut angle = (&v_mag / &h_mag).to_owned();
    angle.mapv_inplace(|x| x.atan());
    Gradient { magnitude, angle }
}

impl HistogramOfGradientsBuilder {
    pub fn build(self) -> HistogramOfGradientsExtractor {
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
            orientations,
            cell_width,
            block_width,
        }
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
    /// Start creating a hog extractor
    pub fn create() -> HistogramOfGradientsBuilder {
        Default::default()
    }

    pub fn get_features<T>(&self, image: &ImageBase<T, Gray>) -> Array1<f64>
    where
        T: DataMut<Elem = f64>,
    {
        let grad = get_image_gradients(&image.data);
        let histograms = self.create_histograms(image.rows(), image.cols(), &grad);
        // descriptor blocks and normalisation
        let mut feature_vector = Vec::new();
        let vec_len = self.feature_len(self.cell_dim((image.rows(), image.cols())));
        feature_vector.reserve(vec_len);
        for window in histograms.windows((self.block_width, self.block_width, self.orientations)) {
            // turn window into block and normalise
            // L2 norm by default
            let norm = window.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let mut block = window.iter().map(|v| v / norm).collect::<Vec<f64>>();
            feature_vector.append(&mut block);
        }

        feature_vector.into()
    }

    fn block_descriptor_len(&self) -> usize {
        self.cell_width.pow(2) * self.orientations
    }

    fn cell_dim(&self, dim: (usize, usize)) -> (usize, usize) {
        (dim.0 / self.cell_width, dim.1 / self.cell_width)
    }

    fn feature_len(&self, cells: (usize, usize)) -> usize {
        let blocks_wide = cells.1 - self.block_width;
        let blocks_tall = cells.0 - self.block_width;
        blocks_wide * blocks_tall * self.block_descriptor_len()
    }

    fn create_histograms(&self, rows: usize, cols: usize, grad: &Gradient) -> Array3<f64> {
        let (v_cells, h_cells) = self.cell_dim((rows, cols));
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
