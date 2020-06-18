use crate::core::*;
use crate::processing::SobelExt;
use ndarray::{prelude::*, s, DataMut};
use std::cmp::max;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Gradient {
    /// Magnitude of the gradient
    magnitude: Array3<f64>,
    /// Angle of the gradient
    angle: Array3<f64>,
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
        let (magnitude, angle) = image.data.full_sobel().unwrap();
        let grad = Gradient {
            magnitude,
            angle
        };
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
        let blocks_wide = cells.1 as isize - self.block_width as isize;
        let blocks_tall = cells.0 as isize - self.block_width as isize;
        let blocks_wide = max(1, blocks_wide) as usize;
        let blocks_tall = max(1, blocks_tall) as usize;
        blocks_wide * blocks_tall * self.block_descriptor_len()
    }

    fn create_histograms(&self, rows: usize, cols: usize, grad: &Gradient) -> Array3<f64> {
        let (v_cells, h_cells) = self.cell_dim((rows, cols));
        let mut result = Array3::zeros((v_cells, h_cells, self.orientations));

        let delta = (2.0 * PI) / (self.orientations as f64);
        println!("Angle delta {}", delta * 180.0/PI);
        // Binning
        for r in 0..h_cells {
            for c in 0..v_cells {
                let r_start = r * self.cell_width;
                let r_end = r_start + self.cell_width;
                let c_start = c * self.cell_width;
                let c_end = c_start + self.cell_width;
                for (a, m) in grad
                    .angle
                    .slice(s![r_start..r_end, c_start..c_end, 0])
                    .iter()
                    .zip(
                        grad.magnitude
                            .slice(s![r_start..r_end, c_start..c_end, 0])
                            .iter(),
                    )
                {
                    let a = if *a < 0.0 {
                        a + 2.0 * PI
                    } else {
                        *a
                    };
                    let bucket = (a / delta).floor() as usize;
                    if let Some(v) = result.get_mut((r, c, bucket)) {
                        *v += m;
                    }
                }
            }
        }
        result
    }
}
