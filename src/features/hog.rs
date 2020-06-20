use crate::core::*;
use crate::processing::SobelExt;
use ndarray::{prelude::*, s, DataMut};
use std::cmp::{max, min};
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

/// Extract the HoG descriptor from an image and optionally visualise it
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

fn draw_line(start: (isize, isize), end: (isize, isize), image: &mut ArrayViewMut2<u8>) {
    let dim = image.dim();
    let dim = (dim.0 as isize, dim.1 as isize);
    let deltax = (end.1 - start.1) as f64;
    let deltay = (end.0 - start.0) as f64;
    if deltax != 0.0 {
        let deltaerr = (deltay / deltax).abs();
        let mut error = 0.0;
        let mut y = start.0;
        for x in start.1..end.1 {
            if x > 0 && y > 0 && x < dim.1 && y < dim.0 {
                image[[y as usize, x as usize]] = 255;
            }
            error += deltaerr;
            if error >= 0.5 {
                y = y + deltay.signum() as isize;
                error = error - 1.0;
            }
        }
    } else {
        for y in min(start.0, end.0)..max(start.0, end.0) {
            if start.1 > 0 && y > 0 && start.1 < dim.1 && y < dim.0 {
                image[[y as usize, start.1 as usize]] = 255;
            }
        }
    }
}

impl HistogramOfGradientsExtractor {
    /// Start creating a hog extractor
    pub fn create() -> HistogramOfGradientsBuilder {
        Default::default()
    }

    pub fn visualise_features(
        &self,
        image_dims: (usize, usize),
        features: ArrayView1<f64>,
    ) -> Image<u8, Gray> {
        let mut result = Image::new(image_dims.0, image_dims.1);
        let cells = self.cell_dim(image_dims);
        let cell_len = self.block_descriptor_len();
        let centre = (self.cell_width/2, self.cell_width/2);
        let delta = self.angle_delta();
        for row in 0..cells.0 {
            for col in 0..cells.1 {
                let start = row * cells.0 + col;
                let mut vector = features
                    .into_iter()
                    .skip(start * cell_len)
                    .take(self.orientations)
                    .copied()
                    .collect::<Vec<_>>();
                if vector.is_empty() {
                    println!("This shouldn't happen if it's working...");
                    break;
                }
                let norm: f64 = vector.iter().sum::<f64>() + f64::EPSILON;
                vector.iter_mut().for_each(|x| *x /= norm);

                let mut cell = result.data.slice_mut(s![
                    (row * self.cell_width)..((row + 1) * self.cell_width),
                    (col * self.cell_width)..((col + 1) * self.cell_width),
                    0
                ]);
                cell[centre] = 255; 
                // From the features get circle coordinate x = r*sin(theta) y = r*cos(theta)
                // theta is angle of the bin, r is normalised magnitude * cell_width
                // Draw a white line from end coordinate to origin 
                for a in 0..self.orientations {
                    let a_f = a as f64;
                    let x = centre.1 as isize + (vector[a] * (self.cell_width as f64) * (delta*a_f).sin()).round() as isize;
                    let y = centre.0 as isize + (vector[a] * (self.cell_width as f64) * (delta*a_f).cos()).round() as isize;
                    draw_line((centre.0 as isize, centre.1 as isize ), (y, x), &mut cell);
                }
            }
        }
        result
    }

    pub fn get_features<T>(&self, image: &ImageBase<T, Gray>) -> Array1<f64>
    where
        T: DataMut<Elem = f64>,
    {
        let (magnitude, angle) = image.data.full_sobel().unwrap();
        let grad = Gradient { magnitude, angle };
        let histograms = self.create_histograms(image.rows(), image.cols(), &grad);
        // descriptor blocks and normalisation
        let mut feature_vector = Vec::new();
        let vec_len = self.feature_len(self.cell_dim((image.rows(), image.cols())));
        feature_vector.reserve(vec_len);
        for window in histograms.windows((self.block_width, self.block_width, self.orientations)) {
            // turn window into block and normalise
            // L2 norm by default
            let norm = (window.iter().map(|x| x.powi(2)).sum::<f64>() + f64::EPSILON.powi(2)).sqrt();
            let mut block = window.iter().map(|v| v / norm).collect::<Vec<f64>>();
            feature_vector.append(&mut block);
        }

        feature_vector.into()
    }

    fn block_descriptor_len(&self) -> usize {
        self.block_width.pow(2) * self.orientations
    }

    fn cell_dim(&self, dim: (usize, usize)) -> (usize, usize) {
        (dim.0 / self.cell_width, dim.1 / self.cell_width)
    }

    fn angle_delta(&self) -> f64 {
        (2.0 * PI) / (self.orientations as f64)
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

        let delta = self.angle_delta();
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
                    let a = if *a < 0.0 { a + 2.0 * PI } else { *a };
                    let bucket = (a / delta - 0.5).floor() as usize % self.orientations;
                    let centre = delta * (bucket as f64 + 0.5);
                    let vote_2 = m * ((a - centre).abs()) / delta;
                    let vote_1 = m - vote_2;
                    if let Some(v) = result.get_mut((r, c, bucket)) {
                        *v += vote_1;
                    }
                    let bucket_2 = (bucket + 1) % self.orientations;
                    if let Some(v) = result.get_mut((r, c, bucket_2)) {
                        *v += vote_2;
                    }
                }
            }
        }
        result
    }
}
