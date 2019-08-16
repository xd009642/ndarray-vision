use crate::core::{ColourModel, Image};
use crate::processing::Error;
use ndarray::prelude::*;
use ndarray::{s, Zip};
use num_traits::{Num, NumAssignOps};
use std::marker::PhantomData;
use std::marker::Sized;

pub struct NoPadding;
pub struct ZeroPadding;
pub struct ExtendPadding;

pub trait ImagePadder where Self: Sized + Copy {
    type Data;

    fn pad(&self, pad_size: (usize, usize)) -> Self;
}


impl<T> ImagePadder<T> for NoPadding {
    fn pad(&self, pad_size: (usize, usize)) -> Self {
        *self
    }
}

/// Perform image convolutions
pub trait ConvolutionExt
where
    Self: Sized,
{
    /// Underlying data type to perform the colution on
    type Data;

    /// Perform a convolution returning the resultant data
    fn conv2d(&self, kernel: ArrayView3<Self::Data>) -> Result<Self, Error>;
    /// Performs the convolution inplace mutating the containers data
    fn conv2d_inplace(&mut self, kernel: ArrayView3<Self::Data>) -> Result<(), Error>;
}

fn kernel_centre(rows: usize, cols: usize) -> (usize, usize) {
    let row_offset = rows / 2 - ((rows % 2 == 0) as usize);
    let col_offset = cols / 2 - ((cols % 2 == 0) as usize);
    (row_offset, col_offset)
}

impl<T> ConvolutionExt for Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps,
{
    type Data = T;

    fn conv2d(&self, kernel: ArrayView3<Self::Data>) -> Result<Self, Error> {
        if self.shape()[2] != kernel.shape()[2] {
            Err(Error::ChannelDimensionMismatch)
        } else {
            let k_s = kernel.shape();
            // Bit icky but handles fact that uncentred convolutions will cross the bounds
            // otherwise
            let (row_offset, col_offset) = kernel_centre(k_s[0], k_s[1]);
            // row_offset * 2 may not equal k_s[0] due to truncation
            let shape_ext = (
                self.shape()[0] + row_offset * 2,
                self.shape()[1] + col_offset * 2,
                self.shape()[2],
            );
            let shape = (self.shape()[0], self.shape()[1], self.shape()[2]);

            if shape.0 > 0 && shape.1 > 0 {
                let mut result = Self::zeros(shape);

                let mut tmp = Self::zeros(shape_ext);
                tmp.slice_mut(s![
                    row_offset..shape_ext.0 - row_offset,
                    col_offset..shape_ext.1 - col_offset,
                    ..
                ])
                .assign(&self);

                Zip::indexed(tmp.windows(kernel.dim())).apply(|(i, j, _), window| {
                    let mult = &window * &kernel;
                    let sums = mult.sum_axis(Axis(0)).sum_axis(Axis(0));
                    result.slice_mut(s![i, j, ..]).assign(&sums);
                });
                Ok(result)
            } else {
                Err(Error::InvalidDimensions)
            }
        }
    }

    fn conv2d_inplace(&mut self, kernel: ArrayView3<Self::Data>) -> Result<(), Error> {
        *self = self.conv2d(kernel)?;
        Ok(())
    }
}

impl<T, C> ConvolutionExt for Image<T, C>
where
    T: Copy + Clone + Num + NumAssignOps,
    C: ColourModel,
{
    type Data = T;
    fn conv2d(&self, kernel: ArrayView3<Self::Data>) -> Result<Self, Error> {
        let data = self.data.conv2d(kernel)?;
        Ok(Self {
            data,
            model: PhantomData,
        })
    }

    fn conv2d_inplace(&mut self, kernel: ArrayView3<Self::Data>) -> Result<(), Error> {
        self.data.conv2d_inplace(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::{Gray, RGB};
    use ndarray::arr3;

    #[test]
    fn bad_dimensions() {
        let error = Err(Error::ChannelDimensionMismatch);
        let error2 = Err(Error::ChannelDimensionMismatch);

        let mut i = Image::<f64, RGB>::new(5, 5);
        let bad_kern = Array3::<f64>::zeros((2, 2, 2));
        assert_eq!(i.conv2d(bad_kern.view()), error);

        let data_clone = i.data.clone();
        let res = i.conv2d_inplace(bad_kern.view());
        assert_eq!(res, error2);
        assert_eq!(i.data, data_clone);

        let good_kern = Array3::<f64>::zeros((2, 2, RGB::channels()));
        assert!(i.conv2d(good_kern.view()).is_ok());
        assert!(i.conv2d_inplace(good_kern.view()).is_ok());
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn basic_conv() {
        let input_pixels = vec![
            1, 1, 1, 0, 0,
            0, 1, 1, 1, 0,
            0, 0, 1, 1, 1,
            0, 0, 1, 1, 0,
            0, 1, 1, 0, 0,
        ];
        let output_pixels = vec![
            2, 2, 3, 1, 1,
            1, 4, 3, 4, 1,
            1, 2, 4, 3, 3,
            1, 2, 3, 4, 1,
            0, 2, 2, 1, 1, 
        ];

        let kern = arr3(
            &[
                [[1], [0], [1]],
                [[0], [1], [0]],
                [[1], [0], [1]]
            ]);

        let input = Image::<u8, Gray>::from_shape_data(5, 5, input_pixels);
        let expected = Image::<u8, Gray>::from_shape_data(5, 5, output_pixels);

        assert_eq!(Ok(expected), input.conv2d(kern.view()));
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn basic_conv_inplace() {
        let input_pixels = vec![
            1, 1, 1, 0, 0,
            0, 1, 1, 1, 0,
            0, 0, 1, 1, 1,
            0, 0, 1, 1, 0,
            0, 1, 1, 0, 0,
        ];

        let output_pixels = vec![
            2, 2, 3, 1, 1,
            1, 4, 3, 4, 1,
            1, 2, 4, 3, 3,
            1, 2, 3, 4, 1,
            0, 2, 2, 1, 1,
        ];

        let kern = arr3(
            &[
                [[1], [0], [1]],
                [[0], [1], [0]],
                [[1], [0], [1]]
            ]);

        let mut input = Image::<u8, Gray>::from_shape_data(5, 5, input_pixels);
        let expected = Image::<u8, Gray>::from_shape_data(5, 5, output_pixels);

        input.conv2d_inplace(kern.view()).unwrap();

        assert_eq!(expected, input);
    }
}
