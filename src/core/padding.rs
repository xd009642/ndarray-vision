use crate::core::{ColourModel, Image, ImageBase};
use ndarray::{prelude::*, s, Data, OwnedRepr};
use num_traits::identities::Zero;
use std::marker::PhantomData;

/// Defines a method for padding the data of an image applied directly to the
/// ndarray type internally. Padding is symmetric
pub trait PaddingStrategy<T>
where
    T: Copy,
{
    /// Taking in the image data and the margin to apply to rows and columns
    /// returns a padded image
    fn pad(
        &self,
        image: ArrayView<T, Ix3>,
        padding: (usize, usize),
    ) -> ArrayBase<OwnedRepr<T>, Ix3>;

    /// Taking in the image data and row and column return the pixel value
    /// if the coordinates are within the image bounds this should probably not
    /// be used in the name of performance
    fn get_pixel(&self, image: ArrayView<T, Ix3>, index: (isize, isize)) -> Option<Array1<T>>;

    /// Gets a value for a channel rows and columns can exceed bounds but the channel index must be
    /// present
    fn get_value(&self, image: ArrayView<T, Ix3>, index: (isize, isize, usize)) -> Option<T>;

    /// Returns true if the padder will return a value for (row, col) or if None if it can pad
    /// an image at all. `NoPadding` is a special instance which will always be false
    fn will_pad(&self, _coord: Option<(isize, isize)>) -> bool {
        true
    }
}

/// Doesn't apply any padding to the image returning it unaltered regardless
/// of padding value
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct NoPadding;

/// Pad the image with a constant value
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct ConstantPadding<T>(T)
where
    T: Copy;

/// Pad the image with zeros. Uses ConstantPadding internally
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct ZeroPadding;

#[inline]
fn is_out_of_bounds(dim: (usize, usize, usize), index: (isize, isize, usize)) -> bool {
    index.0 < 0
        || index.1 < 0
        || index.0 >= dim.0 as isize
        || index.1 >= dim.1 as isize
        || index.2 >= dim.2
}

impl<T> PaddingStrategy<T> for NoPadding
where
    T: Copy,
{
    fn pad(
        &self,
        image: ArrayView<T, Ix3>,
        _padding: (usize, usize),
    ) -> ArrayBase<OwnedRepr<T>, Ix3> {
        image.to_owned()
    }

    fn get_pixel(&self, image: ArrayView<T, Ix3>, index: (isize, isize)) -> Option<Array1<T>> {
        let index = (index.0, index.1, 0);
        if is_out_of_bounds(image.dim(), index) {
            None
        } else {
            Some(image.slice(s![index.0, index.1, ..]).to_owned())
        }
    }

    fn get_value(&self, image: ArrayView<T, Ix3>, index: (isize, isize, usize)) -> Option<T> {
        if is_out_of_bounds(image.dim(), index) {
            None
        } else {
            image
                .get((index.0 as usize, index.1 as usize, index.2))
                .copied()
        }
    }

    fn will_pad(&self, _coord: Option<(isize, isize)>) -> bool {
        false
    }
}

impl<T> PaddingStrategy<T> for ConstantPadding<T>
where
    T: Copy,
{
    fn pad(
        &self,
        image: ArrayView<T, Ix3>,
        padding: (usize, usize),
    ) -> ArrayBase<OwnedRepr<T>, Ix3> {
        let shape = (
            image.shape()[0] + padding.0 * 2,
            image.shape()[1] + padding.1 * 2,
            image.shape()[2],
        );

        let mut result = Array::from_elem(shape, self.0);
        result
            .slice_mut(s![
                padding.0..shape.0 - padding.0,
                padding.1..shape.1 - padding.1,
                ..
            ])
            .assign(&image);

        result
    }

    fn get_pixel(&self, image: ArrayView<T, Ix3>, index: (isize, isize)) -> Option<Array1<T>> {
        let index = (index.0, index.1, 0);
        if is_out_of_bounds(image.dim(), index) {
            let v = vec![self.0; image.dim().2];
            Some(Array1::from(v))
        } else {
            Some(image.slice(s![index.0, index.1, ..]).to_owned())
        }
    }

    fn get_value(&self, image: ArrayView<T, Ix3>, index: (isize, isize, usize)) -> Option<T> {
        if is_out_of_bounds(image.dim(), index) {
            Some(self.0)
        } else {
            image
                .get((index.0 as usize, index.1 as usize, index.2))
                .copied()
        }
    }
}

impl<T> PaddingStrategy<T> for ZeroPadding
where
    T: Copy + Zero,
{
    fn pad(
        &self,
        image: ArrayView<T, Ix3>,
        padding: (usize, usize),
    ) -> ArrayBase<OwnedRepr<T>, Ix3> {
        let padder = ConstantPadding(T::zero());
        padder.pad(image, padding)
    }

    fn get_pixel(&self, image: ArrayView<T, Ix3>, index: (isize, isize)) -> Option<Array1<T>> {
        let padder = ConstantPadding(T::zero());
        padder.get_pixel(image, index)
    }

    fn get_value(&self, image: ArrayView<T, Ix3>, index: (isize, isize, usize)) -> Option<T> {
        let padder = ConstantPadding(T::zero());
        padder.get_value(image, index)
    }
}

/// Padding extension for images
pub trait PaddingExt<T> {
    /// Type of the output image
    type Output;
    /// Pad the object with the given padding and strategy
    fn pad(&self, padding: (usize, usize), strategy: &dyn PaddingStrategy<T>) -> Self::Output;
}

impl<T, U> PaddingExt<T> for ArrayBase<U, Ix3>
where
    U: Data<Elem = T>,
    T: Copy,
{
    type Output = ArrayBase<OwnedRepr<T>, Ix3>;

    fn pad(&self, padding: (usize, usize), strategy: &dyn PaddingStrategy<T>) -> Self::Output {
        strategy.pad(self.view(), padding)
    }
}

impl<T, U, C> PaddingExt<T> for ImageBase<U, C>
where
    U: Data<Elem = T>,
    T: Copy,
    C: ColourModel,
{
    type Output = Image<T, C>;

    fn pad(&self, padding: (usize, usize), strategy: &dyn PaddingStrategy<T>) -> Self::Output {
        Self::Output {
            data: strategy.pad(self.data.view(), padding),
            model: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::{Gray, RGB};

    #[test]
    fn constant_padding() {
        let i = Image::<u8, Gray>::from_shape_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let p = i.pad((1, 1), &ConstantPadding(0));

        let exp = Image::<u8, Gray>::from_shape_data(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
            ],
        );
        assert_eq!(p, exp);

        let p = i.pad((1, 1), &ConstantPadding(2));

        let exp = Image::<u8, Gray>::from_shape_data(
            5,
            5,
            vec![
                2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 2, 4, 5, 6, 2, 2, 7, 8, 9, 2, 2, 2, 2, 2, 2,
            ],
        );
        assert_eq!(p, exp);

        let p = i.pad((2, 0), &ConstantPadding(0));
        let z = i.pad((2, 0), &ZeroPadding {});
        assert_eq!(p, z);

        let exp = Image::<u8, Gray>::from_shape_data(
            7,
            3,
            vec![
                0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,
            ],
        );
        assert_eq!(p, exp);
    }

    #[test]
    fn no_padding() {
        let i = Image::<u8, RGB>::new(5, 5);
        let p = i.pad((10, 10), &NoPadding {});

        assert_eq!(i, p);

        let p = i.pad((0, 0), &NoPadding {});
        assert_eq!(i, p);
    }
}
