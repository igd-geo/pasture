use std::{collections::HashMap, path::Path};

use anyhow::{anyhow, Result};
use las_rs::Builder;

use crate::las::{LASReader, LASWriter};

use super::{PointReader, PointWriter, SeekToPoint};

pub trait PointReadAndSeek: PointReader + SeekToPoint {}

impl<T: PointReader + SeekToPoint> PointReadAndSeek for T {}

type ReaderFactoryFn = dyn Fn(&Path) -> Result<Box<dyn PointReadAndSeek>>;
type WriterFactoryFn = dyn Fn(&Path) -> Result<Box<dyn PointWriter>>;

/// Factory that can create `PointReader` and `PointWriter` objects based on file extensions. Use this if you have a file path
/// and just want to create a `PointReader` or `PointWriter` from this path, without knowing the type of file. The `Default`
/// implementation supports all file formats that Pasture natively works with, custom formats can be registered using the
/// `register_...` functions. An extension in this context is whatever [`Path::extension()`](Path::extension) returns for a valid file path
pub struct IOFactory {
    reader_factories: HashMap<String, Box<ReaderFactoryFn>>,
    writer_factories: HashMap<String, Box<WriterFactoryFn>>,
}

impl IOFactory {
    /// Try to create a `PointReader` that can read from the given `file`. This function will fail if `file` has
    /// a format that is unsupported by Pasture, or if there are any I/O errors while trying to access `file`.
    pub fn make_reader(&self, file: &Path) -> Result<Box<dyn PointReadAndSeek>> {
        let extension = file.extension().ok_or_else(|| {
            anyhow!(
                "File extension could not be determined from path {}",
                file.display()
            )
        })?;
        let extension_str = extension.to_str().ok_or_else(|| {
            anyhow!(
                "File extension of path {} is no valid Unicode string",
                file.display()
            )
        })?;
        let extension_str_lower = extension_str.to_lowercase();
        let factory = self
            .reader_factories
            .get(extension_str_lower.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "Reading from point cloud files with extension {} is not supported",
                    extension_str
                )
            })?;

        factory(file)
    }

    /// Try to create a `PointWriter` for writing into the given `file`. This function will fail if `file` has
    /// a format that is unsupported by Pasture, or if there are any I/O errors while trying to access `file`.
    pub fn make_writer(&self, file: &Path) -> Result<Box<dyn PointWriter>> {
        let extension = file.extension().ok_or_else(|| {
            anyhow!(
                "File extension could not be determined from path {}",
                file.display()
            )
        })?;
        let extension_str = extension.to_str().ok_or_else(|| {
            anyhow!(
                "File extension of path {} is no valid Unicode string",
                file.display()
            )
        })?;
        let extension_str_lower = extension_str.to_lowercase();
        let factory = self
            .writer_factories
            .get(extension_str_lower.as_str())
            .ok_or_else(|| {
                anyhow!(
                    "Writing to point cloud files with extension {} is not supported",
                    extension_str
                )
            })?;

        factory(file)
    }

    /// Returns `true` if the associated `IOFactory` supports creating `PointReader` objects for the given
    /// file `extension`
    pub fn supports_reading_from(&self, extension: &str) -> bool {
        let extension_lower = extension.to_lowercase();
        self.reader_factories.contains_key(extension_lower.as_str())
    }

    /// Returns `true` if the associated `IOFactory` supports creating `PointWriter` objects for the given
    /// file `extension`
    pub fn supports_writing_to(&self, extension: &str) -> bool {
        let extension_lower = extension.to_lowercase();
        self.writer_factories.contains_key(extension_lower.as_str())
    }

    /// Register a new readable file extension with the associated `IOFactory`. The `reader_factory` will be called whenever
    /// `extension` is encountered as a file extension in `make_reader`. Returns the previous reader factory function that
    /// was registered for `extension`, if there was any. File extensions are treated as lower-case internally, so if the
    /// extension `.FOO` is registered here, it will match `file.foo` and `file.FOO` (and all case-variations thereof).
    pub fn register_reader_for_extension<
        F: Fn(&Path) -> Result<Box<dyn PointReadAndSeek>> + 'static,
    >(
        &mut self,
        extension: &str,
        reader_factory: F,
    ) -> Option<Box<ReaderFactoryFn>> {
        let extension_lower = extension.to_lowercase();
        self.reader_factories
            .insert(extension_lower, Box::new(reader_factory))
    }

    /// Register a new writeable file extension with the associated `IOFactory`. The `writer_factory` will be called whenever
    /// `extension` is encountered as a file extension in `make_writer`. Returns the previous writer factory function that
    /// was registered for `extension`, if there was any. File extensions are treated as lower-case internally, so if the
    /// extension `.FOO` is registered here, it will match `file.foo` and `file.FOO` (and all case-variations thereof).
    pub fn register_writer_for_extension<F: Fn(&Path) -> Result<Box<dyn PointWriter>> + 'static>(
        &mut self,
        extension: &str,
        writer_factory: F,
    ) -> Option<Box<WriterFactoryFn>> {
        let extension_lower = extension.to_lowercase();
        self.writer_factories
            .insert(extension_lower, Box::new(writer_factory))
    }
}

impl Default for IOFactory {
    fn default() -> Self {
        let mut factory = Self {
            reader_factories: Default::default(),
            writer_factories: Default::default(),
        };

        factory.register_reader_for_extension("las", |path| {
            let reader = LASReader::from_path(path)?;
            Ok(Box::new(reader))
        });
        factory.register_writer_for_extension("las", |path| {
            let header = Builder::from((1, 4)).into_header()?;
            let writer = LASWriter::from_path_and_header(path, header)?;
            Ok(Box::new(writer))
        });

        factory.register_reader_for_extension("laz", |path| {
            let reader = LASReader::from_path(path)?;
            Ok(Box::new(reader))
        });
        factory.register_writer_for_extension("laz", |path| {
            let header = Builder::from((1, 4)).into_header()?;
            let writer = LASWriter::from_path_and_header(path, header)?;
            Ok(Box::new(writer))
        });

        factory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_factory_ignores_extension_case() {
        let mut factory: IOFactory = Default::default();

        assert!(factory.supports_reading_from("las"));
        assert!(factory.supports_reading_from("LAS"));
        assert!(factory.supports_writing_to("las"));
        assert!(factory.supports_writing_to("LAS"));

        factory.register_reader_for_extension("FOO", |_path| unimplemented!());
        factory.register_writer_for_extension("FOO", |_path| unimplemented!());

        assert!(factory.supports_reading_from("foo"));
        assert!(factory.supports_reading_from("FOO"));
        assert!(factory.supports_writing_to("foo"));
        assert!(factory.supports_writing_to("FOO"));
    }
}
