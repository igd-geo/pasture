#![feature(min_specialization)]

use nalgebra::Vector3;
use pasture_core::{
    containers_v2::{BufferStorageMut, ColumnarStorage, PointBuffer, PointSlice, VectorStorage},
    layout::{attributes::INTENSITY, PointLayout},
};
use pasture_derive::PointType;
use rand::{thread_rng, Rng};

#[derive(Copy, Clone, Debug, PointType)]
#[repr(C, packed)]
struct ExamplePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    intensity: u16,
    #[pasture(BUILTIN_GPS_TIME)]
    gps_time: f64,
}

fn reference_data(count: usize) -> Vec<ExamplePoint> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| ExamplePoint {
            gps_time: rng.gen(),
            intensity: rng.gen(),
            position: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
        })
        .collect()
}

trait Reader<S: BufferStorageMut> {
    fn read(&mut self, count: usize) -> PointBuffer<S>;
    fn read_into(&mut self, count: usize, buffer: &mut PointBuffer<S>);
    fn default_point_layout(&self) -> PointLayout;
}

trait MakeReader<S: BufferStorageMut, S2: BufferStorageMut> {
    fn new() -> Option<Box<dyn Reader<S>>>;
}

impl<S: BufferStorageMut, R: Reader<S2>, S2: BufferStorageMut> MakeReader<S, S2> for R {
    default fn new() -> Option<Box<dyn Reader<S>>> {
        None
    }
}

struct ReaderVector;

impl Reader<VectorStorage> for ReaderVector {
    fn read(&mut self, count: usize) -> PointBuffer<VectorStorage> {
        let layout = Reader::<VectorStorage>::default_point_layout(self);
        let mut buffer = PointBuffer::new(
            VectorStorage::from_layout_with_capacity(&layout, count),
            layout,
        );
        let data: Vec<u8> = vec![0; buffer.point_layout().size_of_point_entry() as usize];
        for _ in 0..count {
            buffer.storage_mut().push(&data);
        }
        buffer
    }

    fn read_into(&mut self, count: usize, buffer: &mut PointBuffer<VectorStorage>) {
        todo!()
    }

    fn default_point_layout(&self) -> PointLayout {
        todo!()
    }
}

impl MakeReader<VectorStorage, VectorStorage> for ReaderVector {
    fn new() -> Option<Box<dyn Reader<VectorStorage>>> {
        None
    }
}

struct ReaderBoth;

impl Reader<VectorStorage> for ReaderBoth {
    fn read(&mut self, count: usize) -> PointBuffer<VectorStorage> {
        todo!()
    }

    fn read_into(&mut self, count: usize, buffer: &mut PointBuffer<VectorStorage>) {
        todo!()
    }

    fn default_point_layout(&self) -> PointLayout {
        todo!()
    }
}

impl Reader<ColumnarStorage> for ReaderBoth {
    fn read(&mut self, count: usize) -> PointBuffer<ColumnarStorage> {
        todo!()
    }

    fn read_into(&mut self, count: usize, buffer: &mut PointBuffer<ColumnarStorage>) {
        todo!()
    }

    fn default_point_layout(&self) -> PointLayout {
        todo!()
    }
}

fn get_general_reader<T: BufferStorageMut>() -> Box<dyn Reader<T>> {
    let f1 = <ReaderBoth as MakeReader<T, VectorStorage>>::new();
    f1.unwrap()
}

fn main() {
    const COUNT: usize = 1024;
    let mut ref_data = reference_data(COUNT);

    // Usage examples to compare our buffer API to:
    for point in &ref_data {
        println!("{:?}", point);
    }

    let _bright_points = ref_data
        .iter()
        .filter(|p| p.intensity > 50000)
        .copied()
        .collect::<Vec<_>>();

    ref_data.sort_by(|a, b| {
        let intensity_a = a.intensity;
        let intensity_b = b.intensity;
        intensity_a.cmp(&intensity_b)
    });

    let _positions = ref_data.iter().map(|p| p.position).collect::<Vec<_>>();

    // Modify data
    for point_mut in &mut ref_data {
        point_mut.gps_time += 1.0;
    }

    let subrange = &ref_data[10..20];
    for point in subrange {
        println!("Point in subrange: {:?}", point);
    }

    // With ours:
    let mut pasture_data: PointBuffer<VectorStorage> = ref_data.into_iter().collect();

    for point in pasture_data.view::<ExamplePoint>() {
        println!("{:?}", point);
    }

    let _bright_points = pasture_data
        .view_attribute::<u16>(&INTENSITY)
        .into_iter()
        .filter(|intensity| *intensity > 50000)
        .collect::<Vec<_>>();

    // Filter by typed points
    let _bright_points = pasture_data
        .view::<ExamplePoint>()
        .into_iter()
        .filter(|point| {
            let intensity = point.intensity;
            intensity > 50_000
        })
        .collect::<PointBuffer<VectorStorage>>();

    pasture_data.view_mut::<ExamplePoint>().sort_by(|a, b| {
        let intensity_a = a.intensity;
        let intensity_b = b.intensity;
        intensity_a.cmp(&intensity_b)
    });

    for point_mut in pasture_data.view_mut::<ExamplePoint>().iter_mut() {
        point_mut.gps_time += 1.0;
    }

    let slice = pasture_data.slice(10..20);
    for point in slice.view::<ExamplePoint>() {
        println!("Point in subrange: {:?}", point);
    }

    // Repeated slicing:
    let slice_of_slice = slice.slice(0..5);
    for point in slice_of_slice.view::<ExamplePoint>() {
        println!("Point in slice of slice: {:?}", point);
    }
}
