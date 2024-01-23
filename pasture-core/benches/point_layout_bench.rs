use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pasture_core::layout::{
    attributes::{
        CLASSIFICATION, CLASSIFICATION_FLAGS, COLOR_RGB, EDGE_OF_FLIGHT_LINE, GPS_TIME, INTENSITY,
        NUMBER_OF_RETURNS, POINT_SOURCE_ID, POSITION_3D, RETURN_NUMBER, SCANNER_CHANNEL,
        SCAN_ANGLE, SCAN_ANGLE_RANK, SCAN_DIRECTION_FLAG, USER_DATA,
    },
    PointAttributeDefinition, PointLayout,
};
use rand::{seq::SliceRandom, thread_rng};

fn attribute_by_name(point_layout: &PointLayout, attribute: &PointAttributeDefinition) {
    let found_attribute = point_layout.get_attribute(attribute);
    black_box(found_attribute);
}

fn bench(c: &mut Criterion) {
    let attribute_set = vec![
        POSITION_3D,
        INTENSITY,
        RETURN_NUMBER,
        NUMBER_OF_RETURNS,
        CLASSIFICATION_FLAGS,
        SCANNER_CHANNEL,
        SCAN_DIRECTION_FLAG,
        EDGE_OF_FLIGHT_LINE,
        CLASSIFICATION,
        SCAN_ANGLE_RANK,
        SCAN_ANGLE,
        USER_DATA,
        POINT_SOURCE_ID,
        COLOR_RGB,
        GPS_TIME,
    ];

    let mut rng = thread_rng();
    let mut group = c.benchmark_group("find_attribute");
    for num_attributes in [1, 2, 4, 8, 12] {
        let random_layout: PointLayout = {
            let mut attributes = attribute_set.clone();
            attributes.shuffle(&mut rng);
            attributes
                .into_iter()
                .take(num_attributes as usize)
                .collect()
        };
        group.throughput(Throughput::Elements(num_attributes));
        for (idx, attribute) in random_layout.attributes().enumerate() {
            let input = (
                random_layout.clone(),
                attribute.attribute_definition().clone(),
            );
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}/{}", (idx + 1), num_attributes)),
                &input,
                |b, (layout, attribute)| b.iter(|| attribute_by_name(layout, attribute)),
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = point_layout;
    config = Criterion::default().sample_size(40);
    targets = bench
}
criterion_main!(point_layout);
