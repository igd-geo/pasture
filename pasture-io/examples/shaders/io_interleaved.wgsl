struct PointBuffer {
    position: vec4<f64>;
    intensity: u32;
};

[[block]]
struct PointBufferSsbo {
    pointBuffer: [[stride(64)]] array<PointBuffer>;
};

[[block]]
struct PointUniform {
    NUM_POINTS: u32;
    model: mat4x4<f64>;
};

[[group(0), binding(0)]] var<storage> pointBufferSsbo: [[access(read_write)]] PointBufferSsbo;

[[group(1), binding(0)]] var<uniform> point_uniform: PointUniform;

[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let idx: u32 = global_id.x;

    if(idx >= point_uniform.NUM_POINTS) {
        return;
    }

    pointBufferSsbo.pointBuffer[idx].intensity = u32(255);
    pointBufferSsbo.pointBuffer[idx].position = point_uniform.model * pointBufferSsbo.pointBuffer[idx].position;
}