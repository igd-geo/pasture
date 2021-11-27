#version 450

layout(location = 0) in vec3 inPos;

layout(set = 0, binding = 0) uniform UBO {
	mat4 matrix;
} ubo;

void main() {
	// float x = int(gl_VertexIndex) - 1;
	// float y = int(gl_VertexIndex & 1u) * 2 - 1;
	// vec3 pos = vec3(x, y, -5);

	vec3 pos = inPos;
	gl_Position = ubo.matrix * vec4(pos, 1);

	// ugh
	float dist = gl_Position.z / gl_Position.w;
	gl_PointSize = 10.f / clamp(dist, 0.1, 1);
}

