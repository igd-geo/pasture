#version 450

// layout(location = 0) in vec3 inPos;

layout(set = 0, binding = 0) uniform UBO {
	mat4 matrix;
} ubo;

void main() {
	// gl_Position = vec4(inPos, 1.0);
	float x = int(gl_VertexIndex) - 1;
	float y = int(gl_VertexIndex & 1u) * 2 - 1;

	vec3 pos = vec3(x, y, -5);

	gl_Position = ubo.matrix * vec4(pos, 1);
}

