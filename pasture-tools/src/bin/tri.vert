#version 450

layout(location = 0) in vec3 inPos;

layout(set = 0, binding = 0) uniform UBO {
	mat4 matrix;
} ubo;

const uint indices[] = {0, 1, 2, 0, 2, 3};

void main() {
	// float x = int(gl_VertexIndex) - 1;
	// float y = int(gl_VertexIndex & 1u) * 2 - 1;
	// vec3 pos = vec3(x, y, -5);
	
	const uint idx = indices[gl_VertexIndex];
	vec2 rectPos = vec2(
		float((idx + 1) & 2) * 0.5f,
		float(idx & 2) * 0.5f
	);

	// TODO: orient along camera
	float size = 0.05;
	vec3 pos = inPos + size * vec3(rectPos, 0);
	gl_Position = ubo.matrix * vec4(pos, 1);
}

