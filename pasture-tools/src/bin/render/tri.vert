#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in uvec3 inColor;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outPos;

layout(set = 0, binding = 0) uniform UBO {
	mat4 matrix;
} ubo;

const bool swizzleYZ = true;

void main() {
	// NOTE: the first line triggers a bug in gfx-rs/naga when translating
	// it to a webgl shader
	// const uint indices[6] = {0, 1, 2, 0, 2, 3};
	uint indices[6]; // = {0, 1, 2, 0, 2, 3};
	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 0;
	indices[4] = 2;
	indices[5] = 3;

	const uint idx = indices[gl_VertexIndex];
	vec2 rectPos = vec2(
		float((idx + 1) & 2) * 0.5f,
		float(idx & 2) * 0.5f
	);

	float size = 0.05;
	vec3 pos = inPos;

	if(swizzleYZ) {
		pos = pos.xzy;
	}

	gl_Position = ubo.matrix * vec4(pos, 1);
	float rectSize = 0.005f;
	gl_Position.xy += rectSize * (-1.0 + 2.f * rectPos);

	outColor = inColor / 255.f;
	outPos = -1.f + 2.f * rectPos;
}

