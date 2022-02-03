fn main() {
    let shaders = [
        "src/bin/tri.frag",
        "src/bin/tri.vert",
    ];

    let mut compiler = shaderc::Compiler::new().unwrap();

    for shader in shaders {
        let path = std::path::Path::new(shader);
        let source = std::fs::read_to_string(path).unwrap();

        let shader_type = path.extension().and_then(|ext| {
                match ext.to_string_lossy().as_ref() {
                    "vert" => Some(shaderc::ShaderKind::Vertex),
                    "frag" => Some(shaderc::ShaderKind::Fragment),
                    "comp" => Some(shaderc::ShaderKind::Compute),
                    _ => None,
                }
            }).unwrap();

        let cs_spirv = compiler.compile_into_spirv(
                &source,
                shader_type,
                shader,
                "main",
                None,
            )
            .unwrap();

        // TODO: don't hardcode src/bin!
        let out_path = format!(
            "src/bin/{}.spv",
            path.file_name().unwrap().to_string_lossy()
        );
        std::fs::write(&out_path, &cs_spirv.as_binary_u8()).unwrap();
    }
}
