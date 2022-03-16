use pasture_core::gpu as pgpu;
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;

use pasture_core::{
    containers::{
        InterleavedVecPointStorage,
        PointBufferExt,
    },
    layout::{
        attributes, PointLayout, PointAttributeDefinition, PointAttributeDataType
    },
};

use winit::window::Window;
use nalgebra::{Vector2, Vector3, Point3, UnitQuaternion, Matrix4};
use crevice::std140::AsStd140;
use instant::Instant;

#[repr(C)]
#[derive(Debug, AsStd140)]
struct UboData {
    view_proj_matrix: mint::ColumnMatrix4<f32>,
}

pub struct Camera {
    // camera position and orientation in world space
    pos: Point3<f32>,
    rot: UnitQuaternion<f32>,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            pos: Point3::new(0.0, 0.0, 0.0),
            rot: UnitQuaternion::identity(),
        }
    }

    pub fn view_dir(&self) -> Vector3::<f32> {
        self.rot * Vector3::new(0.0f32, 0.0f32, -1.0f32)
    }

    pub fn view_mat(&self) -> nalgebra::Matrix4::<f32> {
        let up = Vector3::new(0.0f32, 1.0f32, 0.0f32);
        let target = self.pos + self.view_dir();

        nalgebra::Matrix4::look_at_rh(&self.pos, &target, &up)
    }
}

pub trait CameraController {
    fn update(&mut self, _cam: &mut Camera, _dt: f32) {}
    fn process_mouse_move(&mut self, _cam: &mut Camera, _pos: Vector2::<f32>) {}
    fn process_keyboard_input(&mut self, _cam: &mut Camera, _key: winit::event::VirtualKeyCode, _pressed: bool) {}
    fn process_mouse_button(&mut self, _cam: &mut Camera, _key: winit::event::MouseButton, _pressed: bool) {}
    fn process_mouse_wheel(&mut self, _cam: &mut Camera, _delta: f32) {}
}

pub struct FPSCameraController {
    pitch: f32,
    yaw: f32,

    // controller information
    pub w_down: bool,
    pub a_down: bool,
    pub s_down: bool,
    pub d_down: bool,
    pub q_down: bool,
    pub e_down: bool,
    pub rotating: bool,
    pub mouse_pos: Option<Vector2::<f32>>,
}

pub struct ArcballCameraController {
	// Describes how far the rotation center is in front of the camera.
	offset: f32,

	rotating: bool,
	panning: bool,
    mouse_pos: Option<Vector2::<f32>>,
}

// TODO: we only need a Vec here because gpu_point_buffer needs it
const POINT_ATTRIB_3D_F32: PointAttributeDefinition =
    attributes::POSITION_3D.with_custom_datatype(PointAttributeDataType::Vec3f32);

const POINT_ATTRIBS: [pgpu::BufferInfoPerAttribute; 2] = [
    pgpu::BufferInfoPerAttribute {
        attribute: &POINT_ATTRIB_3D_F32,
        binding: 0,
    },
    pgpu::BufferInfoPerAttribute {
        attribute: &attributes::COLOR_RGB,
        binding: 1,
    },
];

impl FPSCameraController {
    pub fn new() -> FPSCameraController {
        FPSCameraController {
            pitch: 0.0,
            yaw: 0.0,
            w_down: false,
            a_down: false,
            s_down: false,
            d_down: false,
            q_down: false,
            e_down: false,
            rotating: false,
            mouse_pos: None,
        }
    }
}

impl CameraController for FPSCameraController {
    fn update(&mut self, cam: &mut Camera, dt: f32) {
        let dir = cam.view_dir();
        let right = cam.rot * Vector3::new(1.0f32, 0.0f32, 0.0f32);

        let up = Vector3::new(0.0f32, 1.0f32, 0.0f32);
        let real_up = cam.rot * up;
        let fac = dt;

        if self.w_down {
            cam.pos += fac * dir;
        }
        if self.s_down {
            cam.pos -= fac * dir;
        }
        if self.a_down {
            cam.pos -= fac * right;
        }
        if self.d_down {
            cam.pos += fac * right;
        }
        if self.q_down {
            cam.pos += fac * real_up;
        }
        if self.e_down {
            cam.pos -= fac * real_up;
        }
    }

    fn process_mouse_move(&mut self, cam: &mut Camera, pos: Vector2::<f32>) {
        if self.rotating {
            if let Some(mpos) = self.mouse_pos {
                let delta : Vector2::<f32> = pos - mpos;
                let fac = 0.01f32;
                let limit_pitch = true;
                let pitch_eps = 0.001;

                self.yaw -= fac * delta.x;
                self.pitch -= fac * delta.y;

                if limit_pitch {
                    let pi = std::f32::consts::PI;
                    self.pitch = self.pitch.clamp(
                        -0.5 * pi + pitch_eps, 0.5 * pi - pitch_eps);
                }

                // NOTE: pitch, yaw, roll order seems to be switched
                // but nalgebra seems to just assume a different coordinate
                // system. For us - and most rendering software - y is up.
                cam.rot = UnitQuaternion::from_euler_angles(self.pitch, self.yaw, 0.0f32);
            }
        }

        self.mouse_pos = Some(pos)
    }

    fn process_keyboard_input(&mut self, _cam: &mut Camera, key: winit::event::VirtualKeyCode, pressed: bool) {
        match key {
            winit::event::VirtualKeyCode::W => {
                self.w_down = pressed
            }
            winit::event::VirtualKeyCode::S => {
                self.s_down = pressed
            }
            winit::event::VirtualKeyCode::A => {
                self.a_down = pressed
            }
            winit::event::VirtualKeyCode::D => {
                self.d_down = pressed
            }
            winit::event::VirtualKeyCode::Q => {
                self.q_down = pressed
            }
            winit::event::VirtualKeyCode::E => {
                self.e_down = pressed
            }
            _ => {}
        }
    }

    fn process_mouse_button(&mut self, _cam: &mut Camera, button: winit::event::MouseButton, pressed: bool) {
        if button == winit::event::MouseButton::Left {
            self.rotating = pressed
        }
    }
}

impl ArcballCameraController {
    pub fn new() -> ArcballCameraController {
        ArcballCameraController {
            offset: 1.0,
            rotating: false,
            panning: false,
            mouse_pos: None,
        }
    }

    pub fn center(&self, cam: &Camera) -> Point3::<f32> {
        cam.pos + self.offset * cam.view_dir()
    }

    pub fn zoom(&mut self, cam: &mut Camera, fac: f32) {
        let c = self.center(cam); // old center
        self.offset *= fac;
        cam.pos = c - self.offset * cam.view_dir();
    }
}

impl CameraController for ArcballCameraController {
    fn process_mouse_move(&mut self, cam: &mut Camera, pos: Vector2::<f32>) {
        if let Some(mpos) = self.mouse_pos {
            let delta : Vector2::<f32> = pos - mpos;

            let right = cam.rot * Vector3::new(1.0f32, 0.0f32, 0.0f32);
            // let up = Vector3::new(0.0f32, 1.0f32, 0.0f32);
            let up = cam.rot * Vector3::new(0.0f32, 1.0f32, 0.0f32);

            if self.panning {
                // reversed in y direction to account for different orientations
                // of rendering and input coords
                // TODO: make pan fac dependent on offset and used projection
                const PAN_FAC : f32 = 0.01;
                let x = -PAN_FAC * delta.x * right;
                let y = PAN_FAC * delta.y * up;
                cam.pos += x + y;
            }

            if self.rotating {
                const ROT_FAC : f32 = 0.005;
                let c = self.center(cam);

                let yaw = ROT_FAC * delta.x;
                let pitch = ROT_FAC * delta.y;

                let pitch_rot = UnitQuaternion::from_euler_angles(-pitch, 0.0, 0.0);
                let yaw_rot = UnitQuaternion::from_euler_angles(0.0, -yaw, 0.0);
                let rot = cam.rot * pitch_rot;
                let rot = yaw_rot * rot;

                // let rot = cam.rot * UnitQuaternion::from_euler_angles(-pitch, -yaw, 0.0);

                cam.rot = rot;
                cam.pos = c - self.offset * cam.view_dir();
            }
        }

        self.mouse_pos = Some(pos)
    }

    fn process_mouse_button(&mut self, _cam: &mut Camera, button: winit::event::MouseButton, pressed: bool) {
        if button == winit::event::MouseButton::Left {
            self.panning = pressed
        } else if button == winit::event::MouseButton::Middle {
            self.rotating = pressed
        }
    }

    fn process_mouse_wheel(&mut self, cam: &mut Camera, delta: f32) {
        const ZOOM_FAC : f32 = 1.1;
        self.zoom(cam, f32::powf(ZOOM_FAC, -delta));
    }

    // NOTE: mostly for web, mouse wheel input does not seem to work there
    fn process_keyboard_input(&mut self, cam: &mut Camera, key: winit::event::VirtualKeyCode, pressed: bool) {
        const ZOOM_FAC : f32 = 1.1;
        if key == winit::event::VirtualKeyCode::I && pressed {
            println!("zoom in");
            self.zoom(cam, 1.0 / ZOOM_FAC);
        } else if key == winit::event::VirtualKeyCode::O && pressed {
            println!("zoom out");
            self.zoom(cam, ZOOM_FAC);
        }
    }
}


pub struct Renderer {
    device: wgpu::Device,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    queue: wgpu::Queue,

    pipeline: wgpu::RenderPipeline,

    #[allow(dead_code)]
    pipeline_layout: wgpu::PipelineLayout,
    #[allow(dead_code)]
    uniform_bind_group_layout: wgpu::BindGroupLayout,

    uniform_bind_group: wgpu::BindGroup,
    ubo: wgpu::Buffer,
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,

    pub cam: Camera,
    pub cam_controller: Box<dyn CameraController>,
    last_frame: Instant,

    model_mat: Matrix4<f32>,
    depth_view: wgpu::TextureView,

    // TODO: kinda terrible that this abstraction is designed so
    // tightly for compute shaders and creates a lot of stuff
    // we don't need for rendering
    gpu_point_buffer: pgpu::GpuPointBufferPerAttribute<'static>,
    point_count: u32,
}

impl Renderer {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub async fn new(window: &Window) -> Renderer {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command queue
        let limits = wgpu::Limits {
            ..wgpu::Limits::downlevel_webgl2_defaults()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    // TODO: don't enable all features, just what we need:
                    // - 64-bit floats for the default position_3D attrib
                    // - point rendering
                    features: adapter.features(),
                    // TODO: wasm workaround
                    // limits: wgpu::Limits::default(),
                    limits
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let vert_spirv = include_bytes!("tri.vert.spv");
        let vert_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(vert_spirv),
        });

       let frag_spirv = include_bytes!("tri.frag.spv");
        let frag_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(frag_spirv),
        });

        let swapchain_format = surface.get_preferred_format(&adapter).unwrap();

        let mut sizer = crevice::std140::Sizer::new();
        sizer.add::<UboData>();

        // bind groups
        let ubo = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("uniform_buffer"),
                size: sizer.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        let uniform_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None
                    }
                ],
            }
        );

        let uniform_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("uniform_bind_group"),
                layout: &uniform_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ubo.as_entire_binding(),
                    },
                ],
            }
        );

        let bind_group_layouts = [&uniform_bind_group_layout];

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let mut primitive_state = wgpu::PrimitiveState::default();
        primitive_state.cull_mode = None;
        primitive_state.topology = wgpu::PrimitiveTopology::TriangleList;
        primitive_state.polygon_mode = wgpu::PolygonMode::Fill;

        let pos_vertex_attrib_desc = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0
            },
        ];

        let color_vertex_attrib_desc = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32x4,
                offset: 0,
                shader_location: 1
            },
        ];

        let pos_vertex_buf_desc = wgpu::VertexBufferLayout {
            array_stride: 4 * 4, // vec3 but aligned as vec4 by pasture/gpu
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &pos_vertex_attrib_desc,
        };

        let color_vertex_buf_desc = wgpu::VertexBufferLayout {
            array_stride: 4 * 4, // unorm vec4 but treated as uvec4 by pasture/gpu
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &color_vertex_attrib_desc,
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vert_shader,
                entry_point: "main",
                buffers: &[pos_vertex_buf_desc, color_vertex_buf_desc],
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_shader,
                entry_point: "main",
                targets: &[swapchain_format.into()],
            }),
            primitive: primitive_state,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multiview: None,
            multisample: wgpu::MultisampleState::default(),
        });

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &surface_config);
        let depth_view = Self::create_depth_texture(&surface_config, &device);

        Renderer {
            adapter,
            device,
            queue,
            surface,
            uniform_bind_group_layout,
            uniform_bind_group,
            ubo,
            surface_config,
            pipeline: render_pipeline,
            pipeline_layout,
            cam: Camera::new(),
            cam_controller: Box::new(FPSCameraController::new()),
            model_mat: Matrix4::identity(),
            depth_view,
            // cam_controller: Box::new(ArcballCameraController::new()),
            last_frame: Instant::now(),
            gpu_point_buffer: pgpu::GpuPointBufferPerAttribute::new(),
            point_count: 0,
        }
    }

    fn create_depth_texture(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    // z-up: Whether the z-axis is the up-direction for the given point cloud.
    //   If this is false, the y-axis will be treated as up-direction.
    pub fn load_points(&mut self, reader: &mut LASReader, z_up: bool) {
        let default_layout = reader.get_default_point_layout();
        println!("point layout: {}", default_layout);

        let point_count = reader.remaining_points();
        let layout = PointLayout::from_attributes(&[
            POINT_ATTRIB_3D_F32,
            attributes::COLOR_RGB,
        ]);

        self.point_count = point_count as u32;
        let mut point_buffer = InterleavedVecPointStorage::with_capacity(point_count, layout);

        println!("Reading {} points (this can take a while for large point clouds)", point_count);
        reader.read_into(&mut point_buffer, point_count).unwrap();
        println!("Done!");

        let inf = f32::INFINITY;
        let mut aabb_min = Vector3::new(inf, inf, inf);
        let mut aabb_max = Vector3::new(-inf, -inf, -inf);

        println!("computing aabb...");

        for position in point_buffer.iter_attribute::<Vector3<f32>>(&POINT_ATTRIB_3D_F32) {
            let pos = position;
            aabb_min = nalgebra::Matrix::inf(&pos, &aabb_min);
            aabb_max = nalgebra::Matrix::sup(&pos, &aabb_max);
        }

        println!("aabb min: {:?}", aabb_min);
        println!("aabb max: {:?}", aabb_max);

        let center = 0.5f32 * (aabb_min + aabb_max);
        println!("center: {:?}", self.cam.pos);

        let output_color_bounds = true;
        if output_color_bounds {
            println!("computing color bounds...");

            let mut color_min = Vector3::<u16>::new(65535, 65535, 65535);
            let mut color_max = Vector3::<u16>::new(0, 0, 0);

            for color in point_buffer.iter_attribute::<Vector3<u16>>(&attributes::COLOR_RGB) {
                color_min = nalgebra::Matrix::inf(&color, &color_min);
                color_max = nalgebra::Matrix::sup(&color, &color_max);
            }

            println!("color min: {:?}", color_min);
            println!("color max: {:?}", color_max);
        }

        let extent = aabb_max - aabb_min;
        let max_ext = f32::max(extent.x, f32::max(extent.y, extent.z));

        if z_up {
            self.model_mat = self.model_mat * Matrix4::<f32>::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0);
        }

        self.model_mat = self.model_mat * Matrix4::new_scaling(10.0 / max_ext);
        self.model_mat = self.model_mat * Matrix4::new_translation(&(-center));

        self.gpu_point_buffer.malloc(point_count as u64, &POINT_ATTRIBS, &mut self.device, false);
        self.gpu_point_buffer.upload(&mut point_buffer, 0..point_count, &POINT_ATTRIBS, &mut self.device, &self.queue);
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.depth_view = Self::create_depth_texture(&self.surface_config, &self.device);
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn render(&mut self, window: &Window) {
        let frame = self.surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.
            create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });
            rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
            rpass.set_pipeline(&self.pipeline);

            let pos_buf = &self.gpu_point_buffer.buffers.get(POINT_ATTRIB_3D_F32.name()).unwrap();
            let color_buf = &self.gpu_point_buffer.buffers.get(attributes::COLOR_RGB.name()).unwrap();

            rpass.set_vertex_buffer(0, pos_buf.slice(..));
            rpass.set_vertex_buffer(1, color_buf.slice(..));

            // rpass.draw(0..3, 0..1);
            rpass.draw(0..6, 0..self.point_count);
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.cam_controller.update(&mut self.cam, dt);
        let view_mat = self.cam.view_mat();

        let win_size = window.inner_size();
        let aspect = (win_size.width as f32) / (win_size.height as f32);
        let fovy = 1.2f32;
        let near = 0.1f32;
        let far = 20.0f32;
        let proj_mat = nalgebra::Matrix4::<f32>::new_perspective(aspect, fovy, near, far);

        let gpu_data = UboData {
            view_proj_matrix: (proj_mat * view_mat * self.model_mat).into(),
        }.as_std140();

        self.queue.write_buffer(&self.ubo, 0, bytemuck::bytes_of(&gpu_data));

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
