// https://github.com/gfx-rs/wgpu/blob/v0.9/wgpu/examples/hello-triangle/main.rs

use pasture_core::gpu as pgpu;
use pasture_io::base::PointReader;
use pasture_io::las::LASReader;

use pasture_core::{
    containers::{
        PerAttributeVecPointStorage,
        InterleavedVecPointStorage,
        PointBufferExt,
    },
    layout::{
        attributes, PointLayout, PointAttributeDefinition, PointAttributeDataType
    },
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use nalgebra::{Vector2, Vector3, Point3, UnitQuaternion};
use crevice::std140::AsStd140;

use instant::Instant;

struct Camera {
    // camera position and orientation in world space
    pos: Point3<f32>,
    rot: UnitQuaternion<f32>,

    // perspective projection parameters
    aspect: f32,
    fovy: f32,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            pos: Point3::new(0.0, 0.0, 0.0),
            rot: UnitQuaternion::identity(),
            aspect: 1.0,
            fovy: 1.5f32, // 90 degrees
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

trait CameraController {
    fn update(&mut self, cam: &mut Camera, dt: f32) {}
    fn process_mouse_move(&mut self, cam: &mut Camera, pos: Vector2::<f32>) {}
    fn process_keyboard_input(&mut self, key: winit::event::VirtualKeyCode, pressed: bool) {}
    fn process_mouse_button(&mut self, key: winit::event::MouseButton, pressed: bool) {}
    fn process_mouse_wheel(&mut self, cam: &mut Camera, delta: f32) {}
}

struct FPSCameraController {
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

struct ArcballCameraController {
	// Describes how far the rotation center is in front of the camera.
	offset: f32,

	rotating: bool,
	panning: bool,
    mouse_pos: Option<Vector2::<f32>>,
}

// TODO: we only need a Vec here because gpu_point_buffer needs it
const POINT_ATTRIB_3D_F32: PointAttributeDefinition = PointAttributeDefinition::custom(
    "Position3D", PointAttributeDataType::Vec3f32
);

const POINT_ATTRIBS: [pgpu::BufferInfoPerAttribute; 1] = [
    pgpu::BufferInfoPerAttribute {
        attribute: &POINT_ATTRIB_3D_F32,
        binding: 0,
    }
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

        if self.w_down {
            cam.pos += dt * dir;
        }
        if self.s_down {
            cam.pos -= dt * dir;
        }
        if self.a_down {
            cam.pos -= dt * right;
        }
        if self.d_down {
            cam.pos += dt * right;
        }
        if self.q_down {
            cam.pos += dt * real_up;
        }
        if self.e_down {
            cam.pos -= dt * real_up;
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

    fn process_keyboard_input(&mut self, key: winit::event::VirtualKeyCode, pressed: bool) {
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

    fn process_mouse_button(&mut self, button: winit::event::MouseButton, pressed: bool) {
        if button == winit::event::MouseButton::Left {
            self.rotating = pressed
        }
    }
}

// TODO
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

    fn process_mouse_button(&mut self, button: winit::event::MouseButton, pressed: bool) {
        if button == winit::event::MouseButton::Middle {
            self.panning = pressed
        } else if button == winit::event::MouseButton::Right {
            self.rotating = pressed
        }
    }

    fn process_mouse_wheel(&mut self, cam: &mut Camera, delta: f32) {
        const ZOOM_FAC : f32 = 1.1;
        let c = self.center(cam); // old center
        self.offset *= f32::powf(ZOOM_FAC, -delta);
        cam.pos = c - self.offset * cam.view_dir();
    }
}

#[repr(C)]
#[derive(Debug, AsStd140)]
struct UboData {
    view_proj_matrix: mint::ColumnMatrix4<f32>,
}

// TODO: get rid of lifetime by properly factoring out compute-related
// stuff from pasture_core::gpu::Device
struct Renderer {
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

    cam: Camera,
    cam_controller: Box<dyn CameraController>,
    last_frame: Instant,

    // TODO: kinda terrible that this abstraction is designed so
    // tightly for compute shaders and creates a lot of stuff
    // we don't need for rendering
    gpu_point_buffer: pgpu::GpuPointBufferPerAttribute<'static>,
    point_count: u32,
}

impl Renderer {
    async fn new(window: &Window) -> Renderer {
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

        let vertex_attrib_desc = wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 0
        };

        let vertex_buf_desc = wgpu::VertexBufferLayout {
            array_stride: 4 * 4, // aligned like vec4 f32
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[vertex_attrib_desc],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vert_shader,
                entry_point: "main",
                buffers: &[vertex_buf_desc],
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_shader,
                entry_point: "main",
                targets: &[swapchain_format.into()],
            }),
            primitive: primitive_state,
            depth_stencil: None,
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
            // cam_controller: Box::new(ArcballCameraController::new()),
            last_frame: Instant::now(),
            gpu_point_buffer: pgpu::GpuPointBufferPerAttribute::new(),
            point_count: 0,
        }
    }

    pub fn load_points(&mut self, reader: &mut LASReader) {
        let point_count = reader.remaining_points();
        let layout = PointLayout::from_attributes(&[POINT_ATTRIB_3D_F32]);

        self.point_count = point_count as u32;
        let mut point_buffer = InterleavedVecPointStorage::with_capacity(point_count, layout);
        reader.read_into(&mut point_buffer, point_count).unwrap();

        // let position: Vector3<f32> = point_buffer.get_attribute(&POINT_ATTRIB_3D_F32, 0);
        // println!("pos0: {}", position);

        for position in point_buffer.iter_attribute::<Vector3<f32>>(&POINT_ATTRIB_3D_F32) {
            // Notice that `iter_attribute<T>` returns `T` by value. It is available for all point buffer types, at the expense of
            // only receiving a copy of the attribute.
            println!("Position: {:?}", position);
        }

        self.gpu_point_buffer.malloc(point_count as u64, &POINT_ATTRIBS, &mut self.device);
        self.gpu_point_buffer.upload(&mut point_buffer, 0..point_count, &POINT_ATTRIBS, &mut self.device, &self.queue);
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
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
        let buf = &self.gpu_point_buffer.buffers.get(POINT_ATTRIB_3D_F32.name()).unwrap();

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_bind_group(0, &self.uniform_bind_group, &[]);
            rpass.set_pipeline(&self.pipeline);

            rpass.set_vertex_buffer(0, buf.slice(..));
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
            view_proj_matrix: (proj_mat * view_mat).into(),
        }.as_std140();

        self.queue.write_buffer(&self.ubo, 0, bytemuck::bytes_of(&gpu_data));

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut renderer = Renderer::new(&window).await;

    let buf : &[u8] = include_bytes!("/home/jan/code/pasture/pasture-io/examples/in/10_points_format_1.las");
    let c = std::io::Cursor::new(buf);
    let mut reader = LASReader::from_read(c, false).expect("Failed to create LASReader");

    // let mut reader = LASReader::from_path(
    //     // TODO
    //     // "/home/jan/loads/points.laz"
    //     "/home/jan/code/pasture/pasture-io/examples/in/10_points_format_1.las"
    //     // "/home/jan/loads/NEONDSSampleLiDARPointCloud.las"
    // ).expect("Failed to open las file");

    renderer.load_points(&mut reader);
    drop(reader);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent{event, ..} => {
                match event {
                    WindowEvent::Resized(size) => {
                        renderer.resize(size);
                    }
                    WindowEvent::CloseRequested => {
                        println!("Exiting");
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::KeyboardInput{input, ..} => {
                        let pressed = input.state == winit::event::ElementState::Pressed;
                        if let Some(key) = input.virtual_keycode {
                            renderer.cam_controller.process_keyboard_input(key, pressed)
                        }
                    }
                    WindowEvent::CursorMoved{position, ..} => {
                        let posvec = Vector2::<f32>::new(position.x as f32, position.y as f32);
                        renderer.cam_controller.process_mouse_move(&mut renderer.cam, posvec);
                    }
                    WindowEvent::MouseInput{state, button, ..} => {
                        let pressed = state == winit::event::ElementState::Pressed;
                        renderer.cam_controller.process_mouse_button(button, pressed)
                    }
                    WindowEvent::MouseWheel{delta, ..} => {
                        if let winit::event::MouseScrollDelta::LineDelta(_, dy) = delta {
                            renderer.cam_controller.process_mouse_wheel(&mut renderer.cam, dy);
                        }
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                renderer.render(&window);
            }
            Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();


    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
