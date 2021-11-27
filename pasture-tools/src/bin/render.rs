// https://github.com/gfx-rs/wgpu/blob/v0.9/wgpu/examples/hello-triangle/main.rs

use pasture_core::gpu as pgpu;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use std::time::Instant;
use nalgebra::{Vector2, Vector3, Point3, UnitQuaternion};
use crevice::std140::AsStd140;

struct Camera {
    pos: Point3<f32>,
    rot: UnitQuaternion<f32>,

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

impl Camera {
    pub fn new() -> Camera {
        Camera {
            pos: Point3::new(0.0, 0.0, 0.0),
            rot: UnitQuaternion::identity(),
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

    pub fn view_dir(&self) -> Vector3::<f32> {
        self.rot * Vector3::new(0.0f32, 0.0f32, -1.0f32)
    }

    pub fn view_mat(&self) -> nalgebra::Matrix4::<f32> {
        let up = Vector3::new(0.0f32, 1.0f32, 0.0f32);
        let target = self.pos + self.view_dir();

        nalgebra::Matrix4::look_at_rh(&self.pos, &target, &up)
    }

    pub fn update(&mut self, dt: f32) {
        let dir = self.view_dir();
        let right = self.rot * Vector3::new(1.0f32, 0.0f32, 0.0f32);

        let up = Vector3::new(0.0f32, 1.0f32, 0.0f32);
        let real_up = self.rot * up;

        if self.w_down {
            self.pos += dt * dir;
        }
        if self.s_down {
            self.pos -= dt * dir;
        }
        if self.a_down {
            self.pos -= dt * right;
        }
        if self.d_down {
            self.pos += dt * right;
        }
        if self.q_down {
            self.pos += dt * real_up;
        }
        if self.e_down {
            self.pos -= dt * real_up;
        }
    }

    pub fn process_mouse_move(&mut self, pos: Vector2::<f32>) {
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
                self.rot = UnitQuaternion::from_euler_angles(self.pitch, self.yaw, 0.0f32);
            }
        }

        self.mouse_pos = Some(pos)
    }
}

#[repr(C)]
#[derive(Debug, AsStd140)]
struct UboData {
    view_proj_matrix: mint::ColumnMatrix4<f32>,
}

// TODO: get rid of lifetime by properly factoring out compute-related
// stuff from pasture_core::gpu::Device
struct Renderer<'a> {
    dev: pgpu::Device<'a>,

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
    last_frame: std::time::Instant,
}

impl<'a> Renderer<'a> {
    async fn new(window: &Window) -> Renderer<'a> {
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
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // compile glsl shader
        let mut compiler = shaderc::Compiler::new().unwrap();

        let vert_spirv = compiler
            .compile_into_spirv(
                include_str!("tri.vert"),
                shaderc::ShaderKind::Vertex,
                "Vertex shader",
                "main",
                None,
            )
            .unwrap();
        let vert_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(vert_spirv.as_binary_u8()),
        });

        let frag_spirv = compiler
            .compile_into_spirv(
                include_str!("tri.frag"),
                shaderc::ShaderKind::Fragment,
                "Fragment shader",
                "main",
                None,
            )
            .unwrap();
        let frag_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(frag_spirv.as_binary_u8()),
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vert_shader,
                entry_point: "main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &frag_shader,
                entry_point: "main",
                targets: &[swapchain_format.into()],
            }),
            primitive: primitive_state,
            depth_stencil: None,
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

        // Wrap device and queue into pasture device so we can create
        // point cloud buffers
        let pdev = pgpu::Device::new_from_device(adapter, device, queue);

        Renderer {
            dev: pdev,
            surface,
            uniform_bind_group_layout,
            uniform_bind_group,
            ubo,
            surface_config,
            pipeline: render_pipeline,
            pipeline_layout,
            cam: Camera::new(),
            last_frame: Instant::now(),
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.surface.configure(&self.dev.wgpu_device, &self.surface_config);
    }

    pub fn render(&mut self, window: &Window) {
        let frame = self.surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.dev.wgpu_device.
            create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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
            rpass.draw(0..3, 0..1);
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.cam.update(dt);
        let view_mat = self.cam.view_mat();

        let win_size = window.inner_size();
        let aspect = (win_size.width as f32) / (win_size.height as f32);
        let fovy = 1.5f32; // 90 degrees
        let near = 0.1f32;
        let far = 100.0f32;
        let proj_mat = nalgebra::Matrix4::<f32>::new_perspective(aspect, fovy, near, far);

        let gpu_data = UboData {
            view_proj_matrix: (proj_mat * view_mat).into(),
        }.as_std140();

        self.dev.wgpu_queue.write_buffer(&self.ubo, 0, bytemuck::bytes_of(&gpu_data));

        self.dev.wgpu_queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut renderer = Renderer::new(&window).await;

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
                        match input.virtual_keycode {
                            Some(winit::event::VirtualKeyCode::W) => {
                                renderer.cam.w_down = pressed
                            }
                            Some(winit::event::VirtualKeyCode::S) => {
                                renderer.cam.s_down = pressed
                            }
                            Some(winit::event::VirtualKeyCode::A) => {
                                renderer.cam.a_down = pressed
                            }
                            Some(winit::event::VirtualKeyCode::D) => {
                                renderer.cam.d_down = pressed
                            }
                            Some(winit::event::VirtualKeyCode::Q) => {
                                renderer.cam.q_down = pressed
                            }
                            Some(winit::event::VirtualKeyCode::E) => {
                                renderer.cam.e_down = pressed
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved{position, ..} => {
                        let posvec = Vector2::<f32>::new(position.x as f32, position.y as f32);
                        renderer.cam.process_mouse_move(posvec);
                    }
                    WindowEvent::MouseInput{
                        state,
                        button: winit::event::MouseButton::Left,
                        ..
                    } => {
                        renderer.cam.rotating = state == winit::event::ElementState::Pressed;
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

    env_logger::init();
    pollster::block_on(run(event_loop, window));
}
