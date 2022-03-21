#[cfg(feature = "gpu")]
mod detail {

    use clap::{App, Arg};
    use pasture_io::las::LASReader;
    use std::path::PathBuf;

    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::Window,
    };

    use nalgebra::Vector2;

    mod renderer;

    struct Args {
        pub input_file: PathBuf,
    }

    fn get_args() -> Args {
        let matches = App::new("pasture renderer")
            .version("0.1")
            .author("Jan Kelling <jan.kelling@stud.tu-darmstadt.de>")
            .about("Renders point cloud files")
            .arg(
                Arg::with_name("INPUT")
                    .short("i")
                    .takes_value(true)
                    .value_name("INPUT")
                    .help("Input point cloud file")
                    .required(true),
            )
            .get_matches();

        let input_file = PathBuf::from(matches.value_of("INPUT").unwrap());

        Args { input_file }
    }

    async fn run(event_loop: EventLoop<()>, window: Window, path: Option<PathBuf>) {
        let mut renderer = renderer::Renderer::new(&window).await;

        // For wasm: cannot load from file atm. Include statically.
        let mut reader = if let Some(reader_path) = path {
            LASReader::from_path(reader_path).expect("Failed to open lad file")
        } else {
            // let buf : &[u8] = include_bytes!("/home/jan/loads/red-rocks.laz");
            let buf: &[u8] =
                include_bytes!("../../../../pasture-io/examples/in/10_points_format_1.las");
            let c = std::io::Cursor::new(buf);
            LASReader::from_read(c, false).expect("Failed to create LASReader")
        };

        renderer.load_points(&mut reader, true);
        drop(reader);

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(size) => {
                        renderer.resize(size);
                    }
                    WindowEvent::CloseRequested => {
                        println!("Exiting");
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        let pressed = input.state == winit::event::ElementState::Pressed;
                        if let Some(key) = input.virtual_keycode {
                            if key == winit::event::VirtualKeyCode::F {
                                renderer.cam_controller =
                                    Box::new(renderer::FPSCameraController::new());
                            } else if key == winit::event::VirtualKeyCode::B {
                                renderer.cam_controller =
                                    Box::new(renderer::ArcballCameraController::new());
                            } else {
                                renderer.cam_controller.process_keyboard_input(
                                    &mut renderer.cam,
                                    key,
                                    pressed,
                                )
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let posvec = Vector2::<f32>::new(position.x as f32, position.y as f32);
                        renderer
                            .cam_controller
                            .process_mouse_move(&mut renderer.cam, posvec);
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        let pressed = state == winit::event::ElementState::Pressed;
                        renderer.cam_controller.process_mouse_button(
                            &mut renderer.cam,
                            button,
                            pressed,
                        )
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        if let winit::event::MouseScrollDelta::LineDelta(_, dy) = delta {
                            renderer
                                .cam_controller
                                .process_mouse_wheel(&mut renderer.cam, dy);
                        }
                    }
                    _ => {}
                },
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
            let args = get_args();
            env_logger::init();
            pollster::block_on(run(event_loop, window, Some(args.input_file)));
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
            wasm_bindgen_futures::spawn_local(run(event_loop, window, None));
        }
    }
}

#[cfg(feature = "gpu")]
fn main() {
    detail::main();
}

#[cfg(not(feature = "gpu"))]
fn main() {}
