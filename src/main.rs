use nalgebra as na;
use std::time::{Duration, Instant};
use vk_shader_macros::include_glsl;
use wgpu_glyph::{GlyphBrush, GlyphBrushBuilder, Scale, Section};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

struct TimeBuffer {
    contents: Vec<Duration>,
    head: usize,
    size: usize,
}

impl TimeBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            contents: vec![Duration::from_secs(0); capacity],
            head: 0,
            size: 0,
        }
    }

    fn push(&mut self, duration: Duration) {
        self.contents[self.head] = duration;
        self.head = (self.head + 1) % self.contents.len();
        self.size = (self.size + 1).min(self.contents.len());
    }

    fn average(&self) -> Duration {
        let sum: Duration = self.contents[..self.size].iter().sum();

        sum / self.size.max(1) as u32
    }
}

struct Timer {
    last_tick: Instant,
}

impl Timer {
    fn new() -> Self {
        Self {
            last_tick: Instant::now(),
        }
    }

    fn lap_time(&mut self) -> Duration {
        let tick = Instant::now();
        let lap_time = tick.duration_since(self.last_tick);
        self.last_tick = tick;
        lap_time
    }
}

const GUI_FONT: &[u8] = include_bytes!("../assets/InputSansNarrow-Regular.ttf");

struct Gui {
    glyph_brush: GlyphBrush<'static, ()>,
    target_width: u32,
    target_height: u32,
}

impl Gui {
    fn new(
        device: &wgpu::Device,
        target_width: u32,
        target_height: u32,
        render_format: wgpu::TextureFormat,
    ) -> Self {
        let glyph_brush = GlyphBrushBuilder::using_font_bytes(GUI_FONT)
            .expect("Loading fonts")
            .build(&device, render_format);
        Self {
            glyph_brush,
            target_width,
            target_height,
        }
    }

    fn resize(&mut self, target_width: u32, target_height: u32) {
        self.target_width = target_width;
        self.target_height = target_height;
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
    ) -> wgpu::CommandBuffer {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gui_encoder"),
        });
        self.glyph_brush
            .draw_queued(
                &device,
                &mut encoder,
                &frame.view,
                self.target_width,
                self.target_height,
            )
            .expect("Drawing queued glyphs");
        encoder.finish()
    }

    fn render_text(&mut self, text: &str, screen_position: (f32, f32)) {
        self.glyph_brush.queue(Section {
            text,
            screen_position,
            color: [1.0, 1.0, 1.0, 1.0],
            scale: Scale::uniform(24.0),
            ..Section::default()
        });
    }
}

struct Hud {
    mouse_down: bool,
    mouse_position: (f32, f32),
    frame_timer: Timer,
    frame_time_buffer: TimeBuffer,
}

impl Hud {
    fn new() -> Self {
        Self {
            mouse_down: false,
            mouse_position: (0.0, 0.0),
            frame_timer: Timer::new(),
            frame_time_buffer: TimeBuffer::new(100),
        }
    }

    fn update(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                self.mouse_down = true;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                ..
            } => {
                self.mouse_down = false;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = (position.x as f32, position.y as f32);
            }
            _ => {}
        }
    }

    fn render(&mut self, gui: &mut Gui) {
        self.frame_time_buffer.push(self.frame_timer.lap_time());
        let frame_time = format!("{:?}", self.frame_time_buffer.average());

        gui.render_text(&frame_time, (30.0, 30.0));
    }
}

#[repr(u32)]
#[derive(Copy, Clone)]
enum Material {
    Diffuse = 0,
    Reflective = 1,
    Emissive = 2,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Shape {
    a: na::Vector2<f32>,
    b: na::Vector2<f32>,
    color: na::Vector3<f32>,
    material: Material,
}

impl Default for Shape {
    fn default() -> Self {
        Self {
            a: na::Vector2::identity(),
            b: na::Vector2::identity(),
            color: na::Vector3::identity(),
            material: Material::Diffuse,
        }
    }
}

const SHAPE_MAX_COUNT: usize = 512;

#[repr(C)]
#[derive(Copy, Clone)]
struct TracerUniforms {
    shapes: [Shape; SHAPE_MAX_COUNT],
    shape_count: u32,
    sample_count: u32,
    time: f32,
}

unsafe impl bytemuck::Pod for TracerUniforms {}
unsafe impl bytemuck::Zeroable for TracerUniforms {}

impl TracerUniforms {
    fn size() -> wgpu::BufferAddress {
        std::mem::size_of::<Self>() as wgpu::BufferAddress
    }

    fn add_shape(&mut self, shape: &Shape) {
        let count = self.shape_count as usize;
        self.shapes[count] = *shape;
        self.shape_count += 1;
    }
}

struct Tracer {
    uniforms: TracerUniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    start_time: Instant,
}

impl Tracer {
    fn new(device: &wgpu::Device) -> Tracer {
        let uniforms = TracerUniforms {
            shapes: [Shape::default(); SHAPE_MAX_COUNT],
            shape_count: 0,
            sample_count: 0,
            time: 0.0,
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: TracerUniforms::size(),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            label: Some("uniform_buffer"),
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                }],
                label: Some("uniform_bind_group_layout"),
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buffer,
                    range: 0..TracerUniforms::size(),
                },
            }],
            label: Some("uniform_bind_group"),
        });

        let vs_module = device.create_shader_module(include_glsl!("src/tracer.vert"));
        let fs_module = device.create_shader_module(include_glsl!("src/tracer.frag"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&uniform_bind_group_layout],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &render_pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            render_pipeline,
            start_time: Instant::now(),
        }
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
    ) -> wgpu::CommandBuffer {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tracer_encoder"),
        });

        let since_start = Instant::now().duration_since(self.start_time);
        self.uniforms.time = since_start.as_secs_f32();
        self.uniforms.sample_count += 1;

        let temp_uniform_buffer = device.create_buffer_with_data(
            bytemuck::bytes_of(&self.uniforms),
            wgpu::BufferUsage::COPY_SRC,
        );
        encoder.copy_buffer_to_buffer(
            &temp_uniform_buffer,
            0,
            &self.uniform_buffer,
            0,
            TracerUniforms::size(),
        );
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::BLACK,
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw(0..3 * 2, 0..1);
        }
        encoder.finish()
    }
}

struct Gfx {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    tracer: Tracer,
    gui: Gui,
    hud: Hud,
}

impl Gfx {
    async fn request(window: &Window) -> Gfx {
        let width = window.inner_size().width;
        let height = window.inner_size().height;

        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: wgpu::Limits::default(),
            })
            .await;

        let render_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let swap_chain_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: render_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Mailbox,
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_desc);

        let tracer = Tracer::new(&device);
        let gui = Gui::new(&device, width, height, render_format);
        let hud = Hud::new();

        Self {
            surface,
            device,
            queue,
            swap_chain_desc,
            swap_chain,
            tracer,
            gui,
            hud,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.gui.resize(width, height);
        self.swap_chain_desc.width = width;
        self.swap_chain_desc.height = height;
        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swap_chain_desc);
    }

    fn render(&mut self) {
        self.hud.render(&mut self.gui);

        let frame = self
            .swap_chain
            .get_next_texture()
            .expect("Timeout when acquiring next swap chain texture");
        self.queue.submit(&[
            self.tracer.render(&self.device, &frame),
            self.gui.render(&self.device, &frame),
        ]);
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut gfx = Gfx::request(&window).await;

    gfx.tracer.uniforms.add_shape(&Shape {
        a: na::Vector2::new(500.0, 500.0),
        b: na::Vector2::new(800.0, 800.0),
        color: na::Vector3::new(1.0, 1.0, 1.0),
        material: Material::Emissive,
    });
    gfx.tracer.uniforms.add_shape(&Shape {
        a: na::Vector2::new(800.0, 500.0),
        b: na::Vector2::new(800.0, 800.0),
        color: na::Vector3::new(1.0, 0.0, 0.0),
        material: Material::Reflective,
    });
    gfx.tracer.uniforms.add_shape(&Shape {
        a: na::Vector2::new(500.0, 800.0),
        b: na::Vector2::new(800.0, 800.0),
        color: na::Vector3::new(0.0, 0.0, 1.0),
        material: Material::Diffuse,
    });

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                gfx.render();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => {
                    gfx.resize(size.width, size.height);
                }
                WindowEvent::CloseRequested
                | WindowEvent::Destroyed
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Released,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {
                    gfx.hud.update(event);
                }
            },
            _ => {}
        }
    });
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    futures::executor::block_on(run(event_loop, window));
}

// fn orthographic_projection(width: u32, height: u32) -> na::Matrix4<f32> {
//     na::Matrix4::new_orthographic(0.0, width as f32, height as f32, 0.0, -1.0, 1.0)
// }
