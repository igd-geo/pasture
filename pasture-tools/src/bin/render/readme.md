# Point cloud renderer

### TODO

- improve the rendering
	- See [the potree thesis](https://www.cg.tuwien.ac.at/research/publications/2016/SCHUETZ-2016-POT/)
	  for excellent descriptions of this
	- using a data structure for proper point sampling, allowing to scale points,
	  avoiding holes and overdraw.
	- add outlines for depth buffer discontinuities (eye dome lighting EDL)
- add gui (e.g. imgui overlay) to modify parameters and visualized data fields
- wasm: Allow to select point cloud file.
  See e.g. [here](https://stackoverflow.com/questions/67944469/rust-wasm-iterating-over-input-file)
  for an implementation sketch using web_sys.
- wasm: the input seems kinda wonky

## Wasm
### Building for wasm

First, you need to compile the renderer for the wasm target.
```
cargo build --bin render --target wasm32-unknown-unknown
```

We use `wasm32-unknown-unknown` since all other wasm targets caused various issues.
Apart from the render application, many examples and tests don't compile
for wasm, that is expected.

You then need to use `wasm-bindgen`. The tool can be installed via 
`cargo install -f wasm-bindgen-cli`, see [here](https://rustwasm.github.io/wasm-bindgen/reference/cli.html)
for more details.

```
wasm-bindgen --target web --out-dir pasture-tools/src/bin/render/wasm target/wasm32-unknown-unknown/debug/render.wasm
```

This will move the wasm code to the `pasture-tools/src/bin/render/wasm` folder
and generate the javascript binding code.
You can then simply run a web server on `pasture-tools/src/bin/render/wasm/index.html`
to see the website with the renderer.

For instance, run
```
python -m http.server --directory pasture-tools/src/bin/render/wasm/
```

and then access `localhost:8000` in your browser.

### Workarounds

There are several workarounds in place to allow compiling for the wasm
backend:

- The dependencies overwrite in the root `Cargo.toml`.
  We need the latest version of `chrono` for pull request 568.
  When there is a new release of `chrono`, we can just use that
- We need a custom `getrandom` to enable its `js` feature.
- When compiling for wasm, several tests and examples in pasture
  are completely disabled. The `reprojection` algorithm is disabled since
  `proj-sys` dependency (native C++) couldn't be compiled on wasm.
- webgl does not support compute shaders. That's why we can't use any
  compute shaders in the renderer at all and all occurrences of compute
  shader stages (e.g. in bind group layouts) were currently disabled.
