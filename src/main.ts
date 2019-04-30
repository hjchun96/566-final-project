import {vec2, vec3} from 'gl-matrix';
import * as Stats from 'stats-js';
import * as DAT from 'dat-gui';
import Square from './geometry/Square';
import OpenGLRenderer from './rendering/gl/OpenGLRenderer';
import Camera from './Camera';
import {setGL} from './globals';
import ShaderProgram, {Shader} from './rendering/gl/ShaderProgram';
// import Texture from './rendering/gl/Texture';

// Define an object with application parameters and button callbacks
// This will be referred to by dat.GUI's functions that add GUI elements.
const controls = {
  tesselations: 5,
  'Load Scene': loadScene, // A function pointer, essentially
  'Scene': 'Sunny',
  'GodRay': 'On',
};

let square: Square;
let time: number = 0;
// let noiseTex1: Texture;
// let noiseTex2: Texture;

function loadScene() {
  square = new Square(vec3.fromValues(0, 0, 0));
  square.create();

  // noiseTex1 = new Texture('../resources/fbmRep1.png', 0)
  // noiseTex2 = new Texture('../resources/fbmRep2.png', 0)
  //

  // Vapor Emulation

  // Illumination Pass (set lightsource in CPU Side)
}

function main() {
  window.addEventListener('keypress', function (e) {
    // console.log(e.key);
    switch(e.key) {
      // Use this if you wish
    }
  }, false);

  window.addEventListener('keyup', function (e) {
    switch(e.key) {
      // Use this if you wish
    }
  }, false);

  // Initial display for framerate
  const stats = Stats();
  stats.setMode(0);
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.left = '0px';
  stats.domElement.style.top = '0px';
  document.body.appendChild(stats.domElement);

  // Add controls to the gui
  const gui = new DAT.GUI();
  gui.add(controls, 'Scene', ['Sunny', 'Gloomy']);
  gui.add(controls, 'GodRay', ['On', 'Off']);

  // get canvas and webgl context
  const canvas = <HTMLCanvasElement> document.getElementById('canvas');
  const gl = <WebGL2RenderingContext> canvas.getContext('webgl2');
  if (!gl) {
    alert('WebGL 2 not supported!');
  }
  // `setGL` is a function imported above which sets the value of `gl` in the `globals.ts` module.
  // Later, we can import `gl` from `globals.ts` to access it
  setGL(gl);

  // Initial call to load scene
  loadScene();


  let camera = new Camera(vec3.fromValues(0, -8, -15), vec3.fromValues(0, -5, 0));
  if (controls.Scene == "Gloomy") {
    camera = new Camera(vec3.fromValues(0, -2, -5), vec3.fromValues(0, 0, 0));
  }

  const renderer = new OpenGLRenderer(canvas);
  renderer.setClearColor(164.0 / 255.0, 233.0 / 255.0, 1.0, 1);
  gl.enable(gl.DEPTH_TEST);

  const flat = new ShaderProgram([
    new Shader(gl.VERTEX_SHADER, require('./shaders/flat-vert.glsl')),
    new Shader(gl.FRAGMENT_SHADER, require('./shaders/flat-frag.glsl')),
  ]);

  function processKeyPresses() {
    // Use this if you wish
  }

  // flat.bindTexToUnit(flat.unifSampler1, noiseTex1, 0);
  // flat.bindTexToUnit(flat.unifSampler2, noiseTex2, 1);

  // This function will be called every frame
  function tick() {
    camera.update();
    stats.begin();
    gl.viewport(0, 0, window.innerWidth, window.innerHeight);
    renderer.clear();
    processKeyPresses();
    let scene, godray;
    if (controls.Scene == "Sunny") {
      scene = 1;
    } else if (controls.Scene == "Gloomy") {
      scene = 2;
    }

    if (controls.GodRay == "On") {
      godray = 1;
    } else if (controls.GodRay == "Off") {
      godray = 2;
    }

    renderer.render(camera, flat, [
      square,
    ], scene, time);
    time++;
    stats.end();

    // Tell the browser to call `tick` again whenever it renders a new frame
    requestAnimationFrame(tick);
  }

  window.addEventListener('resize', function() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.setAspectRatio(window.innerWidth / window.innerHeight);
    camera.updateProjectionMatrix();
    flat.setDimensions(window.innerWidth, window.innerHeight);
  }, false);

  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.setAspectRatio(window.innerWidth / window.innerHeight);
  camera.updateProjectionMatrix();
  flat.setDimensions(window.innerWidth, window.innerHeight);

  // Start the render loop
  tick();
}

main();
