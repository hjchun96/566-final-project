#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform int u_Scene;
uniform int u_GodRay;

uniform sampler2D u_NoiseTex1, u_NoiseTex2;

in vec2 fs_Pos;
in vec4 fs_LightVec;
out vec4 out_Col;

// ------- Globals ------- //
vec3 sunDir1 = normalize(vec3(-3.0, 2.5, 1.0));
vec3 sunDir2 = normalize(vec3(3.0,2., 1.0));//normalize(vec3(2.0, 0.5, -1.0));
// vec3 sunDir2 = normalize(vec3(0., 2.5, 1.0));//normalize(vec3(2.0, 0.5, -1.0));
vec3 cloudDir1 = normalize(vec3(3.0, 0.0, -3.0));
vec3 cloudDir2 = normalize(vec3(1.0, 0.0, 1.0));
float speed = 0.02;
// ------- Constants ------- //

#define PI = 3.1415926535897932384626433832795;


// Colors
vec3 BLUE = vec3(135./255.,206./255.,250./255.);
vec3 WHITE = vec3(1.0, 1.0, 1.0);

// Ray Marching
const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;


// Fractal Brownian Motion (referenced lecture code)
float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

float random1( vec3 p , vec3 seed) {
  return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

float interpNoise2D( float x, float y, vec2 seed) {

	float intX = floor(x);
	float fractX = fract(x);
	float intY = floor(y);
	float fractY = fract(y);

	float v1 = random1(vec2(intX, intY), seed);
	float v2 = random1(vec2(intX + 1.0, intY), seed);
	float v3 = random1(vec2(intX, intY + 1.0), seed);
	float v4 = random1(vec2(intX + 1.0, intY + 1.0), seed);

	float i1 = mix(v1, v2, fractX);
	float i2 = mix(v3, v4, fractY);
	return mix(i1, i2, fractY);
}

mat3 m = mat3( 0.00,  0.80,  0.60,
              -0.80,  0.36, -0.48,
              -0.60, -0.48,  0.64 );
float hash( float n )
{
    return fract(sin(n)*43758.5453);
}

float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y* 57.0 + 113.0 * p.z;

    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);

    return res;
}

float fbm( vec3 p )
{
    float f;
    f  = 0.5000*noise( p ); p = m*p*2.02;
    f += 0.2500*noise( p ); p = m*p*2.03;
    f += 0.1250*noise( p );
    return f;
}

// ------- Primatives ------- //
float sdEllipsoid(vec3 p,vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf
}

// ------- Rotate Operations ------- //
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}


float rand(vec3 co)
{
	return fract(sin(dot(co.xyz, vec3(12.9898, 78.233, 56.787))) * 43758.5453);
}

//----------------------------------------------------------------------------//
// Hash without Sine by DaveHoskins
// https://www.shadertoy.com/view/4djSRW
//----------------------------------------------------------------------------//

float hash12( vec2 p ) {
    p  = 50.0*fract( p*0.3183099 );
    return fract( p.x*p.y*(p.x+p.y) );
}

float hash13(vec3 p3) {
    p3  = fract(p3 * 1031.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 hash33(vec3 p3) {
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+19.19);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

float valueHash(vec3 p3) {
    p3  = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

//
// Noise functions used for cloud shapes
//
float valueNoise( in vec3 x, float tile ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    return mix(mix(mix( valueHash(mod(p+vec3(0,0,0),tile)),
                        valueHash(mod(p+vec3(1,0,0),tile)),f.x),
                   mix( valueHash(mod(p+vec3(0,1,0),tile)),
                        valueHash(mod(p+vec3(1,1,0),tile)),f.x),f.y),
               mix(mix( valueHash(mod(p+vec3(0,0,1),tile)),
                        valueHash(mod(p+vec3(1,0,1),tile)),f.x),
                   mix( valueHash(mod(p+vec3(0,1,1),tile)),
                        valueHash(mod(p+vec3(1,1,1),tile)),f.x),f.y),f.z);
}


// iq's noise
float pn( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( u_NoiseTex2, (uv+ 0.5)/256.0, 0.0 ).yx;
	return -1.0+2.4*mix( rg.x, rg.y, f.z );
}


float fpn(vec3 p)
{
   return pn(p*.06125)*.5 + pn(p*.125)*.25 + pn(p*.25)*.125;
}

//----------------------------------------------------------------------------//
// SDF & Ray Marching Core                                                    //
//----------------------------------------------------------------------------//
float opU( float d1, float d2 ) { return min(d1,d2); }
float opS( float d1, float d2 ) { return max(-d1,d2); }


float singleCloud(vec3 pnt, vec3 offset) {
  float cloud1 = sdEllipsoid(pnt + offset * 3.0, vec3(7,7, 7));
  float cloud2 = sdEllipsoid(pnt + vec3(4.0, 0.0, 0.0) + offset * 2.0, vec3(7,3, 7));
  float cloud3 = sdEllipsoid(pnt - vec3(4.0, 0.0, 0.0) + offset * 2.0, vec3(7,3, 7));

  float res = cloud1;
  res = opU(res, cloud2);
  res = opU(res, cloud3);
  // res = opU(res, cloud4);
  float cloud_base = sdRoundBox(pnt - vec3(0.,4.2,0.)+ offset.xyz, vec3(7.0, 1.2, 7.0), 1.0);
  res = opS(cloud_base, res);
  return res;
}

float mapCloud(vec3 pnt) {

  float c = fbm(pnt);//noiseTexture(pnt);
  // float c = Clouds(pnt);
  vec4 offset = vec4(c, c, c, c);
  // vec4 offset = vec4(0.);

  float res;
  if (u_Scene ==1) { // SUNNY
    res = singleCloud(pnt, offset.xyz);
    res = opU(res, singleCloud(pnt + vec3(-10.0, 0.0, -10.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(10.0, 0.0, 5.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(10.0, 0.0, -10.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(-10.0, 0.0, 0.0), offset.xyz));

    res = opU(res, singleCloud(pnt + vec3(-10.0, -4.0, -12.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(7.0, -6.0, -10.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(10.0, -4.0, -12.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(20.0, -3.0, -8.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(-23.0, -3.0, -9.0), offset.xyz));

    res = opU(res, singleCloud(pnt + vec3(25.0, 0.0, 12.0), offset.xyz));

    // res = opU(res, singleCloud(pnt + vec3(10.0, 10.0, 5.0), offset.xyz));
    res = opU(res, singleCloud(pnt + vec3(10.0, 0.0, 3.0), offset.xyz));


    vec3 og_pnt  = pnt;
    if (u_Time == 12.0) {
      vec3 negOffset = cloudDir1 * speed * u_Time;
      pnt -= negOffset;
    }
    if (u_Time >= 12.0) {
      // res = opU(res, singleCloud(pnt + vec3(25.0, 10.0, 10.0), offset.xyz));
      // res = opU(res, singleCloud(pnt + vec3(10.0, 0.0, 5.0), offset.xyz));
      // res = opU(res, singleCloud(pnt + vec3(34.0, 0.0, 15.0), offset.xyz));
      res = opU(res, singleCloud(pnt + vec3(28.0, 0.0, 10.0), offset.xyz));

      res = opU(res, singleCloud(pnt + vec3(30.0, -5.0, 12.0), offset.xyz));
      // res = opU(res, singleCloud(og_pnt + vec3(25.0, 0.0, 15.0), offset.xyz));
    }


  } else { // GLOOMY

    res = singleCloud(pnt + vec3(15.0, -1.0, -1.0), offset.xyz); // front left
    //
    float bp1 = opU(res, sdEllipsoid(pnt + vec3(4.0, 14.0, -15) + offset.xyz * 2.0, vec3(20, 15, 5))); // gigantic background one
    float bp2 = opU(res, sdEllipsoid(pnt + vec3(-15, 16.0, -15) + offset.xyz * 2.0, vec3(15, 10, 5))); // gigantic background one
    float bp3 = opU(res, sdEllipsoid(pnt + vec3(15, 10.0, -15) + offset.xyz * 2.0, vec3(15, 10, 5))); // gigantic background one

    float backpiece = opU(bp1, bp2);
    backpiece = opU(backpiece, bp3);
    res = opU(res, backpiece);

    float chunk1 = sdEllipsoid(pnt + vec3(-3.0, -5.0, -3.) + offset.xyz * 2.0, vec3(5, 4, 4)); // centerpiece one
    float chunk2 = sdEllipsoid(pnt + vec3(-3.0, -5.0, -3.0) + offset.xyz *2.0, vec3(10, 7, 5)); // centerpiece one
    float chunk3 = sdEllipsoid(pnt + vec3(1.0, -5.0, 1.0) + offset.xyz *2.0, vec3(2, 3, 5)); // centerpiece one
    float chunk4 = sdEllipsoid(pnt + vec3(-1.0, -6.0, -1.0) + offset.xyz * 2.0, vec3(5, 4, 5)); // centerpiece one
    float chunk5 = sdEllipsoid(pnt + vec3(-9.0, -2.0, -2.0) + offset.xyz * 2.0, vec3(3, 3, 4)); // centerpiece one

    float centerpiece = opU(chunk1, chunk2);
    centerpiece = opU(centerpiece, chunk3);
    centerpiece = opU(centerpiece, chunk4);
    centerpiece = opU(centerpiece, chunk5);

    res = opU(res, centerpiece);

    for (int i = 1; i < 5; i++) {
      res = opU(res, sdEllipsoid(pnt + vec3(2. - float(i)*9., -3.0, -7.) + offset.xyz * 2.0, vec3(7, 7, 5)));
    }

    for (int i = 1; i < 4; i++) {
      res = opU(res, sdEllipsoid(pnt + vec3(0. - float(i)*9., 1.0, -8.5) + offset.xyz * 2.0, vec3(7, 7, 5)));
    }

    for (int i = 1; i < 3; i++) {
      res = opU(res, sdEllipsoid(pnt + vec3(-4. - float(i)*10., 5.0, -10.) + offset.xyz * 2.0, vec3(7, 10, 5)));
    }

    //
    // float farback = singleCloud(pnt + vec3(-20.0, -14.0, -17.0), offset.xyz);
    //
    // res = opU(res, farback);

    // res = opU(res, b .);

  }

  return res;
}

float map(vec3 og_pnt) {
  vec3 cloud_c = og_pnt * rotateZ(3.14) - vec3(0.0, 4., 5.0);
  float cloud = mapCloud(cloud_c);
  float res = cloud;

  return res;
}


//----------------------------------------------------------------------------------------

float Noise3D(vec3 p)
{
    vec2 e = vec2(0.0, 1.0);
    vec3 i = floor(p);
    vec3 f = fract(p);

    float x0 = mix(rand(i + e.xxx), rand(i + e.yxx), f.x);
    float x1 = mix(rand(i + e.xyx), rand(i + e.yyx), f.x);
    float x2 = mix(rand(i + e.xxy), rand(i + e.yxy), f.x);
    float x3 = mix(rand(i + e.xyy), rand(i + e.yyy), f.x);

    float y0 = mix(x0, x1, f.y);
    float y1 = mix(x2, x3, f.y);

    float val = mix(y0, y1, f.z);

    val = val * val * (3.0 - 2.0 * val);
    return val;
}

vec4 godray(vec2 uv, float angle)
{
  vec2 ndc = (2.0 * gl_FragCoord.xy - u_Dimensions.xy) / u_Dimensions.y;
  vec2 godRayOrigin = ndc + vec2(-1.5, -1.);
  if (u_Scene == 2) godRayOrigin = ndc + vec2(0.8, -1.5);

  float inputfunc = atan(godRayOrigin.y, godRayOrigin.x) * 0.7;
  float light = inputfunc * 0.8 + 0.5;
  light = 0.5 * (light+ (sin(inputfunc) * 0.5 ));
  light *= pow(clamp(dot(normalize(-godRayOrigin), normalize(ndc - godRayOrigin)), 0.0, 1.0), 2.5);
  light *= pow(uv.y, 1.1);
  light = pow(light, 1.75);
  return  vec4(vec3(light), 1.0);// * (1.0 - uv.y);
}


// const float EXPOSURE     = 2.5;
// const float INV_GAMMA    = 1.0 / 2.2;
//
// // Uncharted 2 tone mapping
// const float A = 0.15;
// const float B = 0.50;
// const float C = 0.10;
// const float D = 0.20;
// const float E = 0.02;
// const float F = 0.30;
// vec3 unchartedToneMapping(in vec3 x) {
//     return ((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F;
// }
// vec3 toneMapping(in vec3 x) {
//     vec3 color = unchartedToneMapping(x*EXPOSURE);
//     vec3 white = vec3(100);
//     color *= 1.0 / unchartedToneMapping(white);
//     return pow(color, vec3(INV_GAMMA));
// }

//----------------------------------------------------------------------------//
// volumetric integration helpers                                             //
//----------------------------------------------------------------------------//
float powder(float DenSample, float cosAngle, float DenCoef, float PowderCoef) {
  float Powder = 1.0 - exp(-DenSample * DenCoef * 2.0);
  Powder = clamp(Powder * PowderCoef, 0.0, 1.0);
  return mix(1.0, Powder, smoothstep(0.5, -0.5, cosAngle));
}

float BeerLaw(float DenSample, float DenCoef)
{
  return exp( -DenSample * DenCoef);
}

float henyeyGreenstein(float cosAngle, float g) {
  float g2 = g * g;
  float invPi = 0.31830988618;
  return 0.25 * invPi * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cosAngle, 3.0/2.0));
  // return phase;
}

float beerLambert(float d) {
  return max(exp(-d), 0.7*exp(-0.25*d));
}

//----------------------------------------------------------------------------//
// lighting                                                                  //
//----------------------------------------------------------------------------//
vec3 shadeSky( vec3 rd )
{
    vec3 sunDir = sunDir1;
    if (u_Scene == 2) sunDir = sunDir2;

    vec3 color = vec3(0.0);
    vec3 skyColor, sunColor;
    if (u_Scene == 1) {
      skyColor = BLUE;
      sunColor = vec3(1.0, 0.58, 0.3);
      float sunDot = clamp(dot(sunDir, rd), 0.0, 1.0);
      color += sunColor * pow(sunDot, 17.0);
    } else if (u_Scene == 2) {
      // skyColor =vec3(26./255., 42./255., 68./255.);
    // sunColor = vec4(204./255., 222./255., 249./255., 1.0);;//vec4(1.0, 0.58, 0.3, 1.0);

      skyColor =vec3(3./255., 19./255., 44./255.); // vec3(29./255., 44./255.,67./255.);
      sunColor = vec3(204./255., 222./255., 249./255.);;//vec4(1.0, 0.58, 0.3, 1.0);
      // sunColor = vec3(161./255., 191./255., 224./255.);

      // sunColor = vec3(0.3, 0.3, 0.6); // not super distinct. act as lightsource.
      float sunDot = clamp(dot(sunDir, rd), 0.0, 1.0);
      color += sunColor * pow(sunDot, 17.0);
    }
    color += skyColor;
    return color;
}
//
// float rand(vec2 co) {
//     return fract(sin(dot(co*0.123,vec2(12.9898,78.233))) * 43758.5453);
// }

float getScatter(float cosAngle, float fg, float bg, float hg) {

  float fwdScatter = henyeyGreenstein(cosAngle, fg);
  float bkwdScatter = henyeyGreenstein(cosAngle, bg);
  float totalHGPhase = (hg * fwdScatter) + ((1. - hg) * bkwdScatter);
  return totalHGPhase;
  // return fwdScatter;
}


float getAbsorbedLight(vec3 pos, float den, float cosAngle, out float sden) {
  vec3 sunDir = sunDir1;
  if (u_Scene == 2) sunDir = sunDir2;

  // Absorption
  // float sden;
  float stepsize = 0.1;
  for (int i = 0; i < 3; i++) {
      vec3 lsPos = pos + stepsize * float(i) * sunDir;
      if (map(lsPos) < EPSILON) {
        float lsDen = fbm(lsPos);
        if (lsDen > 0.0) {
            sden += lsDen;
        }
      }
  }
  return beerLambert(sden);
}

//----------------------------------------------------------------------------//
// rendering                                                                  //
//----------------------------------------------------------------------------//
vec4 raymarch(vec3 ro, vec3 rd, float start, float maxd, out float res) {

  // set oup variables
  vec3 sunDir, cloudDir;
  if (u_Scene == 1)
  {
    cloudDir = cloudDir1;
    sunDir = sunDir1;
  } else {
    cloudDir = cloudDir2;
    sunDir = sunDir2;
  }

  // background sky
  vec3 colSky = vec3(0.6,0.71,0.75) - rd.y*0.2*vec3(1.0,0.5,1.0) + 0.15*0.5;
  float sun = clamp(dot(sunDir,rd), 0.0, 1.0);
  colSky += 0.25 * vec3(1.0,.6,0.1) * pow(sun, 8.0) + 2.0 * vec3(1.0,.6,0.1) * pow(sun, 1.0);

  vec4 sum = vec4(0.0);

  // setup ray marching variables
  float t             = start;
  float dt            = 0.5;
  float alpha         = 0.0; // acculmulated density.
  float transmittance = 1.0;

  // Lighting Variables
  //based on "Real-Time Volumetric Cloudscapes" by Andrew Schneider
  const float FwdSctG   = 0.5;
  const float BckwdSctG = -0.1;
  const float HGCoef    = .8;
  const float DenCoef   = .75;
  const float PowderCoef = 1.3;

  float cosAngle = dot(normalize(rd), normalize(sunDir));
  float scatter = getScatter(cosAngle, FwdSctG, BckwdSctG, HGCoef);

  vec3 bg = shadeSky(rd);
  for (int i = 0; i < MAX_MARCHING_STEPS; i++)
  {

    vec3 pos = ro + t * rd + cloudDir * speed * u_Time;
    if (t >= maxd) break;
    if (sum.a > 0.99) {
      sum.a = 1.00;
      break;
    }

    float den = fbm(pos);
    float res;
    if (den > 0.01) {

      res = map(pos);


      if (res < EPSILON) {

        // Absorption
        float sden;
        float beer = getAbsorbedLight(pos, den, cosAngle, sden);

        float ext = exp(-0.5 * sden * 0.1);
        transmittance *= ext;

        float powder = powder(exp(-den), cosAngle, DenCoef, PowderCoef);

        vec3 light = vec3(0.6, 0.55, 0.4)* 50.0;
        float lightE = scatter * beer * powder;

        vec4 col;
        col.xyz = 0.5 * light * lightE * transmittance; // * Dt? //(vec3(1.) - ext) * (WHITE * beer * TotalHGPhase * powder) * transmittance;
        col.a = 1. - transmittance;

        // // front to back blending
        if (u_Scene == 1) col.xyz = mix( col.xyz, bg *0.9 +  WHITE * 0.1, 1.0-exp(-0.003*t*t));
        else col.xyz = mix( col.xyz, bg *0.5 +  WHITE * 0.5, 1.0-exp(-0.003*t*t));

        col.a *= 0.35;
        col.rgb *= col.a;
        sum +=  col*(1.0-sum.a);

        if (alpha > 1.) {
          alpha = 1.0;
          break;
        }

      }
    }

    dt = max(0.05, 0.02 * t);
    t += dt;
  }

  return vec4((1. - sum.a) * bg + sum.xyz, 1.0);
}

vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        map(vec3(p.x + EPSILON, p.y, p.z)) - map(vec3(p.x - EPSILON, p.y, p.z)),
        map(vec3(p.x, p.y + EPSILON, p.z)) - map(vec3(p.x, p.y - EPSILON, p.z)),
        map(vec3(p.x, p.y, p.z  + EPSILON)) - map(vec3(p.x, p.y, p.z - EPSILON))
    ));
}


vec4 render(vec3 ro, vec3 rd) {
  vec3 sunDir = sunDir1;
  if (u_Scene == 2) sunDir = sunDir2;
  vec4 col = vec4(1.0);
  float sun = clamp( dot(sunDir,rd), 0.0, 1.0 );

  float res;
  vec4 r_col = raymarch(ro, rd, MIN_DIST, MAX_DIST, res);

  if (res <= MAX_DIST - EPSILON) {
    col.xyz =r_col.xyz;
    col.a = r_col.a;
  }

  // sun glare
  col.xyz += 0.2 * vec3(1.0,0.4,0.2) * pow( sun, 3.0 );

  if (u_GodRay == 1) {
    vec2 uv;
    uv.x  = gl_FragCoord.x / u_Dimensions.x;
    uv.y = gl_FragCoord.y/ u_Dimensions.y;
    vec4 gr = godray(uv, 10.);
    vec4 sunColor = vec4(1.0, 0.58, 0.3, 1.0);
    if (u_Scene == 2) sunColor = vec4(204./255., 222./255., 249./255., 1.0);;//vec4(1.0, 0.58, 0.3, 1.0);
    col += gr *sunColor;
  }
  return col;
  // if (u_Scene == 2) {
  //   return vec4(toneMapping(col.xyz), 1.0);
  // } else {
  //   return col;
  // }
  // vec3 rp = ro + res * rd;
  // vec3 n = estimateNormal(rp);
  // col.xyz+=pow(clamp(-n.y, 0.0,1.0),2.)*WHITE/1.5;

}


//----------------------------------------------------------------------------//
// ray setup                                                                  //
//----------------------------------------------------------------------------//
vec3 calculateRayMarchPoint() {
  vec3 forward = u_Ref - u_Eye;
	vec3 right = normalize(cross(u_Up, forward));
	vec3 up = normalize(cross(forward, right));

	float len = length(u_Ref - u_Eye);
	forward = normalize(forward);
	float aspect = u_Dimensions.x / u_Dimensions.y;

  float fovy = 90.0;
	float alpha = radians(fovy/2.0);
	vec3 V = up * len * tan(alpha);
	vec3 H = right * len * aspect * tan(alpha);

	float sx = 1.0 - (2.0 * gl_FragCoord.x/u_Dimensions.x);
	float sy = (2.0 * gl_FragCoord.y/u_Dimensions.y) - 1.0;
	vec3 p = u_Ref + sx * H + sy * V;
	return p;
}

void main() {
	vec3 p = calculateRayMarchPoint();
  vec3 rd = normalize(p - u_Eye);
  vec3 ro = u_Eye;
  out_Col = render(ro, rd);
}
