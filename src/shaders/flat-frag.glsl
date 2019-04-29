#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
// uniform sampler2D u_NoiseTex1, u_NoiseTex2;

in vec2 fs_Pos;
in vec4 fs_LightVec;
out vec4 out_Col;

// ------- Globals ------- //
vec3 sunDir = normalize( vec3(-1.0,0.0,-1.0) );

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

    float n = p.x + p.y*57.0 + 113.0*p.z;

    // texture(u_NoiseTex1, uv - vec2(0.0, u_Time * 0.01)).r - 1.0;

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


float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
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

// Noise functions
//
// Hash without Sine by DaveHoskins
//
// https://www.shadertoy.com/view/4djSRW
//
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

float voronoi( vec3 x, float tile ) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    float res = 100.;
    for(int k=-1; k<=1; k++){
        for(int j=-1; j<=1; j++) {
            for(int i=-1; i<=1; i++) {
                vec3 b = vec3(i, j, k);
                vec3 c = p + b;

                if( tile > 0. ) {
                    c = mod( c, vec3(tile) );
                }

                vec3 r = vec3(b) - f + hash13( c );
                float d = dot(r, r);

                if(d < res) {
                    res = d;
                }
            }
        }
    }

    return 1.-res;
}

float tilableVoronoi( vec3 p, const int octaves, float tile ) {
    float f = 1.;
    float a = 1.;
    float c = 0.;
    float w = 0.;

    if( tile > 0. ) f = tile;

    for( int i=0; i<octaves; i++ ) {
        c += a*voronoi( p * f, f );
        f *= 2.0;
        w += a;
        a *= 0.5;
    }

    return c / w;
}

float tilableFbm( vec3 p, const int octaves, float tile ) {
    float f = 1.;
    float a = 1.;
    float c = 0.;
    float w = 0.;

    if( tile > 0. ) f = tile;

    for( int i=0; i<octaves; i++ ) {
        c += a*valueNoise( p * f, f );
        f *= 2.0;
        w += a;
        a *= 0.5;
    }

    return c / w;
}

float noiseTexture (vec3 coord) {

     float r = tilableFbm( coord, 16,  3. );
     float g = tilableFbm( coord,  4,  8. );
     float b = tilableFbm( coord,  4, 16. );

     float c = max(0., 1.-(r + g * .5 + b * .25) / 1.75);
     return c;
}

// ------- SDF & Ray Marching Core Functions ------- //
float opU( float d1, float d2 ) { return min(d1,d2); }

vec3 preProcessPnt(vec3 pnt, mat3 rotation) {
	vec3 new_pnt = rotation * pnt;
  vec3 centerOffset = vec3(0.0, 5.0, 5.0);
  return new_pnt - centerOffset;
}

float mapCloud(vec3 pnt) {

  float c = fbm(pnt);//noiseTexture(pnt);
  vec4 offset = vec4(c, c, c, c);
  float cloud1 = sdEllipsoid(pnt + offset.xyz * 3.0, vec3(10, 10, 10));//+ vec3(0.2, 0.33, 1.1)
  // float cloud2 = sdEllipsoid(pnt + vec3(0.0, 5.0, 5.0) + offset.xyz * 3.0, vec3(10, 10, 5));//+ vec3(0.2, 0.33, 1.1)
  // float cloud3 = sdEllipsoid(pnt + vec3(0.0, -5.0, 5.0) + offset.xyz * 3.0, vec3(10, 10, 5));//+ vec3(0.2, 0.33, 1.1)
  // float cloud4 = sdEllipsoid(pnt + vec3(0.0, 5.0, -5.0)+ offset.xyz * 3.0, vec3(10, 10, 5));//+ vec3(0.2, 0.33, 1.1)

  float res = cloud1;
  // res = opU(res, cloud2);
  // res = opU(res, cloud3);
  // res = opU(res, cloud4);
  return res;
}

float map(vec3 og_pnt) {

	vec3 pnt = preProcessPnt(og_pnt, rotateX(1.57));

  // **** Define Components and Position **** //
  // lr, , depth, height

  // Candy
  vec3 cloud_c = pnt - vec3(0.0, -1.3, 2.0);
  float cloud = mapCloud(cloud_c);

  float res = cloud;

  return res;
}

// *** Lighting ***/
// Henyey-Greenstein
float Phase(in float g, in float theta) {
    float invPi = 0.31830988618;
    return 0.25 * invPi * (1.0 - g * g) / (1.0 + g * g - 2.0 * g * pow(theta, 1.5));
}

vec4 integrate( in vec4 sum, in float dif, in float den, in vec3 bgcol, in float t )
{
    // lighting
    vec3 lin = vec3(0.65,0.7,0.75)*1.4 + vec3(1.0, 0.6, 0.3)*dif;
    vec4 col = vec4( mix( vec3(1.0,0.95,0.8), vec3(0.25,0.3,0.35), den ), den );
    col.xyz *= lin;
    col.xyz = mix( col.xyz, bgcol, 1.0-exp(-0.003*t*t) );

    // front to back blending
    col.a *= 0.4;
    col.rgb *= col.a;
    return sum + col*(1.0-sum.a);
}

vec4 raymarch(vec3 ro, vec3 rd, float start, float maxd, out float res) {
  float t = start;
  vec4 sum = vec4(0.0);

  for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    if (t >= maxd || sum.a > 0.99) {break;}
    vec3 pos = ro + t * rd;

    float density = fbm(pos);//noiseTexture(pos);
    if (density > 0.01) {
      // float ext = exp(-scatteringCoeff * density * dt);
      // transmittance *= ext;
      float res = map(pos);
      if (res < EPSILON) {
        // lighting here
        float dif =  clamp((density - fbm(pos))/0.6, 0.0, 1.0 );
        sum = integrate( sum, dif, density, BLUE, t );
      }
    }
    t += max(0.05, 0.02 * t);
  }
  return sum;

  // for(int i=0; i< MAX_MARCHING_STEPS; i++) {
  //   vec3  pos = ro + t*rd;
  //   if( pos.y<-3.0 || pos.y>2.0 || sum.a > 0.99 ) break;
  //   float den = fbm(pos);
  //
  //   if( den> 0.01 ) {
  //     vec3 tmp_pos = pos + 0.3 * sunDir;
  //     float dif =  clamp((den - fbm(tmp_pos))/0.6, 0.0, 1.0 );
  //     sum = integrate( sum, dif, den, BLUE, t );
  //   }
  //   t += max(0.05,0.02*t);
  // }

  // return clamp( sum, 0.0, 1.0 );
  // float t = start; // aka. depth
  // float transmittance = 1.0;
  // float dt;
  // vec3 pos;
  // vec4 color = vec4(0.0);
  // float scatteringCoeff = 0.5;
  //
  // pos = ro + t * rd;
  // float density = noiseTexture(pos);
  //
	// for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
  //
  //   if(color.a > 0.99 ) break;
  //   float accumDensity = 0.0;
  //
  //   res = map(pos);
  //
	// 	if (res < EPSILON) {
  //
  //     // transmittance *= exp(-scatteringCoeff * density * dt);
  //     // color.xyz = BLUE;
  //     color.xyz = WHITE;
  //     color.a = density;//transmittance;
  //
  //     // Compute samples toward light source
  //     // float stepSize = 0.1;
  //     // vec3 light = vec3(0.6, 0.55, 0.4) * 50.0;
  //     //
  //     // float accumDensity = 0.0;
  //     // accumDensity += noiseTexture(pos + dt * sunDir) * dt;
  //     // vec3 incidentLight = light * exp(-scatteringCoeff * accumDensity * dt);
  //     // color.xyz += scatteringCoeff * density * Phase(scatteringCoeff, abs(dot(rd, sunDir))) * incidentLight * transmittance * dt;
  //
  //     //**  beginning of lighting function **//
  //     // color += lighting(accumDensity, );
  //
  //     // float scatteringCoeff = 0.5;
  //     // vec4 light;
  //     // // Sun
  //     // // Background
  //     //
  //     // incidentLight = light * exp(-scatteringCoeff * accumDensity * stepSize);
  //     //
  //     // //Absorbed-light calculation
  //     // light += sunColor * pL;
  //     //
  //     // // Scattered-light calculation
  //     // light += scatteringCoeff * density * Phase(scatteringCoeff, abs(dot(rd, sunDir))) * incidentLight * transmittance * dt;
  //     //
  //     // // Front to back blending
  //     // col.a *= 0.4;
  //     // col.rgb *= col.a;
  //     //
  //     // // Calculate Cloud Color
  //     // return sum + col*(1.0-sum.a);
  //
  //     // **  end of lighting function **//
  //
  //     //if (transmittance <= 0.01) break;
  //   }
  //   dt = max(0.05, 0.02 * t);
  //   t += dt;
	// 	if (t >= maxd) {break;}
	// }
	// return color;

}


vec3 ShadeBackground( vec3 rd )
{
    vec3 color = vec3(0.0);
    //vec3 sunDir = normalize(vec3(cos(iTime), 2.0, sin(iTime)));
    // Sun
    float sunDot = clamp(dot(sunDir, rd), 0.0, 1.0);
    vec3 sunColor = vec3(1.0, 0.58, 0.3) * 20.0;
    color += sunColor * pow(sunDot, 17.0);

    // Sky
    vec3 skyColor = vec3(0.22, 0.44, 0.58) * 0.65;
    color += skyColor;

    return color;
}

vec4 render(vec3 ro, vec3 rd) {
  vec4 col = vec4(1.0);
  // float sun = clamp( dot(sunDir,rd), 0.0, 1.0 );
  // col.xyz = vec3(0.6,0.71,0.75) - rd.y*0.2*vec3(1.0,0.5,1.0) + 0.15*0.5;
  // col.xyz += 0.2*vec3(1.0,.6,0.1) * pow( sun, 8.0 );
  // //clouds
  // vec4 res = raymarch(ro, rd, MIN_DIST, MAX_DIST);
  // col = col*(1.0-res.w) + res.xyz;

  float res;
  vec4 r_col = raymarch(ro, rd, MIN_DIST, MAX_DIST, res);

  // float ambiance = 0.8;

  if (res <= MAX_DIST - EPSILON) {
    col.xyz =r_col.xyz;
    col.a = r_col.a;
    return col;

    // vec3 rp = ro + res * rd;

    // col = WHITE.xyz;
    // vec3 n = estimateNormal(rp);

  } //else {
  col.xyz = BLUE;
  return col;
  //}

  //sun glare
  // col.xyz += 0.2 * vec3(1.0,0.4,0.2) * pow( sun, 3.0 );
  // return col;

}


// ------- Ray March Direction Calc ------- //
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
