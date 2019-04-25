#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

uniform sampler2D u_DensityTex;

in vec2 fs_Pos;
in vec4 fs_LightVec;
out vec4 out_Col;

// ------- Globals ------- //
vec3 sunDir = normalize( vec3(-1.0,0.0,-1.0) );

// ------- Constants ------- //
const float PI = 3.1415926535897932384626433832795;

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


//
// float fbm( float x, float y, vec2 seed) {
//
// 	float total = 0.0;
// 	float persistance = 0.5;
// 	float octaves = 8.0;
//
// 	for (float i = 0.0; i < octaves; i = i + 1.0) {
// 		float freq = pow(2.0, i);
// 		float amp = pow(persistance, i);
// 		total += interpNoise2D(x * freq, y * freq, seed) * amp;
// 	}
// 	return total;
// }


/*

// ------ Math Helpers -----//

float dist (vec2 p1, vec2 p2) {
  float diffX = p1.x - p2.x;
  float diffY = p1.y - p2.y;
  return sqrt(diffX*diffX + diffY*diffY);
}


float dot2( in vec3 v ) { return dot(v,v); }

// ------- Primatives ------- //
// All taken from IQ: http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

float sdEllipsoid(vec3 p,vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
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

// ------- Operations ------- //
// All taken from IQ, constants modified appropriately

float opU( float d1, float d2 ) { return min(d1,d2); }

float opI( float d1, float d2 ) { return max(d1,d2); }

float opSmoothUnion( float d1, float d2, float k ) {
  float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
  return mix( d2, d1, h ) - k*h*(1.0-h);
}

float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}


float opDisplace(vec3 p)
{
  float total = floor(p.x*float(u_Dimensions.x)) +
        floor(p.y*float(u_Dimensions.y));
  bool isEven = mod(total, 2.0)==0.0;
  return (isEven)? 0.4:0.1;
}

// ------- Toolbox Functions ------- //
float impulse( float k, float x )
{
    float h = k*x;
    return h*exp(1.0-h);
}


float cubicPulse( float c, float w, float x )
{
    x = abs(x - c);
    if( x > w ) return 0.0;
    x /= w;
    return 1.0 - x*x*(3.0-2.0*x);
}

float square_wave(float x, float freq, float amplitude) {
	return abs(mod(floor(x * freq), 2.0)* amplitude);
}

float bias (float b, float t) {
  return pow(t, log(b)/log(0.5));
}

float gain (float g, float t) {
  if (t < 0.5) {
    return bias(1.-g, 2.*t)/2.;
  } else {
    return 1. - bias(1.- g, 2. - 2.*t)/2.;
  }
}

float rand(vec2 co){return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);}
float rand (vec2 co, float l) {return rand(vec2(rand(co), l));}
float rand (vec2 co, float l, float t) {return rand(vec2(rand(co, l), t));}

float perlin(vec2 p, float dim, float time) {
	vec2 pos = floor(p * dim);
	vec2 posx = pos + vec2(1.0, 0.0);
	vec2 posy = pos + vec2(0.0, 1.0);
	vec2 posxy = pos + vec2(1.0);

	float c = rand(pos, dim, time);
	float cx = rand(posx, dim, time);
	float cy = rand(posy, dim, time);
	float cxy = rand(posxy, dim, time);

	vec2 d = fract(p * dim);
	d = -0.5 * cos(d * 3.141592) + 0.5;

	float ccx = mix(c, cx, d.x);
	float cycxy = mix(cy, cxy, d.x);
	float center = mix(ccx, cycxy, d.y);

	return center * 2.0 - 1.0;
}

float turbulence(float x, float y, float size)
{
  float value = 0.0;
  float initialSize = size;

  while(size >= 1.)
  {
    value += smoothstep(x/size, y / size, size);
    size /= 2.0;
  }

  return(128.0 * value / initialSize);
}
/*
// ------- SDF & Ray Marching Core Functions ------- //

vec3 preProcessPnt(vec3 pnt, mat3 rotation) {
	vec3 new_pnt = rotation * pnt;
  vec3 centerOffset = vec3(0.0, 5.0, 5.0);
  return new_pnt - centerOffset;
}

float mapCloud(vec3 pnt) {

  float tube = sdTorus(pnt * rotateX(1.57), vec2(0.7, 0.2));
  return tube;
}

vec2 map(vec3 og_pnt) {

	vec3 pnt = preProcessPnt(og_pnt, rotateX(1.57));

  // **** Define Components and Position **** //
  // lr, , depth, height

  // Candy
  vec3 cloud_c = pnt -vec3(0.0, -1.3, 2.0);
  float cloud = mapCloud(candy_c);
  vec2 res = vec2(cloud, 0.1);

  float pupil1 = sdEllipsoid(bear_c+ vec3(0.2, 0.33, 1.1), vec3(0.05, 0.06, 0.08));
  float pupil2 = sdEllipsoid(bear_c+ vec3(-0.2, 0.33, 1.1), vec3(0.05, 0.06, 0.08));
  float pupils = opU(pupil1, pupil2);
  if (pupils < res.x){ res = vec2(pupils, 0.9);}

  // Plane (floor)
  float plane = sdPlane(pnt - vec3(0.0, 0.0, 2.0), vec4(0.0, 0.0, -1.0, 1.0));
  if (plane < res.x){ res = vec2(plane, 1.0);}

  return res;
}

vec2 raymarch(vec3 ro, vec3 rd, float start, float maxd) {// eye = ray orientation

	float depth = start;

	for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
		vec3 pnt = ro + depth * rd;
    vec2 res = map(pnt);
		float dist = res.x;
		if (dist < EPSILON) {return vec2(depth, res.y);}
		depth += dist;
		if (depth >= maxd) {break;}
	}

	return vec2(maxd, 0.0);
}

// ------- Normal ------- //
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        map(vec3(p.x + EPSILON, p.y, p.z)).x - map(vec3(p.x - EPSILON, p.y, p.z)).x,
        map(vec3(p.x, p.y + EPSILON, p.z)).x - map(vec3(p.x, p.y - EPSILON, p.z)).x,
        map(vec3(p.x, p.y, p.z  + EPSILON)).x - map(vec3(p.x, p.y, p.z - EPSILON)).x
    ));
}

float softshadow(vec3 ro,vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t < maxt; )
    {
        float h = map(ro + rd*t).x;
        if( h< 0.001 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h;
    }
    return res;
}

vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));

    float dotLN = dot(L, N);
    float dotRV = dot(R, V);

    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    }

    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.6 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;

    vec3 light1Pos = vec3(10.0, 20.0, 2.0);
    vec3 light1Intensity = vec3(0.1, 0.1, 0.1);

    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);

    vec3 light2Pos = vec3(7.0,
                          20.0,
                          20.0);
    vec3 light2Intensity = vec3(0.1, 0.1, 0.1);

    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity);

    return color;
}


// From simon green and others
float ambientOcclusion(vec3 p, vec3 n)
{
    int num_steps = 5;
    float delta = 3.0;

    float occ= 0.0;
    float w = 2.0;

    for(int i=1; i <= num_steps; i++) {
        float d = (float(i) / float(num_steps)) * delta;
        vec2 res = map(p + n*d);
        occ += w*(d - res.x);
        w *= 0.6;
    }
    return clamp(1.0 - 1.5*occ, 0.0, 1.0);
}

*/


// Pseudocode from Paper Translation WIP
// function getCandidates(vec3 ro, vec3 rd, int i) {
//
// 	    for j = 0< number of spheres in boundingbox[i] do
// 	        temp = ro - sphere.center;
//           a = dot2(rd);
// 	        b = 2.0 * rd * temp;
// 	        c = temp * temp * radius * radius;
// 	        delta = b * b - 4 * a * c;
// 	        if Δ>0 then {There is a collision}
// 	           σ←Δ−−√
// 	           λin = -b −σ2.0×a
// 	           λout= -b +σ2.0×a
// 	           candidates[k] = (λin,λout,j,i)
// 	           k=k+1
// 	        end if
// 	    end for
// 	    return (candidates,k)
// 	end function
// }

// 	function sortCandidates(candidates, n) {//Insertion-sort algorithm
// 	    for (int k = 1; k < n; k++) {
// 	       aux = candidates[k];
// 	       h=k−1;
// 	       while ((h>=0) && (aux λin< candidates[h] λin)) {
// 	           candidates[h + 1] = candidates[h]
// 	           h=h−1
// 	       }
// 	       candidates[h + 1] = aux
// 	    }
// }

// vec4 rayTrace(vec3 ro,vec3 rd) {
//
// 	    B = boundingboxDetection(ro,rd);
// 	    tau = 1;
// 	    R = vec4(0,0,0,0) //{Consider alpha-channel}
// 	    for each (i in B) do
// 	        (C,n) =getCandidates(rayOrigin,rayDirection→,i)
// 	        candidates←sortCandidates(C,n)
// 	        for j←0<n do
// 	           λ = candidates[j]λin
// 	           λout = candidates[j]λout
// 	           while λ≤λout do// {Trace pseudo-spheroid}
// 	               rp = ro + λ⋅rd
// 	               dens = fbm(rayPos)
// 	               γ←e−∥rayPos−sphereCenter→∥/(r((1−κ)+2kρ))
// 	               if ρ<γ then
// 	                   R += lighting(dens ,rp,rd,sunDir, voxelGrid[candidates[j]i],sunColor)
// 	                   if τ<10−6 then
// 	                       break for (36:)
// 	                   end if
// 	               end if     {Level of Detail (LOD)}
// 	               λ+= A⋅e−(∥rayOrigin−cloudCenter→∥⋅δ)
// 	           end while
// 	        end for
// 	    end foreach
// 	    return R {Blend with the sky}
// }


// vec3 lighting(den, rp, rd, gridIndex, sunColor) {
//     delta =  e−ϕρ
//     //Calculate Voxel
//     voxelIndex = (rp−gridMin)/(gridMax−gridMin)
//     // Precomputed-light retrieval
//     pL = texture(gridIndex,voxelIndex)
//     //Absorbed-light calculation
//     aL= sunColor * pL
//     // Scattered-light calculation}
//     sL= aL * phase(g, rd, sunDir) * γ;
//     // Total-light calculation
//     tL = aL + sL;
//     // Calculate cloud color
//     color = (1−Δτ)*tL*τ
//     τ = τ*Δτ
//     return (color,1−τ)
// }

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

vec4 raymarch(vec3 ro, vec3 rd, float start, float maxd) {// eye = ray orientation

  vec4 sum = vec4(0.0);
	float t = 0.0;
  vec2 seed = vec2(1.302, 5.203);

  for(int i=0; i< MAX_MARCHING_STEPS; i++) {
    vec3  pos = ro + t*rd;
    if( pos.y<-3.0 || pos.y>2.0 || sum.a > 0.99 ) break;
    float den = fbm(pos);
    if( den> 0.01 ) {
      vec3 tmp_pos =pos+0.3*sunDir;
      float dif =  clamp((den - fbm(tmp_pos))/0.6, 0.0, 1.0 );
      sum = integrate( sum, dif, den, BLUE, t );
    }
    t += max(0.05,0.02*t);
  }

  return clamp( sum, 0.0, 1.0 );
	// float depth = start;
  //
	// for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
	// 	vec3 pnt = ro + depth * rd;
  //   vec2 res = map(pnt);
	// 	float dist = res.x;
	// 	if (dist < EPSILON) {return vec2(depth, res.y);}
	// 	depth += dist;
	// 	if (depth >= maxd) {break;}
	// }
  //
	// return vec2(maxd, 0.0);
}

vec3 render(vec3 ro, vec3 rd) {

  float sun = clamp( dot(sunDir,rd), 0.0, 1.0 );
	vec3 col = vec3(0.6,0.71,0.75) - rd.y*0.2*vec3(1.0,0.5,1.0) + 0.15*0.5;
	col += 0.2*vec3(1.0,.6,0.1)*pow( sun, 8.0 );

  // clouds
  vec4 res = raymarch(ro, rd, MIN_DIST, MAX_DIST);
  col = col*(1.0-res.w) + res.xyz;

  // sun glare
  col += 0.2 *vec3(1.0,0.4,0.2)*pow( sun, 3.0 );
  return col;

  // float ambiance = 0.8;
  //
  // if (res.x <= MAX_DIST - EPSILON) {
  //
  //   vec3 rp = ro + res.x * rd;
  //   col = WHITE;
  //   vec3 n = estimateNormal(rp);
  //
  //   // Lighting
  //   vec3 light_source = vec3(10.0, 20.0, 2.0);
  //   vec3 light_dir = normalize(light_source);
  //   vec3 light_color = K_d;
  //   vec3 pntToLight = vec3(light_source - rp);
  //   float dist = length(pntToLight);
  //   return obj_color * ambiance;
  // }
  //
  //  return BLUE;
}


// ------- Ray March Direction Calc ------- //
vec3 calculateRayMarchPoint()
{
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

void main()
{
	vec3 p = calculateRayMarchPoint();
  vec3 rd = normalize(p - u_Eye);
  vec3 ro = u_Eye;
  vec3 col = render(ro, rd);
  out_Col = vec4(col, 1.0);
}



//vec3 render(vec3 ro, vec3 rd) {

  /* vec2 res = raymarch(ro, rd, MIN_DIST, MAX_DIST);
  float ambiance = 0.8;

  if (res.x <= MAX_DIST - EPSILON) {

    vec3 rp = ro + res.x * rd;
    vec3 col = getObjectColor(res.y);
    // if (res.y == 0.1) {
    //   vec3 P2 = vec3(255./255., 178./255., 195./255.);
    //   float noise =(perlin(rp.yz, 30.0,10.0));
    //   col = col *noise  +  P2 * (1. - noise);
    // } else if (res.y == 0.6) {
    //   vec3 P2 = vec3(255./255., 178./255., 195./255.);
    //   float noise =(perlin(rp.yz, 3.0,1.0));
    //   col = col *noise  +  P2 * (1. - noise);
    // }
     //+ mix(0.f, square_wave(0.2, 0.2, 4.0),(rp.z)/50.f);
    vec3 n = estimateNormal(rp);
    // if (res.y == 0.1) {
    //   float noise =(perlin(rp.yz, 20.0,1.0));
    //   n =n*(1.-noise) ;
    // }

    // if (res.y == 1.0) {
    //   float radius = 13.0;
    //   float softness = 1.0;
    //   vec3 position = rp;
    //   position.z -= 4.5;
    //   float len = length(position);
    //   float vignette = smoothstep(radius, radius-softness, len);
    //   col = mix(col.rgb, col.rgb * vignette, 0.5);
    // }

    // Apply Phong Illumination
    vec3 p = calculateRayMarchPoint();
    vec3 rd = normalize(p - u_Eye);
    vec3 K_a = col;
    vec3 K_d = PINK;
    vec3 K_s = vec3(0.2, 0.2, 0.2);

    vec3 obj_color = phongIllumination(K_a, K_d, K_s, shininess, rp, ro);

    // // Lighting

    vec3 light_source = vec3(10.0, 20.0, 2.0);
    vec3 light_dir = normalize(light_source);
    vec3 light_color = K_d;
    // float light = dot(light_dir, n); // Lambertian Reflective Lighting
    vec3 pntToLight = vec3(light_source - rp);
    float dist = length(pntToLight);
    // float attenuation = 1.0 / dist;

    // vec3 spec_color = YELLOW;
    // float ssd = softshadow(rp,light_dir, 0.0, 20.0, 2.0);
    // float ao = ambientOcclusion(rp, light_dir);
    // vec3 specularReflection = vec3(0.0, 0.0, 0.0);
    // if (dot(n, light_dir) > 0.0)
    // {
    //   vec3 halfway = normalize(light_dir + rd);
    //   float w = pow(1.0 - max(0.0, dot(halfway, rd)), 5.0);
    //   specularReflection = attenuation * light_color * mix(spec_color, vec3(1.0), w)  * pow(max(0.0, dot(reflect(-light_dir, n), rd)),shininess);
    // }

    return obj_color * ambiance;//(obj_color + ssd + specularReflection + ao) * ambiance;
  }*/

//    return BLUE;
// }
