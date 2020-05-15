#version 450

#define PI 3.1415927

#define SHAPE_MAX_COUNT 512

#define MAT_DIFFUSE 0
#define MAT_REFLECTIVE 1
#define MAT_EMISSIVE 2

struct Intersection {
    float tMin;
    float tMax;
    float mat;
    vec3 color;
    vec2 n;
};

struct Ray {
    vec2 origin;
    vec2 dir;
};

struct Shape {
    vec2 a;
    vec2 b;
    vec3 color;
    uint mat;
};

layout(set=0, binding=0) uniform Uniforms {
    Shape shapes[SHAPE_MAX_COUNT];
    uint shapeCount;
    uint sampleCount;
    float time;
};

layout(location = 0) out vec4 o_Target;

float rand(vec2 scale, float seed) {
    return fract(sin(dot(gl_FragCoord.xy + seed, scale)) * 43758.5453 + seed);
}

vec2 sampleCircle(float seed) {
    float xi = rand(vec2(6.7264, 10.873), seed);
    float theta = 2.0 * PI * xi;
    return vec2(cos(theta), sin(theta));
}

void fetchShape(uint i, out Shape s) {
    s = shapes[i];
}

void intersectLine(Ray r, vec2 a, vec2 b, float mat, vec3 color, inout Intersection isect) {
    vec2 sT = b - a;
    vec2 sN = vec2(-sT.y, sT.x);
    float t = dot(sN, a - r.origin) / dot(sN, r.dir);
    float u = dot(sT, r.origin + r.dir*t - a);
    if (t < isect.tMin || t >= isect.tMax || u < 0.0 || u > dot(sT, sT))
        return;

    isect.tMax = t;
    isect.mat = mat;
    isect.color = color;
    isect.n = normalize(sN);
}

void intersect(Ray r, inout Intersection isect) {
    Shape s;
    for (uint i = 0; i < shapeCount; i++) {
        fetchShape(i, s);
        intersectLine(r, s.a, s.b, s.mat, s.color, isect);
    }
}

#define BOUNCE_COUNT 4
#define T_MIN 1e-4
#define T_MAX 1e30

#define DIST_COEF 0.35
#define LIGHT_COEF 2.0
#define REFL_COEF 0.7

vec3 traceRay(Ray r) {
    vec3 color = vec3(0.0);
    vec3 mask = vec3(1.0);

    for (uint ibounce = 0; ibounce < BOUNCE_COUNT; ibounce++) {
        Intersection isect;
        isect.tMin = T_MIN;
        isect.tMax = T_MAX;
        intersect(r, isect);

        if (isect.tMax == T_MAX) {
            // no hit
            break;
        }
        if (isect.mat == MAT_EMISSIVE) {
            float cosTheta = abs(dot(normalize(r.dir), isect.n));
            float d = 1.0 - isect.tMax;
            color += mask * isect.color * cosTheta * d * LIGHT_COEF;
            break;
        }
        else if (isect.mat == MAT_REFLECTIVE) {
            vec2 hit = r.origin + r.dir * isect.tMax;
            r.origin = hit;
            r.dir = reflect(r.dir, isect.n);
            mask *= isect.color;
        }
        else if (isect.mat == MAT_DIFFUSE) {
            vec2 hit = r.origin + r.dir * isect.tMax;
            r.origin = hit;
            r.dir = (isect.n + sampleCircle(time + sampleCount + ibounce)) * 200;
            mask *= isect.color;
        }
    }

    return color;
}

vec3 computeColor(vec2 coord) {
    Ray r;
    r.origin = coord;
    r.dir = sampleCircle(time + sampleCount) * 500;
    return traceRay(r);
}

void main() {
    vec3 color = computeColor(gl_FragCoord.xy);
    o_Target = vec4(color, 1.0);
}
