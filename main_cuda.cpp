#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#define EPSILON 0.00001f
#define EPSILON_RAY 0.0001f
#define IS_ZERO(val) val > -EPSILON && val < EPSILON
#define M_PI 3.14159265359f

using namespace std;

class Vector2Int {
public:
    int x;
    int y;

    Vector2Int() : x(0), y(0) {}
    Vector2Int(int x, int y) : x(x), y(y) {}
};

class Vector3 {
public:
    float x;
    float y;
    float z;

    __host__ __device__ Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vector3(const Vector3& other) : x(other.x), y(other.y), z(other.z) {}

    __host__ __device__ Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Vector3 operator*(const Vector3& other) const {
        return Vector3(x * other.x, y * other.y, z * other.z);
    }

    __host__ __device__ Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ Vector3 operator/(float scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    __host__ __device__ float dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ Vector3 cross(const Vector3& other) const {
        return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    __host__ __device__ float magnitude() const {
        return sqrt(x * x + y * y + z * z);
    }

    __host__ __device__ Vector3 normalize() const {
        float mag = magnitude();
        if (mag != 0.0f) {
            return Vector3(x / mag, y / mag, z / mag);
        }
        return Vector3();
    }

    __host__ __device__ Vector3 copy() const {
        return Vector3(x, y, z);
    }
};

class Color {
public:
    float red;
    float green;
    float blue;

    __host__ __device__ Color() : red(0.0f), green(0.0f), blue(0.0f) {}
    __host__ __device__ Color(float red, float green, float blue) : red(min(red, 1.0f)), green(min(green, 1.0f)), blue(min(blue, 1.0f)) {}

    __host__ __device__ Color operator+(const Color& other) const {
        return Color(red + other.red, green + other.green, blue + other.blue);
    }

    __host__ __device__ Color operator-(const Color& other) const {
        return Color(red - other.red, green - other.green, blue - other.blue);
    }

    __host__ __device__ Color operator*(const Color& other) const {
        return Color(red * other.red, green * other.green, blue * other.blue);
    }

    __host__ __device__ Color operator*(float scalar) const {
        return Color(red * scalar, green * scalar, blue * scalar);
    }

    __host__ __device__ Color operator/(float scalar) const {
        return Color(red / scalar, green / scalar, blue / scalar);
    }

    __host__ __device__ Color copy() const {
        return Color(red, green, blue);
    }
};

class ColorInt {
public:
    int red;
    int green;
    int blue;

    __host__ __device__ ColorInt() : red(0), green(0), blue(0) {}
    __host__ __device__ ColorInt(int red, int green, int blue) : red(red), green(green), blue(blue) {}
    __host__ __device__ ColorInt(Color other) : red((int)floor(other.red * 255.0f)), green((int)floor(other.green * 255.0f)), blue((int)floor(other.blue * 255.0f)) {}

    __host__ __device__ ColorInt operator+(const ColorInt& other) const {
        return ColorInt(red + other.red, green + other.green, blue + other.blue);
    }

    __host__ __device__ ColorInt operator-(const ColorInt& other) const {
        return ColorInt(red - other.red, green - other.green, blue - other.blue);
    }

    __host__ __device__ ColorInt operator*(float scalar) const {
        return ColorInt((int)floor(red * scalar), (int)floor(green * scalar), (int)floor(blue * scalar));
    }

    __host__ __device__ ColorInt operator/(float scalar) const {
        return ColorInt((int)floor(red / scalar), (int)floor(green / scalar), (int)floor(blue / scalar));
    }

    __host__ __device__ ColorInt copy() const {
        return ColorInt(red, green, blue);
    }
};

class Matrix3 {
public:
    float matrix[3][3];

    __host__ __device__ Matrix3(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
        matrix[0][0] = a11;
        matrix[0][1] = a12;
        matrix[0][2] = a13;
        matrix[1][0] = a21;
        matrix[1][1] = a22;
        matrix[1][2] = a23;
        matrix[2][0] = a31;
        matrix[2][1] = a32;
        matrix[2][2] = a33;
    }

    __host__ __device__ Vector3 operator*(const Vector3& v) const {
        float x = matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z;
        float y = matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z;
        float z = matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z;
        return Vector3(x, y, z);
    }

    __host__ __device__ Matrix3 operator*(const Matrix3& other) const {
        Matrix3 result(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    result.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return result;
    }
};

struct PointLight {
    Vector3 center;
    float intensity;
    Color color;
};

struct Triangle {
    Vector3 point_A;
    Vector3 point_B;
    Vector3 point_C;
    Color color;
    float reflectivity;
};

struct Ray {
    Vector3 origin;
    Vector3 direction;
    float t;
};

struct Plane {
    Vector3 center;
    Vector3 normal;
};

struct RayTraceRet {
    float dist;
    Vector3 hit_point;
    Vector3 normal;
    Color color;
    float reflectivity;
};

struct ReflectionBounceRet {
    Color shaded;
    Vector3 reflected;
    Vector3 normal;
    Vector3 hit_point;
    float reflectivity;

    ReflectionBounceRet() = default;
};

vector<string> split_string(const string& str, const string& delimiter) {
    vector<string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

vector<Triangle> obj_to_triangles(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "Failed to open file: " << path << endl;
        return {};
    }

    vector<Vector3> vertices;
    vector<Vector3> triangle_indices;
    vector<Triangle> triangles;

    string line;
    while (getline(file, line)) {
        vector<string> split_v = split_string(line, "v ");
        vector<string> split_f = split_string(line, "f ");
        if (split_v.size() > 1) {
            vector<string> params = split_string(split_v[1], " ");
            float x = stod(params[0]);
            float y = stod(params[1]);
            float z = stod(params[2]);
            Vector3 v = Vector3(x, y, z);
            vertices.push_back(v);
        }
        if (split_f.size() > 1) {
            vector<string> params = split_string(split_f[1], " ");
            int a = stoi(split_string(params[0], "/")[0]) - 1;
            int b = stoi(split_string(params[1], "/")[0]) - 1;
            int c = stoi(split_string(params[2], "/")[0]) - 1;
            Vector3 i = Vector3(a, b, c);
            triangle_indices.push_back(i);
        }
    }

    for (Vector3 index : triangle_indices) {
        int a = (int)index.x;
        int b = (int)index.y;
        int c = (int)index.z;
        triangles.push_back({
            vertices[a],
            vertices[b],
            vertices[c],
            Color(1.0f, 1.0f, 1.0f),
            0.5f
            });
    }

    return triangles;
}

void show_triangles(const vector<Triangle>& triangles) {
    cout << triangles.size() << endl;
    for (const Triangle& triangle : triangles) {
        cout << "Triangle:" << endl;
        cout << "Point A: (" << triangle.point_A.x << ", " << triangle.point_A.y << ", " << triangle.point_A.z << ")" << endl;
        cout << "Point B: (" << triangle.point_B.x << ", " << triangle.point_B.y << ", " << triangle.point_B.z << ")" << endl;
        cout << "Point C: (" << triangle.point_C.x << ", " << triangle.point_C.y << ", " << triangle.point_C.z << ")" << endl;
        cout << "Color: (" << triangle.color.red << ", " << triangle.color.green << ", " << triangle.color.blue << ")" << endl;
        cout << "Reflectivity: " << triangle.reflectivity << endl;
        cout << endl;
    }
}

__device__ Vector3 rotateVector(const Vector3& v, const float yaw, const float pitch) {
    Matrix3 yawMatrix(
        cos(yaw), 0.0f, -sin(yaw),
        0.0f, 1.0f, 0.0f,
        sin(yaw), 0.0f, cos(yaw)
    );

    Matrix3 pitchMatrix(
        1.0f, 0.0f, 0.0f,
        0.0f, cos(pitch), sin(pitch),
        0.0f, -sin(pitch), cos(pitch)
    );

    Matrix3 resultMatrix = yawMatrix * pitchMatrix;
    const Vector3 rotatedVector = resultMatrix * v;

    return rotatedVector;
}

__device__ float get_yaw(const Vector3& u, Vector3* v) {
    Vector3 u_norm = u.normalize();
    Vector3 v_norm = v->normalize();

    return atan2(v_norm.z, v_norm.x) - atan2(u_norm.z, u_norm.x);
}

__device__ float get_pitch(const Vector3& u, Vector3* v) {
    Vector3 u_norm = u.normalize();
    Vector3 v_norm = v->normalize();

    return asin(v_norm.y) - asin(u_norm.y);
}

__device__ float mix(const float factor, const float A, const float B) {
    return A + (B - A) * factor;
}

__device__ Color mix(const float factor, const Color& A, const Color& B) {
    return A + (B - A) * factor;
}

__device__ ColorInt a(int x, int y, int width, int height) {
    ColorInt ret = ColorInt();
    ret.red = floor(((float)(x + y) / (float)width) * 255.0f);
    ret.green = floor(((float)(x + y) / (float)width) * 255.0f);
    ret.blue = floor(((float)(x + y) / (float)width) * 255.0f);
    return ret;
}

__device__ RayTraceRet ray_trace(const Triangle& triangle, Ray ray) {
    RayTraceRet false_ret = { 1000.0f, Vector3(), Vector3(), Color(), 0.0f };

    Vector3 tri_edge_1 = triangle.point_B - triangle.point_A;
    Vector3 tri_edge_2 = triangle.point_C - triangle.point_A;
    Vector3 tri_flat_normal = tri_edge_1.cross(tri_edge_2).normalize();

    Plane tri_plane = {
        triangle.point_A,
        tri_flat_normal
    };

    float n_dot_d = tri_plane.normal.dot(ray.direction);

    if (IS_ZERO(n_dot_d))
        return false_ret;

    float n_dot_ps = tri_plane.normal.dot(tri_plane.center - ray.origin);
    ray.t = n_dot_ps / n_dot_d;

    if (ray.t < 0.0)
        return false_ret;

    Vector3 plane_point = ray.origin + ray.direction * ray.t;

    Vector3 a_to_b_edge = triangle.point_B - triangle.point_A;
    Vector3 b_to_c_edge = triangle.point_C - triangle.point_B;
    Vector3 c_to_a_edge = triangle.point_A - triangle.point_C;

    Vector3 a_to_point = plane_point - triangle.point_A;
    Vector3 b_to_point = plane_point - triangle.point_B;
    Vector3 c_to_point = plane_point - triangle.point_C;

    Vector3 a_test_vec = a_to_b_edge.cross(a_to_point);
    Vector3 b_test_vec = b_to_c_edge.cross(b_to_point);
    Vector3 c_test_vec = c_to_a_edge.cross(c_to_point);

    bool a_test_vec_match = a_test_vec.dot(tri_flat_normal) > -EPSILON;
    bool b_test_vec_match = b_test_vec.dot(tri_flat_normal) > -EPSILON;
    bool c_test_vec_match = c_test_vec.dot(tri_flat_normal) > -EPSILON;

    if (a_test_vec_match && b_test_vec_match && c_test_vec_match)
        return { ray.t, plane_point, tri_flat_normal, triangle.color, triangle.reflectivity };

    return false_ret;
}

__device__ RayTraceRet ray_trace_all(Ray ray, Triangle* triangles, size_t num_triangles) {
    RayTraceRet closest_ret = { 1000.0f, Vector3(), Vector3(), Color(), 0.0f };
    for (size_t i = 0; i < num_triangles; i++) {
        Triangle triangle = triangles[i];
        RayTraceRet ret = ray_trace(triangle, ray);
        if (ret.dist < closest_ret.dist)
            closest_ret = ret;
    }
    float frenel = pow((closest_ret.normal.dot(ray.direction) + 1), 1.7);
    closest_ret.reflectivity = mix(frenel, 1 - closest_ret.reflectivity, 1.0);
    return closest_ret;
}

__device__ Color point_light(const Vector3& normal, const Vector3& ray_origin, const Vector3& light_pos, const float intensity, const Color& color, Triangle* triangles, size_t num_triangles) {
    if (IS_ZERO(normal.magnitude())) {
        return Color();
    }
    Vector3 dir = light_pos - ray_origin;
    float dir_len = dir.magnitude();
    Vector3 light_vector = dir.normalize();
    float dotted = max(normal.dot(light_vector), 0.0f);
    float shaded_hit_point = dotted / (dir_len * dir_len);
    RayTraceRet ret_shadow_ray = ray_trace_all({ ray_origin + light_vector * EPSILON_RAY, light_vector }, triangles, num_triangles);
    float mult = ret_shadow_ray.dist > dir_len ? 1.0f : 0.0f;
    Color ret_shading = color * (shaded_hit_point * mult * intensity);
    return ret_shading;
}

__device__ Color shade_hit_point(const Vector3& normal, const Vector3& hit_point, const Color& color, Triangle* triangles, size_t num_triangles, PointLight* point_lights, size_t num_point_lights) {
    Color shading = Color();
    for (size_t i = 0; i < num_point_lights; i++) {
        PointLight light = point_lights[i];
        shading = shading + point_light(normal, hit_point, light.center, light.intensity, light.color, triangles, num_triangles);
    }
    return shading * color;
}

__device__ Vector3 reflect(const Vector3& vector, const Vector3& normal) {
    return vector - normal * (vector.dot(normal) * 2.0f);
}

__device__ ReflectionBounceRet get_reflection_bounce(const Ray& ray, const Vector3& ray_normal, Triangle* triangles, size_t num_triangles, PointLight* point_lights, size_t num_point_lights) {
    Vector3 reflected = reflect(ray.direction, ray_normal);
    RayTraceRet ray_ret = ray_trace_all({ ray.origin + reflected * EPSILON_RAY * 2.0f, reflected }, triangles, num_triangles);
    Color shaded_ret = shade_hit_point(ray_ret.normal, ray_ret.hit_point, ray_ret.color, triangles, num_triangles, point_lights, num_point_lights);
    return { shaded_ret, reflected, ray_ret.normal, ray_ret.hit_point, ray_ret.reflectivity };
}

__global__ void trace_pixel(Vector2Int* positions, ColorInt* colors, Triangle* triangles, PointLight* point_lights, Vector3* camera_pos, Vector3* camera_dir, int width, int height, int size, size_t num_triangles, size_t num_point_lights) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        float yaw = get_yaw(Vector3(0.0f, 0.0f, -1.0f), camera_dir);
        float pitch = get_pitch(Vector3(0.0f, 0.0f, -1.0f), camera_dir);

        int x = positions[tid].x;
        int y = positions[tid].y;

        float n_x = (float)x / (float)width - 0.5f;
        float n_y = (float)y / (float)height - 0.5f;

        Vector3 camera_pos_n = Vector3(camera_pos->x, camera_pos->y, camera_pos->z);
        Vector3 dir = rotateVector(Vector3(n_x, n_y, -1).normalize(), yaw, pitch);

        Ray ray = { camera_pos_n, dir, -1.0f };

        RayTraceRet ret = ray_trace_all(ray, triangles, num_triangles);

        ret.normal = ret.normal.normalize();
        ret.normal.x = max(min(ret.normal.x, 1.0f), -1.0f);
        ret.normal.y = max(min(ret.normal.y, 1.0f), -1.0f);
        ret.normal.z = max(min(ret.normal.z, 1.0f), -1.0f);

        Color shaded = shade_hit_point(ret.normal, ret.hit_point, ret.color, triangles, num_triangles, point_lights, num_point_lights);

        ReflectionBounceRet reflected_point_1 = get_reflection_bounce({ ret.hit_point + dir * EPSILON_RAY, dir }, ret.normal, triangles, num_triangles, point_lights, num_point_lights);
        ReflectionBounceRet reflected_point_2 = get_reflection_bounce({ reflected_point_1.hit_point + reflected_point_1.reflected * EPSILON_RAY, reflected_point_1.reflected }, reflected_point_1.normal, triangles, num_triangles, point_lights, num_point_lights);

        Color reflective = mix(reflected_point_1.reflectivity, reflected_point_1.shaded, reflected_point_2.shaded);
        Color shaded_reflective = mix(1 - ret.reflectivity, shaded, reflective);

        ColorInt normal_color = ColorInt(floor(abs(ret.normal.x) * 255.0f), floor(abs(ret.normal.y) * 255.0f), floor(abs(ret.normal.z) * 255.0f));
        ColorInt shaded_color = ColorInt(floor(shaded.red * 255.0f), floor(shaded.green * 255.0f), floor(shaded.blue * 255.0f));
        ColorInt reflectivity_color = ColorInt(floor(ret.reflectivity * 255.0f), floor(ret.reflectivity * 255.0f), floor(ret.reflectivity * 255.0f));
        ColorInt shaded_reflective_color = ColorInt(floor(shaded_reflective.red * 255.0f), floor(shaded_reflective.green * 255.0f), floor(shaded_reflective.blue * 255.0f));

        colors[tid] = shaded_reflective_color;
    }
}

class RayTracer {
public:
    string name;
    int width;
    int height;
    unsigned char* imageData;
    Vector3 camera_pos;
    Vector3 camera_dir;
    vector<Triangle> triangles_v;
    vector<PointLight> point_lights_v;
    bool show_percent;

    RayTracer() : name("out"), width(100), height(100), camera_pos(Vector3(0.0f, 0.0f, -4.0f)), camera_dir(Vector3(0.0f, 0.0f, 1.0f)), triangles_v(obj_to_triangles("C:\\Users\\anton\\OneDrive\\Desktop\\obj.obj")), show_percent(false) {}

    RayTracer(const string& name, int width, int height, const Vector3& camera_pos, const Vector3& camera_dir, const vector<Triangle>& triangles, const vector<PointLight>& point_lights, bool show_percent) : name(name), width(width), height(height), camera_pos(camera_pos), camera_dir(camera_dir), triangles_v(triangles), point_lights_v(point_lights), show_percent(show_percent) {}

    void save_bitmap(const string& filename)
    {
        // Define the bitmap file header
        unsigned char bitmapFileHeader[14] = {
                'B', 'M',                     // Signature
                0, 0, 0, 0,           // File size (to be filled later)
                0, 0, 0, 0,           // Reserved
                54, 0, 0, 0        // Pixel data offset
        };

        // Define the bitmap info header
        unsigned char bitmapInfoHeader[40] = {
                40, 0, 0, 0,            // Info header size
                0, 0, 0, 0,             // Image width (to be filled later)
                0, 0, 0, 0,           // Image height (to be filled later)
                1, 0,                         // Number of color planes
                24, 0,                        // Bits per pixel (24 bits for RGB)
                0, 0, 0, 0,          // Compression method (none)
                0, 0, 0, 0,          // Image size (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Horizontal resolution (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Vertical resolution (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Number of colors in the palette (not used for 24-bit images)
                0, 0, 0, 0           // Number of important colors (not used for 24-bit images)
        };

        // Calculate the padding bytes
        int paddingSize = (4 - (width * 3) % 4) % 4;

        // Calculate the file size
        int fileSize = 54 + (width * height * 3) + (paddingSize * height);

        // Fill in the file size in the bitmap file header
        bitmapFileHeader[2] = (unsigned char)(fileSize);
        bitmapFileHeader[3] = (unsigned char)(fileSize >> 8);
        bitmapFileHeader[4] = (unsigned char)(fileSize >> 16);
        bitmapFileHeader[5] = (unsigned char)(fileSize >> 24);

        // Fill in the image width in the bitmap info header
        bitmapInfoHeader[4] = (unsigned char)(width);
        bitmapInfoHeader[5] = (unsigned char)(width >> 8);
        bitmapInfoHeader[6] = (unsigned char)(width >> 16);
        bitmapInfoHeader[7] = (unsigned char)(width >> 24);

        // Fill in the image height in the bitmap info header
        bitmapInfoHeader[8] = (unsigned char)(height);
        bitmapInfoHeader[9] = (unsigned char)(height >> 8);
        bitmapInfoHeader[10] = (unsigned char)(height >> 16);
        bitmapInfoHeader[11] = (unsigned char)(height >> 24);

        // Open the output file
        ofstream file(filename, ios::binary);

        // Write the bitmap headers
        file.write(reinterpret_cast<const char*>(bitmapFileHeader), sizeof(bitmapFileHeader));
        file.write(reinterpret_cast<const char*>(bitmapInfoHeader), sizeof(bitmapInfoHeader));

        // Write the pixel data (BGR format) row by row
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                // Calculate the pixel position
                int position = (x + y * width) * 3;

                // Write the pixel data (BGR order)
                file.write(reinterpret_cast<const char*>(&imageData[position + 2]), 1); // Blue
                file.write(reinterpret_cast<const char*>(&imageData[position + 1]), 1); // Green
                file.write(reinterpret_cast<const char*>(&imageData[position]), 1);     // Red
            }

            // Write the padding bytes
            for (int i = 0; i < paddingSize; i++)
            {
                file.write("\0", 1);
            }
        }

        // Close the file
        file.close();
    }

    void put_pixel(const int x, const int y, const int r, const int g, const int b) {
        int position = (x + y * width) * 3;
        imageData[position] = r;
        imageData[position + 1] = g;
        imageData[position + 2] = b;
    }

    void generate() {
        imageData = new unsigned char[width * height * 3];

        const int size = width * height;

        // DEFINE PARAMETERS

        Vector2Int* positions = new Vector2Int[size];
        ColorInt* colors = new ColorInt[size];

        size_t num_triangles = triangles_v.size();
        Triangle* triangles = new Triangle[num_triangles];
        for (size_t i = 0; i < num_triangles; i++) {
            triangles[i] = triangles_v[i];
        }

        size_t num_point_lights = point_lights_v.size();
        PointLight* point_lights = new PointLight[num_point_lights];
        for (size_t i = 0; i < num_point_lights; i++) {
            point_lights[i] = point_lights_v[i];
        }

        // DEFINE SIZES

        const long vector2int_size = size * sizeof(Vector2Int);
        const long colorint_size = size * sizeof(ColorInt);
        const long triangles_size = num_triangles * sizeof(Triangle);
        const long point_lights_size = num_point_lights * sizeof(PointLight);
        const long vector3_size = sizeof(Vector3);

        // UPDATE I/O PARAMERTERS

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            positions[i] = Vector2Int(x, y);
            colors[i] = ColorInt();
        }

        // DEFINE D_PARAMETERS

        Vector2Int* d_positions;
        ColorInt* d_colors;
        Triangle* d_triangles;
        PointLight* d_point_lights;
        Vector3* d_camera_pos;
        Vector3* d_camera_dir;

        cudaMalloc((void**)&d_positions, vector2int_size);
        cudaMalloc((void**)&d_colors, colorint_size);
        cudaMalloc((void**)&d_triangles, triangles_size);
        cudaMalloc((void**)&d_point_lights, point_lights_size);
        cudaMalloc((void**)&d_camera_pos, vector3_size);
        cudaMalloc((void**)&d_camera_dir, vector3_size);

        // MEMORY COPY PARAMETERS

        cudaMemcpy(d_positions, positions, vector2int_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_colors, colors, colorint_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_triangles, triangles, triangles_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_point_lights, point_lights, point_lights_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_camera_pos, &camera_pos, vector3_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_camera_dir, &camera_dir, vector3_size, cudaMemcpyHostToDevice);

        // RUN

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        trace_pixel << < blocksPerGrid, threadsPerBlock >> > (d_positions, d_colors, d_triangles, d_point_lights, d_camera_pos, d_camera_dir, width, height, size, num_triangles, num_point_lights);

        cudaMemcpy(colors, d_colors, colorint_size, cudaMemcpyDeviceToHost);

        // PROCESS OUTPUT

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            ColorInt color = colors[i];
            put_pixel(x, height - y - 1, color.red, color.green, color.blue);
        }

        save_bitmap("./" + name + ".bmp");

        // FREE MEMORY

        cudaFree(d_positions);
        cudaFree(d_colors);
        cudaFree(d_triangles);
        cudaFree(d_point_lights);
        cudaFree(d_camera_pos);
        cudaFree(d_camera_dir);

        delete[] positions;
        delete[] colors;
        delete[] triangles;
        delete[] point_lights;

        delete[] imageData;
    }
};

int main() {
    const float light_radius = 3.0f;
    const float camera_radius = 5.0f;
    const float light_intensity = 5.0f;

    vector<PointLight> point_lights = {};
    const vector<Triangle> triangles = obj_to_triangles("C:\\Users\\anton\\OneDrive\\Desktop\\obj.obj");
    const vector<Color> cols = {
        Color(1.0f, 0.0f, 0.0f),
        Color(0.0f, 1.0f, 0.0f),
        Color(0.0f, 0.0f, 1.0f)
    };
    for (int i = 0; i < 3; i++) {
        point_lights.push_back({ Vector3(sin((float)i / 3 * M_PI * 2) * light_radius, 3, cos((float)i / 3 * M_PI * 2) * light_radius), light_intensity, cols[i] });
    }

    chrono::system_clock::time_point start_time = chrono::system_clock::now();;
    int count = 400;
    for (int i = 0; i < count; i++) {
        chrono::system_clock::time_point start_frame = chrono::system_clock::now();;

        string name = "out_" + to_string(i);

        int width = 1200;
        int height = 1200;

        Vector3 camera_pos = Vector3(sin((float)i / count * M_PI * 2.0f) * camera_radius, 0.0f, cos((float)i / count * M_PI * 2.0f) * camera_radius);
        Vector3 camera_dir = (Vector3() - camera_pos).normalize();

        bool show_percent = false;

        RayTracer tracer = RayTracer(name, width, height, camera_pos, camera_dir, triangles, point_lights, show_percent);
        tracer.generate();

        chrono::time_point<chrono::system_clock> end_frame = chrono::system_clock::now();
        chrono::duration<float> duration_frame = end_frame - start_frame;

        cout << i + 1 << "/" << count << " (" << floor((float)(i + 1.0f) / (float)count * 1000.0f) / 10.0f << "%)" << " " << duration_frame.count() << " seconds" << endl;
    }
    chrono::time_point<chrono::system_clock> end_time = chrono::system_clock::now();
    chrono::duration<float> duration = end_time - start_time;
    cout << "Total time taken: " << duration.count() << " seconds" << endl;

    return 0;
}
