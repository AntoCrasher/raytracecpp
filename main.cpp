#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <thread>

using namespace std;

class Vector2 {
public:
    double x;
    double y;

    Vector2() : x(0.0), y(0.0) {}
    Vector2(double x, double y) : x(x), y(y) {}

    Vector2(const Vector2& other) : x(other.x), y(other.y) {}

    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }

    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }

    Vector2 operator*(double scalar) const {
        return Vector2(x * scalar, y * scalar);
    }

    Vector2 operator/(double scalar) const {
        return Vector2(x / scalar, y / scalar);
    }

    double dot(const Vector2& other) const {
        return x * other.x + y * other.y;
    }

    double magnitude() const {
        return sqrt(x * x + y * y);
    }

    Vector2 normalize() const {
        double mag = magnitude();
        if (mag != 0.0f) {
            return Vector2(x / mag, y / mag);
        }
        return Vector2(0.0, 0.0);
    }

    string toString() const {
        return "(" + to_string(x) + ", " + to_string(y) + ")";
    }
};

class Vector3 {
public:
    double x;
    double y;
    double z;

    Vector3() : x(0.0), y(0.0), z(0.0) {}
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3(const Vector3& other) : x(other.x), y(other.y), z(other.z) {}

    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    Vector3 operator*(double scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    Vector3 operator/(double scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    double dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3 cross(const Vector3& other) const {
        return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    double magnitude() const {
        return sqrt(x * x + y * y + z * z);
    }

    Vector3 normalize() const {
        double mag = magnitude();
        if (mag != 0.0f) {
            return Vector3(x / mag, y / mag, z / mag);
        }
        return Vector3(0.0, 0.0, 0.0);
    }

    Vector3 copy() const {
        return Vector3(x, y, z);
    }

    string toString() const {
        return "(" + to_string(x) + ", " + to_string(y) + ", " + to_string(z) + ")";
    }
};

class Color {
public:
    double red;
    double green;
    double blue;

    Color() : red(0.0), green(0.0), blue(0.0) {}
    Color(double red, double green, double blue) : red(min(red, 1.0)), green(min(green, 1.0)), blue(min(blue, 1.0)) {}

    Color operator+(const Color& other) const {
        return Color(red + other.red, green + other.green, blue + other.blue);
    }

    Color operator-(const Color& other) const {
        return Color(red - other.red, green - other.green, blue - other.blue);
    }

    Color operator*(const Color& other) const {
        return Color(red * other.red, green * other.green, blue * other.blue);
    }

    Color operator*(double scalar) const {
        return Color(red * scalar, green * scalar, blue * scalar);
    }

    Color operator/(double scalar) const {
        return Color(red / scalar, green / scalar, blue / scalar);
    }

    Color copy() const {
        return Color(red, green, blue);
    }

    string toString() const {
        return "(" + to_string(red) + ", " + to_string(green) + ", " + to_string(blue) + ")";
    }
};

class ColorInt {
public:
    int red;
    int green;
    int blue;

    ColorInt() : red(0), green(0), blue(0) {}
    ColorInt(int red, int green, int blue) : red(red), green(green), blue(blue) {}
    ColorInt(Color other) : red(floor(other.red * 255.0)), green(floor(other.green * 255.0)), blue(floor(other.blue * 255.0)) {}

    ColorInt operator+(const ColorInt& other) const {
        return ColorInt(red + other.red, green + other.green, blue + other.blue);
    }

    ColorInt operator-(const ColorInt& other) const {
        return ColorInt(red - other.red, green - other.green, blue - other.blue);
    }

    ColorInt operator*(double scalar) const {
        return ColorInt(red * scalar, green * scalar, blue * scalar);
    }

    ColorInt operator/(double scalar) const {
        return ColorInt(red / scalar, green / scalar, blue / scalar);
    }

    ColorInt copy() const {
        return ColorInt(red, green, blue);
    }

    string toString() const {
        return "(" + to_string(red) + ", " + to_string(green) + ", " + to_string(blue) + ")";
    }
};

struct Sphere {
    Vector3 center;
    double radius;
    Color color;
    double reflectivity;
};

struct PointLight {
    Vector3 center;
    double intensity;
    Color color;
};

struct ShootRayRet {
    double dist;
    Vector3 normal;
    Vector3 hit_point;
    Color color;
    double reflectivity;
};

struct AddSphereRet {
    double dist;
    Vector3 normal;
    Color color;
    double reflectivity;
};

struct RayCastRet {
    Color final;
};

struct ReflectionBounceRet {
    Color shaded;
    Vector3 reflected;
    Vector3 normal;
    Vector3 hit_point;
    double reflectivity;
};

class RayTrace {
public:
    int size;
    int width;
    int height;
    Vector3 camera_pos;
    vector<Sphere> spheres;
    vector<PointLight> point_lights;
    double rotation;
    unsigned char* imageData;
    bool show_percent;
    bool multi_thread;
    int max_threads;
    int image_index;

    RayTrace() : size(1000),
                 camera_pos(Vector3(-0.02, 0.09, -2.091)),
                 spheres({
                     { Vector3(-0.58, -0.28, -0.94), 0.31, Color(1.0, 0.112, 0.115), 0.1 },
                     { Vector3(0.8, -0.35, -0.35), 0.59, Color(0.159475, 0.435126, 1.0), 0.1 },
                     { Vector3(-0.19, -0.17, 0.03), 0.94, Color(0.584398, 1.0, 0.121992), 0.2 },
                     { Vector3(0.41, -3000.4, 0.03), 3000.0, Color(1, 1, 1), 0.4 }
                 }),
                 point_lights({
                     { Vector3(-2.6, 2.0, -2.8), 9.2, Color(1, 1, 1) },
                     { Vector3(1.2, 2.4, -2.6), 4.8, Color(1, 1, 1) }
                 }),
                 rotation(0.0),
                 show_percent(true),
                 multi_thread(true),
                 max_threads(200),
                 image_index(0)
    {Init();}

    RayTrace(int size, Vector3 camera_pos, vector<Sphere> spheres, vector<PointLight> point_lights, double rotation, bool show_percent, bool multi_thread, int max_threads, int image_index) : size(size),
                                                                                                                                                             camera_pos(camera_pos),
                                                                                                                                                             spheres(spheres),
                                                                                                                                                             point_lights(point_lights),
                                                                                                                                                             rotation(rotation),
                                                                                                                                                             show_percent(show_percent),
                                                                                                                                                             multi_thread(multi_thread),
                                                                                                                                                             max_threads(max_threads),
                                                                                                                                                             image_index(image_index)
    {Init();}

    void save_bitmap(string filename)
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

    void put_pixel(int x, int y, int r, int g, int b) {
        int position = (x + y * width) * 3;
        imageData[position] = r;
        imageData[position + 1] = g;
        imageData[position + 2] = b;
    }

    Vector3 generate_ray(Vector2 pos) {
        Vector3 pixel_pos = Vector3(pos.x / (0.5 * size), pos.y / (0.5 * size), 1);
        return pixel_pos.normalize();
    }

    AddSphereRet add_sphere(double dist, Vector3 normal, Color color, double reflectivity, Vector3 ray_origin, Vector3 ray_dir, double radius, Vector3 pos, Color current_color, double current_reflectivity) {
        pos = rotate_y(pos, rotation);
        Vector3 new_sphere_pos = pos - ray_origin;
        double l = max(new_sphere_pos.dot(ray_dir), 0.0);
        Vector3 b_point = ray_origin + ray_dir * l;
        double b = (b_point - pos).magnitude();

        double ret_dist = dist;
        Vector3 ret_normal = normal.copy();
        Color ret_color = color.copy();
        double ret_reflectivity = reflectivity;

        double mult = ((pos - ray_origin).magnitude() > radius) ? 1.0 : 0.0;
        if (b < radius) {
            double a = sqrt(radius * radius - b * b);
            double d = l - a;
            ret_dist = 1000 + mult * (d - 1000);

            Vector3 hit_point = ray_origin + ray_dir * ret_dist;
            Vector3 new_normal = (hit_point - pos).normalize();

            double mult2 = ret_dist < dist ? 1.0 : 0.0;

            ret_dist = min(dist, ret_dist);
            ret_normal = normal + (new_normal - normal) * mult2;
            ret_color = color + (current_color - color) * mult2;
            ret_reflectivity = reflectivity + (current_reflectivity - reflectivity) * mult2;
        }
        return { ret_dist, ret_normal, ret_color, ret_reflectivity };
    }

    ShootRayRet shoot_ray(Vector3 ray_origin, Vector3 ray_dir) {
        AddSphereRet last_ret = { 1000, Vector3(0.0, 0.0, 0.0), Color(0.0, 0.0, 0.0), 0.0 };
        for (Sphere sphere : spheres) {
            last_ret = add_sphere(last_ret.dist, last_ret.normal, last_ret.color, last_ret.reflectivity, ray_origin, ray_dir, sphere.radius, sphere.center, sphere.color, sphere.reflectivity);
        }
        double frenel = pow((last_ret.normal.dot(ray_dir) + 1), 1.3);
        return { last_ret.dist, last_ret.normal, ray_origin + ray_dir * last_ret.dist, last_ret.color, frenel };
    }

    Color point_light(Vector3 normal, Vector3 ray_origin, Vector3 light_pos, double intensity, Color color) {
        light_pos = rotate_y(light_pos, rotation);
        Vector3 dir = light_pos - ray_origin;
        double dir_len = dir.magnitude();
        Vector3 light_vector = (light_pos - ray_origin).normalize();
        double dotted = max(normal.dot(light_vector), 0.0);
        double shaded_hit_point = dotted / (dir_len * dir_len);
        ShootRayRet ret_shadow_ray = shoot_ray(ray_origin, light_vector);
        double mult = ret_shadow_ray.dist > dir_len ? 1.0 : 0.0;
        Color ret_shading = color * (shaded_hit_point * mult * intensity);
        return ret_shading;
    }

    Color shade_hit_point(Vector3 normal, Vector3 hit_point, Color color) {
        Color shading = Color(0.0, 0.0, 0.0);
        for (PointLight light : point_lights) {
            shading = shading + point_light(normal, hit_point, light.center, light.intensity, light.color);
        }
        return shading * color;
    }

    Vector3 reflect(Vector3 vector, Vector3 normal) {
        return vector - normal * (vector.dot(normal) * 2);
    }

    ReflectionBounceRet get_reflection_bounce(Vector3 ray_origin, Vector3 ray_direction, Vector3 ray_normal) {
        Vector3 reflected = reflect(ray_direction, ray_normal);
        ShootRayRet ray_ret = shoot_ray(ray_origin, reflected);
        Color shaded_ret = shade_hit_point(ray_ret.normal, ray_ret.hit_point, ray_ret.color);
        return { shaded_ret, reflected, ray_ret.normal, ray_ret.hit_point, ray_ret.reflectivity };
    }

    Color mix(double factor, Color A, Color B) {
        return A + (B - A) * factor;
    }

    RayCastRet ray_cast(Vector2 pos) {
        Vector3 ray_dir = generate_ray(pos);
        ShootRayRet ray_ret = shoot_ray(camera_pos, ray_dir);
        Color shaded = shade_hit_point(ray_ret.normal, ray_ret.hit_point, ray_ret.color);
        ReflectionBounceRet reflected_point_1 = get_reflection_bounce(ray_ret.hit_point, ray_dir, ray_ret.normal);
        ReflectionBounceRet reflected_point_2 = get_reflection_bounce(reflected_point_1.hit_point, reflected_point_1.reflected, reflected_point_1.normal);
        Color mixed_1 = mix(reflected_point_1.reflectivity, reflected_point_1.shaded, reflected_point_2.shaded);
        Color final = mix(ray_ret.reflectivity, shaded, mixed_1);
        return { final };
    }

    ColorInt calc_pixel(int x, int y) {
        const Vector2 pos(x - size * 0.5, y - size * 0.5);
        RayCastRet ret = ray_cast(pos);
        ColorInt col = ColorInt(ret.final);
        return col;
    }

    Vector3 rotate_y(Vector3 vector, double angle) {
        angle = -angle * 3.14159/180.0;

        double cos_angle = cos(angle);
        double sin_angle = sin(angle);

        Vector3 rotatedVector;

        rotatedVector.x = vector.x * cos_angle + vector.z * sin_angle;
        rotatedVector.y = vector.y;
        rotatedVector.z = -vector.x * sin_angle + vector.z * cos_angle;

        return rotatedVector;
    }

    void Init() {
        width = size;
        height = size;
        imageData = new unsigned char[width * height * 3];
    }

    void Generate() {
        if (multi_thread) {
            vector<thread> threads;
            int pixelsPerThread = width * height / max_threads;
            for (int t = 0; t < max_threads; t++) {
                int startPixel = t * pixelsPerThread;
                int endPixel = (t == max_threads - 1) ? width * height : (t + 1) * pixelsPerThread;
                threads.push_back(thread([startPixel, endPixel, this]() {
                                             for (int p = startPixel; p < endPixel; p++) {
                                                 int x = p % width;
                                                 int y = p / width;
                                                 ColorInt ret = calc_pixel(x, y);
                                                 put_pixel(x, height-y-1, ret.red, ret.green, ret.blue);
                                             }
                                         }));
                if (show_percent) {
                    cout << "THREAD START --> " << (double) floor(((double) t / (double) max_threads) * 1000.0) / 10.0 << "%" << endl;
                }
            }
            int threadi = 0;
            for (auto& thread : threads) {
                threadi++;
                thread.join();
                if (show_percent) {
                    cout << "THREAD FINISH --> " << (double) floor(((double) threadi / (double) max_threads) * 1000.0) / 10.0 << "%" << endl;
                }
            }

        } else {
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    ColorInt col = calc_pixel(x, y);
                    put_pixel(x, height-y-1, col.red, col.green, col.blue);
                }
                if (show_percent) {
                    cout << "FINISHED --> " << (double) floor(((double) x / (double) width) * 1000.0) / 10.0 << "%" << endl;
                }
            }
        }
        save_bitmap("..//out_" + to_string(image_index) + ".bmp");
        delete[] imageData;
    }
};

int main() {
    bool multi_thread = true;
    bool show_percent = false;

    int maxthreads = 200;
    const int size = 1000;

    int width = size;
    int height = size;

    unsigned char* imageData = new unsigned char[width * height * 3];

    const Vector3 camera_pos = Vector3(-0.02, 0.09, -2.091);
    const vector<Sphere> spheres = {
        { Vector3(-0.58, -0.28, -0.94), 0.31, Color(0.961, 0.184, 0.388), 0.3 },
        { Vector3(0.8, -0.35, -0.35), 0.59, Color(0.506, 0.184, 0.961), 0.1 },
        { Vector3(-0.29, 0.83, 0.53), 1.04, Color(1.0, 1.0, 1.0), 1 },
        { Vector3(0.41, -3000.4, 0.03), 3000.0, Color(1.0, 1.0, 1.0), 0.4 }
    };
    const vector<PointLight> point_lights = {
        { Vector3(-2.6, 2.0, -2.8), 9.2, Color(0.279755, 0.523567, 1.0) },
        { Vector3(1.2, 2.4, -2.6), 5.1, Color(1.0, 0.360682, 0.060892) },
        { Vector3(-0.1, 3.8, -3.5), 5.2, Color(1.0, 1.0, 1.0) }
    };

    for (int i = 0; i < 360; i++) {
        double rotation = i;
        RayTrace rayTrace = RayTrace(size, camera_pos, spheres, point_lights, rotation, show_percent, multi_thread, maxthreads, i);
        rayTrace.Generate();
        cout << i + 1 << "/" << 360 << " (" << (double) floor(((double) (i + 1) / 360.0) * 1000.0) / 10.0 << "%)" << endl;
    }
    return 0;
}
