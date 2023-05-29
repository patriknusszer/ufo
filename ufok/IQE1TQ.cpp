//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nusszer Palltrik Marcell
// Neptun : IQE1TQ
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include <vector>
#include <string>
#include <math.h>
#include <list>
#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>        // must be downloaded
#include <GL/freeglut.h>    // must be downloaded unless you have an Apple
#endif

using namespace std;

typedef unsigned int uint32;

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
    #version 330                // Shader 3.3
    precision highp float;        // normal floats, makes no difference on desktop computers

    uniform mat4 MVP;            // uniform variable, the Model-View-Projection transformation matrix
    layout(location = 0) in vec2 vp;    // Varying input: vp = vertex position is expected in attrib array 0

    void main() {
        gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;        // transform vp from modeling space to normalized device space
    }
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
    #version 330            // Shader 3.3
    precision highp float;    // normal floats, makes no difference on desktop computers
    
    uniform vec3 color;        // uniform variable, the color of the primitive
    out vec4 outColor;        // computed color of the current pixel

    void main() {
        outColor = vec4(color, 1);    // computed color is the color of the primitive
    }
)";

// The following is just a ripoff framework.h

// Resolution of screen
const unsigned int windowWidth = 600, windowHeight = 600;

//--------------------------
struct vec2 {
//--------------------------
    float x, y;

    vec2(float x0 = 0, float y0 = 0) { x = x0; y = y0; }
    vec2 operator*(float a) const { return vec2(x * a, y * a); }
    vec2 operator/(float a) const { return vec2(x / a, y / a); }
    vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
    vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
    vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }
    vec2 operator-() const { return vec2(-x, -y); }
};

inline float dot(const vec2& v1, const vec2& v2) {
    return (v1.x * v2.x + v1.y * v2.y);
}

inline float length(const vec2& v) { return sqrtf(dot(v, v)); }

inline vec2 normalize(const vec2& v) { return v * (1 / length(v)); }

inline vec2 operator*(float a, const vec2& v) { return vec2(v.x * a, v.y * a); }

//--------------------------
struct vec3 {
//--------------------------
    float x, y, z;

    vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
    vec3(vec2 v) { x = v.x; y = v.y; z = 0; }

    vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
    vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }
    vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    vec3 operator-()  const { return vec3(-x, -y, -z); }
};

inline float dot(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

inline float length(const vec3& v) { return sqrtf(dot(v, v)); }

inline vec3 normalize(const vec3& v) { return v * (1 / length(v)); }

inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

inline vec3 operator*(float a, const vec3& v) { return vec3(v.x * a, v.y * a, v.z * a); }

//--------------------------
struct vec4 {
//--------------------------
    float x, y, z, w;

    vec4(float x0 = 0, float y0 = 0, float z0 = 0, float w0 = 0) { x = x0; y = y0; z = z0; w = w0; }
    float& operator[](int j) { return *(&x + j); }
    float operator[](int j) const { return *(&x + j); }

    vec4 operator*(float a) const { return vec4(x * a, y * a, z * a, w * a); }
    vec4 operator/(float d) const { return vec4(x / d, y / d, z / d, w / d); }
    vec4 operator+(const vec4& v) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
    vec4 operator-(const vec4& v)  const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
    vec4 operator*(const vec4& v) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
    void operator+=(const vec4 right) { x += right.x; y += right.y; z += right.z; w += right.w; }
};

inline float dot(const vec4& v1, const vec4& v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
}

inline vec4 operator*(float a, const vec4& v) {
    return vec4(v.x * a, v.y * a, v.z * a, v.w * a);
}

//---------------------------
struct mat4 { // row-major matrix 4x4
//---------------------------
    vec4 rows[4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33) {
        rows[0][0] = m00; rows[0][1] = m01; rows[0][2] = m02; rows[0][3] = m03;
        rows[1][0] = m10; rows[1][1] = m11; rows[1][2] = m12; rows[1][3] = m13;
        rows[2][0] = m20; rows[2][1] = m21; rows[2][2] = m22; rows[2][3] = m23;
        rows[3][0] = m30; rows[3][1] = m31; rows[3][2] = m32; rows[3][3] = m33;
    }
    mat4(vec4 it, vec4 jt, vec4 kt, vec4 ot) {
        rows[0] = it; rows[1] = jt; rows[2] = kt; rows[3] = ot;
    }

    vec4& operator[](int i) { return rows[i]; }
    vec4 operator[](int i) const { return rows[i]; }
    operator float*() const { return (float*)this; }
};

inline vec4 operator*(const vec4& v, const mat4& mat) {
    return v[0] * mat[0] + v[1] * mat[1] + v[2] * mat[2] + v[3] * mat[3];
}

inline mat4 operator*(const mat4& left, const mat4& right) {
    mat4 result;
    for (int i = 0; i < 4; i++) result.rows[i] = left.rows[i] * right;
    return result;
}

inline mat4 TranslateMatrix(vec3 t) {
    return mat4(vec4(1,   0,   0,   0),
                vec4(0,   1,   0,   0),
                vec4(0,   0,   1,   0),
                vec4(t.x, t.y, t.z, 1));
}

inline mat4 ScaleMatrix(vec3 s) {
    return mat4(vec4(s.x, 0,   0,   0),
                vec4(0,   s.y, 0,   0),
                vec4(0,   0,   s.z, 0),
                vec4(0,   0,   0,   1));
}

inline mat4 RotationMatrix(float angle, vec3 w) {
    float c = cosf(angle), s = sinf(angle);
    w = normalize(w);
    return mat4(vec4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s, w.x*w.z*(1 - c) - w.y*s, 0),
                vec4(w.x*w.y*(1 - c) - w.z*s, c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s, 0),
                vec4(w.x*w.z*(1 - c) + w.y*s, w.y*w.z*(1 - c) - w.x*s, c * (1 - w.z*w.z) + w.z*w.z, 0),
                vec4(0, 0, 0, 1));
}

//---------------------------
class Texture {
//---------------------------
    std::vector<vec4> load(std::string pathname, bool transparent, int& width, int& height) {
        FILE * file = fopen(pathname.c_str(), "r");
        if (!file) {
            printf("%s does not exist\n", pathname.c_str());
            width = height = 0;
            return std::vector<vec4>();
        }
        unsigned short bitmapFileHeader[27];                    // bitmap header
        fread(&bitmapFileHeader, 27, 2, file);
        if (bitmapFileHeader[0] != 0x4D42) printf("Not bmp file\n");
        if (bitmapFileHeader[14] != 24) printf("Only true color bmp files are supported\n");
        width = bitmapFileHeader[9];
        height = bitmapFileHeader[11];
        unsigned int size = (unsigned long)bitmapFileHeader[17] + (unsigned long)bitmapFileHeader[18] * 65536;
        fseek(file, 54, SEEK_SET);
        std::vector<unsigned char> bImage(size);
        fread(&bImage[0], 1, size, file);     // read the pixels
        fclose(file);
        std::vector<vec4> image(width * height);
        int i = 0;
        for (unsigned int idx = 0; idx < size; idx += 3) { // Swap R and B since in BMP, the order is BGR
            float alpha = (transparent) ? (bImage[idx] + bImage[idx + 1] + bImage[idx + 2]) / 3.0f / 256.0f : 1.0f;
            image[i++] = vec4(bImage[idx + 2] / 256.0f, bImage[idx + 1] / 256.0f, bImage[idx] / 256.0f, alpha);
        }
        return image;
    }

public:
    unsigned int textureId = 0;

    Texture() { textureId = 0; }

    Texture(std::string pathname, bool transparent = false) {
        textureId = 0;
        create(pathname, transparent);
    }

    Texture(int width, int height, const std::vector<vec4>& image, int sampling = GL_LINEAR) {
        textureId = 0;
        create(width, height, image, sampling);
    }

    Texture(const Texture& texture) {
        printf("\nError: Texture resource is not copied on GPU!!!\n");
    }

    void operator=(const Texture& texture) {
        printf("\nError: Texture resource is not copied on GPU!!!\n");
    }

    void create(std::string pathname, bool transparent = false) {
        int width, height;
        std::vector<vec4> image = load(pathname, transparent, width, height);
        if (image.size() > 0) create(width, height, image);
    }

    void create(int width, int height, const std::vector<vec4>& image, int sampling = GL_LINEAR) {
        if (textureId == 0) glGenTextures(1, &textureId);                  // id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, &image[0]); // To GPU
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sampling); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sampling);
    }

    ~Texture() {
        if (textureId > 0) glDeleteTextures(1, &textureId);
    }
};


//---------------------------
class GPUProgram {
//--------------------------
    unsigned int shaderProgramId = 0;
    unsigned int vertexShader = 0, geometryShader = 0, fragmentShader = 0;
    bool waitError = true;

    void getErrorInfo(unsigned int handle) { // shader error report
        int logLen, written;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0) {
            std::string log(logLen, '\0');
            glGetShaderInfoLog(handle, logLen, &written, &log[0]);
            printf("Shader log:\n%s", log.c_str());
            if (waitError) getchar();
        }
    }

    bool checkShader(unsigned int shader, std::string message) { // check if shader could be compiled
        int OK;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
        if (!OK) {
            printf("%s!\n", message.c_str());
            getErrorInfo(shader);
            return false;
        }
        return true;
    }

    bool checkLinking(unsigned int program) {     // check if shader could be linked
        int OK;
        glGetProgramiv(program, GL_LINK_STATUS, &OK);
        if (!OK) {
            printf("Failed to link shader program!\n");
            getErrorInfo(program);
            return false;
        }
        return true;
    }

    int getLocation(const std::string& name) {    // get the address of a GPU uniform variable
        int location = glGetUniformLocation(shaderProgramId, name.c_str());
        if (location < 0) printf("uniform %s cannot be set\n", name.c_str());
        return location;
    }

public:
    GPUProgram(bool _waitError = true) { shaderProgramId = 0; waitError = _waitError; }

    GPUProgram(const GPUProgram& program) {
        if (program.shaderProgramId > 0) printf("\nError: GPU program is not copied on GPU!!!\n");
    }

    void operator=(const GPUProgram& program) {
        if (program.shaderProgramId > 0) printf("\nError: GPU program is not copied on GPU!!!\n");
    }

    unsigned int getId() { return shaderProgramId; }

    bool create(const char * const vertexShaderSource,
                const char * const fragmentShaderSource, const char * const fragmentShaderOutputName,
                const char * const geometryShaderSource = nullptr)
    {
        // Create vertex shader from string
        if (vertexShader == 0) vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) {
            printf("Error in vertex shader creation\n");
            exit(1);
        }
        glShaderSource(vertexShader, 1, (const GLchar**)&vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        if (!checkShader(vertexShader, "Vertex shader error")) return false;

        // Create geometry shader from string if given
        if (geometryShaderSource != nullptr) {
            if (geometryShader == 0) geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
            if (!geometryShader) {
                printf("Error in geometry shader creation\n");
                exit(1);
            }
            glShaderSource(geometryShader, 1, (const GLchar**)&geometryShaderSource, NULL);
            glCompileShader(geometryShader);
            if (!checkShader(geometryShader, "Geometry shader error")) return false;
        }

        // Create fragment shader from string
        if (fragmentShader == 0) fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) {
            printf("Error in fragment shader creation\n");
            exit(1);
        }

        glShaderSource(fragmentShader, 1, (const GLchar**)&fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        if (!checkShader(fragmentShader, "Fragment shader error")) return false;

        shaderProgramId = glCreateProgram();
        if (!shaderProgramId) {
            printf("Error in shader program creation\n");
            exit(1);
        }
        glAttachShader(shaderProgramId, vertexShader);
        glAttachShader(shaderProgramId, fragmentShader);
        if (geometryShader > 0) glAttachShader(shaderProgramId, geometryShader);

        // Connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgramId, 0, fragmentShaderOutputName);    // this output goes to the frame buffer memory

        // program packaging
        glLinkProgram(shaderProgramId);
        if (!checkLinking(shaderProgramId)) return false;

        // make this program run
        glUseProgram(shaderProgramId);
        return true;
    }

    void Use() {         // make this program run
        glUseProgram(shaderProgramId);
    }

    void setUniform(int i, const std::string& name) {
        int location = getLocation(name);
        if (location >= 0) glUniform1i(location, i);
    }

    void setUniform(float f, const std::string& name) {
        int location = getLocation(name);
        if (location >= 0) glUniform1f(location, f);
    }

    void setUniform(const vec2& v, const std::string& name) {
        int location = getLocation(name);
        if (location >= 0) glUniform2fv(location, 1, &v.x);
    }

    void setUniform(const vec3& v, const std::string& name) {
        int location = getLocation(name);
        if (location >= 0) glUniform3fv(location, 1, &v.x);
    }

    void setUniform(const vec4& v, const std::string& name) {
        int location = getLocation(name);
        if (location >= 0) glUniform4fv(location, 1, &v.x);
    }

    void setUniform(const mat4& mat, const std::string& name) {
        int location = getLocation(name);
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, mat);
    }

    void setUniform(const Texture& texture, const std::string& samplerName, unsigned int textureUnit = 0) {
        int location = getLocation(samplerName);
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, texture.textureId);
        }
    }

    ~GPUProgram() { if (shaderProgramId > 0) glDeleteProgram(shaderProgramId); }
};

GPUProgram gpuProgram; // vertex and fragment shaders

// pragma: let x, y be plane coordinates, let z be height coordinate

namespace tools {
namespace hyperboloid {
        float lorentz(const vec3& v) {
            return pow(v.x, 2) + pow(v.y, 2) - pow(v.z, 2);
        }

        float lorentz(const vec3& v1, const vec3& v2) {
            return v1.x * v2.x + v1.y * v2.y - v1.z * v2.z;
        }
    
        float len(const vec3& v) {
            return sqrtf(lorentz(v));
        }
    
        vec3 normalise(const vec3& v) {
            return v * (1 / len(v));
        }
    
        static vec3 getPointAway(const vec3& p, const vec3& tangent, const float& dst) {
            vec3 _tangent = normalise(tangent);
            return p * cosh(dst) + _tangent * sinh(dst);
        }
        
        void getPointInformation(const vec3& q, const vec3& p, vec3* tangent, float* dst) {
            // distance can be specified wihout knowing the direction.
            // the equation of uniform movement is scalar multiplied by p, and the scalar of p and v must be zero
            float _dst = acosh(-1 * lorentz(p, q));
            // formula after substituting the distance back, and rearranging it to solve for the tangent
            if (tangent != 0)
                *tangent = vec3(
                            (q.x - p.x * cosh(_dst)) / sinh(_dst),
                            (q.y - p.y * cosh(_dst)) / sinh(_dst),
                            (q.z - p.z * cosh(_dst)) / sinh(_dst)
                            );
                
            if (dst != 0)
                *dst = _dst;
        }

        bool close(const vec3& pos1, const vec3& pos2, float tolerance) {
            float dst;
            getPointInformation(pos1, pos2, 0, &dst);
            return dst <= tolerance;
        }
            
        void pointCorrection(vec3& p) {
            // let the plane (x, y) coordinates stay the same, we are solving for z, so that p is again on the hyperboloid
            p.z = sqrt(pow(p.x, 2) + pow(p.y, 2) + 1);
        }
    
        void tangentCorrection(const vec3& p, vec3& tangent) {
            float lambda = (p.z * tangent.z - p.x * tangent.x - p.y * tangent.y) / (pow(p.x, 2) + pow(p.y, 2) - pow(p.z, 2));
            // calculated by rearrangement of p dot (v + p * lambda) = 0
            tangent = tangent + p * lambda;
        }
    
        vec3 hycross(const vec3& v1, const vec3& v2) {
            return cross(vec3(v1.x, v1.y, -1 * v1.z), vec3(v2.x, v2.y, -1 * v2.z));
        }
    
        // rotating a vector by a given angle in hyperbolic geometry in the tangent plane of a point
        vec3 rotateTangent(const float& angle, const vec3& tangent, const vec3& p) {
            vec3 cr = hycross(tangent, p);
            cr = normalise(cr);
            vec3 _tangent = normalise(tangent);
            return _tangent * cos(angle) + cr * sin(angle);
        }
    
        void moveInTangentDirection(const vec3& tangent, const vec3& pos, const double& t, vec3* tangentout, vec3* posout) {
            vec3 _tangent = normalise(tangent);
            *posout = pos * cosh(t) + _tangent * sinh(t);
            *tangentout = normalise(pos * sinh(t) + _tangent * cosh(t));
        }
    
        vec2 project2Disc(const vec3& hypt) {
            // based on triangle similarity
            double scale = hypt.z + 1;
            return vec2(
                        hypt.x / scale,
                        hypt.y / scale
                        );
        }

        vec3 project2Hyperboloid(const vec2& dipt) {
            // scale based on triangle similarity
            // (kx)^2+(ky)^2-(k - 1)=-1 is the equation from which the scale is obtained
            // a point (x, y, 1) can be projected to the hyperboloid x^2+y^2-(z-1)^2=-1
            // which is a hyperboloid shifted one up
            // at this time, k is the coordinate of height for the image point
            // however, for hyperboloid x^2+y^2-z^2=-1 x and y are the same, and k-1 is the height
            // that being said, x and y should be on the unit circle, perimeter excluded
            double scale = 2 / (1 - pow(dipt.x, 2) - pow(dipt.y, 2));
            return vec3(dipt.x * scale,
                        dipt.y * scale,
                        scale - 1);
        }
    
        vector<vec2> projectHyperbolicCircle(const vec3& hyCenter, const vec3& hyTangent, const float& hyRadius, const int& precision) {
            vector<vec2> vertices = vector<vec2>();
            vertices.push_back(project2Disc(hyCenter));
            double delta = 2 * M_PI / precision;
            vec3 _tangent = hyTangent;
            
            for (int i = 0; i < precision; i++) {
                _tangent = rotateTangent(delta, _tangent, hyCenter);
                vertices.push_back(project2Disc(getPointAway(hyCenter, _tangent, hyRadius)));
            }
            
            vertices.push_back(vertices[1]);
            return vertices;
        }
    };

    uint32 createVAO() {
        uint32 vaoID;
        glGenVertexArrays(1, &vaoID);
        return vaoID;
    }

    uint32 createAttribVBO(const vector<vec2>& vertices, const uint32& vaoID) {
        unsigned int vboID;
        glBindVertexArray(vaoID);
        glGenBuffers(1, &vboID);
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        //glBufferData(GL_ARRAY_BUFFER,
          //  sizeof(vec2) * vertices.size(),
            //vertices.data(),
            //GL_DYNAMIC_DRAW);
        
        /*
        !!!!!
         
        Eleg rossz, hogy csak openGL 4.3 tol lehet kulon meghatarozni az AttribArray indexet es a vertex formatumot.
        Emiatt van az, hogy amennyiben egy VAO tobb VBO-t tarol magaban,
            akkor muszaj ugyanazt a formatumot, es mas-mas AttribArray
            indexet meghatarozni a glVeretxAttribPoitner hivasban.
        Amennyiben pedig egyetlen AttribArray-t kivanunk hasznalni, akkor pedig elkerulhetetlen, hogy minden egyes kirajzolasnal meghatarozzuk az AttribArray-t es a veretx formatumto ujra es ujra egyetlen hizvasban, mert a glVeretxAttribPointer metodus a legutoljara aktivalt VBO-hoz koti a formatumot az attribute index-szel egyutt.
         
         Meglehetosen ronda konstrukcio, hogy oszinte legyek.
         https://stackoverflow.com/questions/40652905/render-one-vao-containing-two-vbos
         
        !!!!!
        */
        glEnableVertexAttribArray(0);
            glVertexAttribPointer(0,
                2, GL_FLOAT, GL_FALSE,
                0, NULL);
        return vboID;
    }

    void drawVertices(const GLenum& mode, const vector<vec2>& vertices, const vec3& colour, const uint32& vaoID, const uint32& vboID, const int& bindStructureAndAttribIndex = -1) {
        glBindVertexArray(vaoID);
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        /*glEnableVertexAttribArray(0);
            glVertexAttribPointer(0,
                2, GL_FLOAT, GL_FALSE,
                0, NULL);*/
        glBufferData(GL_ARRAY_BUFFER,
            sizeof(vec2) * vertices.size(),
                     vertices.data(),
            GL_STATIC_DRAW);
        //glBindVertexArray(vaoID);
        
        int address = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(address, colour.x, colour.y, colour.z); // 3 floats
        
        float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                                  0, 1, 0, 0,    // row-major!
                                  0, 0, 1, 0,
                                  0, 0, 0, 1 };
        
        address = glGetUniformLocation(gpuProgram.getId(), "MVP");    // Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(address, 1, GL_TRUE, &MVPtransf[0][0]);    // Load a 4x4 row-major float matrix to the specified location
        
        if (bindStructureAndAttribIndex != -1) {
            glEnableVertexAttribArray(bindStructureAndAttribIndex);
                glVertexAttribPointer(bindStructureAndAttribIndex,
                    2, GL_FLOAT, GL_FALSE,
                    0, NULL);
        }
        
        //address = glGetUniformLocation(gpuProgram.getId(), "z");
        //glUniform1f(address, layer);
        // binding buffer to attribute
        // amennyiben egyetlen VAO lenne, itt kene meghivni a glAttribArrayPointer fgv-t
        glDrawArrays(mode, 0 /*startIdx*/, vertices.size() /*# Elements*/);
        if (glGetError() == GL_INVALID_OPERATION) {
            int g = 10;
        }
    }
};

class GameObject {
protected:
    bool allowRender = true;
    vector<vec2> vertices;
    uint32 vboID;
    uint32 vaoID;
public:
    virtual void render() = 0;
    virtual void destroy() = 0;
    bool destroyed() { return !allowRender; }
};

class LineSegment : GameObject {
private:
    vec3 colour;
    vec2 b, e;
public:
    LineSegment() {
        vaoID = tools::createVAO();
        vboID = tools::createAttribVBO(vertices, vaoID);
    }
    
    vec2 begin() { return b; }
    vec2 end() { return e; }
    
    void setBeg(vec2 b) { this->b = b; }
    void setEnd(vec2 e) { this->e = e; }
    
    void setColour(vec3 colour) {
        this->colour = colour;
    }
    
    LineSegment(vec3 colour, vec2 beg, vec2 end) {
        this->colour = colour;
        vertices.push_back(beg);
        vertices.push_back(end);
        vaoID = tools::createVAO();
        vboID = tools::createAttribVBO(vertices, vaoID);
    }
    
    void render() override {
        if (!allowRender) return;
        vertices.clear();
        vertices.push_back(b);
        vertices.push_back(e);
        tools::drawVertices(GL_LINES, vertices, colour, vaoID, vboID);
    }
    
    void destroy() override {
        if (!allowRender) return;
        allowRender = false;
        glDeleteBuffers(1, &vboID);
        glDeleteVertexArrays(1, &vaoID);
    }
};

class Line : GameObject {
private:
    vec3 colour;
    vec3 dskColour = vec3(66 / 255.0, 66 / 255.0, 66 / 255.0);
    int maxLen = 0;
    LineSegment* ls;
    int len = 0, rw = 0;
public:
    Line() {}
    
    void destroy() override {
        if (!allowRender) return;
        allowRender = false;
        for (int i = 0; i < maxLen; i++)
            ls[i].destroy();
        delete[] ls;
    }
    
    Line(vec3 colour, int maxLen) {
        //vaoID = tools::createVAO();
        //vboID = tools::createAttribVBO(vertices, vaoID);
        this->colour = colour;
        this->maxLen = maxLen;
        ls = new LineSegment[maxLen];
    }
    
    void setMax(int mx) {
        if (!allowRender) return;
        delete[] ls;
        maxLen = abs(mx);
        ls = new LineSegment[maxLen];
        len = rw = 0;
    }
    
    int mod(int a, int m) {
        if (a >= 0) return a % m;
        else return m - (abs(a) % m);
    }
    
    void update() {
        if (!allowRender) return;
        float dr = dskColour.x - colour.x;
        float dg = dskColour.y - colour.y;
        float db = dskColour.z - colour.z;
        float d = 1.0 / len;
        float m = 1;
        
        for (int i = 0; i < len; i++) {
            int ind = mod(rw + i, maxLen);
            ls[ind].setColour(vec3(
                                                   colour.x + m * dr,
                                                   colour.y + m * dg,
                                                   colour.z + m * db));
            m -= d;
        }
    }
    
    void push(vec2 pt) {
        if (!allowRender) return;
        if (len == maxLen) {
            ls[rw].setEnd(pt);
            ls[rw].setBeg(ls[mod(rw - 1, maxLen)].end());
            rw = mod(rw + 1, maxLen);
        }
        else {
            if (len == 0) {
                ls[0].setBeg(pt);
                ls[0].setEnd(pt);
            }
            else {
                ls[len].setBeg(ls[len - 1].end());
                ls[len].setEnd(pt);
            }
            
            len++;
        }
        
        update();
    }
    
    void render() {
        if (!allowRender) return;
        for (int i = 0; i < len; i++) {
            int ind = mod(rw + i, maxLen);
            ls[ind].render();
        }
    }
};

class Disk : public GameObject {
private:
    vec2 center;
    vec3 colour;
    double radius;
    int precision;
public:
    Disk() {}
    
    Disk(vec3 colour) {
        this->colour = colour;
        vaoID = tools::createVAO();
        vboID = tools::createAttribVBO(vertices, vaoID);
    }
    
    Disk(vec3 colour, vec2 center, double radius, double precision) : Disk(colour) {
        this->radius = radius;
        this->center = center;
        this->precision = precision;
    }
    
    void render() override {
        if (!allowRender) return;
        vertices.clear();
        //vertices.push_back(center);
        double delta = 2 * M_PI / precision;
        double angle = M_PI / 2;
        
        for (int i = 0; i < precision; i++) {
            double unity = sin(angle);
            double unitx = cos(angle);
            vertices.push_back(vec2(unitx * radius + center.x, unity * radius + center.y));
            angle += delta;
        }
        
        vertices.push_back(vertices[1]);
        tools::drawVertices(GL_TRIANGLE_FAN, vertices, colour, vaoID, vboID);
    }
    
    void render(const vector<vec2>& vertices) {
        if (!allowRender) return;
        tools::drawVertices(GL_TRIANGLE_FAN, vertices, colour, vaoID, vboID);
    }
    
    void destroy() override {
        if (!allowRender) return;
        allowRender = false;
        glDeleteBuffers(1, &vboID);
        glDeleteVertexArrays(1, &vaoID);
    }
};

class Hami : public GameObject {
protected:
    vec3 colour;
    vec3 hyTangent;
    vec3 hyPos;
    // hyperboloid radius is not a real radius. It is a curved radius from the center position to teh nose through the hyperbolic surface
    float hyRadius;
    int precision;
    Disk body;
    Disk reye;
    Disk leye;
    Disk reyeb;
    Disk leyeb;
    Disk mouth;
    Line mucus;
    Hami* other;
    float mouthFunc = M_PI / 2;
    
public:
    void setFriend(Hami* other) {
        this->other = other;
    }
    
    void setMaxMucus(int mx) {
        mucus.setMax(mx);
    }
    
    void destroy() {
        if (!allowRender) return;
        allowRender = false;
        mucus.destroy();
        reye.destroy();
        leye.destroy();
        reyeb.destroy();
        leyeb.destroy();
        body.destroy();
        mouth.destroy();
    }
    
    vec3 getHyPos() { return hyPos; }
    
    Hami() {}
    
    Hami(vec3 colour, float hyRadius, int precision) {
        this->colour = colour;
        this->hyRadius = hyRadius;
        this->precision = precision;
        body = Disk(colour);
        reye = leye = Disk(vec3(1, 1, 1));
        reyeb = leyeb = Disk(vec3(0,0,0));
        mouth = Disk(vec3(0.1,0.1,0.1));
        mucus = Line(vec3(1,1,1), 200);
    }
    
    virtual ~Hami() {}
};

// a jatekos UFO hamija
class PirosHami : public Hami {
private:
    void correction() {
        tools::hyperboloid::pointCorrection(hyPos);
        tools::hyperboloid::tangentCorrection(hyPos, hyTangent);
    }
public:
    PirosHami() {}
    
    PirosHami(vec3 colour) : Hami(colour, 0.2, 500) {
        // PirosHami is placed at (0, 0) on thee disk, at this point the position on the hyperboloid is clear
        hyPos = vec3(0, 0, 1);
        // just a random position on the disk, upwards on the disk from our disk position
        vec2 hamiDiskDirection = vec2(0, 0.1);
        vec3 q = tools::hyperboloid::project2Hyperboloid(hamiDiskDirection);
        // getting the direction of the hyperbolic projection of the random point
        tools::hyperboloid::getPointInformation(q, hyPos, &hyTangent, 0);
    }
    
    void render() override {
        if (!allowRender) return;
        correction();
        
        vec3 reyeDir = tools::hyperboloid::rotateTangent(0.7, hyTangent, hyPos);
        vec3 leyeDir = tools::hyperboloid::rotateTangent(-0.7, hyTangent, hyPos);
        
        vec3 reyeTan, leyeTan, reyePos, leyePos;
        tools::hyperboloid::moveInTangentDirection(reyeDir, hyPos, 0.2, &reyeTan, &reyePos);
        tools::hyperboloid::moveInTangentDirection(leyeDir, hyPos, 0.2, &leyeTan, &leyePos);
        
        mucus.render();
        
        body.render(tools::hyperboloid::projectHyperbolicCircle(hyPos, hyTangent, hyRadius, 500));
        reye.render(tools::hyperboloid::projectHyperbolicCircle(reyePos, reyeTan, 0.06, 50));
        leye.render(tools::hyperboloid::projectHyperbolicCircle(leyePos, leyeTan, 0.06, 50));
        
        vec3 mouthPos, mouthTan;
        tools::hyperboloid::moveInTangentDirection(hyTangent, hyPos, 0.2, &mouthTan, &mouthPos);
        
        mouth.render(tools::hyperboloid::projectHyperbolicCircle(mouthPos, mouthTan, 0.05 * sin(mouthFunc), 50));
        mouthFunc += 0.07;
        
        if (other->destroyed()) {
            reyeb.render(tools::hyperboloid::projectHyperbolicCircle(reyePos, reyeTan, 0.03, 50));
            leyeb.render(tools::hyperboloid::projectHyperbolicCircle(leyePos, leyeTan, 0.03, 50));
            return;
        }
        
        vec3 rotan;
        tools::hyperboloid::getPointInformation(other->getHyPos(), reyePos, &rotan, 0);
        vec3 reyebPos, reyebTan;
        tools::hyperboloid::moveInTangentDirection(rotan, reyePos, 0.03, &reyebTan, &reyebPos);
        
        vec3 lotan;
        tools::hyperboloid::getPointInformation(other->getHyPos(), leyePos, &lotan, 0);
        vec3 leyebPos, leyebTan;
        tools::hyperboloid::moveInTangentDirection(lotan, leyePos, 0.03, &leyebTan, &leyebPos);
        
        reyeb.render(tools::hyperboloid::projectHyperbolicCircle(reyebPos, reyebTan, 0.03, 50));
        leyeb.render(tools::hyperboloid::projectHyperbolicCircle(leyebPos, leyebTan, 0.03, 50));
    }
    
    void rotateLeft() {
        if (!allowRender) return;
        hyTangent = tools::hyperboloid::rotateTangent(-0.3, hyTangent, hyPos);
        mucus.push(tools::hyperboloid::project2Disc(hyPos));
    }
    
    void rotateRight() {
        if (!allowRender) return;
        hyTangent = tools::hyperboloid::rotateTangent(0.3, hyTangent, hyPos);
        mucus.push(tools::hyperboloid::project2Disc(hyPos));
    }
    
    void move() {
        if (!allowRender) return;
        tools::hyperboloid::moveInTangentDirection(hyTangent, hyPos, 0.1, &hyTangent, &hyPos);
        mucus.push(tools::hyperboloid::project2Disc(hyPos));
    }
    
    void stand() {
        if (!allowRender) return;
        mucus.push(tools::hyperboloid::project2Disc(hyPos));
    }
};

class ZoldHami : public Hami {
private:
    vec3 hyRotCenter;
    vec3 hyRotTangent;
    float hyRotRadius = 0.6;
public:
    ZoldHami() {}
    
    ZoldHami(vec3 colour) : Hami(colour, 0.2, 500) {
        vec3 ht = vec3(1, 0, 0);
        vec3 p = vec3(0, 0, 1);
        tools::hyperboloid::moveInTangentDirection(ht, p, 1.4, &hyRotTangent, &hyRotCenter);
    }
    
    void rotate() {
        if (!allowRender) return;
        hyRotTangent = tools::hyperboloid::rotateTangent(-0.1, hyRotTangent, hyRotCenter);
        hyPos = tools::hyperboloid::getPointAway(hyRotCenter, hyRotTangent, hyRotRadius);
        mucus.push(tools::hyperboloid::project2Disc(hyPos));
    }
    
    void render() override {
        if (!allowRender) return;
        //hyTangent = vec3(1,2,3);
        tools::hyperboloid::getPointInformation(hyRotCenter, hyPos, &hyTangent, 0);
        // Hami is looking to the right
        vec3 perp = tools::hyperboloid::hycross(hyTangent, hyPos);
        vec3 reyeDir = tools::hyperboloid::rotateTangent(0.7, perp, hyPos);
        vec3 leyeDir = tools::hyperboloid::rotateTangent(-0.7, perp, hyPos);
        vec3 reyeTan, leyeTan, reyePos, leyePos;
        tools::hyperboloid::moveInTangentDirection(reyeDir, hyPos, 0.2, &reyeTan, &reyePos);
        tools::hyperboloid::moveInTangentDirection(leyeDir, hyPos, 0.2, &leyeTan, &leyePos);
        
        vec3 rotan;
        tools::hyperboloid::getPointInformation(other->getHyPos(), reyePos, &rotan, 0);
        vec3 reyebPos, reyebTan;
        tools::hyperboloid::moveInTangentDirection(rotan, reyePos, 0.03, &reyebTan, &reyebPos);
        
        vec3 lotan;
        tools::hyperboloid::getPointInformation(other->getHyPos(), leyePos, &lotan, 0);
        vec3 leyebPos, leyebTan;
        tools::hyperboloid::moveInTangentDirection(lotan, leyePos, 0.03, &leyebTan, &leyebPos);
        
        vec3 mouthPos, mouthTan;
        tools::hyperboloid::moveInTangentDirection(perp, hyPos, 0.2, &mouthTan, &mouthPos);
        
        tools::hyperboloid::tangentCorrection(hyPos, hyTangent);
        
        mucus.render();
        
        body.render(tools::hyperboloid::projectHyperbolicCircle(hyPos, hyTangent, hyRadius, 500));
        reye.render(tools::hyperboloid::projectHyperbolicCircle(reyePos, reyeTan, 0.06, 50));
        reyeb.render(tools::hyperboloid::projectHyperbolicCircle(reyebPos, reyebTan, 0.03, 50));
        leye.render(tools::hyperboloid::projectHyperbolicCircle(leyePos, leyeTan, 0.06, 50));
        leyeb.render(tools::hyperboloid::projectHyperbolicCircle(leyebPos, leyebTan, 0.03, 50));
        mouth.render(tools::hyperboloid::projectHyperbolicCircle(mouthPos, mouthTan, 0.05 * sin(mouthFunc), 50));
        mouthFunc += 0.07;
    }
};

Disk dsk;
PirosHami ph;
ZoldHami zh;

bool firstMove = true;
bool doMove = false;
bool doRotateR = false;
bool doRotateL = false;

int frames2Play = 1;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
    dsk = Disk(vec3(66 / 255.0, 66 / 255.0, 66 / 255.0), vec2(0, 0), 1.0, 500);
    ph = PirosHami(vec3(224 / 255.0, 40 / 255.0,40 / 255.0));
    zh = ZoldHami(vec3(1 / 255.0, 178 / 255.0, 63 / 255.0));
    ph.setFriend(&zh);
    zh.setFriend(&ph);
    ph.setMaxMucus(2500);
    zh.setMaxMucus(50);
    // zh = PirosHami(vec3(1 / 255.0, 178 / 255.0, 63 / 255.0), 0.01, 500);
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
    dsk.render();
    ph.render();
    zh.render();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
        case 'e':
            //ph.moveUp();
            doMove = true;
            break;
        case 's':
            //ph.rotateRight();
            doRotateL = true;
            break;
        case 'f':
            doRotateR = true;
            // ph.rotateLeft();
            break;
        default: return;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
// TODO: UFOHAMI blinking
long lastCheckIn = -1;
void onIdle() {
    if (tools::hyperboloid::close(zh.getHyPos(), ph.getHyPos(), 0.1))
        zh.destroy();
    long cur = glutGet(GLUT_ELAPSED_TIME);
    
    if (lastCheckIn != -1) {
        int delta = cur - lastCheckIn;
        if (firstMove && (doMove || doRotateL || doRotateR)) {
            firstMove = false;
            
            if (delta < 350)
                frames2Play = 1;
            else frames2Play = delta / 27 + 1;
        }
        else if (delta < 35) frames2Play = 1;
        else frames2Play = delta / 27 + 1;
        lastCheckIn = cur;
    }
    else {
        lastCheckIn = cur;
        frames2Play = 1;
    }
    
    for (int i = 0; i < frames2Play; i++) {
        if (doMove)
            ph.move();
        if (doRotateR)
            ph.rotateRight();
        if (doRotateL)
            ph.rotateLeft();
        if (!doMove && !doRotateL && !doRotateR)
            ph.stand();
        zh.rotate();
    }

    doMove = doRotateL = doRotateR = false;
    glutPostRedisplay();
}
