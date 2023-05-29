//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
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
// Nev    : Kova'cs Levente
// Neptun : N3XCCL
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

#include "framework.h"

// Default vshader for transforming stuff
const char * const vertexSource = R"(
    #version 330                // Shader 3.3
    precision highp float;        // normal floats, makes no difference on desktop computers

    uniform mat4 MVP;            // uniform variable, the Model-View-Projection transformation matrix
    layout(location = 0) in vec2 vp;    // Varying input: vp = vertex position is expected in attrib array 0

    void main() {
        gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;        // transform vp from modeling space to normalized device space
    }
)";

// VShader for texturing background
const char * const vertexTexture = R"(
    #version 330                // Shader 3.3
    precision highp float;        // normal floats, makes no difference on desktop computers

    layout(location = 0) in vec2 vtxPos;    // Varying input: vp = vertex position is expected in attrib array 0
    layout(location = 1) in vec2 vtxUV;
    
    out vec2 texcoord;

    void main() {
        gl_Position = vec4(vtxPos, 0, 1);        // transform vp from modeling space to normalized device space
        texcoord = vtxUV;
    }
)";

// Default fshader for transforming stuff
const char * const fragmentSource = R"(
    #version 330            // Shader 3.3
    precision highp float;    // normal floats, makes no difference on desktop computers
    
    uniform vec3 color;        // uniform variable, the color of the primitive
    out vec4 outColor;        // computed color of the current pixel

    void main() {
        outColor = vec4(color.x, color.y, color.z, 1);    // computed color is the color of the primitive
    }
)";

// FShader for texturing background
const char * const fragmentTexture = R"(
    #version 330            // Shader 3.3
    precision highp float;    // normal floats, makes no difference on desktop computers
    
    uniform sampler2D samplerUnit;
    in vec2 texcoord;
    out vec4 outColor;        // computed color of the current pixel

    void main() {
        outColor = texture(samplerUnit, texcoord);    // computed color is the color of the primitive
    }
)";

GPUProgram gpuProgram, textureProgram; // textureProgram for texturing, gpuProgram for everything else
unsigned int vao, vaot;       // vaot for texturing, vao for everything else
uint32_t textureID;

const float g = 9.81;

bool isSpaceOn = false;

bool closeTo(float f1, float f2, float epsilon=0.0001) {
    return fabs(f1 - f2) < epsilon;
}

vec2 operator*(float f, vec2 v) {
    return v * f;
}

/*vec2 operator/(vec2 v, float f) {
    return v * (1/f);
}*/

vec2& operator+=(vec2& v1, vec2 v2) {
    v1 = v1 + v2;
    return v1;
}

class ExplicitKochanekBartels {
private:
    std::vector<vec2> controlPoints;
    
    float tau;
    
    float interpolate(float x0, float y0, float v0, float x1, float y1, float v1, float x) {
        float a0 = y0;
        float a1 = v0;
        float a2 = 3 * (y1 - y0) / pow(x1 - x0, 2) - (v1 + 2 * v0) / (x1 - x0);
        float a3 = 2 * (y0 - y1) / pow(x1 - x0, 3) + (v1 + v0) / pow(x1 - x0, 2);
        return a3 * pow(x - x0, 3.0f) + a2 * pow(x - x0, 2.0f) + a1 * (x - x0) + a0;
    }
    
    float interpolate_derivate(float x0, float y0, float v0, float x1, float y1, float v1, float x) {
        float a0 = y0;
        float a1 = v0;
        float a2 = 3 * (y1 - y0) / pow(x1 - x0, 2) - (v1 + 2 * v0) / (x1 - x0);
        float a3 = 2 * (y0 - y1) / pow(x1 - x0, 3) + (v1 + v0) / pow(x1 - x0, 2);
        return 3 * a3 * pow(x - x0, 2.0f) + 2 * a2 * (x - x0) + a1;
    }
    
    float get_v_helper(int idx) {
        return (controlPoints[idx + 1].y - controlPoints[idx].y) / (controlPoints[idx + 1].x - controlPoints[idx].x) * (1 - tau);
    }
    
    float get_v(int idx) {
        if (idx == 0) {
            return get_v_helper(idx);
        } else if (idx == controlPoints.size() - 1) {
            return get_v_helper(idx - 1);
        }
        return (get_v_helper(idx) + get_v_helper(idx - 1)) * (0.5f);
    }
    
public:
    ExplicitKochanekBartels(float tau, float height) {
        controlPoints.push_back(vec2(-1, height));
        controlPoints.push_back(vec2(1, height));
        this->tau = tau;
    }
    
    int len() {
        return controlPoints.size();
    }
    
    void addPoint(float x, float y) {
        addPoint(vec2(x, y));
    }
    
    void addPoint(vec2 point) {
        for (size_t i = 0; i < controlPoints.size(); i++) {
            if (controlPoints[i].x > point.x) {
                controlPoints.insert(controlPoints.begin() + i, point);
                return;
            }
        }
    }
    
    float y(float x) {
        for (int i = 0; i < controlPoints.size()-1; i++) {
            if (controlPoints[i].x <= x && x <= controlPoints[i + 1].x + 0.0001) {
                float y0 = controlPoints[i].y;
                float y1 = controlPoints[i + 1].y;
                float x0 = controlPoints[i].x;
                float x1 = controlPoints[i + 1].x;
                float v0 = get_v(i);
                float v1 = get_v(i + 1);
                float y = interpolate(x0, y0, v0, x1, y1, v1, x);
                return y;
            }
        }
        return 0;
    }
    
    float y_der(float x) {
        for (int i = 0; i < controlPoints.size()-1; i++) {
            if (controlPoints[i].x <= x && x <= controlPoints[i + 1].x + 0.0001) {
                float y0 = controlPoints[i].y;
                float y1 = controlPoints[i + 1].y;
                float x0 = controlPoints[i].x;
                float x1 = controlPoints[i + 1].x;
                float v0 = get_v(i);
                float v1 = get_v(i + 1);
                float y = interpolate_derivate(x0, y0, v0, x1, y1, v1, x);
                return y;
            }
        }
        return 0;
    }
    
    std::vector<vec2> triangles;
    
    void triangulate() {
        std::vector<vec2> splineSegments;
        for (float x = -1.0f; x <= 1.0f + 0.0001; x += 0.001) {
            float y_ = y(x);
            splineSegments.push_back(vec2(x, y_));
        }
        
        triangles.clear();
        for (vec2 spsg : splineSegments) {
            triangles.push_back(vec2(spsg.x, -1));
            triangles.push_back(vec2(spsg.x, spsg.y));
        }
    }
} ground(-0.5, -0.5), mountains(3, 0);

class Bike {
public:
    const float m = 0.8; // mass
    std::vector<vec2> verticesWheel;
    std::vector<vec2> verticesBody;
    float r = 0.1; // wheel radius
    float headR = 0.05; // head radius
    float bodyLen = 0.3; // distance between head and wheel centre
    float angle = 0; // wheel rotation
    float legR = 0.05; // Foot (or pedal) radius
    
    vec2 legStart = vec2(0, r + 0.05); // Waist point
    vec2 legEnd = vec2(cos(angle) * legR, sin(angle) * legR); // Foot point
    float legLen = (legStart.y + legR) / 2; // From waist to knee == from knee to foot
    
    Bike() {
        
        // Wheel
        verticesWheel.push_back(vec2(r, 0));
        for (float phi = 0; phi <= 2 * M_PI; phi += 0.01) {
            verticesWheel.push_back(vec2(cos(phi) * r, sin(phi) * r));
            verticesWheel.push_back(vec2(cos(phi) * r, sin(phi) * r));
        }
        verticesWheel.push_back(vec2(r, 0));
        
        for (int i = 0; i < 8; i++) {
            float phi = 2 * M_PI / 8 * i;
            verticesWheel.push_back(vec2(cos(phi) * r, sin(phi) * r));
            verticesWheel.push_back(vec2(0, 0));
        }
        
        // Head
        verticesBody.push_back(vec2(headR, bodyLen));
        for (float phi = 0; phi <= 2 * M_PI; phi += 0.01) {
            verticesBody.push_back(vec2(cos(phi) * headR, sin(phi) * headR + bodyLen));
            verticesBody.push_back(vec2(cos(phi) * headR, sin(phi) * headR + bodyLen));
        }
        verticesBody.push_back(vec2(headR, bodyLen));
        
        // Torso
        verticesBody.push_back(vec2(0, bodyLen - headR));
        verticesBody.push_back(legStart);
        
        // Upper leg
        verticesBody.push_back(legStart);
        verticesBody.push_back(genLegMid());
        
        // Lower leg
        verticesBody.push_back(genLegMid());
        verticesBody.push_back(legEnd);
        
    }
    vec2 position; // Where does the bike touch the ground
    vec2 posdelta; // Center of the wheel minus position
    float dir = 1; // Direction, 1 or -1
    
    // Circle intersection
    vec2 genLegMid() {
        float d = length(legStart - legEnd);
        vec2 midp = (legEnd - legStart) / 2;

        float disthalf = d / 2.0;
        float y = sqrt(legLen * legLen - disthalf * disthalf);
        
        return vec2(legStart.x + midp.x - y * (legEnd.y - legStart.y) / d,
                    legStart.y + midp.y + y * (legEnd.x - legStart.x) / d);
    }
} bike;

uint32_t groundVBO, bikeWheelVBO, bikeVBO/*One vbo per object*/, textureVBOs[2] /*Different vao and two vbos for texturing bg*/;

// Generates backgorund
void generateTexture() {
    std::vector<vec4> textureData(windowHeight * windowWidth);
    for (uint16_t y = 0; y < windowHeight; y++) {
        for (uint16_t x = 0; x < windowWidth; x++) {
            float x_vert = (float)x / windowWidth * 2 - 1;
            float y_vert = (float)y / windowHeight * 2 - 1;
            float y_ref_vert = mountains.y(x_vert);
            uint16_t yref = (y_ref_vert + 1) / 2 * windowHeight;
            float r, g, b;
            float mul = 5;
            if (yref < y) {
                r = cosf(fmin(abs(x_vert * mul), 2.4f)) + cosf(fmin(abs((y_vert - 0.4) * mul), 2.4f));
                b = 0.6;
                g = fmax(r / 2, b);
            } else {
                g = pow((y_vert + 1) / 2, 0.5);
                b = pow((y_vert + 1) / 2, 0.5);
                r = fmin(g, b);
            }
            textureData[y * windowWidth + x] = vec4(r, g, b, 1);
        }
    }
    
    
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &textureData[0]);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glGenVertexArrays(1, &vaot);
    glBindVertexArray(vaot);
    glGenBuffers(2, textureVBOs);
    glBindBuffer(GL_ARRAY_BUFFER, textureVBOs[0]);
    float vtxs[] = {-1, -1, 1, -1, 1, 1, -1, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(vtxs), vtxs, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    
    glBindBuffer(GL_ARRAY_BUFFER, textureVBOs[1]);
    float uvs[] = {0, 0, 1, 0, 1, 1, 0, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    
}

// Create new vbo in binded vao
uint32_t gen_vbo(std::vector<vec2>& tris) {
    unsigned int vbo;        // vertex buffer object
    glGenBuffers(1, &vbo);    // Generate 1 buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    vec2* vertices = tris.data();
    glBufferData(GL_ARRAY_BUFFER,
        tris.size() * sizeof(vec2),
        vertices,
        GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);  // AttribArray 0
    glVertexAttribPointer(0,       // vbo -> AttribArray 0
        2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
        0, NULL);              // stride, offset: tightly packed
    return vbo;
}

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    // create programs for the GPU
    gpuProgram.Create(vertexSource, fragmentSource, "outColor");
    textureProgram.Create(vertexTexture, fragmentTexture, "outColor");
    
    ground.triangulate();
    groundVBO = gen_vbo(ground.triangles);
    
    mountains.addPoint(-0.2, 0.4);
    mountains.addPoint(0.7, 0.6);
    mountains.addPoint(0.0, -0.1);
    
    mountains.triangulate();
    
    bikeWheelVBO = gen_vbo(bike.verticesWheel);
    bikeVBO = gen_vbo(bike.verticesBody);
    
    generateTexture();
}

void renderGround() {
    int location = glGetUniformLocation(gpuProgram.getId(), "color");
    
    glUniform3f(location, 0.2, 0.2, 0.3);

    float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                              0, 1, 0, 0,    // row-major!
                              0, 0, 1, 0,
                              isSpaceOn ? -bike.position.x-bike.posdelta.x : 0, isSpaceOn ? -bike.position.y-bike.posdelta.y : 0, 0, 1 };

    location = glGetUniformLocation(gpuProgram.getId(), "MVP");
    glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, groundVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,
        2, GL_FLOAT, GL_FALSE,
        0, NULL);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, ground.triangles.size());
}

void renderBike() {
    // Wheel rendering -- with rotation
    int location = glGetUniformLocation(gpuProgram.getId(), "color");
    
    glUniform3f(location, 1, 0, 0);
    
    float MVPtransf[4][4] = { cosf(bike.angle), -sinf(bike.angle), 0, 0,    // MVP matrix,
                              sinf(bike.angle), cosf(bike.angle), 0, 0,    // row-major!
                              0, 0, 1, 0,
                              isSpaceOn ? 0 : bike.position.x + bike.posdelta.x, isSpaceOn ? 0 : bike.position.y + bike.posdelta.y, 0, 1 };

    location = glGetUniformLocation(gpuProgram.getId(), "MVP");
    glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, bikeWheelVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,
        2, GL_FLOAT, GL_FALSE,
        0, NULL);

    glBindVertexArray(vao);
    glLineWidth(2);
    glDrawArrays(GL_LINES, 0, bike.verticesWheel.size());
    
    
    // Biker rendering -- no rotation
    float MVPtransf2[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                              0, 1, 0, 0,    // row-major!
                              0, 0, 1, 0,
                              isSpaceOn ? 0 : bike.position.x + bike.posdelta.x, isSpaceOn ? 0 : bike.position.y + bike.posdelta.y, 0, 1 };

    location = glGetUniformLocation(gpuProgram.getId(), "MVP");
    glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf2[0][0]);
    
    glBindBuffer(GL_ARRAY_BUFFER, bikeVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,
        2, GL_FLOAT, GL_FALSE,
        0, NULL);
    glDrawArrays(GL_LINES, 0, bike.verticesBody.size());
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    // Rendering background
    textureProgram.Use();
    glBindVertexArray(vaot);
    int location = glGetUniformLocation(textureProgram.getId(), "samplerUnit");
    if (location >= 0) {
        glUniform1i(location, 0);
        glActiveTexture(GL_TEXTURE0 + 0);
        glBindTexture(GL_TEXTURE_2D, textureID);
    } else {
        printf("uniform samplerUnit cannot be set\n");
    }
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    
    // Rendering everything else
    gpuProgram.Use();
    glBindVertexArray(vao);
    
    renderGround();
    renderBike();

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == ' ') {
        isSpaceOn = !isSpaceOn;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;
    
    if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
        ground.addPoint(vec2(cX, cY) + (isSpaceOn ? bike.position + bike.posdelta : vec2(0,0)));
        ground.triangulate();
        
        groundVBO = gen_vbo(ground.triangles);
        
        glutPostRedisplay();
    }
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    
    const float dt = 0.01;
    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        float der = ground.y_der(bike.position.x); // Derivate
        float alpha = atan(der); // Direction in angles
        float v = (8.0f * bike.dir - bike.m * g * sin(alpha)) / 10000; // speed
        float ds = v * Dt;
        float dx = ds / sqrt(1 + pow(der, 2));
        bike.angle += ds / bike.r;
        
        bike.position = vec2(bike.position.x + dx, ground.y(bike.position.x + dx));
        bike.posdelta = vec2(-sin(alpha) * bike.r, cos(alpha) * bike.r);
        
        // Turning back at the edges
        if (bike.position.x > 1 ) {
            bike.dir = -1;
            bike.position.x = 1;
        }
        if (bike.position.x < -1) {
            bike.dir = 1;
            bike.position.x = -1;
        }
        
    }
    
    // Calculating leg positions
    bike.legEnd = vec2(-cos(bike.angle) * bike.legR, sin(bike.angle) * bike.legR);
    bike.verticesBody[bike.verticesBody.size() - 1] = bike.legEnd;
    vec2 legMid = bike.genLegMid();
    bike.verticesBody[bike.verticesBody.size() - 2] = legMid;
    bike.verticesBody[bike.verticesBody.size() - 3] = legMid;
    
    // overwriting vbo with new leg data
    glBindBuffer(GL_ARRAY_BUFFER, bikeVBO);
    vec2* data = bike.verticesBody.data();
    glBufferData(GL_ARRAY_BUFFER,
        bike.verticesBody.size() * sizeof(vec2),
        data,
        GL_STATIC_DRAW);
    
    glutPostRedisplay();
}
