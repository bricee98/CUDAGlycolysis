#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>
#include <map>
#include "SimulationSpace.h"
#include "Molecule.cuh"
#include <ft2build.h>
#include FT_FREETYPE_H
#include "visualization.h"
#include <math.h>

// Remove this line as it's not needed anymore
// #include "MoleculeType.h"

// Function prototypes
void initVisualization();
void renderSimulation();
void cleanupVisualization();
void initTextRendering();
glm::vec3 getMoleculeColor(MoleculeType type);
float getMoleculeSize(MoleculeType type);
void renderMoleculeCounts(const std::vector<Molecule>& molecules);
void renderText(const std::string &text, float x, float y, float scale, glm::vec3 color);

// Add this declaration to use the function from main.cu
const char* getMoleculeTypeName(MoleculeType type);

// Global variables
extern GLFWwindow* window;
GLuint shaderProgram;
GLuint VBO, VAO;
glm::mat4 projection, view;
float cameraDistance = 100.0f;
float cameraRotationX = 0.0f;
float cameraRotationY = 0.0f;
const float NANOMETER_TO_WORLD_SCALE = 10.0f;

// Update these constants at the top of the file
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const float TEXT_SCALE = 0.3f;
const float LINE_HEIGHT = 24.0f;
const float MAX_CAMERA_DISTANCE = 1000.0f * NANOMETER_TO_WORLD_SCALE;
const float PAN_SPEED = 50.0f * NANOMETER_TO_WORLD_SCALE;  // Increased speed for noticeable movement

// Add these constants at the top of the file
const float MIN_PARTICLE_SIZE = 0.5f; // Minimum size in nanometers
const float DISTANCE_SCALE_FACTOR = 50.0f; // Adjust to control size scaling with distance

// Add these global variables for camera position
glm::vec3 cameraPosition(0.0f, 0.0f, cameraDistance);
glm::vec3 cameraFront(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

// Add these global variables for camera orientation
float yaw = -90.0f;    // Initialized to -90.0 degrees to look towards the negative Z-axis
float pitch = 0.0f;

// Shader source code (we'll implement these later)
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform float pointSize;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        gl_PointSize = pointSize;
    }
)";

// Update the fragment shader to include a more pronounced glow effect
const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    uniform vec3 color;
    
    void main()
    {
        // Calculate distance from fragment to center of molecule
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        float dist = length(circCoord);
        
        // Core color
        vec3 coreColor = color;
        
        // Glow color (brighter version of core color)
        vec3 glowColor = color * 2.0 + vec3(0.2);
        
        // Mix core and glow based on distance
        vec3 finalColor = mix(coreColor, glowColor, smoothstep(0.5, 1.0, dist));
        
        // Fade out alpha at the edges
        float alpha = 1.0 - smoothstep(0.8, 1.0, dist);
        
        FragColor = vec4(finalColor, alpha);
    }
)";

// Add these shader source codes at the top of the file, after other shader sources
const char* textVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
    out vec2 TexCoords;

    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
        TexCoords = vertex.zw;
    }
)";

const char* textFragmentShaderSource = R"(
    #version 330 core
    in vec2 TexCoords;
    out vec4 color;

    uniform sampler2D text;
    uniform vec3 textColor;

    void main()
    {    
        vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
        color = vec4(textColor, 1.0) * sampled;
    }
)";

// Global variables for simulation control
extern bool isPaused;

// Global variables for text rendering
FT_Library ft;
FT_Face face;
GLuint textVAO, textVBO;
GLuint textShaderProgram;

// 1. Declare textProjection globally with other global variables
glm::mat4 textProjection;

// Include necessary headers
#include <map>

// Define a structure to hold information about each character
struct Character {
    GLuint TextureID;   // ID handle of the glyph texture
    glm::ivec2 Size;    // Size of glyph
    glm::ivec2 Bearing; // Offset from baseline to left/top of glyph
    GLuint Advance;     // Offset to advance to next glyph
};

// A map to store glyphs for quick access
std::map<GLchar, Character> Characters;

// At the top of visualization.cu, add this declaration to access the variable from main.cu
extern bool isRenderingPaused;

// Function to handle key input
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        isPaused = !isPaused;
    }

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        isRenderingPaused = !isRenderingPaused;
    }
}

// Declare these global variables at the top of the file
float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;
bool firstMouse = true;

// Update the mouseCallback function
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = float(xpos);
        lastY = float(ypos);
        firstMouse = false;
    }

    float xoffset = float(xpos) - lastX;
    float yoffset = lastY - float(ypos); // Reversed since y-coordinates go from bottom to top
    lastX = float(xpos);
    lastY = float(ypos);

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain the pitch angle to prevent screen flip
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    // Update cameraFront vector based on updated yaw and pitch
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

// Update the scrollCallback function to maintain zoom functionality
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    cameraDistance -= yoffset * 10.0f * NANOMETER_TO_WORLD_SCALE;
    if (cameraDistance < 10.0f * NANOMETER_TO_WORLD_SCALE)
        cameraDistance = 10.0f * NANOMETER_TO_WORLD_SCALE;
    if (cameraDistance > MAX_CAMERA_DISTANCE)
        cameraDistance = MAX_CAMERA_DISTANCE;
}

// Define getMoleculeColor and getMoleculeSize before they're used
// Define getMoleculeColor and getMoleculeSize before they're used
glm::vec3 getMoleculeColor(MoleculeType type) {
    switch (type) {
        // Substrates and products
        case GLUCOSE: return glm::vec3(1.0f, 0.0f, 0.0f);  // Red
        case ATP: return glm::vec3(0.0f, 1.0f, 0.0f);      // Green
        case ADP: return glm::vec3(0.0f, 0.0f, 1.0f);      // Blue
        case GLUCOSE_6_PHOSPHATE: return glm::vec3(1.0f, 0.5f, 0.0f);  // Orange
        case FRUCTOSE_6_PHOSPHATE: return glm::vec3(1.0f, 1.0f, 0.0f);  // Yellow
        case FRUCTOSE_1_6_BISPHOSPHATE: return glm::vec3(0.5f, 1.0f, 0.0f);  // Lime
        case DIHYDROXYACETONE_PHOSPHATE: return glm::vec3(0.0f, 1.0f, 0.5f);  // Spring Green
        case GLYCERALDEHYDE_3_PHOSPHATE: return glm::vec3(0.0f, 1.0f, 1.0f);  // Cyan
        case _1_3_BISPHOSPHOGLYCERATE: return glm::vec3(0.5f, 0.0f, 1.0f);  // Purple
        case _3_PHOSPHOGLYCERATE: return glm::vec3(1.0f, 0.0f, 1.0f);  // Magenta
        case _2_PHOSPHOGLYCERATE: return glm::vec3(1.0f, 0.5f, 0.5f);  // Pink
        case PHOSPHOENOLPYRUVATE: return glm::vec3(0.5f, 0.5f, 1.0f);  // Light Blue
        case PYRUVATE: return glm::vec3(0.5f, 0.25f, 0.0f);  // Brown
        case NAD_PLUS: return glm::vec3(0.75f, 0.75f, 0.75f);  // Light Gray
        case NADH: return glm::vec3(0.25f, 0.25f, 0.25f);  // Dark Gray
        case PROTON: return glm::vec3(1.0f, 1.0f, 1.0f);  // White
        case INORGANIC_PHOSPHATE: return glm::vec3(0.5f, 0.5f, 0.5f);  // Gray
        case WATER: return glm::vec3(0.0f, 0.5f, 1.0f);  // Light Blue
        case AMP: return glm::vec3(0.9f, 0.9f, 0.5f);  // Light Yellow
        case CITRATE: return glm::vec3(0.5f, 0.9f, 0.9f);  // Light Cyan
        case FRUCTOSE_2_6_BISPHOSPHATE: return glm::vec3(0.9f, 0.5f, 0.9f);  // Light Magenta

        // Enzymes
        case HEXOKINASE: return glm::vec3(0.8f, 0.2f, 0.2f);  // Dark Red
        case PHOSPHOGLUCOSE_ISOMERASE: return glm::vec3(0.2f, 0.8f, 0.2f);  // Dark Green
        case PHOSPHOFRUCTOKINASE_1: return glm::vec3(0.2f, 0.2f, 0.8f);  // Dark Blue
        case ALDOLASE: return glm::vec3(0.8f, 0.8f, 0.2f);  // Dark Yellow
        case TRIOSEPHOSPHATE_ISOMERASE: return glm::vec3(0.8f, 0.2f, 0.8f);  // Dark Magenta
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE: return glm::vec3(0.2f, 0.8f, 0.8f);  // Dark Cyan
        case PHOSPHOGLYCERATE_KINASE: return glm::vec3(0.6f, 0.4f, 0.2f);  // Dark Orange
        case PHOSPHOGLYCERATE_MUTASE: return glm::vec3(0.4f, 0.6f, 0.2f);  // Olive
        case ENOLASE: return glm::vec3(0.2f, 0.4f, 0.6f);  // Steel Blue
        case PYRUVATE_KINASE: return glm::vec3(0.6f, 0.2f, 0.4f);  // Maroon

        // Enzyme complexes
        case HEXOKINASE_GLUCOSE_COMPLEX: return glm::vec3(0.9f, 0.3f, 0.3f);  // Light Red
        case HEXOKINASE_GLUCOSE_ATP_COMPLEX: return glm::vec3(0.9f, 0.4f, 0.4f);  // Lighter Red
        case GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return glm::vec3(0.3f, 0.9f, 0.3f);  // Light Green
        case FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return glm::vec3(0.4f, 0.9f, 0.4f);  // Lighter Green
        case PHOSPHOFRUCTOKINASE_1_COMPLEX: return glm::vec3(0.3f, 0.3f, 0.9f);  // Light Blue
        case PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX: return glm::vec3(0.4f, 0.4f, 0.9f);  // Lighter Blue
        case FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX: return glm::vec3(0.9f, 0.9f, 0.3f);  // Light Yellow
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX: return glm::vec3(0.9f, 0.9f, 0.4f);  // Lighter Yellow
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX: return glm::vec3(0.9f, 0.9f, 0.5f);  // Even Lighter Yellow
        case DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return glm::vec3(0.9f, 0.3f, 0.9f);  // Light Magenta
        case GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return glm::vec3(0.9f, 0.4f, 0.9f);  // Lighter Magenta
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX: return glm::vec3(0.3f, 0.9f, 0.9f);  // Light Cyan
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX: return glm::vec3(0.4f, 0.9f, 0.9f);  // Lighter Cyan
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX: return glm::vec3(0.5f, 0.9f, 0.9f);  // Even Lighter Cyan
        case PHOSPHOGLYCERATE_KINASE_COMPLEX: return glm::vec3(0.7f, 0.5f, 0.3f);  // Light Orange
        case PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX: return glm::vec3(0.8f, 0.6f, 0.4f);  // Lighter Orange
        case PHOSPHOGLYCERATE_MUTASE_COMPLEX: return glm::vec3(0.5f, 0.7f, 0.3f);  // Light Olive
        case ENOLASE_COMPLEX: return glm::vec3(0.3f, 0.5f, 0.7f);  // Light Steel Blue
        case PYRUVATE_KINASE_COMPLEX: return glm::vec3(0.7f, 0.3f, 0.5f);  // Light Maroon
        case PYRUVATE_KINASE_ADP_COMPLEX: return glm::vec3(0.8f, 0.4f, 0.6f);  // Lighter Maroon

        default: return glm::vec3(0.5f, 0.5f, 0.5f);  // Gray for unknown types
    }
}

float getMoleculeSize(MoleculeType type) {
    switch (type) {
        // Substrates and products
        case GLUCOSE: return 1.0f;
        case ATP: return 1.2f;
        case ADP: return 1.1f;
        case GLUCOSE_6_PHOSPHATE: return 1.1f;
        case FRUCTOSE_6_PHOSPHATE: return 1.1f;
        case FRUCTOSE_1_6_BISPHOSPHATE: return 1.2f;
        case DIHYDROXYACETONE_PHOSPHATE: return 0.9f;
        case GLYCERALDEHYDE_3_PHOSPHATE: return 0.9f;
        case _1_3_BISPHOSPHOGLYCERATE: return 1.0f;
        case _3_PHOSPHOGLYCERATE: return 0.9f;
        case _2_PHOSPHOGLYCERATE: return 0.9f;
        case PHOSPHOENOLPYRUVATE: return 0.9f;
        case PYRUVATE: return 0.8f;
        case NAD_PLUS: return 1.1f;
        case NADH: return 1.1f;
        case PROTON: return 0.3f;
        case INORGANIC_PHOSPHATE: return 0.7f;
        case WATER: return 0.5f;
        case AMP: return 1.0f;
        case CITRATE: return 1.0f;
        case FRUCTOSE_2_6_BISPHOSPHATE: return 1.2f;

        // Enzymes (generally larger than substrates and products)
        case HEXOKINASE: return 2.0f;
        case PHOSPHOGLUCOSE_ISOMERASE: return 1.9f;
        case PHOSPHOFRUCTOKINASE_1: return 2.1f;
        case ALDOLASE: return 2.0f;
        case TRIOSEPHOSPHATE_ISOMERASE: return 1.8f;
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE: return 2.2f;
        case PHOSPHOGLYCERATE_KINASE: return 2.0f;
        case PHOSPHOGLYCERATE_MUTASE: return 1.9f;
        case ENOLASE: return 2.0f;
        case PYRUVATE_KINASE: return 2.1f;

        // Enzyme complexes (slightly larger than individual enzymes)
        case HEXOKINASE_GLUCOSE_COMPLEX: return 2.1f;
        case HEXOKINASE_GLUCOSE_ATP_COMPLEX: return 2.2f;
        case GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return 2.0f;
        case FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return 2.0f;
        case PHOSPHOFRUCTOKINASE_1_COMPLEX: return 2.2f;
        case PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX: return 2.3f;
        case FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX: return 2.1f;
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX: return 2.1f;
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX: return 2.2f;
        case DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return 1.9f;
        case GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return 1.9f;
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX: return 2.3f;
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX: return 2.4f;
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX: return 2.5f;
        case PHOSPHOGLYCERATE_KINASE_COMPLEX: return 2.1f;
        case PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX: return 2.2f;
        case PHOSPHOGLYCERATE_MUTASE_COMPLEX: return 2.0f;
        case ENOLASE_COMPLEX: return 2.1f;
        case PYRUVATE_KINASE_COMPLEX: return 2.2f;
        case PYRUVATE_KINASE_ADP_COMPLEX: return 2.3f;

        default: return 0.8f;  // Default size for unknown types
    }
}



void initTextRendering() {
    if (FT_Init_FreeType(&ft)) {
        fprintf(stderr, "Could not init FreeType Library\n");
        return;
    }

    // Update the font path to a more common location or use a relative path
    const char* fontPath = "Roboto-Thin.ttf";  // For Windows
    // const char* fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";  // For Linux
    // const char* fontPath = "/System/Library/Fonts/Helvetica.ttc";  // For macOS

    printf("Attempting to load font: %s\n", fontPath);
    int error = FT_New_Face(ft, fontPath, 0, &face);
    if (error == FT_Err_Unknown_File_Format) {
        fprintf(stderr, "Font file found, but format unsupported\n");
    } else if (error) {
        fprintf(stderr, "Font file not found or other error\n");
    }

    // Increase the font size (originally 48)
    FT_Set_Pixel_Sizes(face, 0, 64);


    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // 1. Assign the projection matrix using the correct window dimensions
    textProjection = glm::ortho(0.0f, static_cast<float>(WINDOW_WIDTH), 0.0f, static_cast<float>(WINDOW_HEIGHT));
    glUseProgram(textShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(textShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(textProjection));

    // Load first 128 ASCII characters
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Disable byte-alignment restriction

    for (GLubyte c = 0; c < 128; c++) {
        // Load character glyph
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            fprintf(stderr, "ERROR::FREETYPE: Failed to load Glyph %c\n", c);
            continue;
        }
        // Generate texture
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );
        // Set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Prevent artifacts
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Linear filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Store character for later use
        Character character = {
            texture,
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
            static_cast<GLuint>(face->glyph->advance.x)
        };
        Characters.insert(std::pair<GLchar, Character>(c, character));
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    // Clean up FreeType resources
    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    // Create and compile text shader program
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &textVertexShaderSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &textFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    textShaderProgram = glCreateProgram();
    glAttachShader(textShaderProgram, vertexShader);
    glAttachShader(textShaderProgram, fragmentShader);
    glLinkProgram(textShaderProgram);

    // Check for shader compilation and linking errors
    GLint success;
    GLchar infoLog[512];

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
    }

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
    }

    glGetProgramiv(textShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(textShaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Set up the projection matrix for text rendering
    textProjection = glm::ortho(0.0f, static_cast<float>(WINDOW_WIDTH), 0.0f, static_cast<float>(WINDOW_HEIGHT));
    glUseProgram(textShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(textShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(textProjection));

    printf("Text rendering initialized\n");
}

// Add this function to compile shaders
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// Global variables for sphere rendering
GLuint sphereVAO = 0;
GLuint sphereVBO = 0;
GLuint sphereEBO = 0;
unsigned int indexCount;

// Function to generate sphere mesh
void generateSphere(float radius, unsigned int sectorCount, unsigned int stackCount) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius;    // vertex normal
    float s, t;                                     // vertex texCoord

    float sectorStep = 2 * 3.14159265359 / sectorCount;
    float stackStep = 3.14159265359 / stackCount;
    float sectorAngle, stackAngle;

    for(unsigned int i = 0; i <= stackCount; ++i)
    {
        stackAngle = 3.14159265359 / 2 - i * stackStep;        // from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        // add (sectorCount+1) vertices per stack
        for(unsigned int j = 0; j <= sectorCount; ++j)
        {
            sectorAngle = j * sectorStep;           // from 0 to 2pi

            // vertex position
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    // indices
    unsigned int k1, k2;
    for(unsigned int i = 0; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(unsigned int j = 0; j < sectorCount; ++j, ++k1, ++k2)
        {
            if(i != 0)
            {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            if(i != (stackCount-1))
            {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }

    indexCount = indices.size();

    // Generate buffers
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // Vertex Positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

// Update the drawSphere function
void drawSphere() {
    if(sphereVAO == 0)
        generateSphere(1.0f, 36, 18); // Generate a unit sphere once

    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// Add this callback function to handle window resizing
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // Adjust the viewport to the new window dimensions
    glViewport(0, 0, width, height);

    // Update the projection matrix with the new aspect ratio
    projection = glm::perspective(glm::radians(45.0f), static_cast<float>(width) / height, 0.1f, 1000.0f * NANOMETER_TO_WORLD_SCALE);

    // Update the text projection matrix
    textProjection = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height));

    // Update the text shader with the new projection matrix
    glUseProgram(textShaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(textShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(textProjection));
}

void initVisualization() {
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Molecular Simulation Visualization", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Add this line
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Create and compile shaders
    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);

    // Set up vertex buffer and vertex array object
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Set up a single vertex at the origin
    float vertex[] = {0.0f, 0.0f, 0.0f};
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), vertex, GL_STATIC_DRAW);

    // Set up vertex attributes (position only)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Adjust the camera distance to match the new scale
    cameraDistance = 100.0f * NANOMETER_TO_WORLD_SCALE;

    // Set up projection and view matrices
    projection = glm::perspective(glm::radians(45.0f), static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT, 0.1f, 1000.0f * NANOMETER_TO_WORLD_SCALE);
    view = glm::lookAt(glm::vec3(0.0f, 0.0f, cameraDistance),
                       glm::vec3(0.0f, 0.0f, 0.0f),
                       glm::vec3(0.0f, 1.0f, 0.0f));

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Set up input callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Add this line to set the framebuffer size callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Enable cursor capture for camera control
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Enable blending for text rendering
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Initialize text rendering
    initTextRendering();

    // Generate sphere mesh with a default radius (will be scaled per molecule)
    generateSphere(1.0f, 36, 18); // Base radius of 1.0f

    // Adjust the initial camera position to match the new scale
    cameraPosition = glm::vec3(0.0f, 0.0f, cameraDistance);
    cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);  // Ensure initial front vector
}

// Update the processInput function to handle continuous key input and separate vertical movement
void processInput(GLFWwindow* window, float deltaTime) {
    float cameraSpeed = PAN_SPEED * deltaTime;

    // Move forward and backward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPosition += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPosition -= cameraSpeed * cameraFront;

    // Strafe left and right
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPosition -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPosition += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

    // Move up and down
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPosition += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
        cameraPosition -= cameraSpeed * cameraUp;
}

// Add a callback for handling mouse movement to adjust camera rotation


// Update the renderSimulation function
void renderSimulation(const SimulationSpace& space, const std::vector<Molecule>& molecules, float total_simulated_time, float deltaTime) {
    //printf("Rendering simulation\n");

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use the shader program
    glUseProgram(shaderProgram);

    // Process input
    processInput(window, deltaTime);

    // Update view matrix based on camera position and front vector
    glm::mat4 view = glm::lookAt(cameraPosition,
                                 cameraPosition + cameraFront,
                                 cameraUp);

    // Set projection and view matrices in the shader
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Set the view position uniform
    GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    glUniform3fv(viewPosLoc, 1, glm::value_ptr(cameraPosition));

    // Render each molecule
    for (const auto& molecule : molecules) {
        // Get the position of the molecule
        float x, y, z;
        molecule.getPosition(x, y, z);
        glm::vec3 position(x * NANOMETER_TO_WORLD_SCALE, y * NANOMETER_TO_WORLD_SCALE, z * NANOMETER_TO_WORLD_SCALE);

        // Set color based on molecule type
        glm::vec3 color = getMoleculeColor(molecule.getType());
        float size = getMoleculeSize(molecule.getType());

        // Create model matrix
        glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
        model = glm::scale(model, glm::vec3(size * NANOMETER_TO_WORLD_SCALE));

        // Set uniforms
        GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
        glUniform3fv(colorLoc, 1, glm::value_ptr(color));

        // Draw the molecule as a sphere
        drawSphere();
    }

    // Set up OpenGL state for 2D rendering
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Render total simulated time at the top left
    char timeText[50];
    sprintf(timeText, "Simulated Time: %.6f s", total_simulated_time);
    renderText(timeText, 10.0f, 30.0f, TEXT_SCALE, glm::vec3(1.0f, 1.0f, 1.0f));

    // Render molecule counts
    renderMoleculeCounts(molecules);

    // Restore OpenGL state
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void renderMoleculeCounts(const std::vector<Molecule>& molecules) {
    std::map<MoleculeType, int> moleculeCounts;
    for (const auto& molecule : molecules) {
        moleculeCounts[molecule.getType()]++;
    }

    float yOffset = 2 * LINE_HEIGHT; // Start below the simulated time text
    for (const auto& pair : moleculeCounts) {
        MoleculeType type = pair.first;
        int count = pair.second;
        std::string text = std::string(getMoleculeTypeName(type)) + ": " + std::to_string(count);
        renderText(text, 10.0f, yOffset, TEXT_SCALE, glm::vec3(1.0f, 1.0f, 1.0f));
        yOffset += LINE_HEIGHT;
    }
}

void renderText(const std::string &text, float x, float y, float scale, glm::vec3 color) {
    // Activate corresponding render state
    glUseProgram(textShaderProgram);
    glUniform3f(glGetUniformLocation(textShaderProgram, "textColor"), color.x, color.y, color.z);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(textVAO);

    // Disable depth testing and enable blending
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Initialize cursor position
    float x_cursor = x;

    // Increase the text scale (adjust as needed)
    scale *= 1.5f;

    // Iterate over all characters
    std::string::const_iterator c;
    for (c = text.begin(); c != text.end(); c++) {
        Character ch = Characters[*c];

        // Adjust the y-position calculation
        float xpos = x_cursor + ch.Bearing.x * scale;
        float ypos = WINDOW_HEIGHT - (y - ch.Bearing.y * scale);  // Invert y-coordinate

        float w = ch.Size.x * scale;
        float h = ch.Size.y * scale;

        // Update VBO for each character
        float vertices[6][4] = {
            { xpos,     ypos - h,   0.0f, 1.0f },
            { xpos + w, ypos,       1.0f, 0.0f },
            { xpos,     ypos,       0.0f, 0.0f },

            { xpos,     ypos - h,   0.0f, 1.0f },
            { xpos + w, ypos - h,   1.0f, 1.0f },
            { xpos + w, ypos,       1.0f, 0.0f }
        };

        // Render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.TextureID);

        // Update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Advance cursor for the next character
        x_cursor += (ch.Advance >> 6) * scale;
    }

    // Re-enable depth testing and disable blending after text rendering
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST); // Ensure depth testing is re-enabled

    glUseProgram(0); // Optional: unbind any shader program
}

void cleanupVisualization() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();

    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
}