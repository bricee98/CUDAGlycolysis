#include <GL/glew.h>
#include <GL/gl.h>
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

// Remove this line as it's not needed anymore
// #include "MoleculeType.h"

// Function prototypes
void initVisualization();
void renderSimulation(const SimulationSpace& space, const std::vector<Molecule>& molecules);
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

// Shader source code (we'll implement these later)
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform float size;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        gl_PointSize = size;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    uniform vec3 color;
    
    void main()
    {
        FragColor = vec4(color, 1.0);
    }
)";

// Global variables for simulation control
extern bool isPaused;

// Global variables for text rendering
FT_Library ft;
FT_Face face;
GLuint textVAO, textVBO;
GLuint textShaderProgram;

// Function to handle key input
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        isPaused = !isPaused;
    }
}

// Function to handle mouse input for camera control
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    static float lastX = 400, lastY = 300;
    static bool firstMouse = true;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    cameraRotationY += xoffset;
    cameraRotationX += yoffset;

    if (cameraRotationX > 89.0f) cameraRotationX = 89.0f;
    if (cameraRotationX < -89.0f) cameraRotationX = -89.0f;
}

// Function to handle mouse scroll for zoom
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    printf("Scroll callback called: %f\n", yoffset);
    printf("Camera distance: %f\n", cameraDistance);
    cameraDistance -= yoffset * 5.0f;
    if (cameraDistance < 10.0f) cameraDistance = 10.0f;
    if (cameraDistance > 200.0f) cameraDistance = 200.0f;
}

// Define getMoleculeColor and getMoleculeSize before they're used
glm::vec3 getMoleculeColor(MoleculeType type) {
    switch (type) {
        case GLUCOSE: return glm::vec3(1.0f, 0.0f, 0.0f);  // Red
        case ATP: return glm::vec3(0.0f, 1.0f, 0.0f);      // Green
        case ADP: return glm::vec3(0.0f, 0.0f, 1.0f);      // Blue
        // Add more cases for other molecule types
        default: return glm::vec3(0.5f, 0.5f, 0.5f);       // Gray for unknown types
    }
}

float getMoleculeSize(MoleculeType type) {
    switch (type) {
        case GLUCOSE: return 1.0f;
        case ATP: return 1.2f;
        case ADP: return 1.1f;
        // Add more cases for other molecule types
        default: return 0.8f;
    }
}

void initTextRendering() {
    if (FT_Init_FreeType(&ft)) {
        fprintf(stderr, "Could not init FreeType Library\n");
        return;
    }

    // Update the font path to a more common location or use a relative path
    const char* fontPath = "C:/Windows/Fonts/arial.ttf";  // For Windows
    // const char* fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";  // For Linux
    // const char* fontPath = "/System/Library/Fonts/Helvetica.ttc";  // For macOS

    printf("Attempting to load font: %s\n", fontPath);
    int error = FT_New_Face(ft, fontPath, 0, &face);
    if (error == FT_Err_Unknown_File_Format) {
        fprintf(stderr, "Font file found, but format unsupported\n");
    } else if (error) {
        fprintf(stderr, "Font file not found or other error\n");
    }

    FT_Set_Pixel_Sizes(face, 0, 48);

    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    printf("Text rendering initialized\n");

    // Create and compile text shader program
    // ... (implement this part)
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

void initVisualization() {
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(1024, 768, "Molecular Simulation Visualization", NULL, NULL);
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

    // Set up vertex attributes (position, color)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Set up projection and view matrices
    projection = glm::perspective(glm::radians(45.0f), 1024.0f / 768.0f, 0.1f, 1000.0f);
    view = glm::lookAt(glm::vec3(0.0f, 0.0f, cameraDistance),
                       glm::vec3(0.0f, 0.0f, 0.0f),
                       glm::vec3(0.0f, 1.0f, 0.0f));

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Set up input callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Enable cursor capture for camera control
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Initialize text rendering
    initTextRendering();
}

void renderSimulation(const SimulationSpace& space, const std::vector<Molecule>& molecules) {
    printf("Rendering simulation\n");

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use the shader program
    glUseProgram(shaderProgram);

    // Update view matrix based on camera rotation and distance
    glm::mat4 view = glm::mat4(1.0f);
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -cameraDistance));
    view = glm::rotate(view, glm::radians(cameraRotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, glm::radians(cameraRotationY), glm::vec3(0.0f, 1.0f, 0.0f));

    // Set projection and view matrices in the shader
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Render each molecule
    for (const auto& molecule : molecules) {
        // Get the position of the molecule
        float x, y, z;
        molecule.getPosition(x, y, z);
        glm::vec3 position(x, y, z);

        // Set model matrix for each molecule
        glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
        GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        // Set color and size based on molecule type
        glm::vec3 color = getMoleculeColor(molecule.getType());
        float size = getMoleculeSize(molecule.getType());
        GLint colorLoc = glGetUniformLocation(shaderProgram, "color");
        GLint sizeLoc = glGetUniformLocation(shaderProgram, "size");
        glUniform3fv(colorLoc, 1, glm::value_ptr(color));
        glUniform1f(sizeLoc, size);

        // Draw the molecule (as a point sprite)
        glDrawArrays(GL_POINTS, 0, 1);
    }

    // Render molecule counts
    renderMoleculeCounts(molecules);

    // Swap front and back buffers
    glfwSwapBuffers(window);

    // Poll for and process events
    glfwPollEvents();
}

void renderMoleculeCounts(const std::vector<Molecule>& molecules) {
    std::map<MoleculeType, int> moleculeCounts;
    for (const auto& molecule : molecules) {
        moleculeCounts[molecule.getType()]++;
    }

    int yOffset = 10;
    for (const auto& pair : moleculeCounts) {
        MoleculeType type = pair.first;
        int count = pair.second;
        std::string text = std::string(getMoleculeTypeName(type)) + ": " + std::to_string(count);
        renderText(text, 10, yOffset, 0.5f, glm::vec3(1.0f, 1.0f, 1.0f));
        yOffset += 20;
    }
}

void renderText(const std::string &text, float x, float y, float scale, glm::vec3 color) {
    glUseProgram(textShaderProgram);
    glUniform3f(glGetUniformLocation(textShaderProgram, "textColor"), color.x, color.y, color.z);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(textVAO);

    for (char c : text) {
        FT_Load_Char(face, c, FT_LOAD_RENDER);
        
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

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        float xpos = x + face->glyph->bitmap_left * scale;
        float ypos = y - (face->glyph->bitmap.rows - face->glyph->bitmap_top) * scale;

        float w = face->glyph->bitmap.width * scale;
        float h = face->glyph->bitmap.rows * scale;

        float vertices[6][4] = {
            { xpos,     ypos + h,   0.0f, 0.0f },            
            { xpos,     ypos,       0.0f, 1.0f },
            { xpos + w, ypos,       1.0f, 1.0f },

            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos + w, ypos,       1.0f, 1.0f },
            { xpos + w, ypos + h,   1.0f, 0.0f }           
        };

        glBindBuffer(GL_ARRAY_BUFFER, textVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        x += (face->glyph->advance.x >> 6) * scale;

        glDeleteTextures(1, &texture);
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void cleanupVisualization() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}