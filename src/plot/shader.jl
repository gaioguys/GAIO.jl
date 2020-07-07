function compile_shader(src::String, ty)
    shader = glCreateShader(ty)
    # Attempt to compile the shader
    glShaderSource(shader, 1, convert(Ptr{UInt8}, pointer([convert(Ptr{GLchar}, pointer(src))])), C_NULL)
    glCompileShader(shader)

    # Get the compile status
    status = GLint[0]
    glGetShaderiv(shader, GL_COMPILE_STATUS, status)

    # Fail on error
    if first(status) == GL_FALSE
        len = GLint[0]
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, len)

        buffer = zeros(GLchar, first(len))
        sizei = GLsizei[0]
        glGetShaderInfoLog(shader, first(len), sizei, buffer)

        error(unsafe_string(pointer(buffer), first(sizei)))
    end

    return shader
end

function link_program(vs, fs)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    # Get the link status
    status = GLint[0]
    glGetProgramiv(program, GL_LINK_STATUS, status)

    # Fail on error
    if first(status) == GL_FALSE
        len = GLint[0]
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, len)

        buffer = zeros(GLchar, first(len))
        sizei = GLsizei[0]
        glGetProgramInfoLog(program, first(len), sizei, buffer)

        error(unsafe_string(pointer(buffer), first(sizei)))
    end

    return program
end
