mutable struct WindowState
    width::Int
    height::Int
end

mutable struct MouseState
    pos::Tuple{Float64,Float64}
    pressed::Bool
    scroll::Float64
end

struct CameraState
    offset::Vector{Float32}
    scale::Vector{Float32}
end

function glGenOne(glGenFn)
    id = pointer(GLuint[0])
    glGenFn(1, id)
    return unsafe_load(id)
end

glGenBuffer() = glGenOne(glGenBuffers)
glGenVertexArray() = glGenOne(glGenVertexArrays)

"""
    plot(boxes::Vector{Box{2,Float32}})

Plot a set of boxes with `N=2` on a two-dimensional surface and open a new window to look at the result.
"""
function plot(boxes::Vector{Box{2,Float32}})
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    GLFW.WindowHint(GLFW.RESIZABLE, true)
    GLFW.WindowHint(GLFW.SAMPLES, 8)

    window = GLFW.CreateWindow(640, 640, "plot")
    GLFW.MakeContextCurrent(window)

    glEnable(GL_MULTISAMPLE)

    vertex_shader = compile_shader(read(joinpath(@__DIR__, "box2d.vert"), String), GL_VERTEX_SHADER)
    fragment_shader = compile_shader(read(joinpath(@__DIR__, "box2d.frag"), String), GL_FRAGMENT_SHADER)
    shader_program = link_program(vertex_shader, fragment_shader)

    camera_offset_loc = glGetUniformLocation(shader_program, "camera_offset")
    camera_scale_loc = glGetUniformLocation(shader_program, "camera_scale")

    camera_state = CameraState(Float32[0.0, 0.0], Float32[1.0, 1.0])
    mouse_state = MouseState((0.0, 0.0), false, 0.0)
    window_state = WindowState(640, 640)

    mouse_button_callback = let mouse_state = mouse_state
        (window, button, action, mods) -> begin
            if (button == GLFW.MOUSE_BUTTON_LEFT && action == GLFW.PRESS)
                mouse_state.pressed = true
            end

            if (button == GLFW.MOUSE_BUTTON_LEFT && action == GLFW.RELEASE)
                mouse_state.pressed = false
            end
        end
    end

    cursor_position_callback = let mouse_state = mouse_state
        (window, xpos, ypos) -> begin
            mouse_state.pos = (xpos, ypos)
        end
    end

    scroll_callback = let mouse_state = mouse_state
        (window, xoffset, yoffset) -> begin
            mouse_state.scroll += yoffset
        end
    end

    framebuffer_size_callback = let window_state = window_state
        (window, width, height) -> begin
            window_state.width = width
            window_state.height = height
            glViewport(0, 0, width, height)
        end
    end

    GLFW.SetScrollCallback(window, scroll_callback)
    GLFW.SetCursorPosCallback(window, cursor_position_callback)
    GLFW.SetMouseButtonCallback(window, mouse_button_callback)
    GLFW.SetFramebufferSizeCallback(window, framebuffer_size_callback)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    glUseProgram(shader_program)

    vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(boxes),
        boxes,
        GL_STATIC_DRAW,
    )

    vao = glGenVertexArray()
    glBindVertexArray(vao)
    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        16,
        Ptr{Cvoid}(0),
    )
    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        16,
        Ptr{Cvoid}(8),
    )

    glVertexAttribDivisor(0, 1)
    glVertexAttribDivisor(1, 1)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendEquation(GL_FUNC_ADD)

    last_pos = mouse_state.pos
    last_scroll = mouse_state.scroll

    while !GLFW.WindowShouldClose(window)
        m_to_c = xy -> (2.0*xy[1] / window_state.width - 1.0, 1.0 - 2.0*xy[2] / window_state.height)

        if mouse_state.pressed
            move_diff = collect(mouse_state.pos .- last_pos)
            move_diff[2] *= -1

            move_diff[1] *= 2.0 / window_state.width
            move_diff[2] *= 2.0 / window_state.height

            camera_state.offset .+= move_diff
        end

        if mouse_state.scroll != last_scroll
            yoffset = mouse_state.scroll - last_scroll

            if yoffset > 0.5
                camera_state.scale .*= 1.2

                camera_state.offset .-= m_to_c(mouse_state.pos)
                camera_state.offset .*= 1.2
                camera_state.offset .+= m_to_c(mouse_state.pos)
            elseif yoffset < -0.5
                camera_state.scale .*= inv(1.2)

                camera_state.offset .-= m_to_c(mouse_state.pos)
                camera_state.offset .*= inv(1.2)
                camera_state.offset .+= m_to_c(mouse_state.pos)
            end
        end

        camera_state.scale[2] = camera_state.scale[1] * window_state.width / window_state.height

        last_pos = mouse_state.pos
        last_scroll = mouse_state.scroll

        glUniform2f(camera_offset_loc, camera_state.offset[1], camera_state.offset[2])
        glUniform2f(camera_scale_loc, camera_state.scale[1], camera_state.scale[2])

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, length(boxes))

        GLFW.SwapBuffers(window)

        # change to wait events?
        GLFW.PollEvents()
    end

    GLFW.DestroyWindow(window)
    return nothing
end

function projection_matrix(fovy, aspect, zNear, zFar)
    f = 1.0 / tan(fovy / 2.0)

    return [
        f / aspect 0 0                                    0;
        0          f 0                                    0;
        0          0 (zFar + zNear) / (zNear - zFar)     -1;
        0          0 (2 * zFar * zNear) / (zNear - zFar)  0;
    ]
end

"""
    plot(boxes::Vector{Box{3,Float32}})

Plot a set of boxes with `N=3` into a three-dimensional space and open a new window to look at the result.
"""
function plot(boxes::Vector{Box{3,Float32}})
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    GLFW.WindowHint(GLFW.RESIZABLE, true)

    window = GLFW.CreateWindow(640, 640, "plot")
    GLFW.MakeContextCurrent(window)

    # Enable depth test
    glEnable(GL_DEPTH_TEST)
    # Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS)

    vertex_shader = compile_shader(read(joinpath(@__DIR__, "box3d.vert"), String), GL_VERTEX_SHADER)
    fragment_shader = compile_shader(read(joinpath(@__DIR__, "box3d.frag"), String), GL_FRAGMENT_SHADER)
    shader_program = link_program(vertex_shader, fragment_shader)

    camera_loc = glGetUniformLocation(shader_program, "camera")

    window_state = WindowState(640, 640)
    camera_state = CameraState(Float32[0.0, 0.0], Float32[1.0, 1.0])
    mouse_state = MouseState((0.0, 0.0), false, 0.0)

    mouse_button_callback = let mouse_state = mouse_state
        (window, button, action, mods) -> begin
            if (button == GLFW.MOUSE_BUTTON_LEFT && action == GLFW.PRESS)
                mouse_state.pressed = true
            end

            if (button == GLFW.MOUSE_BUTTON_LEFT && action == GLFW.RELEASE)
                mouse_state.pressed = false
            end
        end
    end

    cursor_position_callback = let mouse_state = mouse_state
        (window, xpos, ypos) -> begin
            mouse_state.pos = (xpos, ypos)
        end
    end

    scroll_callback = let mouse_state = mouse_state
        (window, xoffset, yoffset) -> begin
            mouse_state.scroll += yoffset
        end
    end

    key_state = Dict{Int,Bool}()

    key_callback = let key_state = key_state
        (window, key, scancode, action, mods) -> begin
            if action == GLFW.PRESS
                key_state[Int(key)] = true
            end

            if action == GLFW.RELEASE
                key_state[Int(key)] = false
            end
        end
    end

    framebuffer_size_callback = let window_state = window_state
        (window, width, height) -> begin
            window_state.width = width
            window_state.height = height
            glViewport(0, 0, width, height)
        end
    end

    GLFW.SetInputMode(window, GLFW.CURSOR, GLFW.CURSOR_DISABLED)

    GLFW.SetKeyCallback(window, key_callback)
    GLFW.SetScrollCallback(window, scroll_callback)
    GLFW.SetCursorPosCallback(window, cursor_position_callback)
    GLFW.SetMouseButtonCallback(window, mouse_button_callback)
    GLFW.SetFramebufferSizeCallback(window, framebuffer_size_callback)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    glUseProgram(shader_program)

    vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(boxes),
        boxes,
        GL_STATIC_DRAW,
    )

    vao = glGenVertexArray()
    glBindVertexArray(vao)
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        24,
        Ptr{Cvoid}(0),
    )
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        24,
        Ptr{Cvoid}(12),
    )

    glVertexAttribDivisor(0, 1)
    glVertexAttribDivisor(1, 1)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendEquation(GL_FUNC_ADD)

    last_pos = mouse_state.pos
    last_scroll = mouse_state.scroll

    camerafps = CameraFps()

    lastpos = mouse_state.pos

    t = time()

    while !GLFW.WindowShouldClose(window)
        next_t = time()
        dt = 60*(next_t - t)
        t = next_t

        if get(key_state, Int(GLFW.KEY_W), false)
            walk!(camerafps, -0.3*dt)
        end

        if get(key_state, Int(GLFW.KEY_A), false)
            strafe!(camerafps, -0.3*dt)
        end

        if get(key_state, Int(GLFW.KEY_S), false)
            walk!(camerafps, 0.3*dt)
        end

        if get(key_state, Int(GLFW.KEY_D), false)
            strafe!(camerafps, 0.3*dt)
        end

        pitch!(camerafps, -(mouse_state.pos[2] - lastpos[2])*dt / 200.0)
        yaw!(camerafps, (mouse_state.pos[1] - lastpos[1])*dt / 200.0)

        lastpos = mouse_state.pos

        camera_mat = projection_matrix(0.87266, window_state.width / window_state.height, 1.0, 1000.0)
        camera_mat *= view_matrix(camerafps)

        glUniformMatrix4fv(camera_loc, 1, GL_FALSE, Float32.(camera_mat))

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 14, length(boxes))

        GLFW.SwapBuffers(window)

        # change to wait events?
        GLFW.PollEvents()
    end

    GLFW.DestroyWindow(window)
    return nothing
end

"""
    plot(boxset)

Plot a `boxset` in two or three dimensions using OpenGL, and open a new window to look at the result.
"""
function plot(boxset::BoxSet)
    return plot([Box(Float32.(box.center), Float32.(box.radius)) for box in boxset])
end
