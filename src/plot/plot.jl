mutable struct WindowState
    width::Int
    height::Int
end

mutable struct MouseState
    pos::Tuple{Float64,Float64}
    pressed::Bool
    scroll::Float64
    stale::Bool
end

MouseState() = MouseState((0.0, 0.0), false, 0.0, true)

struct CameraState
    offset::Vector{Float32}
    scale::Vector{Float32}
end

struct GLBox{N}
    center::NTuple{N,Float32}
    radius::NTuple{N,Float32}
    color::NTuple{4,Float32}
end

struct GLBar1D
    center::Float32
    radius::Float32
    height::Float32
end

function glGenOne(glGenFn)
    id = pointer(GLuint[0])
    glGenFn(1, id)
    return unsafe_load(id)
end

glGenBuffer() = glGenOne(glGenBuffers)
glGenVertexArray() = glGenOne(glGenVertexArrays)

function setup_mouse_callbacks!(mouse_state, window)
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
            mouse_state.stale = false
        end
    end

    scroll_callback = let mouse_state = mouse_state
        (window, xoffset, yoffset) -> begin
            mouse_state.scroll += yoffset
        end
    end

    GLFW.SetScrollCallback(window, scroll_callback)
    GLFW.SetCursorPosCallback(window, cursor_position_callback)
    GLFW.SetMouseButtonCallback(window, mouse_button_callback)

    return nothing
end

function setup_window()
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
    GLFW.WindowHint(GLFW.RESIZABLE, true)

    window = GLFW.CreateWindow(640, 640, "plot")
    GLFW.MakeContextCurrent(window)

    window_state = WindowState(640, 640)

    framebuffer_size_callback = let window_state = window_state
        (window, width, height) -> begin
            window_state.width = width
            window_state.height = height
            glViewport(0, 0, width, height)
        end
    end

    GLFW.SetFramebufferSizeCallback(window, framebuffer_size_callback)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBlendEquation(GL_FUNC_ADD)

    return window, window_state
end

function setup_shader_program(vert, frag)
    vertex_shader = compile_shader(read(joinpath(@__DIR__, vert), String), GL_VERTEX_SHADER)
    fragment_shader = compile_shader(read(joinpath(@__DIR__, frag), String), GL_FRAGMENT_SHADER)
    shader_program = link_program(vertex_shader, fragment_shader)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    glUseProgram(shader_program)

    return shader_program
end

function update_camera2d!(camera_state, window_state, mouse_state, last_pos, last_scroll)
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
    return nothing
end

function plot(boxes::Vector{GLBar1D})
    GLFW.WindowHint(GLFW.SAMPLES, 8)

    window, window_state = setup_window()

    glEnable(GL_MULTISAMPLE)

    shader_program = setup_shader_program("bar1d.vert", "bar1d.frag")

    camera_offset_loc = glGetUniformLocation(shader_program, "camera_offset")
    camera_scale_loc = glGetUniformLocation(shader_program, "camera_scale")

    mouse_state = MouseState()
    setup_mouse_callbacks!(mouse_state, window)

    vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxes), boxes, GL_STATIC_DRAW)

    vao = glGenVertexArray()
    glBindVertexArray(vao)
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 12, Ptr{Cvoid}(0))
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 12, Ptr{Cvoid}(4))
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 12, Ptr{Cvoid}(8))

    glVertexAttribDivisor(0, 1)
    glVertexAttribDivisor(1, 1)
    glVertexAttribDivisor(2, 1)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glEnableVertexAttribArray(2)

    last_pos = mouse_state.pos
    last_scroll = mouse_state.scroll

    camera_state = CameraState(Float32[0.0, 0.0], Float32[1.0, 1.0])

    while !GLFW.WindowShouldClose(window)
        update_camera2d!(camera_state, window_state, mouse_state, last_pos, last_scroll)

        last_pos = mouse_state.pos
        last_scroll = mouse_state.scroll

        glUniform1f(camera_offset_loc, camera_state.offset[1])
        glUniform1f(camera_scale_loc, camera_state.scale[1])

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

function plot(boxes::Vector{GLBox{1}})
    window, window_state = setup_window()

    shader_program = setup_shader_program("box1d.vert", "boxnd.frag")

    camera_offset_loc = glGetUniformLocation(shader_program, "camera_offset")
    camera_scale_loc = glGetUniformLocation(shader_program, "camera_scale")

    mouse_state = MouseState()
    setup_mouse_callbacks!(mouse_state, window)

    vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxes), boxes, GL_STATIC_DRAW)

    vao = glGenVertexArray()
    glBindVertexArray(vao)
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 24, Ptr{Cvoid}(0))
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 24, Ptr{Cvoid}(4))
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 24, Ptr{Cvoid}(8))

    glVertexAttribDivisor(0, 1)
    glVertexAttribDivisor(1, 1)
    glVertexAttribDivisor(2, 1)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glEnableVertexAttribArray(2)

    last_pos = mouse_state.pos
    last_scroll = mouse_state.scroll

    camera_state = CameraState(Float32[0.0, 0.0], Float32[1.0, 1.0])

    while !GLFW.WindowShouldClose(window)
        update_camera2d!(camera_state, window_state, mouse_state, last_pos, last_scroll)

        last_pos = mouse_state.pos
        last_scroll = mouse_state.scroll

        glUniform1f(camera_offset_loc, camera_state.offset[1])
        glUniform1f(camera_scale_loc, camera_state.scale[1])

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

function plot(boxes::Vector{GLBox{2}})
    GLFW.WindowHint(GLFW.SAMPLES, 8)

    window, window_state = setup_window()

    glEnable(GL_MULTISAMPLE)

    shader_program = setup_shader_program("box2d.vert", "boxnd.frag")

    camera_offset_loc = glGetUniformLocation(shader_program, "camera_offset")
    camera_scale_loc = glGetUniformLocation(shader_program, "camera_scale")

    mouse_state = MouseState()
    setup_mouse_callbacks!(mouse_state, window)

    vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxes), boxes, GL_STATIC_DRAW)

    vao = glGenVertexArray()
    glBindVertexArray(vao)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 32, Ptr{Cvoid}(0))
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, Ptr{Cvoid}(8))
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 32, Ptr{Cvoid}(16))

    glVertexAttribDivisor(0, 1)
    glVertexAttribDivisor(1, 1)
    glVertexAttribDivisor(2, 1)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glEnableVertexAttribArray(2)

    last_pos = mouse_state.pos
    last_scroll = mouse_state.scroll

    camera_state = CameraState(Float32[0.0, 0.0], Float32[1.0, 1.0])

    while !GLFW.WindowShouldClose(window)
        update_camera2d!(camera_state, window_state, mouse_state, last_pos, last_scroll)

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

function plot(boxes::Vector{GLBox{3}})
    window, window_state = setup_window()

    shader_program = setup_shader_program("box3d.vert", "boxnd.frag")

    camera_loc = glGetUniformLocation(shader_program, "camera")

    mouse_state = MouseState()
    setup_mouse_callbacks!(mouse_state, window)

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

    GLFW.SetInputMode(window, GLFW.CURSOR, GLFW.CURSOR_DISABLED)
    GLFW.SetKeyCallback(window, key_callback)

    # Enable depth test
    glEnable(GL_DEPTH_TEST)
    # Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS)

    vbo = glGenBuffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxes), boxes, GL_STATIC_DRAW)

    vao = glGenVertexArray()
    glBindVertexArray(vao)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 40, Ptr{Cvoid}(0))
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 40, Ptr{Cvoid}(12))
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 40, Ptr{Cvoid}(24))

    glVertexAttribDivisor(0, 1)
    glVertexAttribDivisor(1, 1)
    glVertexAttribDivisor(2, 1)

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glEnableVertexAttribArray(2)

    camerafps = CameraFps()

    last_pos = nothing

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

        if !mouse_state.stale
            if last_pos !== nothing
                yaw!(camerafps, (mouse_state.pos[1] - last_pos[1])*dt / 200.0)
            end

            last_pos = mouse_state.pos
        end

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

function plot(boxset::BoxSet{<:BoxPartition{<:Box{N}}}) where N
    return plot([GLBox{N}(box.center.data, box.radius.data, (1.0f0, 1.0f0, 1.0f0, 1.0f0)) for box in boxset])
end

function plot(boxfun::BoxFun)
    barlist = GLBar1D[]

    for (key, value) in boxfun.dict
        box = key_to_box(boxfun.partition, key)
        push!(barlist, GLBar1D(box.center[1], box.radius[1], value))
    end

    plot(barlist)
end
