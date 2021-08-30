struct CameraFps
    position::Vector{Float64}
    up::Vector{Float64}
    along::Vector{Float64}
    forward::Vector{Float64}
end

function CameraFps()
    return CameraFps([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0])
end

function view_matrix(camera)
    x = dot(camera.along, camera.position)
    y = dot(camera.up, camera.position)
    z = dot(camera.forward, camera.position)

    return [
        camera.along  camera.up -camera.forward zeros(3);
       -x            -y          z              1.0
    ]'
end

function walk!(camera, delta)
    camera.position .-= delta .* camera.forward
    return camera
end

function strafe!(camera, delta)
    camera.position .+= delta .* camera.along
    return camera
end

function yaw!(camera, angle)
    @. camera.forward = cos(angle) * camera.forward + sin(angle) * camera.along

    normalize!(camera.forward)

    camera.along .= cross(camera.forward, camera.up)

    return camera
end
