import warp as wp
import warp.render
import warp.examples

import numpy as np

import math

import os
import time

CLOTH_SIZE = 20.0
CLOTH_HEIGHT = 15.0
DAMPING = 0.99
FRAMES = 1000
SUBSTEPS = 10

@wp.kernel
def initialize_vertices_springs(vertices: wp.array(dtype=wp.vec3),
                                pinned_vertices: wp.array(dtype=int),
                                springs: wp.array(dtype=wp.vec2i),
                                spring_colors: wp.array(dtype=int),
                                vertices_count: int,
                                square: int,
                                springs_count: int):

    tid = wp.tid()

    # Initialize vertices
    if tid < vertices_count:

        #if tid == 0:
        #    pinned_vertices[tid] = 1

        x = tid // square
        z = tid - x * square

        pos_x = -CLOTH_SIZE/2.0 + (CLOTH_SIZE*((wp.float32(x)+1.0)/wp.float32(square)))
        pos_z = -CLOTH_SIZE/2.0 + (CLOTH_SIZE*((wp.float32(z)+1.0)/wp.float32(square)))
        pos = wp.vec3(pos_x, CLOTH_HEIGHT, pos_z)
        vertices[tid] = pos

    # Initialize edges
    if tid < springs_count:

        springs_per_direction = square * (square - 1)
        shear_per_direction = (square - 1) * (square - 1)

        if tid < springs_per_direction:
            # Horizontal springs
            i = tid // (square - 1)
            j = tid - i * (square - 1)

            v0 = i * square + j
            v1 = v0 + 1

            # Either 0 or 1
            spring_colors[tid] = i % 2
            springs[tid] = wp.vec2i(v0, v1)

        elif tid < 2 * springs_per_direction:
            # Vertical springs
            t = tid - springs_per_direction
            i = t // square
            j = t - i * square

            v0 = i * square + j
            v1 = v0 + square

            # Either 2 or 3
            spring_colors[tid] = 2 + (j % 2)
            springs[tid] = wp.vec2i(v0, v1)

        elif tid < 2 * springs_per_direction + shear_per_direction:
            # Shear springs (i, j) to (i+1, j+1)
            t = tid - 2 * springs_per_direction
            i = t // (square - 1)
            j = t - i * (square - 1)

            v0 = i * square + j
            v1 = (i + 1) * square + j + 1

            spring_colors[tid] = 4
            springs[tid] = wp.vec2i(v0, v1)
        
        else:
            # Shear springs (i+1, j) to (i, j+1)
            t = tid - 2 * springs_per_direction - shear_per_direction
            i = t // (square - 1)
            j = t - i * (square - 1)

            v0 = (i + 1) * square + j
            v1 = i * square + (j + 1)

            spring_colors[tid] = 5
            springs[tid] = wp.vec2i(v0, v1)


@wp.kernel
def step(vertices: wp.array(dtype=wp.vec3),
         previous_vertices: wp.array(dtype=wp.vec3),
         pinned_vertices: wp.array(dtype=int),
         damping: float,
         dt: float):
    
    tid = wp.tid()

    if pinned_vertices[tid] == 1:
        return

    g = wp.vec3(0.0, -9.81, 0.0)

    temp = vertices[tid]
    v = (vertices[tid] - previous_vertices[tid]) * damping
    vertices[tid] = vertices[tid] + v + g * (dt * dt)
    previous_vertices[tid] = temp

@wp.kernel
def distance_constraint(vertices: wp.array(dtype=wp.vec3),
                        pinned_vertices: wp.array(dtype=int),
                        springs: wp.array(dtype=wp.vec2i),
                        spring_colors: wp.array(dtype=int),
                        color: int,
                        rest_length: float):

    tid = wp.tid()

    if spring_colors[tid] != color:
        return

    i0 = springs[tid][0]
    i1 = springs[tid][1]

    v0 = vertices[i0]
    v1 = vertices[i1]

    delta = v0 - v1
    length = wp.length(delta)

    if length == 0.0:
        return

    error = length - rest_length

    direction = delta / length
    stiffness = 0.8
    correction = direction * error * 0.5 * stiffness

    if pinned_vertices[i0] == 1:
        vertices[i1] += direction * error
    elif pinned_vertices[i1] == 1:
        vertices[i0] -= direction * error
    else:
        vertices[i0] -= correction
        vertices[i1] += correction

@wp.kernel
def collision(vertices: wp.array(dtype=wp.vec3),
              previous_vertices: wp.array(dtype=wp.vec3),
              pinned_vertices: wp.array(dtype=int),
              volume: wp.uint64,
              radius: float,
              dt: float):
    
    tid = wp.tid()

    if pinned_vertices[tid] == 1:
        return

    vertex = vertices[tid]

    local = wp.volume_world_to_index(volume, vertex)
    distance = wp.volume_sample_f(volume, local, wp.Volume.LINEAR)

    if distance < radius:

        grad = wp.vec3()
        wp.volume_sample_grad_f(volume, local, wp.Volume.LINEAR, grad)

        if wp.length(grad) > 0.0:

            normal = grad / wp.length(grad)

            penetration = radius - distance
            new_pos = vertex + normal * penetration
            vertices[tid] = new_pos

            velocity = (vertices[tid] - previous_vertices[tid]) / dt
            velocity_tangent = velocity - wp.dot(velocity, normal) * normal

            friction = 0.5
            if wp.length(velocity_tangent) < 0.1:
                velocity_tangent = wp.vec3(0.0, 0.0, 0.0)
            else:
                velocity_tangent *= friction

            previous_vertices[tid] = vertices[tid] - (velocity_tangent * dt)


if __name__ == "__main__":

    dt = 1.0/60.0

    wp.init()
    device = wp.get_preferred_device()

    vertices_count = 10000
    if math.isqrt(vertices_count) * math.isqrt(vertices_count) != vertices_count:
        print(f"Error! {vertices_count} is not a perfect square.")
        exit(1)

    vertices = wp.zeros(shape=vertices_count, dtype=wp.vec3)
    square = int(math.sqrt(vertices_count))
    structural_spring_rest_length = CLOTH_SIZE / square
    shear_spring_rest_length = structural_spring_rest_length * math.sqrt(2.0)

    pinned_vertices = wp.zeros(shape=vertices_count, dtype=int)

    shear_springs = 2 * (square - 1) * (square - 1)
    springs_count = 2 * square * (square - 1) + shear_springs
    springs = wp.zeros(shape=springs_count, dtype=wp.vec2i)

    spring_colors = wp.zeros(shape=springs_count, dtype=int)

    wp.launch(initialize_vertices_springs,
              dim=springs_count,
              inputs=[vertices,
                      pinned_vertices,
                      springs,
                      spring_colors,
                      vertices_count,
                      square,
                      springs_count],
              device=device)
    
    previous_vertices = wp.clone(vertices)

    with open(os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb"), "rb") as file:
            # create Volume object
            volume = wp.Volume.load_from_nvdb(file)

    renderer = warp.render.UsdRenderer(stage="stage.usd")
    #renderer.render_ground(size=100.0)

    start_time = time.time()

    for frame in range(FRAMES):

        renderer.begin_frame(frame)

        renderer.render_ref(name="object",
                            path=os.path.join(warp.examples.get_asset_directory(), "rocks.usd"),
                            pos=(0.0, 0.0, 0.0),
                            rot=(0.0, 0.0, 0.0, 0.0),
                            scale=(1.0, 1.0, 1.0),
                            color=(0.0, 0.0, 0.0))
        
        np_vertices = vertices.numpy()
        renderer.render_points(name="balls",
                            points=np_vertices,
                            radius=0.01)
                    
        renderer.end_frame()

        substep_dt = dt / SUBSTEPS

        for s in range(SUBSTEPS):

            # Prediction
            wp.launch(step,
                    dim=vertices_count,
                    inputs=[vertices,
                            previous_vertices,
                            pinned_vertices,
                            DAMPING,
                            substep_dt],
                    device=device)
            wp.synchronize()

            # Solve distance constraints
            for _ in range(3):
                # Do for structural springs
                for color in range(4):
                    wp.launch(distance_constraint,
                            dim=springs_count,
                            inputs=[vertices,
                                    pinned_vertices,
                                    springs,
                                    spring_colors,
                                    color,
                                    structural_spring_rest_length],
                            device=device)
                    wp.synchronize()

                # Do for shear springs 
                for color in range(4, 6):
                    wp.launch(distance_constraint,
                            dim=springs_count,
                            inputs=[vertices,
                                    pinned_vertices,
                                    springs,
                                    spring_colors,
                                    color,
                                    shear_spring_rest_length],
                            device=device)
                    wp.synchronize()

            # Solve collision with rocks
            wp.launch(collision,
                    dim=vertices_count,
                    inputs=[vertices,
                            previous_vertices,
                            pinned_vertices,
                            volume.id,
                            structural_spring_rest_length,
                            substep_dt],
                    device=device)
            wp.synchronize()
                    
    renderer.save()

    print(f"Render completed in {round(time.time() - start_time, 2)} seconds")