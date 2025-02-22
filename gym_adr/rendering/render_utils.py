import numpy as np

from panda3d.core import LineSegs, NodePath, TextureStage


def make_circle(radius, n_points, center=(0, 0, 0), rotation=(0, 0, 0)):
    """
    Make a circle in 3D space

    Parameters
    ----------
    radius : float
        Radius of the circle
    n_points : int
        Number of points to make the circle
    center : tuple
        Center of the circle
    rotation : tuple
        Rotation of the circle


    Returns
    -------
    np.array
        Array of points on the circle
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.zeros_like(theta) + center[2]

    # Apply rotation
    rotation_matrix = get_rotation_matrix(rotation)
    points = np.column_stack((x, y, z))
    rotated_points = np.dot(points, rotation_matrix.T)

    # convert to list of tuples
    rotated_points = [tuple(point) for point in rotated_points]
    return rotated_points


def get_rotation_matrix(rotation):
    alpha, beta, gamma = rotation
    rotation_matrix = np.array(
        [
            [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta) * np.sin(gamma)
                - np.sin(alpha) * np.cos(gamma),
                np.cos(alpha) * np.sin(beta) * np.cos(gamma)
                + np.sin(alpha) * np.sin(gamma),
            ],
            [
                np.sin(alpha) * np.cos(beta),
                np.sin(alpha) * np.sin(beta) * np.sin(gamma)
                + np.cos(alpha) * np.cos(gamma),
                np.sin(alpha) * np.sin(beta) * np.cos(gamma)
                - np.cos(alpha) * np.sin(gamma),
            ],
            [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)],
        ]
    )
    return rotation_matrix


class LineManager:
    def __init__(self, render):
        self.render = render
        self.lines = {}

    def make_line(self, name, points, color=(1, 0, 0, 1), thickness=2.0):
        # Check if line already exists
        if name in self.lines:
            print(f"Line with name '{name}' already exists. Updating instead.")
            self.update_line(name, points, color, thickness)
            return

        # Create LineSegs object
        lines = LineSegs()
        lines.setThickness(thickness)
        lines.setColor(*color)

        # Add points to LineSegs
        for point in points:
            lines.moveTo(point) if point == points[0] else lines.drawTo(point)

        # Create NodePath
        line_geom_node = lines.create(False)
        node_path = NodePath(line_geom_node)
        node_path.reparentTo(self.render)

        # Store the line data
        self.lines[name] = (lines, node_path)

    def update_line(self, name, points, color=(1, 0, 0, 1), thickness=2.0):
        if name not in self.lines:
            # print(f"No line with name '{name}' found. Creating a new one.")
            self.make_line(name, points, color, thickness)
            return

        # Retrieve the existing LineSegs object; NodePath is not needed here
        lines, _ = self.lines[name]

        # Reset and update LineSegs with new properties and points
        lines.reset()
        lines.setThickness(thickness)
        lines.setColor(*color)
        for point in points:
            lines.moveTo(point) if point == points[0] else lines.drawTo(point)

        # Remove the old geom node from the scene graph
        self.lines[name][1].removeNode()

        # Create a new geom node and reparent it to the render
        line_geom_node = lines.create(False)
        node_path = NodePath(line_geom_node)
        node_path.reparentTo(self.render)

        # Update the stored NodePath for this line
        self.lines[name] = (lines, node_path)


def setup_skybox(render, loader):
    size = 20
    distance = 20

    texture_list = []
    names = ["top", "bottom", "right", "left", "front", "back"]
    for i in range(1, 7):
        texture_list.append(
            loader.loadTexture(f"gym_adr/assets/skybox/{names[i - 1]}.png")
        )

    path = "gym_adr/assets/models/plane.obj"
    plane_1 = loader.loadModel(path)
    plane_2 = loader.loadModel(path)
    plane_3 = loader.loadModel(path)
    plane_4 = loader.loadModel(path)
    plane_5 = loader.loadModel(path)
    plane_6 = loader.loadModel(path)

    base_ts = TextureStage("base_ts")
    base_ts.setMode(TextureStage.MReplace)

    plane_1.setTexture(base_ts, texture_list[0])
    plane_1.setScale(size)
    plane_2.setTexture(base_ts, texture_list[1])
    plane_2.setScale(size)
    plane_3.setTexture(base_ts, texture_list[2])
    plane_3.setScale(size)
    plane_4.setTexture(base_ts, texture_list[3])
    plane_4.setScale(size)
    plane_5.setTexture(base_ts, texture_list[4])
    plane_5.setScale(size)
    plane_6.setTexture(base_ts, texture_list[5])
    plane_6.setScale(size)

    plane_1.setPos(0, 0, distance)
    plane_1.setHpr(0, -90, 0)

    plane_2.setPos(0, 0, -distance)
    plane_2.setHpr(0, 90, 0)

    plane_3.setPos(distance, 0, 0)
    plane_3.setHpr(90, 0, 180)

    plane_4.setPos(-distance, 0, 0)
    plane_4.setHpr(-90, 0, 180)

    plane_5.setPos(0, distance, 0)
    plane_5.setHpr(-180, 0, 180)

    plane_6.setPos(0, -distance, 0)
    plane_6.setHpr(0, 0, 180)

    plane_1.reparentTo(render)
    plane_2.reparentTo(render)
    plane_3.reparentTo(render)
    plane_4.reparentTo(render)
    plane_5.reparentTo(render)
    plane_6.reparentTo(render)

    return [plane_1, plane_2, plane_3, plane_4, plane_5, plane_6]


def move_object(object, xyz):
    x = object.getX()
    y = object.getY()
    z = object.getZ()

    object.setX(x + xyz[0])
    object.setY(y + xyz[1])
    object.setZ(z + xyz[2])


def rotate_object(object, xyz):
    x = object.getH()
    y = object.getP()
    z = object.getR()

    object.setH(x + xyz[0])
    object.setP(y + xyz[1])
    object.setR(z + xyz[2])
