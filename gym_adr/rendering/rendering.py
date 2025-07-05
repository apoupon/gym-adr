import numpy as np
import pandas as pd
from screeninfo import get_monitors

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from gym_adr.rendering.utils import LineManager, rotate_object, setup_skybox
from pathlib import Path

from panda3d.core import (
    AntialiasAttrib,
    CardMaker,
    ClockObject,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    KeyboardButton,
    Mat4,
    MouseButton,
    NodePath,
    Point2,
    Point3,
    PointLight,
    Shader,
    Texture,
    TextureStage,
    TextNode,
    Vec3,
    WindowProperties,
    loadPrcFileData,
)


# Detect the screen resolution
monitor = get_monitors()[0]
window_width = monitor.width // 2
window_height = monitor.height // 2

# Launch the rendering in a window
loadPrcFileData("", "fullscreen 0")
loadPrcFileData("", f"win-size {window_width} {window_height}")


class RenderEngine(ShowBase):

    def __init__(self, df: pd.DataFrame, fps: int = 30):
        """Initialize the rendering engine."""
        super().__init__()
        self.current_file = Path(__file__).resolve()
        self.set_antialiasing(is_on=True)

        # Data and state
        self.data = df
        self.current_frame = 0
        self.n_frames = len(self.data)
        row_0 = self.data.loc[self.current_frame]

        self.n_debris = len(row_0) - 3
        self.current_target = row_0["target_index"]
        self.already_deorbited = []

        # Scene components
        self.quad = None
        self.sun = None
        self.otv_node = None
        self.setup_scene()

        # Input handling
        self.accept("space", self.on_space_pressed)
        self.accept("a", self.on_a_pressed)  # toggle atmosphere
        self.accept("c", self.on_c_pressed)  # toggle clouds
        self.accept("d", self.on_d_pressed)  # toggle diagram
        self.accept("f", self.on_f_pressed)  # toggle full circle trajectory
        self.accept("h", self.on_h_pressed)  # toggle HUD
        self.accept("x", self.userExit)

        # Task scheduling
        self.taskMgr.add(self.handle_key_events, "handle_key_events_task")
        self.taskMgr.doMethodLater(1 / fps, self.render_frame, "render_frame")

        # Framerate settings
        global_clock = ClockObject.getGlobalClock()
        global_clock.setMode(ClockObject.MLimited)
        global_clock.setFrameRate(fps)
        self.setFrameRateMeter(True)

        self.game_is_paused = False

    def setup_scene(self):
        """Setup the scene."""
        self.skybox = setup_skybox(self.render, self.loader)
        self.setup_camera()

        self.setup_nodes()
        self.setup_lights()
        self.setup_hud()

        self.setup_offscreen_buffer()
        self.load_shaders()

    def setup_offscreen_buffer(self):
        """Set up an offscreen buffer for rendering."""
        # Configure window and framebuffer properties
        win_size = (self.win.getXSize(), self.win.getYSize())
        win_props = WindowProperties.size(*win_size)

        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setDepthBits(1)

        self.buffer = self.graphicsEngine.makeOutput(
            self.pipe,
            "offscreen buffer",
            -2,
            fb_props,
            win_props,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(),
            self.win,
        )

        # Create textures for color and depth
        self.color_texture = Texture()
        self.depth_texture = Texture()
        self.depth_texture.setFormat(Texture.FDepthStencil)

        # Attach textures to the buffer
        self.buffer.addRenderTexture(
            self.color_texture, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor
        )
        self.buffer.addRenderTexture(
            self.depth_texture,
            GraphicsOutput.RTMCopyRam,
            GraphicsOutput.RTPDepthStencil,
        )

        # Set up the offscreen rendering camera
        self.cam.node().setActive(False)  # Deactivate the main camera
        self.buffer_cam = self.makeCamera(self.buffer, lens=self.camLens)
        self.buffer_cam.reparentTo(self.camera)

    def load_shaders(self):
        """Load post-processing shaders and configure the fullscreen quad."""

        # Load GLSL shader files
        self.shader = Shader.load(
            Shader.SL_GLSL,
            str(self.current_file.parent.parent / 'assets' / 'shaders' / 'post_process.vert'),
            str(self.current_file.parent.parent / 'assets' / 'shaders' / 'post_process.frag'),
        )

        # Create fullscreen quad for post-processing
        cm = CardMaker("fullscreen_quad")
        cm.setFrameFullscreenQuad()
        self.quad = self.render2d.attachNewNode(cm.generate())
        self.quad.setTexture(self.color_texture)
        self.quad.setShader(self.shader)

        # Set shader inputs
        texel_size = (1.0 / self.win.getXSize(), 1.0 / self.win.getYSize())
        self.quad.setShaderInput("tex", self.color_texture)
        self.quad.setShaderInput("depthTex", self.depth_texture)
        self.quad.setShaderInput("texel_size", texel_size)
        self.quad.setShaderInput("diagramValue", self.diagram_value)
        self.quad.setShaderInput("uCameraPosition", self.camera.getPos())

        # Compute and pass inverse projection matrix
        projection_matrix = Mat4(self.camLens.getProjectionMat())
        inverse_projection_matrix = Mat4(projection_matrix)
        inverse_projection_matrix.invertInPlace()
        self.quad.setShaderInput("uInverseProjectionMatrix", inverse_projection_matrix)

        # Compute and pass inverse view matrix
        view_matrix = Mat4(self.buffer_cam.getMat(self.render))
        self.quad.setShaderInput("uInverseViewMatrix", view_matrix)

        # Compute atmospheric scattering coefficients
        wavelengths = [700, 530, 440]
        strength = 20.0
        scattering_coeffs = tuple((400 / w) ** 4 * strength for w in wavelengths)
        self.quad.setShaderInput("scatteringCoefficients", scattering_coeffs)

        self.quad.setShaderInput("atmosphereValue", self.atmosphere_value)

    def setup_nodes(self):
        self.create_earth()
        self.create_sun()
        self.create_otv()

        self.debris_nodes = []
        for _ in range(self.n_debris):
            node = self.create_satellite()
            self.debris_nodes.append(node)

        self.line_manager = LineManager(self.render)

    def setup_lights(self):
        """Setup the lights."""
        # Add a light
        plight = PointLight("plight")
        plight.setColor((1, 1, 1, 1))
        self.light_np = self.render.attachNewNode(plight)
        self.light_np.setPos(10, 0, 0)
        self.render.setLight(self.light_np)

        # Create a shadow buffer
        self.shadowBuffer = self.win.makeTextureBuffer("Shadow Buffer", 1024, 1024)
        self.earth.setShaderInput("shadowMapSize", 1024)
        self.shadowTexture = self.shadowBuffer.getTexture()

        self.depthmap = NodePath("depthmap")
        self.depthmap.setShader(
            Shader.load(
                Shader.SL_GLSL,
                str(self.current_file.parent.parent / 'assets' / 'shaders' / 'shadow_v.glsl'),
                str(self.current_file.parent.parent / 'assets' / 'shaders' / 'shadow_f.glsl'),
            )
        )

        self.earth.setShaderInput("shadowMap", self.shadowTexture)
        self.earth.setShaderInput("lightSpaceMatrix", self.depthmap.getMat())
        self.earth.setShaderInput("lightPos", self.light_np.getPos())
        self.earth.setShaderInput("diagramValue", self.diagram_value)

        self.earth.setShader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=str(self.current_file.parent.parent / 'assets' / 'shaders' / 'pbr.vert'),
                fragment=str(self.current_file.parent.parent / 'assets' / 'shaders' / 'pbr.frag'),
            )
        )

    def setup_camera(self):
        """Setup the camera."""
        self.disableMouse()  # Enable mouse control for the camera

        self.rotation_speed = 50.0
        self.elevation_speed = -50.0

        self.distance_to_origin = 10.0
        self.distance_speed = 0.1
        self.min_dist = 3
        self.max_dist = 16

        self.angle_around_origin = 59.40309464931488
        self.elevation_angle = 4.781174659729004

        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.camera.setPos(8.57774, -5.07224, 0.833504)
        self.camera.lookAt(0, 0, 0)

        self.camLens.setNear(0.1)
        self.camLens.setFar(100.0)

        self.accept("mouse1", self.handle_mouse_click)

        self.taskMgr.add(self.update_camera_task, "update_camera_task")

    def setup_hud(self):
        """Setup the HUD."""
        self.all_labels = []

        # HUD Positions
        y_st = 0.9
        y_sp = 0.07
        left_margin = 0.05

        self.pause_label = self.add_text_label(text="II", pos=(0, y_st))
        self.pause_label.hide()

        # Info Labels (Top Left)
        self.frame_label = self.add_text_label(
            text="Current frame/Total frames",
            pos=(left_margin, y_st),
            alignment_mode=TextNode.ALeft,
            parent=self.a2dLeftCenter,
        )
        self.fuel_label = self.add_text_label(
            text="Fuel: #",
            pos=(left_margin, y_st - y_sp),
            alignment_mode=TextNode.ALeft,
            parent=self.a2dLeftCenter,
        )
        self.target_label = self.add_text_label(
            text="Target: #",
            pos=(left_margin, y_st - 2 * y_sp),
            alignment_mode=TextNode.ALeft,
            parent=self.a2dLeftCenter,
        )
        self.all_labels += [self.frame_label, self.fuel_label, self.target_label]

        # Debris and OTV Labels (Over each debris)
        self.otv_label = self.add_text_label(
            text="OTV", pos=(0, 0), scale=0.05, fg=(0, 1, 0, 1)
        )
        self.all_labels.append(self.otv_label)
        self.debris_labels = [
            self.add_text_label(text=f"Debris {i}", pos=(0, 0), scale=0.05)
            for i in range(1, self.n_debris)
        ]
        self.all_labels += self.debris_labels

        # Control Instructions (Bottom Left)
        control_texts = [
            "X: Exit",
            "Esc: Toggle Fullscreen",
            "Up/Down: Zoom In/Out",
            "Left/Right: Rotate Camera",
            "C: Toggle Clouds",
            "Space: Pause/Resume",
            "D: Toggle diagram",
            "A: Toggle Atmosphere",
            "H: Toggle HUD",
            "F: Toggle Full Trajectory",
        ]

        for i, text in enumerate(reversed(control_texts)):
            label = self.add_text_label(
                text=text,
                pos=(left_margin, -0.9 + y_sp * i),
                scale=0.04,
                alignment_mode=TextNode.ALeft,
                parent=self.a2dLeftCenter,
            )
            self.all_labels.append(label)

    def create_otv(self):
        """Create the OTV model and set its texture."""
        self.otv_node = self.loader.loadModel(str(self.current_file.parent.parent / 'assets' / 'models' / 'otv.dae'))
        albedo_tex = self.loader.loadTexture(str(self.current_file.parent.parent / 'assets' / 'textures' / 'otv_albedo.png'))
        self.otv_node.reparentTo(self.render)
        ts_albedo = TextureStage("albedo")
        self.otv_node.setTexture(ts_albedo, albedo_tex)
        self.otv_node.setScale(0.005)

    def create_satellite(self):
        """Create the satellite model and set its texture."""
        node = self.loader.loadModel(str(self.current_file.parent.parent / 'assets' / 'models' / 'sat.dae'))
        node.reparentTo(self.render)
        albedo_tex = self.loader.loadTexture(str(self.current_file.parent.parent / 'assets' / 'textures' / 'sat_texture.png'))
        ts_albedo = TextureStage("albedo")
        node.setTexture(ts_albedo, albedo_tex)
        node.setScale(0.005)
        return node

    def create_sun(self):
        """Create the sun model and set its shader."""
        self.sun = self.create_sphere(size=0.5)
        self.sun.reparentTo(self.render)
        self.sun.setPos(0, -20, 0)
        self.sun.setShader(
            Shader.load(
                Shader.SL_GLSL,
                vertex=str(self.current_file.parent.parent / 'assets' / 'shaders' / 'sun.vert'),
                fragment=str(self.current_file.parent.parent / 'assets' / 'shaders' / 'sun.frag'),
            )
        )
        self.update_sun()

    def create_earth(self):
        """Create the Earth model and set its textures and shader."""
        self.earth = self.create_sphere(size=0.7)
        self.earth.reparentTo(self.render)
        self.earth.setPos(0, 0, 0)
        self.earth.setHpr(0, 90, 0)

        self.atmosphere_value = 1
        self.cloud_value = 1
        self.skybox_value = 1
        self.diagram_value = 0
        self.full_trajectory_value = 0
        self.full_traj_is_computed = 0
        self.hud_value = 1

        albedo_tex = self.loader.loadTexture(str(self.current_file.parent.parent / 'assets' / 'textures' / 'earth_bm3.png'))
        emission_tex = self.loader.loadTexture(
            str(self.current_file.parent.parent / 'assets' / 'textures' / 'earth_emission.png')
        )
        specular_tex = self.loader.loadTexture(
            str(self.current_file.parent.parent / 'assets' / 'textures' / 'earth_specular.jpg')
        )
        cloud_tex = self.loader.loadTexture(str(self.current_file.parent.parent / 'assets' / 'textures' / 'cloud.png'))
        topography_tex = self.loader.loadTexture(
            str(self.current_file.parent.parent / 'assets' / 'textures' / 'earth_topography.png')
        )

        ts_albedo = TextureStage("albedo")
        ts_emission = TextureStage("emission")
        ts_specular = TextureStage("specular")
        ts_cloud = TextureStage("cloud")
        ts_topography = TextureStage("topography")

        self.earth.setTexture(ts_albedo, albedo_tex)
        self.earth.setTexture(ts_emission, emission_tex)
        self.earth.setTexture(ts_specular, specular_tex)
        self.earth.setTexture(ts_cloud, cloud_tex)
        self.earth.setTexture(ts_topography, topography_tex)

        self.earth.setAttrib(AntialiasAttrib.make(AntialiasAttrib.MAuto))
        self.earth.setRenderModeFilled()

        self.earth.setShaderInput("albedoMap", albedo_tex)
        self.earth.setShaderInput("emissionMap", emission_tex)
        self.earth.setShaderInput("specularMap", specular_tex)
        self.earth.setShaderInput("cloudMap", cloud_tex)
        self.earth.setShaderInput("topographyMap", topography_tex)

        self.update_shader_inputs()

    def create_sphere(
        self,
        size: float = 1,
        low_poly: bool = False,
    ):
        """Create a sphere model."""
        path = str(self.current_file.parent.parent / 'assets' / 'models' / 'high_poly_sphere.obj')
        if low_poly:
            path = str(self.current_file.parent.parent / 'assets' / 'models' / 'low_poly_sphere.obj')
        sphere = self.loader.loadModel(path)
        sphere.setScale(size)
        return sphere

    def add_text_label(
        self,
        text: str = "PlaceHolder",
        pos: tuple = (-1, 1),
        scale: float = 0.06,
        alignment_mode=TextNode.ALeft,
        fg: tuple = (1, 1, 1, 1),
        parent=None,
    ):
        """Add a text label to the HUD."""
        text_label = OnscreenText(
            text=text,
            pos=pos,  # Position on the screen
            scale=scale,  # Text scale
            fg=fg,  # Text color (R, G, B, A)
            bg=(0, 0, 0, 0),  # Background color (R, G, B, A)
            align=alignment_mode,  # Text alignment
            parent=parent,
        )
        return text_label

    def render_frame(self, task):
        """Main rendering task."""
        self.update_hud()
        self.update_shader_inputs()

        if not self.game_is_paused:
            rotate_object(self.earth, [0.025, 0, 0])
            self.update_environment_visuals()

        return Task.cont

    def update_environment_visuals(self):
        """Update the environment for the current frame."""
        # Update the frame
        current_row = self.data.loc[self.current_frame]
        self.current_frame += 1

        if self.current_frame == self.n_frames - 1:
            self.current_frame = 0
            for debris_node in self.debris_nodes:
                debris_node.show()

            row_0 = self.data.loc[self.current_frame]
            self.current_target = row_0["target_index"]

        # --- OTV ---
        self.update_trail("otv", "otv_trail", color=(0.3, 1, 1, 1), thickness=0.5)

        otv_pos = np.array(current_row["otv"])
        self.otv_node.setPos(*otv_pos)

        next_row = self.data.loc[self.current_frame + 1]
        otv_next_pos = np.array(next_row["otv"])
        otv_dir = otv_next_pos - otv_pos
        otv_dir /= np.linalg.norm(otv_dir)

        self.otv_node.setH(np.degrees(np.arctan2(otv_dir[1], otv_dir[0])))
        self.otv_node.setP(90 + np.degrees(np.arcsin(otv_dir[2])))
        self.otv_node.setR(0)

        # --- Debris ---
        for i in range(1, self.n_debris):
            debris_key = f"debris{i}"
            trail_color = (
                (1, 0, 0, 1) if i == self.current_target + 1 else (0.2, 0.3, 0.3, 1)
            )

            self.update_trail(
                debris_key, f"{debris_key}_trail", color=trail_color, thickness=0.5
            )

            debris_pos = np.array(current_row[debris_key])
            self.debris_nodes[i].setPos(*debris_pos)

            debris_next_pos = np.array(next_row[debris_key])
            debris_dir = debris_next_pos - debris_pos
            debris_dir /= np.linalg.norm(debris_dir)

            self.debris_nodes[i].setH(
                np.degrees(np.arctan2(debris_dir[1], debris_dir[0]))
            )
            self.debris_nodes[i].setP(90 + np.degrees(np.arcsin(debris_dir[2])))
            self.debris_nodes[i].setR(0)

        # --- Fuel Display ---
        fuel_init = self.data["fuel"].iloc[0]
        current_fuel = current_row["fuel"]
        fuel_percentage = round(100 * (current_fuel / fuel_init))
        self.fuel_label.setText(f"Fuel: {fuel_percentage}%")

        # --- Target Update ---
        if self.current_target != current_row["target_index"]:
            if self.current_target not in self.already_deorbited:
                self.already_deorbited.append(self.current_target)

        self.current_target = current_row["target_index"]
        self.target_label.setText(f"Target: {self.current_target + 1}")

    def update_sun(self):
        """Update the sun shader inputs."""
        # Retrieve the current model, view, and projection matrices
        model_matrix = self.sun.getMat()
        view_matrix = self.camera.getMat(self.render)
        view_matrix.invert_in_place()
        projection_matrix = self.camLens.getProjectionMat()

        # Set the shader inputs
        self.sun.setShaderInput("model", model_matrix)
        self.sun.setShaderInput("view", view_matrix)
        self.sun.setShaderInput("projection", projection_matrix)

    def update_shader_inputs(self):
        """Update the shader inputs for the Earth and quad."""
        # Get camera position in world space
        view_pos = self.camera.getPos(self.render)
        self.earth.setShaderInput("viewPos", view_pos)
        self.earth.setShaderInput("cloudValue", self.cloud_value)
        self.earth.setShaderInput("diagramValue", self.diagram_value)

        if self.sun is not None:
            self.update_sun()

        if self.quad is not None:
            self.quad.setShaderInput("diagramValue", self.diagram_value)
            self.quad.setShaderInput("atmosphereValue", self.atmosphere_value)
            self.quad.setShaderInput("uCameraPosition", self.camera.getPos())

            # Compute and pass inverse projection and view matrices
            projection_matrix = Mat4(self.camLens.getProjectionMat())
            inverse_projection_matrix = Mat4(projection_matrix)
            inverse_projection_matrix.invertInPlace()
            self.quad.setShaderInput(
                "uInverseProjectionMatrix", inverse_projection_matrix
            )

            view_matrix = Mat4(self.buffer_cam.getMat(self.render))
            self.quad.setShaderInput("uInverseViewMatrix", view_matrix)

    def update_hud(self):
        """Update the HUD labels."""
        self.frame_label.setText(f"{self.current_frame}/{self.n_frames}")

        otv_screen_pos = self.get_object_screen_pos(self.otv_node)
        if otv_screen_pos is not None:
            self.otv_label.setPos(otv_screen_pos[0] + 0.05, otv_screen_pos[1])

        if self.hud_value == 1:
            for i in range(1, self.n_debris):
                debris_screen_pos = self.get_object_screen_pos(self.debris_nodes[i])

                if debris_screen_pos is not None:
                    self.debris_labels[i - 1].setPos(
                        debris_screen_pos[0] + 0.05, debris_screen_pos[1]
                    )
                    self.debris_labels[i - 1].show()
                else:
                    self.debris_labels[i - 1].hide()

                if i == self.current_target + 1:
                    self.debris_labels[i - 1].setText(f"Debris {i} (Target)")
                else:
                    self.debris_labels[i - 1].setText(f"Debris {i}")

    def update_trail(
        self,
        name_in_df: str,
        name_in_line_manager: str,
        n_points: int = 20,
        color: tuple = (0, 1, 1, 1),
        thickness: float = 0.5,
    ):
        """Update the trajectory trail for a given object."""
        if self.full_traj_is_computed == 6:
            return
        elif self.full_trajectory_value == 1:
            self.full_traj_is_computed += 1

        current_frame = self.current_frame + 1  # last
        frame_minus_n_points = max(0, self.current_frame - n_points)  # first

        if self.full_trajectory_value == 1:
            frame_minus_n_points = 0
            current_frame = self.n_frames

        all_points = []
        for i in range(frame_minus_n_points, current_frame, 5):
            pos = tuple(self.data.iloc[i][name_in_df])
            all_points.append(pos)

        self.line_manager.update_line(
            name_in_line_manager, all_points, color=color, thickness=thickness
        )

    def update_camera_task(self, task):
        """Update the camera position based on mouse movement."""
        # Check if the left mouse button is still down
        if self.mouseWatcherNode.isButtonDown(MouseButton.one()):
            # Get the mouse position
            if self.mouseWatcherNode.hasMouse():
                current_mouse_x = self.mouseWatcherNode.getMouseX()
                current_mouse_y = self.mouseWatcherNode.getMouseY()
            else:
                self.taskMgr.remove("update_camera_task")
                return task.done

            # Check if the mouse has moved horizontally
            if current_mouse_x != self.last_mouse_x:
                # Adjust the camera rotation based on the mouse horizontal movement
                self.angle_around_origin -= (
                    current_mouse_x - self.last_mouse_x
                ) * self.rotation_speed

            # Check if the mouse has moved vertically
            if current_mouse_y != self.last_mouse_y:
                # Adjust the camera elevation based on the mouse vertical movement
                self.elevation_angle += (
                    current_mouse_y - self.last_mouse_y
                ) * self.elevation_speed
                self.elevation_angle = max(
                    -90, min(90, self.elevation_angle)
                )  # Clamp the elevation angle

            self.update_camera_position()

            self.last_mouse_x = current_mouse_x
            self.last_mouse_y = current_mouse_y

            return task.cont
        else:
            # Disable the mouse motion task when the left button is released
            self.taskMgr.remove("update_camera_task")
            return task.done

    def update_camera_position(self):
        """Update the camera position."""
        # Camera
        if self.angle_around_origin > 360:
            self.angle_around_origin -= 360
        if self.angle_around_origin < 0:
            self.angle_around_origin += 360

        radian_angle = np.radians(self.angle_around_origin)
        radian_elevation = np.radians(self.elevation_angle)
        x_pos = (
            self.distance_to_origin * np.sin(radian_angle) * np.cos(radian_elevation)
        )
        y_pos = (
            -self.distance_to_origin * np.cos(radian_angle) * np.cos(radian_elevation)
        )
        z_pos = self.distance_to_origin * np.sin(radian_elevation)

        self.camera.setPos(Vec3(x_pos, y_pos, z_pos))
        self.camera.lookAt(Point3(0, 0, 0))

    def handle_key_events(self, task):
        """Check if the key is down."""
        if self.mouseWatcherNode.is_button_down(KeyboardButton.up()):
            self.move_forward()
        if self.mouseWatcherNode.is_button_down(KeyboardButton.down()):
            self.move_backward()
        if self.mouseWatcherNode.is_button_down(KeyboardButton.left()):
            self.angle_around_origin -= 1
            self.update_camera_position()
        if self.mouseWatcherNode.is_button_down(KeyboardButton.right()):
            self.angle_around_origin += 1
            self.update_camera_position()

        return task.cont

    def move_forward(self):
        """Move the camera forward."""
        if self.distance_to_origin > self.min_dist:
            self.distance_to_origin -= self.distance_speed
            self.update_camera_position()

    def move_backward(self):
        """Move the camera backward."""
        if self.distance_to_origin < self.max_dist:
            self.distance_to_origin += self.distance_speed
            self.update_camera_position()

    def handle_mouse_click(self):
        """Handle mouse click to enable camera rotation."""
        # Check if the left mouse button is down
        if self.mouseWatcherNode.isButtonDown(MouseButton.one()):
            # Enable mouse motion task
            self.last_mouse_x = self.mouseWatcherNode.getMouseX()
            self.last_mouse_y = self.mouseWatcherNode.getMouseY()
            self.taskMgr.add(self.update_camera_task, "update_camera_task")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        wp = WindowProperties()

        if self.fullscreen:
            # Switch to windowed mode
            wp.setFullscreen(False)
            wp.setSize(800, 600)
        else:
            # Switch to fullscreen mode
            wp.setFullscreen(True)
            wp.setSize(window_width, window_height)

        self.win.requestProperties(wp)
        self.fullscreen = not self.fullscreen

    def set_antialiasing(self, is_on):
        """Enable anti-aliasing."""
        if is_on:
            loadPrcFileData("", "multisamples 4")  # Enable MSAA
            self.render.setAntialias(AntialiasAttrib.MAuto)

    def toggle_skybox(self):
        """Toggle the skybox visibility."""
        self.skybox_value = 1 - self.skybox_value

        if self.skybox_value == 1:
            self.show_skybox()
        else:
            self.hide_skybox()

    def show_skybox(self):
        """Show the skybox."""
        self.skybox_value = 1
        for plane in self.skybox:
            plane.show()

    def hide_skybox(self):
        """Hide the skybox."""
        self.skybox_value = 0
        for plane in self.skybox:
            plane.hide()

    def on_space_pressed(self):
        """Toggle pause state"""
        self.game_is_paused = not self.game_is_paused

        if self.game_is_paused:
            self.pause_label.show()
        else:
            self.pause_label.hide()

    def on_a_pressed(self):
        """Toggle atmosphere."""
        self.atmosphere_value = 1 - self.atmosphere_value

        if self.atmosphere_value == 1:
            self.diagram_value = 0
            self.show_skybox()

    def on_c_pressed(self):
        """Change cloud value between 0 and 1."""
        self.cloud_value = 1 - self.cloud_value

    def on_d_pressed(self):
        """Toggle diagram mode."""
        self.diagram_value = 1 - self.diagram_value
        self.toggle_skybox()

        if self.diagram_value == 1:
            for label in self.all_labels:
                # set text to black
                label.fg = (0, 0, 0, 1)

        elif self.diagram_value == 0:
            for label in self.all_labels:
                # set text to white
                label.fg = (1, 1, 1, 1)

    def on_f_pressed(self):
        """Toggle full trajectory value."""
        self.full_trajectory_value = 1 - self.full_trajectory_value

        if self.full_trajectory_value == 0:
            self.full_traj_is_computed = 0

    def on_h_pressed(self):
        """Toggle HUD visibility."""
        self.hud_value = 1 - self.hud_value

        if self.hud_value == 1:
            for label in self.all_labels:
                label.show()
        else:
            for label in self.all_labels:
                label.hide()

    def get_object_screen_pos(self, obj):
        """Get the object's position relative to the camera and project it to 2D screen coordinates."""
        pos3d = self.camera.getRelativePoint(obj, Point3(0, 0, 0))

        # Project the 3D point to 2D screen coordinates
        pos2d = Point2()
        if self.camLens.project(pos3d, pos2d):
            screen_x = pos2d.getX() * self.getAspectRatio()
            screen_y = pos2d.getY()

            return screen_x, screen_y
        else:
            return None
