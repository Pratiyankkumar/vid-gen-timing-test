import patch_moderngl
import patch_manim_camera

from manimlib import *
import numpy as np

class SimpleAnimation(Scene):
    CONFIG = {
        "camera_config": {
            "pixel_width": 1920,
            "pixel_height": 720,
            "samples": 1,
            "anti_alias": False,
            "use_z_index": False
        }
    }

    def construct(self):
        num_dots = 10

        dots = VGroup(*[
            Dot(radius=0.05, color=WHITE).move_to(
                np.array([
                    np.random.uniform(-6, 6),
                    np.random.uniform(-3.5, 3.5),
                    0
                ])
            )
            for _ in range(num_dots)
        ])

        self.add(dots)

        def update_dot(dot, dt, index):
            x, y, _ = dot.get_center()
            dot.move_to([
                x + 0.05 * np.sin(TAU * (index + self.time) / 2),
                y + 0.05 * np.cos(TAU * (index + self.time) / 2),
                0
            ])

        for i, dot in enumerate(dots):
            dot.add_updater(lambda d, dt, i=i: update_dot(d, dt, i))

        self.wait(5)

        for dot in dots:
            dot.clear_updaters()

        self.play(FadeOut(dots))