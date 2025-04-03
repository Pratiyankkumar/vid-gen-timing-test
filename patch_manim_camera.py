import moderngl
import manimlib.camera.camera as camera

def get_fbo_fixed(self, samples=0):
    print("✅ [Patch] Camera.get_fbo monkey-patched successfully")
    try:
        return self.ctx.framebuffer(
            color_attachments=[
                self.ctx.texture(self.default_pixel_shape, components=self.n_channels, samples=samples)
            ],
            depth_attachment=self.ctx.depth_renderbuffer(self.default_pixel_shape, samples=samples)
        )
    except Exception as e:
        print("❌ [Patch] Default FBO config failed:", e)
        print("✅ [Patch] Using fallback FBO config")
        tex = self.ctx.texture(self.default_pixel_shape, 4)
        depth = self.ctx.depth_renderbuffer(self.default_pixel_shape)
        return self.ctx.framebuffer(color_attachments=[tex], depth_attachment=depth)