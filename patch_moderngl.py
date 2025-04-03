import moderngl as _moderngl

original_create_context = _moderngl.create_context

def patched_create_context(*args, **kwargs):
    print("✅ [Patch] Forcing standalone EGL context via create_context")
    return original_create_context(**kwargs)

_moderngl.create_context = patched_create_context