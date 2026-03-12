from CodeVideoRenderer import CameraFollowCursorCV
video = CameraFollowCursorCV(
    code_string="print('Hello, World!" \
    "IM KADO')",
    language="python",
)
video.render()