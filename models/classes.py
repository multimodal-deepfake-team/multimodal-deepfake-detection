# FIXED: previous version of this file had the WRONG mapping.
# The real mapping (confirmed from training/train_video.py, which uses
# torchvision.datasets.ImageFolder -- it sorts class folder names
# ALPHABETICALLY -- and is also hardcoded to match in
# detect_social_video.py and app/streamlit_app.py) is:

CLASSES = {
    0: "ai_video",
    1: "audio_video_fake",
    2: "edited",
    3: "faceswap",
    4: "real",
    5: "voiceclone",
}

# Old (incorrect / unused) mapping, kept here only for reference:
# 0: "real", 1: "ai_video", 2: "faceswap", 3: "edited",
# 4: "voiceclone", 5: "audio_video_fake"
#
# That old mapping does NOT match how the model was actually trained
# or how detect_social_video.py / streamlit_app.py interpret its output.
# If anything in the codebase ever imported the old classes.py mapping
# to interpret model predictions, it would have produced WRONG labels.
