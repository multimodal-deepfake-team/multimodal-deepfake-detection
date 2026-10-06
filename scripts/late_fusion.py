import sys


VIDEO_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4
THRESHOLD = 0.5


def fuse_scores(video_fake_score, audio_fake_score):
    combined_fake_score = (
        VIDEO_WEIGHT * video_fake_score
        + AUDIO_WEIGHT * audio_fake_score
    )

    if combined_fake_score >= THRESHOLD:
        final_result = "DEEPFAKE"
    else:
        final_result = "REAL"

    return {
        "combined_fake_score": combined_fake_score,
        "combined_real_score": 1.0 - combined_fake_score,
        "final_result": final_result,
    }


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python -m scripts.late_fusion "
            "<video_fake_score> <audio_fake_score>"
        )
        raise SystemExit(1)

    video_fake_score = float(sys.argv[1])
    audio_fake_score = float(sys.argv[2])

    result = fuse_scores(
        video_fake_score,
        audio_fake_score
    )

    print()
    print("==============================")
    print("MULTIMODAL LATE FUSION RESULT")
    print("==============================")

    print(
        f"Video fake score   : "
        f"{video_fake_score:.4f}"
    )

    print(
        f"Audio fake score   : "
        f"{audio_fake_score:.4f}"
    )

    print(
        f"Video weight       : "
        f"{VIDEO_WEIGHT:.2f}"
    )

    print(
        f"Audio weight       : "
        f"{AUDIO_WEIGHT:.2f}"
    )

    print()

    print(
        f"Combined fake score: "
        f"{result['combined_fake_score']:.4f}"
    )

    print(
        f"Combined real score: "
        f"{result['combined_real_score']:.4f}"
    )

    print(
        f"Final result       : "
        f"{result['final_result']}"
    )