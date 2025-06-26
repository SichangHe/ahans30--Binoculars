"""Run Binoculars on Modal."""

from pathlib import Path, PurePosixPath
from threading import Lock

import huggingface_hub
import modal
from modal import parameter

from binoculars.detector import Binoculars

app = modal.App("binoculars-falcon")


def prefetch_models():
    for observer_name_or_path, performer_name_or_path in (
        ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),
        ("SichangHe/falcon-7b-FP8-Dynamic", "SichangHe/falcon-7b-instruct-FP8-Dynamic"),
    ):
        huggingface_hub.snapshot_download(observer_name_or_path)
        huggingface_hub.snapshot_download(performer_name_or_path)


# These are from <https://modal.com/docs/examples/sgl_vlm>.
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    # NOTE: Avoid downloading from HuggingFace every time a container starts.
    Path("~/.cache").absolute().as_posix(): modal.Volume.from_name(
        "falcon7b-cache", create_if_missing=True
    ),
}
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .pip_install(
        "numpy",
        "transformers[torch]>=4.51.0",
        "huggingface_hub[hf_xet]",
        "torch",
    )
    # NOTE: Download the model from HuggingFace.
    .run_function(prefetch_models, volumes=volumes)
)


@app.cls(
    gpu="H100:1",
    timeout=180,
    scaledown_window=15,
    image=image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=64)
class BinoModal:
    observer_name_or_path: str = parameter(default="tiiuae/falcon-7b")
    performer_name_or_path: str = parameter(default="tiiuae/falcon-7b-instruct")
    torch_dtype: str = parameter(default="bfloat16")
    max_token_observed: int = parameter(default=512)
    mode: str = parameter(default="low-fpr")
    compile: bool = parameter(default=False)
    check_tokenizer_consistency: bool = parameter(default=True)

    @modal.enter()  # what should a container do after it starts but before it gets input?
    def load_bino(self):
        self.bino = Binoculars(
            observer_name_or_path=self.observer_name_or_path,
            performer_name_or_path=self.performer_name_or_path,
            torch_dtype=self.torch_dtype,
            max_token_observed=self.max_token_observed,
            mode=self.mode,
            compile=self.compile,
            check_tokenizer_consistency=self.check_tokenizer_consistency,
        )
        self._lock = Lock()

    @modal.method()
    def compute_score(self, strings: list[str]):
        with self._lock:
            return self.bino.compute_score(strings)


@app.local_entrypoint()
def main():
    with open(
        "data/classify/google_prelim/www.mensfitclub.com20250205-071202.html"
    ) as f:
        string = f.read()
    bino = BinoModal()  # type:ignore[arg-type]
    [score] = bino.compute_score.remote([string])
    assert abs(score - 0.9806451797485352) < 0.01
