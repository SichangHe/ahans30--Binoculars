import os
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from typing import Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from binoculars import BINOCULARS_ACCURACY_THRESHOLD, BINOCULARS_FPR_THRESHOLD

from .metrics import entropy, perplexity
from .utils import assert_tokenizer_consistency

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

DEVICE_1 = "cuda:0"
DEVICE_2 = "cuda:1"

assert torch.cuda.device_count() >= 2, "requires 2 GPU for cross perplexity"


class Binoculars(object):
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        torch.set_float32_matmul_precision("medium")
        self.change_mode(mode)
        self.executor = ThreadPoolExecutor(max_workers=4)
        observer_model_future = self.executor.submit(
            AutoModelForCausalLM.from_pretrained,
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        )
        self.performer_model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                performer_name_or_path,
                device_map={"": DEVICE_2},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"],
            ).eval(),
        )
        self.observer_model = torch.compile(observer_model_future.result().eval())

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )
        return encodings

    @torch.inference_mode()
    def _get_observer_logits(
        self, encodings_obs: transformers.BatchEncoding
    ) -> torch.Tensor:
        return self.observer_model(**encodings_obs).logits

    @torch.inference_mode()
    def _get_performer_logits(
        self, encodings_perf: transformers.BatchEncoding
    ) -> torch.Tensor:
        return self.performer_model(**encodings_perf).logits

    def _get_logits(
        self,
        encodings_obs: transformers.BatchEncoding,
        encodings_perf: transformers.BatchEncoding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_future = self.executor.submit(self._get_observer_logits, encodings_obs)
        performer_logits = self._get_performer_logits(encodings_perf)
        observer_logits = observer_future.result()
        return observer_logits, performer_logits

    def compute_encodings_score(
        self, encodings: transformers.BatchEncoding
    ) -> np.ndarray:
        obs_device = self.observer_model.device
        perf_device = self.performer_model.device
        # NOTE: `BatchEncoding.to()` mutates `self`.
        encodings_obs = copy(encodings).to(obs_device, non_blocking=True)
        encodings_perf = copy(encodings).to(perf_device, non_blocking=True)
        observer_logits, performer_logits = self._get_logits(
            encodings_obs, encodings_perf
        )
        ppl_future = self.executor.submit(perplexity, encodings_obs, observer_logits)
        x_ppl = entropy(
            copy(observer_logits).to(perf_device, non_blocking=True),
            performer_logits,
            encodings_perf,
            self.tokenizer.pad_token_id,
        )
        ppl = ppl_future.result()
        assert isinstance(ppl, torch.Tensor), ppl
        assert isinstance(x_ppl, torch.Tensor), x_ppl
        binoculars_scores = ppl.to("cpu", non_blocking=True) / x_ppl.to(
            "cpu", non_blocking=True
        )
        scores = binoculars_scores.to("cpu").float().numpy()
        del (
            encodings_obs,
            encodings_perf,
            observer_logits,
            performer_logits,
            ppl,
            x_ppl,
            binoculars_scores,
        )
        return scores

    def compute_score(
        self, input_text: Union[list[str], str]
    ) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        binoculars_scores = self.compute_encodings_score(encodings)
        binoculars_scores = binoculars_scores.tolist()
        return (
            binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores
        )

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(
            binoculars_scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred
