import os
from concurrent.futures import ThreadPoolExecutor
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

assert torch.cuda.device_count() > 2, "requires 2 GPU for cross perplexity"


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
        self.observer_model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                observer_name_or_path,
                device_map={"": DEVICE_1},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"],
            ).eval()
        )
        self.performer_model = torch.compile(
            AutoModelForCausalLM.from_pretrained(
                performer_name_or_path,
                device_map={"": DEVICE_2},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"],
            ).eval()
        )

        self.executor = ThreadPoolExecutor(max_workers=4)

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
        ).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_observer_logits(
        self, encodings: transformers.BatchEncoding
    ) -> torch.Tensor:
        return self.observer_model(**encodings.to(DEVICE_1)).logits

    @torch.inference_mode()
    def _get_performer_logits(
        self, encodings: transformers.BatchEncoding
    ) -> torch.Tensor:
        return self.performer_model(**encodings.to(DEVICE_2)).logits

    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        future_observer = self.executor.submit(self._get_observer_logits, encodings)
        future_performer = self.executor.submit(self._get_performer_logits, encodings)

        observer_logits = future_observer.result()
        performer_logits = future_performer.result()

        return observer_logits, performer_logits

    def compute_encodings_score(
        self, encodings: transformers.BatchEncoding
    ) -> np.ndarray:
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        return binoculars_scores

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
