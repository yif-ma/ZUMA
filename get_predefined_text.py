from dataclasses import dataclass
from typing import List, Optional, ClassVar
import torch

@dataclass(frozen=True)
class TextSlice:
    pos: List[str]
    neg: List[str]

    _clip: ClassVar[Optional[object]] = None
    _text_model: ClassVar[Optional[object]] = None
    _device: ClassVar[Optional[object]] = None

    @classmethod
    def bind(cls, clip, text_model, device):
        cls._clip = clip
        cls._text_model = text_model
        cls._device = device

    def get_text_feature(self):
        assert self._clip is not None and self._text_model is not None and self._device is not None, \
            "TextSlice not bound: call TextSlice.bind(clip, text_model, device) first."

        pos_features = self._clip.tokenize(self.pos).cuda(device=self._device)
        neg_features = self._clip.tokenize(self.neg).cuda(device=self._device)

        pos_features = self._text_model.encode_text(pos_features)
        neg_features = self._text_model.encode_text(neg_features)

        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

        pos_features = torch.mean(pos_features, dim=0, keepdim=True)
        neg_features = torch.mean(neg_features, dim=0, keepdim=True)

        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

        text_features = torch.cat([pos_features, neg_features], dim=0).float()
        text_features = text_features.unsqueeze(0)
        return text_features

from typing import Dict, List, Optional

def predefined_text_descriptions(
    obj_name: str,
    txt_path: str = "prompts.txt",
    placeholder: str = "{obj}",
) -> Dict[str, List[str]]:

    o = obj_name.replace("_", " ")

    positive: List[str] = []
    negative: List[str] = []
    section: Optional[str] = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            low = line.lower()
            if low.startswith("#"):
                if "positive" in low:
                    section = "positive"
                elif "negative" in low:
                    section = "negative"
                else:
                    section = None
                continue

            if section is None:
                continue

            line = line.replace(placeholder, o)

            if section == "positive":
                positive.append(line)
            elif section == "negative":
                negative.append(line)

    if not positive or not negative:
        raise ValueError(
            f"Failed to load prompts from {txt_path}. "
            "Please ensure it contains '# positive' and '# negative' sections."
        )

    return {"positive": positive, "negative": negative}
