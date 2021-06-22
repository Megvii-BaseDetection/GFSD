import torch


class ClassSplit():
    def __init__(self):
        # Original class ids, overwrite this for custom class split.
        self._base_classes = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self._novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                               44, 62, 63, 64, 67, 72]
        self.register()

    def register(self):
        self._all_classes = sorted(self._base_classes + self._novel_classes)
        self.base2continuous = {v: i for i, v in enumerate(self._base_classes)}
        self.novel2continuous = {
                v: i for i, v in enumerate(self._novel_classes)}
        self.all2continuous = {v: i for i, v in enumerate(self._all_classes)}

    def num_classes(self, type="all"):
        assert type in ["base", "novel", "all"]
        if type == "base":
            return len(self._base_classes)
        elif type == "novel":
            return len(self._novel_classes)
        else:
            return len(self._all_classes)

    def to_continuous(self, type):
        assert type in ["base", "novel", "all"]
        if type == "base":
            return self.base2continuous
        elif type == "novel":
            return self.novel2continuous
        else:
            return self.all2continuous

    def to_all_classes(self, type):
        assert type in ["base", "novel"]
        if type == "base":
            return {self.base2continuous[k]: self.all2continuous[k]
                    for k in self._base_classes}
        else:
            return {self.novel2continuous[k]: self.all2continuous[k]
                    for k in self._novel_classes}

    def from_all_classes(self, type):
        reverse_dict = self.to_all_classes(type)
        return {v: k for k, v in reverse_dict.items()}

    def to_all_classes_bool(self, type, include_bg=False, device=None):
        assert type in ["base", "novel"]
        mask = torch.zeros(len(self._all_classes) + (1 if include_bg else 0),
                           device=device).to(torch.bool)
        if type == "base":
            indices = [self.all2continuous[k] for k in self._base_classes]
            mask[indices] = True
        else:
            indices = [self.all2continuous[k] for k in self._novel_classes]
            mask[indices] = True
        return mask

    def map_func(self, type):
        assert type in ["base", "novel"]
        mapping = {}
        for id in self._all_classes:
            idx = self.all2continuous[id]
            if id in getattr(self, f"_{type}_classes"):
                mapping[idx] = getattr(self, f"{type}2continuous")[id]
            else:
                mapping[idx] = -1
        mapping[self.num_classes()] = -1
        return lambda x: mapping[x]


class_split = ClassSplit()
