class CurriculumSampler:
    def __init__(self, dataset, difficulties):
        self.dataset = dataset
        self.difficulties = difficulties

    def get_indices(self, epoch, max_epochs):
        frac = min(1.0, 0.3 + epoch / max_epochs)
        k = int(len(self.dataset) * frac)
        sorted_ids = sorted(range(len(self.dataset)), key=lambda i: self.difficulties[i])
        return sorted_ids[:k]
