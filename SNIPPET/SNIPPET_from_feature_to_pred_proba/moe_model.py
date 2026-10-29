import numpy as np
import pandas as pd
from sklearn.base import clone

class ScannerMixtureOfExpertsBootstrap:
    def __init__(
        self,
        base_pipeline,
        scanner_labels,
        min_samples=50,
        n_bootstrap=20,
        bootstrap_size=50,
        balanced_bootstrap=True,
        random_state=42,
        min_minority_for_smote=6,
    ):
        self.base_pipeline = base_pipeline
        self.scanner_labels = pd.Series(scanner_labels)
        self.min_samples = min_samples
        self.n_bootstrap = n_bootstrap
        self.bootstrap_size = bootstrap_size
        self.balanced_bootstrap = balanced_bootstrap
        self.random_state = random_state
        self.min_minority_for_smote = min_minority_for_smote

        self.experts_ = {}
        self.valid_scanners_ = []
        self.scanner_counts_ = None
        self.bootstrap_info_ = []

    def _sample_bootstrap_indices(self, y_scanner, rng):
        y_scanner = pd.Series(y_scanner)

        idx0 = y_scanner[y_scanner == 0].index.to_numpy()
        idx1 = y_scanner[y_scanner == 1].index.to_numpy()

        if len(idx0) == 0 or len(idx1) == 0:
            return None

        n_total = min(self.bootstrap_size, len(y_scanner)) if not self.balanced_bootstrap else self.bootstrap_size

        if self.balanced_bootstrap:
            n0 = n_total // 2
            n1 = n_total - n0

            boot0 = rng.choice(idx0, size=n0, replace=True)
            boot1 = rng.choice(idx1, size=n1, replace=True)
            boot_idx = np.concatenate([boot0, boot1])
            rng.shuffle(boot_idx)
            return boot_idx
        else:
            boot_idx = rng.choice(y_scanner.index.to_numpy(), size=n_total, replace=True)
            return boot_idx

    def _maybe_disable_sampler(self, pipeline, y_boot):
        y_boot = pd.Series(y_boot)
        min_class_count = y_boot.value_counts().min()

        if ("sampler" in pipeline.named_steps) and (min_class_count < self.min_minority_for_smote):
            pipeline = clone(pipeline)
            pipeline.set_params(sampler="passthrough")
        return pipeline

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        scanner_series = self.scanner_labels.loc[X.index]
        self.scanner_counts_ = scanner_series.value_counts()
        self.valid_scanners_ = self.scanner_counts_[self.scanner_counts_ >= self.min_samples].index.tolist()

        self.experts_ = {}
        self.bootstrap_info_ = []

        y = pd.Series(y, index=X.index)

        for scanner_name in self.valid_scanners_:
            mask_scanner = (scanner_series == scanner_name)

            X_scanner = X.loc[mask_scanner].copy()
            y_scanner = y.loc[mask_scanner].copy()

            if y_scanner.nunique() < 2:
                continue

            experts_scanner = []

            for b in range(self.n_bootstrap):
                boot_idx = self._sample_bootstrap_indices(y_scanner, rng)

                if boot_idx is None:
                    continue

                X_boot = X_scanner.loc[boot_idx].copy()
                y_boot = y_scanner.loc[boot_idx].copy()

                if y_boot.nunique() < 2:
                    continue

                model = clone(self.base_pipeline)
                model = self._maybe_disable_sampler(model, y_boot)

                try:
                    model.fit(X_boot, y_boot)
                    experts_scanner.append(model)

                    class_counts = y_boot.value_counts().to_dict()
                    self.bootstrap_info_.append({
                        "scanner": scanner_name,
                        "bootstrap_id": b,
                        "n_samples": len(X_boot),
                        "class_0": class_counts.get(0, 0),
                        "class_1": class_counts.get(1, 0),
                        "sampler_active": int("sampler" in model.named_steps and model.named_steps["sampler"] != "passthrough")
                    })
                except Exception:
                    continue

            if len(experts_scanner) > 0:
                self.experts_[scanner_name] = experts_scanner

        return self

    def predict_proba(self, X):
        if len(self.experts_) == 0:
            raise ValueError("Nessun expert disponibile. Esegui prima fit().")

        scanner_level_probs = []

        for scanner_name, model_list in self.experts_.items():
            if len(model_list) == 0:
                continue

            boot_proba = np.column_stack([
                model.predict_proba(X)[:, 1]
                for model in model_list
            ])

            scanner_mean_proba = boot_proba.mean(axis=1)
            scanner_level_probs.append(scanner_mean_proba)

        if len(scanner_level_probs) == 0:
            raise ValueError("Nessuna probabilità disponibile dagli expert.")

        proba_matrix = np.column_stack(scanner_level_probs)
        p1 = proba_matrix.mean(axis=1)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def get_expert_proba_df(self, X):
        if len(self.experts_) == 0:
            raise ValueError("Nessun expert disponibile. Esegui prima fit().")

        out = {}
        for scanner_name, model_list in self.experts_.items():
            if len(model_list) == 0:
                continue

            boot_proba = np.column_stack([
                model.predict_proba(X)[:, 1]
                for model in model_list
            ])
            out[scanner_name] = boot_proba.mean(axis=1)

        return pd.DataFrame(out, index=X.index)