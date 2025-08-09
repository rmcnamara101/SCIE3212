import numpy as np

class MatrixConstructor:

    def __init__(self, cfg, populations):
        self.cfg = cfg
        self.pops = populations
        self.labels = list(populations.keys())
        self.M = len(self.labels)


    def population_params(self):
        # stack per-population params
        self.lambda_  = np.array([p["dynamics"]["lambda"] for p in self.pops.values()],  dtype=np.float32)
        self.mu       = np.array([p["dynamics"]["mu"]     for p in self.pops.values()],  dtype=np.float32)
        self.mobility = np.array([p["dynamics"]["mobility"] for p in self.pops.values()],  dtype=np.float32)
        self.n_sat    = np.array([p["dynamics"]["nutrient_threshold"]  for p in self.pops.values()],  dtype=np.float32)
        self.u        = np.array([p["dynamics"]["nutrient_consumption"] for p in self.pops.values()],  dtype=np.float32)
        self.n_prod   = np.array([p["dynamics"]["nutrient_production"] for p in self.pops.values()],  dtype=np.float32)

    
    def build_transfer_matrix(self):
        """
        populations: mapping like
        {
            "Healthy":  { "label": "...", "dynamics": {...}, "transfer": {"to":"Diseased","rate":0.99} },
            "Diseased": { "label": "...", "dynamics": {...} }
        }

        Returns:
        P: (M,M) float32, column-stochastic
        name_to_idx: dict {name -> idx} using the YAML key order
        """
        populations = self.pops
        names = list(populations.keys())             # stable order from YAML
        M = len(names)
        name_to_idx = {n: i for i, n in enumerate(names)}

        P = np.eye(M, dtype=np.float32)              # default: stay as self

        for src_name, cfg in populations.items():
            j = name_to_idx[src_name]
            trans = cfg.get("transfer")
            if not trans:
                continue

            # Allow single dict or list of dicts: {"to": "X", "rate": 0.3} OR [{"to":"X","rate":0.3}, ...]
            trans_list = trans if isinstance(trans, (list, tuple)) else [trans]

            # start with zero column, then fill, then add remainder to diagonal
            col = np.zeros(M, dtype=np.float32)

            for item in trans_list:
                try:
                    dest = item["to"]
                    rate = float(item["rate"])
                except Exception:
                    raise ValueError(f"Bad transfer spec for '{src_name}': {item!r}")

                if not (0.0 <= rate <= 1.0):
                    raise ValueError(f"transfer rate out of [0,1]: {src_name}->{dest} = {rate}")

                if dest not in name_to_idx:
                    # nice error that lists allowed names
                    raise KeyError(f"Unknown destination '{dest}' in transfer from '{src_name}'. "
                                f"Known: {list(name_to_idx.keys())}")

                i = name_to_idx[dest]
                col[i] += rate

            s = float(col.sum())
            if s > 1.0 + 1e-7:
                raise ValueError(f"Outgoing fractions from '{src_name}' sum to {s:.6f} (>1).")

            # remainder stays as self
            col[j] += 1.0 - s
            P[:, j] = col

        # final tiny normalisation to squash float drift
        P /= P.sum(axis=0, keepdims=True)
        return P, name_to_idx