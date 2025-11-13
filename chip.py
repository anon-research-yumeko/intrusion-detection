import math
from numba import jit
import numpy as np


class Chip:
    def __init__(self):
        self.nameAlg = 'Chip'
        self.ts = 1
        self.current = {}
        self.total = {}
        self.time_table = {}
        self.time_cache = {}
        self.last_ts = None
        self.last_edge_cache = {}

    @staticmethod
    @jit(nopython=True)
    def chi_squared_test(a, s, t):
        mask = (s != 0) & (t != 1)
        result = np.zeros_like(a, dtype=np.float64)
        result[mask] = ((a[mask] - s[mask] / t[mask]) * t[mask]) ** 2 / (s[mask] * (t[mask] - 1))
        return result

    def chip_no_collision(self, data_src, data_dst, ts):
        data_src = data_src.to_numpy() if hasattr(data_src, 'to_numpy') else np.array(data_src)
        data_dst = data_dst.to_numpy() if hasattr(data_dst, 'to_numpy') else np.array(data_dst)
        ts = ts.to_numpy() if hasattr(ts, 'to_numpy') else np.array(ts)

        batch_size = len(data_src)
        scores = np.zeros(batch_size)

        edge_cache = self.last_edge_cache if ts[0] == self.last_ts else {}
        current_ts = ts[0]
        seg_start = 0  # start index of the current timestamp segment

        for i in range(batch_size):
            if ts[i] != current_ts:
                # finalize the just-finished segment [seg_start, i)
                self._update_scores(scores, edge_cache, data_src, data_dst, ts, current_ts, seg_start, i)
                edge_cache = {}
                current_ts = ts[i]
                seg_start = i  # new segment starts here

            if self.ts < ts[i]:
                self.current.clear()
                self.time_cache.clear()
                self.ts = ts[i]

            edge = (data_src[i], data_dst[i])

            if edge not in self.time_cache:
                self.time_cache[edge] = True
                self.time_table[edge] = self.time_table.get(edge, 0) + 1

            self.current[edge] = self.current.get(edge, 0) + 1
            self.total[edge] = self.total.get(edge, 0) + 1

            current_count = self.current[edge]
            total_count = self.total[edge]
            time_count = self.time_table[edge]

            scores[i] = self.chi_squared_test(
                np.array([current_count]),
                np.array([total_count]),
                np.array([time_count])
            )[0]
            edge_cache[edge] = scores[i]

        # finalize the last open segment [seg_start, batch_size)
        self._update_scores(scores, edge_cache, data_src, data_dst, ts, current_ts, seg_start, batch_size)

        self.last_ts = ts[-1]
        self.last_edge_cache = edge_cache if ts[-1] == current_ts else {}

        return scores

    def _update_scores(self, scores, edge_cache, data_src, data_dst, ts, current_ts, start, end):
        for i in range(start, end):
            if ts[i] == current_ts:
                edge = (data_src[i], data_dst[i])
                scores[i] = edge_cache.get(edge, scores[i])
