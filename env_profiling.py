"""Opt-in wall-clock and Python-call profiling; no CUDA synchronization added."""
import cProfile
from contextlib import contextmanager
import io
import json
from pathlib import Path
import pstats
import time


class RolloutProfiler:
    def __init__(self):
        self.calls = cProfile.Profile()
        self.sections = {}
        self.wall_seconds = 0.

    @contextmanager
    def engine_calls(self, manager):
        """Time the whole Lab 2.1 step, not separate simulate/fetch calls.

        The newer physics_manager is unavailable in Lab 2.1. This inclusive
        timer is deliberately named differently; it is not GPU kernel timing.
        """
        original = manager.step
        def measured(*args, **kwargs):
            with self.section('engine.step_inclusive'):
                return original(*args, **kwargs)
        manager.step = measured
        try:
            yield
        finally:
            manager.step = original

    @contextmanager
    def section(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            entry = self.sections.setdefault(name, dict(seconds=0., count=0))
            entry['seconds'] += time.perf_counter() - start
            entry['count'] += 1

    @contextmanager
    def rollout(self):
        start = time.perf_counter()
        self.calls.enable()
        try:
            yield
        finally:
            self.calls.disable()
            self.wall_seconds = time.perf_counter() - start

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calls.dump_stats(str(output_dir / 'rollout.pstats'))
        stats = pstats.Stats(self.calls)
        functions = []
        for (filename, line, name), (primitive, count, exclusive, cumulative, callers) in stats.stats.items():
            functions.append(dict(file=filename, line=line, function=name, calls=count,
                primitive_calls=primitive, exclusive_seconds=exclusive, cumulative_seconds=cumulative))
        functions.sort(key=lambda row: row['exclusive_seconds'], reverse=True)
        sections = {name: dict(**entry,
            milliseconds_per_call=1000 * entry['seconds'] / max(1, entry['count']),
            percent_of_rollout=100 * entry['seconds'] / max(1e-12, self.wall_seconds))
            for name, entry in self.sections.items()}
        report = dict(wall_seconds=self.wall_seconds, sections=sections,
            cprofile_exclusive_seconds=sum(row['exclusive_seconds'] for row in functions),
            function_timings=functions,
            notes=["Wall-clock attribution, not isolated GPU kernel timing; CUDA work may be charged at a later synchronization.",
                   "command/scene/observe are disjoint; scene.write/scene.physics/scene.update are nested within scene.",
                   "cProfile adds overhead. Compare with an unprofiled identical-case run before claiming speedup.",
                   "Native-engine work can be absent from cProfile; section wall timers are authoritative for phase totals.",
                   "Lab 2.1 engine.step_inclusive includes the complete SimulationContext.step; no separate simulate/fetch attribution is available.",
                   "cumulative function timings overlap; do not sum them."])
        (output_dir / 'profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        stream = io.StringIO()
        stats.stream = stream
        stats.strip_dirs().sort_stats('tottime').print_stats(45)
        stats.sort_stats('cumtime').print_stats(45)
        (output_dir / 'profile.txt').write_text(stream.getvalue(), encoding='utf-8')
        return report
