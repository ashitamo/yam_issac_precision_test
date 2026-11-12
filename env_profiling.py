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
        """Temporarily time native calls without modifying installed Isaac Lab.

        simulate may return before GPU completion; fetch includes waiting, not
        necessarily just transfer cost. This does not add synchronization.
        """
        if isinstance(manager, str):
            from isaaclab.utils.string import string_to_callable
            manager = string_to_callable(str(manager))
        interface = manager._physx_sim
        profiler = self
        class TimedInterface:
            def __getattr__(self, name):
                method = getattr(interface, name)
                if name not in ("simulate", "fetch_results"):
                    return method
                def measured(*args, **kwargs):
                    with profiler.section('engine.' + name):
                        return method(*args, **kwargs)
                return measured
        manager._physx_sim = TimedInterface()
        try:
            yield
        finally:
            manager._physx_sim = interface

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
                   "engine.simulate/fetch_results are nested in scene.physics; fetch includes engine completion waits, not just data transfer.",
                   "cumulative function timings overlap; do not sum them."])
        (output_dir / 'profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        stream = io.StringIO()
        stats.stream = stream
        stats.strip_dirs().sort_stats('tottime').print_stats(45)
        stats.sort_stats('cumtime').print_stats(45)
        (output_dir / 'profile.txt').write_text(stream.getvalue(), encoding='utf-8')
        return report
