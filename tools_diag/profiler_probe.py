import inspect, json, os, sys, traceback
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core  # noqa
print("PROBE torch", torch.__version__, "profile cls", torch.profiler.profile, torch.profiler.profile.__module__)
try:
    import habana_frameworks.torch.activity_profiler.activity_profiler as ap
    src = inspect.getsource(ap).splitlines()
    for i, l in enumerate(src):
        if "use_cuda" in l or "class " in l or "def __init__" in l or "torch.profiler" in l:
            print("PROBE ap:%d: %s" % (i + 1, l.rstrip()[:200]))
except Exception as e:
    print("PROBE ap source err", repr(e))
orig = getattr(torch.profiler.profiler, "profile", None)
print("PROBE torch.profiler.profiler.profile", orig, orig is torch.profiler.profile)
A = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]

def work():
    x = torch.randn(1024, 1024, device="hpu", dtype=torch.bfloat16)
    for _ in range(5):
        x = (x @ x).relu()
    ht.hpu.synchronize()

def check(path):
    try:
        with open(path) as f:
            d = json.load(f)
        ev = d.get("traceEvents", d) if isinstance(d, dict) else d
        pids = {}
        for e in ev:
            if isinstance(e, dict) and e.get("ph") == "X":
                pids[str(e.get("pid"))] = pids.get(str(e.get("pid")), 0) + 1
        print("PROBE trace", path, os.path.getsize(path), "X per pid", dict(list(pids.items())[:12]))
    except Exception as e:
        print("PROBE trace read err", repr(e))

def attempt(name, factory, export):
    print("PROBE ---", name)
    try:
        p = factory()
        p.__enter__()
        work()
        p.__exit__(None, None, None)
        path = "/root/results/probe_%s.json" % name
        export(p, path)
        check(path)
    except Exception:
        print("PROBE FAIL", name, traceback.format_exc()[-1200:])

def exp_default(p, path):
    p.export_chrome_trace(path)

def exp_inner(p, path):
    try:
        p.profiler.export_chrome_trace(path)
    except Exception as e:
        print("PROBE inner export err", repr(e))
        p.profiler.kineto_results.save(path)

attempt("wrapped", lambda: torch.profiler.profile(activities=A), exp_default)
attempt("wrapped_inner", lambda: torch.profiler.profile(activities=A), exp_inner)
if orig is not None and orig is not torch.profiler.profile:
    attempt("orig", lambda: orig(activities=A), exp_default)
    attempt("orig_inner", lambda: orig(activities=A), exp_inner)
try:
    from torch.autograd.profiler import profile as ap2
    attempt("autograd_kineto", lambda: ap2(use_kineto=True, use_device="hpu"), exp_default)
except Exception:
    print("PROBE autograd import fail", traceback.format_exc()[-400:])
print("PROBE done")
