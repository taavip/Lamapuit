import pickle, torch, pathlib, json, traceback

p = pathlib.Path("output/tile_labels/ensemble_model.pkl")
info = {"path": str(p), "exists": p.exists()}
try:
    if not p.exists():
        open("/tmp/model_inspect.json", "w").write(json.dumps(info))
        raise SystemExit(0)
    try:
        # Try safe torch.load first
        try:
            obj = torch.load(p, map_location="cpu")
            info["loaded"] = "torch"
            info["type"] = str(type(obj))
        except Exception as e:
            info["torch_error"] = repr(e)
            # Try loading with weights_only=False (older checkpoints)
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                info["loaded"] = "torch_weights_only_false"
                info["type"] = str(type(obj))
            except Exception as e2:
                info["torch_error_2"] = repr(e2)
                # Fall back to pickle
                try:
                    with open(p, "rb") as f:
                        obj = pickle.load(f)
                    info["loaded"] = "pickle"
                    info["type"] = str(type(obj))
                except Exception as e3:
                    info["pickle_error"] = repr(e3)
                    info["traceback"] = traceback.format_exc()
                    open("/tmp/model_inspect.json", "w").write(json.dumps(info))
                    raise
    except Exception:
        raise
    if isinstance(obj, dict):
        info["keys"] = list(obj.keys())
        for k in ("model_name", "arch", "meta", "build_fn", "state_dict", "build_fn_name"):
            if k in obj:
                try:
                    if k == "state_dict":
                        info[k] = f"state_dict_len={len(obj[k])}"
                    else:
                        info[k] = str(obj[k])
                except Exception as e:
                    info[k] = f"err:{repr(e)}"
    else:
        try:
            info["repr"] = repr(obj)[:1000]
        except Exception:
            info["repr"] = "<unreprable>"
except Exception as e:
    info.setdefault("error", "exception")
    info["exception"] = repr(e)
    info["traceback"] = traceback.format_exc()
finally:
    open("/tmp/model_inspect.json", "w").write(json.dumps(info))
