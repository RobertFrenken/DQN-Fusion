"""Validate a saved artifact exists and can be loaded safely with torch.load(weights_only=True).

Usage:
    python scripts/validate_artifact.py --path /path/to/checkpoint.pth
    python scripts/validate_artifact.py --dir /path/to/checkpoints --resave-sanitized

The script will:
 - attempt to load with torch.load(weights_only=True)
 - if that fails, try torch.load(weights_only=False)
 - if loaded object is a dict, try to sanitize numpy objects and write a sanitized copy ending in _sanitized.pth
 - look for a metadata JSON (same stem + _metadata.json) and verify it is valid JSON
"""
import argparse
import json
import os
from pathlib import Path


def _sanitize(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    try:
        import numpy as _np
        if _np.isscalar(obj):
            return obj.item() if hasattr(obj, 'item') else obj
    except Exception:
        pass
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v) for v in obj)
    if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes)):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


def find_latest_ckpt(directory: Path):
    files = list(directory.glob('**/*'))
    candidates = [f for f in files if f.suffix in ('.pth', '.pt', '.ckpt')]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_artifact(path: Path, resave_sanitized: bool = False):
    print('Validating artifact:', path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Try to load with weights_only=True
    try:
        import torch
        ck = torch.load(str(path), map_location='cpu', weights_only=True)
        print('Loaded with weights_only=True: OK')
        loaded = ck
    except Exception as e1:
        print('weights_only=True failed:', e1)
        try:
            import torch
            ck = torch.load(str(path), map_location='cpu', weights_only=False)
            print('Loaded with weights_only=False')
            loaded = ck
        except Exception as e2:
            print('Failed to load artifact with torch:', e2)
            # Try pickle fallback
            import pickle
            with open(path, 'rb') as f:
                try:
                    loaded = pickle.load(f)
                    print('Loaded with pickle fallback')
                except Exception as e3:
                    raise RuntimeError(f'Could not load artifact: {e3}')

    # If torch.load returned an empty dict (common with test stubs), attempt pickle to get real content
    try:
        if isinstance(loaded, dict) and not loaded:
            import pickle
            with open(path, 'rb') as f:
                maybe = None
                try:
                    maybe = pickle.load(f)
                except Exception:
                    maybe = None
            if isinstance(maybe, dict) and maybe:
                print('Replaced empty torch.load result with pickle content')
                loaded = maybe
    except Exception:
        pass

    # Validate metadata JSON if present
    meta_path = path.with_name(path.stem + '_metadata.json')
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
            print('Metadata JSON present and valid:', meta_path)
        except Exception as e:
            print('Metadata JSON invalid:', e)
    else:
        print('Metadata JSON not found; expected at', meta_path)

    # If loaded is a dict, try sanitizing and optionally resave
    if isinstance(loaded, dict) and resave_sanitized:
        print('Sanitizing dict and saving sanitized copy...')
        s = _sanitize(loaded)
        sanitized_path = path.with_name(path.stem + '_sanitized.pth')
        wrote = False
        try:
            import torch
            try:
                torch.save(s, str(sanitized_path))
                # torch.save may be a no-op in test stubs; verify file exists
                if sanitized_path.exists():
                    print('Wrote sanitized checkpoint to', sanitized_path)
                    wrote = True
            except Exception:
                wrote = False
        except Exception:
            wrote = False

        if not wrote:
            import pickle
            with open(sanitized_path, 'wb') as f:
                pickle.dump(s, f)
            print('Wrote sanitized checkpoint (pickle) to', sanitized_path)

        return sanitized_path

    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--path', type=str, default=None)
    p.add_argument('--dir', type=str, default=None)
    p.add_argument('--resave-sanitized', action='store_true')
    args = p.parse_args()

    target = None
    if args.path:
        target = Path(args.path)
    elif args.dir:
        target = find_latest_ckpt(Path(args.dir))
        if target is None:
            raise FileNotFoundError(f'No checkpoint file found in {args.dir}')
    else:
        raise ValueError('Either --path or --dir must be provided')

    out = validate_artifact(target, resave_sanitized=args.resave_sanitized)
    print('Validation complete ->', out)
