---
layout: post
title: Pre-Commit Error
---

![img1](https://memegenerator.net/img/instances/67061932.jpg)

Have you seen this error before? Is this error annoying you as well?

If you're on OSX I might've found a solution for you.

```

An unexpected error has occurred: OperationalError: unable to open database file
Failed to write to log at /Users/neeliyer/.cache/pre-commit/pre-commit.log
### version information


pre-commit version: 2.14.0
sys.version:
    3.9.5 (v3.9.5:0a7dcbdb13, May  3 2021, 13:17:02) 
    [Clang 6.0 (clang-600.0.57)]
sys.executable: /usr/local/bin/python3.9
os.name: posix
sys.platform: darwin


### error information


An unexpected error has occurred: OperationalError: unable to open database file



Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/error_handler.py", line 65, in error_handler
    yield
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/main.py", line 360, in main
    return hook_impl(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/commands/hook_impl.py", line 232, in hook_impl
    return retv | run(config, store, ns)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/commands/run.py", line 399, in run
    for hook in all_hooks(config, store)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/repository.py", line 228, in all_hooks
    return tuple(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/repository.py", line 231, in <genexpr>
    for hook in _repository_hooks(repo, store, root_config)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/repository.py", line 206, in _repository_hooks
    return _cloned_repository_hooks(repo_config, store, root_config)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/repository.py", line 172, in _cloned_repository_hooks
    manifest_path = os.path.join(store.clone(repo, rev), C.MANIFEST_FILE)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/store.py", line 187, in clone
    return self._new_repo(repo, ref, deps, clone_strategy)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/store.py", line 131, in _new_repo
    result = _get_result()
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/store.py", line 124, in _get_result
    with self.connect() as db:
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/contextlib.py", line 117, in __enter__
    return next(self.gen)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pre_commit/store.py", line 101, in connect
    with contextlib.closing(sqlite3.connect(db_path)) as db:
sqlite3.OperationalError: unable to open database file

```

I usually received the above error after running `pre-commit install` then `git commit`. 

Here's what worked for me:

```terminal
pre-commit install
sudo git commit -m "WIP lol"
```

And that's it!

I couldn't find a lot of information on this error. So I hope this helps someone out there.

Trust me, I spent a while looking. Far too long.



