## llama.cpp configuration files

### CMakeUserPresets.json
This file contains presets for the development environments that I uses and is
stored in this branch. To use the preset it can be copied or the following git
alias can be used:
```console
    get-presets = "!git show origin/danbev-configs:CMakeUserPresets.json > CMakeUserPresets.json"
```

### cmake --workflow option
This was not enabled in the bash completion script even though the option exist
in my cmake version. But we can get the completion to work by updating
`/usr/share/bash-completion/completions/cmake`:
```console
COMPREPLY=( $(compgen -W '$( _parse_help "$1" --help ) --workflow' -- ${cur}) )
```

### scripts
Scripts that I find useful for quickly testing various tools in llama.cpp can be
found in [scripts](./scripts) directory.  The requires that the folllowing
environment variables are set:
```console
export LLAMA_CPP_DIR=~/work/ai/llama.cpp
export LLAMA_PRESET_PREFIX=metal
```
And the idea is that on each system we have this repository checked out as
a git worktree:
```console
$ git worktree add ../danbev-config origin/danbev-configs
```
And if we update the PATH to include the scripts directory then we can run
them from llama.cpp or anywhere else:
```console
$ run-cli.sh [-d/--debug]
```
