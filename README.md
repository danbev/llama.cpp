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
