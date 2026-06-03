# third_party/ — vendored upstream MAPF solvers

The `third_party/` directory contains git **submodules** pointing at the
upstream source trees of the baseline solvers we benchmark against in the
ASYU 2026 paper. Each submodule is pinned to a specific commit hash for
reproducibility — every figure and table in the paper can be regenerated
against exactly the source used in the camera-ready run by checking out
the recorded commits.

> **Status.** Commit hashes below are placeholders. They must be replaced
> with the actual upstream commits used for the camera-ready experiments
> *before* publication. Run `scripts/install_baselines.sh` to fetch the
> submodules and build the binaries; then run
> `scripts/update_baseline_pins.sh` to refresh this table from
> `git submodule status` and the SHA-256 of the produced binaries.

## Pinned upstream commits

| Submodule path                  | Upstream URL                                       | Commit hash                  | License                | Last verified build |
|---------------------------------|----------------------------------------------------|------------------------------|------------------------|---------------------|
| `third_party/EECBS`             | https://github.com/Jiaoyang-Li/EECBS               | `TODO_PIN_COMMIT_HASH`       | USC research license   | TODO                |
| `third_party/lacam`             | https://github.com/Kei18/lacam                     | `TODO_PIN_COMMIT_HASH`       | MIT                    | TODO                |
| `third_party/lacam_star`        | https://github.com/Kei18/lacam2                    | `TODO_PIN_COMMIT_HASH`       | MIT                    | TODO                |
| `third_party/pibt2`             | https://github.com/Kei18/pibt2                     | `TODO_PIN_COMMIT_HASH`       | MIT                    | TODO                |
| `third_party/btpg`              | https://github.com/JingtianYan/BTPG                | `TODO_PIN_COMMIT_HASH`       | (see upstream)         | TODO                |
| `third_party/delay-introduction`| https://github.com/aria-systems-group/Delay-Robust-MAPF | `TODO_PIN_COMMIT_HASH`       | `TODO_VERIFY` (check upstream `LICENSE` once cloned) | TODO                |

Notes:

- **LaCAM*** ships in the upstream `lacam2` repository, so the
  `third_party/lacam_star` submodule actually points at `Kei18/lacam2`.
- **delay-introduction** is the offline min-delay-introduction solver
  from Kottinger et al. 2024 (arXiv:2307.11252). The upstream source
  lives at `aria-systems-group/Delay-Robust-MAPF`; pin the specific
  commit used for the camera-ready run via the upstream
  `git log` (maintainer: verify the commit hash and the license tag in
  `third_party/delay-introduction/LICENSE` once the submodule is
  initialised). The build produces the binary
  `src/slackcertify/solvers/external_bin/kottinger`; the wrapper
  falls back to the in-process reimplementation in
  `baselines/kottinger/_reimpl.py` when the binary is absent (CI
  sandboxes).

## Build dependencies

All vendored solvers are C++ projects driven by CMake. The slowest moving
dependency is `Boost` (required by EECBS).

### Debian / Ubuntu (apt)

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libboost-all-dev \
    libeigen3-dev \
    libgoogle-glog-dev
```

### macOS (Homebrew)

```bash
brew install cmake ninja git boost eigen glog
```

### Minimum versions

| Tool   | Minimum version |
|--------|-----------------|
| CMake  | 3.16            |
| g++    | 11 (C++17 with concepts; some lacam variants need C++20) |
| Boost  | 1.74            |
| Eigen3 | 3.3             |

## License attributions

| Solver               | Upstream license                                    |
|----------------------|-----------------------------------------------------|
| EECBS (USC IDM-Lab)  | USC research-only license — see `third_party/EECBS/LICENSE` once cloned |
| LaCAM, LaCAM*, PIBT2 | MIT — see `third_party/<solver>/LICENSE` once cloned |
| BTPG                 | (see upstream `LICENSE` file)                       |

This wrapper repository (`slack-certify-mapf`) is MIT-licensed
independently; see the top-level `LICENSE` file. Use of any of the
vendored upstream sources is governed by their own licenses.

## Updating the pins

After a successful `scripts/install_baselines.sh` run, refresh the
table above and the binary checksum file with:

```bash
bash scripts/update_baseline_pins.sh
```

That script writes the resolved commit hashes back into this file and
populates `src/slackcertify/solvers/external_bin/CHECKSUMS`.
