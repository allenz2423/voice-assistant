# Linux Distribution and Desktop Portability Audit

**Scope:** Static review of installer behavior, runtime platform detection, audio routing, desktop observation and control, and the local configuration/template relationship. This report describes what the current source says; it does not claim successful installation or operation on every listed distribution or desktop. No tests or cross-distribution runs were performed for this audit.

## Summary

Adam has a meaningful portability foundation: `setup.sh` recognizes several Linux package-manager families; the setup flow does not try to rewrite NixOS declarative configuration; systemd service installation is conditional; the desktop layer has dedicated implementations for several popular desktops and window managers; and X11 and Wayland are distinguished in the visual controller.

The practical portability is narrower than those broad lists suggest. Per-host `config.yaml` overrides can contain named audio devices, GPU UUIDs, and absolute model paths; the file is gitignored, while `config.yaml.example` is the portable template. Desktop observation and visual interaction cover fewer Wayland compositors than desktop action routing does. Audio setup and echo monitoring center on the PulseAudio compatibility interface (`pactl`, PortAudio's `pulse` device, and `PULSE_*` environment variables). The installer attempts to install one large, mostly compositor-independent set of desktop packages, then treats package failures as optional.

**Assessment:** The code is Linux-focused and plausibly adaptable to many mainstream Linux distributions. The example configuration is substantially more portable than per-host overrides. “Wayland supported” should be read as a collection of compositor-specific paths rather than full support for arbitrary Wayland desktops. NixOS and Alpine have additional constraints already mentioned in the README.

## What already helps portability

### Distribution handling

- `setup.sh` reads `/etc/os-release`, uses both `ID` and `ID_LIKE`, and maps Debian/Ubuntu, Fedora/RHEL, openSUSE, Arch, Gentoo, Alpine, Void, and Solus families to package managers and package names ([setup.sh](../setup.sh#L184), [setup.sh](../setup.sh#L211)).
- It retries package installation one package at a time when a batch is rejected, then reports failures as optional ([setup.sh](../setup.sh#L403)). This avoids making every desktop utility a hard installation requirement.
- NixOS is detected separately and the script gives declarative package guidance rather than invoking a package manager or editing system configuration ([setup.sh](../setup.sh#L352)).
- The setup flow creates a systemd user unit only when systemd and a working user manager are available; otherwise it prints a foreground launch command ([setup.sh](../setup.sh#L639)). The README also documents manual startup for non-systemd init systems ([README.md](../README.md#L36)).
- The README accurately warns that Alpine's musl environment may not have wheels for every ML dependency ([README.md](../README.md#L22)).

### Desktop and display handling

- Desktop selection has dedicated backends for Hyprland, Sway, i3, Niri, KDE Plasma, GNOME, COSMIC, and a generic fallback ([desktop.py](../src/tools/desktop.py#L1961)).
- The visual controller prefers Wayland when both Wayland and X11 compatibility variables are present, and has a distinct X11 path ([computer_control.py](../src/tools/computer_control.py#L72)).
- Several desktop operations check whether a command-line helper is installed before using it, and the generic X11 screenshot path can try more than one capture utility ([desktop.py](../src/tools/desktop.py#L1948)).

These are useful foundations. They do not mean every capability works on every backend; the capability sets and fallbacks differ by backend.

## Confirmed portability limitations

### 1. Per-host configuration overrides reduce portability — high impact

Per-host `config.yaml` overrides may name specific audio devices, pin computer-vision workloads to a GPU UUID, or use absolute model/cache paths. By contrast, the tracked `config.yaml.example` uses default audio devices and omits host-specific computer-vision values ([config.yaml.example](../config.yaml.example#L1)).

The setup script copies the example only when `config.yaml` is absent and explicitly preserves existing settings ([setup.sh](../setup.sh#L761); [README.md](../README.md#L34)). Since `.gitignore` excludes `config.yaml`, a fresh normal clone receives the portable example rather than a local override. The risk is when a per-host override is copied, backed up, or used as a template on another computer: device identifiers, GPU UUIDs, and absolute paths may not exist there.

**Recommended disposition:** Keep using the example as the portable template and retain the local override as per-host state. Clearly label the active file as machine-specific if it is shared manually. Prefer user-relative model/cache defaults and GPU auto-selection or unset UUIDs in portable config; keep explicit device/GPU pinning as opt-in customization.

### 2. Visual mouse and keyboard control on Wayland is compositor-limited — high impact

`ComputerController` declares its Wayland window adapter supported only when it can identify Hyprland or Sway ([computer_control.py](../src/tools/computer_control.py#L157)). Its `available` property requires that adapter and `ydotool` ([computer_control.py](../src/tools/computer_control.py#L243)). Consequently, this primary screenshot-guided input path is unavailable on Wayland sessions such as GNOME/Mutter, KDE/KWin, COSMIC, and Niri, even though the separate desktop action layer has backend classes for some of them.

Wayland mouse actions use `ydotool`; startup tries a systemd user service and then a direct `ydotoold` process, and failure reports a requirement for `/dev/uinput` access ([computer_control.py](../src/tools/computer_control.py#L433), [computer_control.py](../src/tools/computer_control.py#L506)). This introduces both a systemd-specific convenience path and a kernel permission/device requirement. Text input can use `wtype` or `ydotool`, but that does not add compositor support to the adapter check ([computer_control.py](../src/tools/computer_control.py#L667)).

The README's description that the controller detects Wayland and uses `ydotool`/`wtype` is directionally true but does not state the Hyprland/Sway adapter restriction ([README.md](../README.md#L123)).

**Recommended disposition:** Document capability availability by session/backend, not only by display protocol. For broader Wayland coverage, define adapters for the relevant compositor APIs or portal-based input where the platform permits it; keep capabilities disabled when no adapter is available. Report the exact unmet requirement (unsupported compositor, missing helper, or missing input permission) separately.

### 3. Desktop observation has a narrower Wayland window inventory than desktop actions — high impact

The observer's `_windows()` implementation tries Hyprland's client API and then X11 `xdotool`; it does not query Sway, Niri, KDE, GNOME, or COSMIC window APIs ([observe_desktop.py](../src/tools/observe_desktop.py#L22)). On those non-Hyprland Wayland sessions, it can report window metadata as unavailable. Screenshot capture is a separate operation and may still succeed, so this limitation affects structured window/app/accessibility context rather than proving that all screenshots fail.

The AT-SPI helper interpreter defaults to `/usr/bin/python3` ([observe_desktop.py](../src/tools/observe_desktop.py#L50)). `ADAM_ATSPI_PYTHON` can override it, but `/usr/bin/python3` is not a universal path (notably on NixOS and some custom/minimal environments). A missing interpreter makes the accessibility read fail even if a Python runtime exists elsewhere.

**Recommended disposition:** Reuse the desktop backend registry for window enumeration rather than maintaining a Hyprland/X11-only observer. Resolve the helper interpreter with the active Python/environment or a documented configurable path; preserve an explicit override.

### 4. Desktop backend detection can select a compositor from helper presence alone — medium impact

The main desktop resolver selects Sway when `swaymsg` exists and either `SWAYSOCK` **or any `WAYLAND_DISPLAY`** is present ([desktop.py](../src/tools/desktop.py#L1996)). On another Wayland compositor with `swaymsg` installed but no Sway socket, this can select the wrong backend. KDE selection also accepts the presence of `kdotool` as sufficient evidence even without KDE session markers ([desktop.py](../src/tools/desktop.py#L2008)).

There is a separate detector with stricter socket/environment checks ([desktop.py](../src/tools/desktop.py#L119)); having multiple detection paths with different criteria increases the chance that skill selection and runtime action dispatch disagree.

**Recommended disposition:** Identify a compositor from session identity (XDG session values, compositor-specific environment variables, or a validated live IPC socket). Use binary presence to decide whether an already-identified backend's capability is installed, not as proof of which compositor is running. Consolidate detection behind one resolver.

### 5. Audio configuration is PulseAudio-compatibility-centric — medium impact

The interactive setup enumerates audio inputs and outputs through `pactl` and keeps the current config when that command is missing or fails ([setup_wizard.py](../tools/setup_wizard.py#L177), [setup_wizard.py](../tools/setup_wizard.py#L203)). The installer asks for PipeWire/WirePlumber and PulseAudio utilities across package-manager branches ([setup.sh](../setup.sh#L258)). Runtime audio routing sets `PULSE_SINK`/`PULSE_SOURCE`; echo reference capture searches for PortAudio's `pulse` device and monitors `@DEFAULT_SINK@.monitor` ([earcon.py](../src/audio/earcon.py#L6), [stream.py](../src/audio/stream.py#L24), [stream.py](../src/audio/stream.py#L44)).

Basic capture can still use PortAudio's default device when the `pulse` bridge is not found (`device=None` in the input stream) ([stream.py](../src/audio/stream.py#L176)). However, the setup device picker and explicit Pulse routing are not backend-neutral, and the reference monitor is skipped when no Pulse device index is available. The setup and runtime also use special placeholder values (`Adam_Clean_Mic`, `Adam_Playback_Sink`) that are not themselves enumerated devices ([config.py](../src/config.py#L7)).

**Recommended disposition:** Make “system default audio” a clearly supported baseline. Treat `pactl` selection and loopback/reference monitoring as an optional Pulse-compatible enhancement. If direct ALSA or native PipeWire selection is intended, implement and document those paths rather than implying the Pulse variables cover them.

### 6. Package installation is broad rather than environment-specific — medium impact

Each supported distro branch includes a similar set of `wtype`, `ydotool`, `xdotool`, `wmctrl`, `grim`, and `slurp` packages regardless of whether the detected session uses Wayland, X11, or neither ([setup.sh](../setup.sh#L258), [setup.sh](../setup.sh#L282), [setup.sh](../setup.sh#L308)). The installer recovers from unavailable package names, but this can produce avoidable warnings, install irrelevant utilities, and still miss desktop-specific tools such as GNOME Screenshot or KDE Spectacle. Package availability and exact package names are not validated against each distribution in this repository.

**Recommended disposition:** Separate required runtime dependencies from optional desktop capabilities. Detect the session and offer/install only relevant packages; maintain per-family names for actual required packages. Show users the resulting capability matrix when optional dependencies are unavailable.

### 7. Non-systemd support is startup-only, not service-manager-neutral — low/medium impact

Setup deliberately falls back to foreground startup without systemd ([setup.sh](../setup.sh#L639)). Some runtime conveniences still invoke `systemctl --user`, including restoring GUI environment variables ([desktop.py](../src/tools/desktop.py#L41)) and starting `ydotool.service` ([computer_control.py](../src/tools/computer_control.py#L433)). Both calls are guarded or have fallbacks in some places, so this is not a claim that foreground mode cannot run. It means environment discovery and daemon startup are not fully init-system-neutral.

Default desktop macros also include commands such as `systemctl suspend` and `loginctl lock-session` in several backends ([desktop.py](../src/tools/desktop.py#L1788)). Those are optional actions that will not be portable to every init/session stack.

**Recommended disposition:** Keep foreground startup as the baseline. Make service-manager-dependent helpers optional adapters and return a precise unsupported/unavailable result for macros whose commands are missing.

### 8. The advertised portability boundary is Linux, with known ML runtime caveats

The README describes Linux distributions and Linux init/session mechanisms. Runtime code directly relies on Linux facilities such as `/proc`, `/run/user/<uid>`, XDG environment variables, and Linux display tools. This audit found no basis for claiming macOS, Windows, or BSD support. The package metadata requires Python 3.13 or newer ([pyproject.toml](../pyproject.toml#L1)); setup uses `uv`, which can manage Python, but system Python alone may be older on some distributions. Alpine's musl wheel limitation is documented; other minimal/immutable systems may need manual native libraries or GPU runtime setup.

**Recommended disposition:** Describe the target as Linux and publish a tested support matrix. Distinguish “installer recognizes this distro” from “all optional ML and desktop capabilities are validated on it.”

## Portability matrix from source inspection

| Area | Current source coverage | Important limit |
|---|---|---|
| Linux package managers | apt, dnf/yum, zypper, pacman, emerge, apk, xbps, eopkg; NixOS guidance | Not evidence of a live install check on each distro; Alpine/musl caveat |
| Init | systemd user unit when available; foreground command otherwise | Runtime convenience paths still call systemctl/loginctl |
| X11 visual input | `xdotool` path | Requires X11 session and installed helper |
| Wayland visual input | Hyprland/Sway adapter plus `ydotool`; `wtype` or `ydotool` for text | Not generic across Wayland compositors; mouse path may need `/dev/uinput` |
| Desktop action backend | Hyprland, Sway, i3, Niri, KDE Plasma, GNOME, COSMIC, generic | Capabilities vary and one resolver uses helper presence as session evidence |
| Structured desktop observation | Hyprland API or X11 `xdotool`; AT-SPI helper | No Sway/Niri/KDE/GNOME/COSMIC Wayland enumeration in observer; helper defaults to `/usr/bin/python3` |
| Audio | PortAudio default device and Pulse-compatible selection/routing | Setup picker and echo reference path assume `pactl`/Pulse bridge |
| OS beyond Linux | No support established by source reviewed | Linux paths, session tools, and runtime assumptions are pervasive |

## Suggested order for a follow-up portability implementation

1. **Separate configuration from the host.** Ensure release/install defaults use `config.yaml.example`-style values, eliminate or derive absolute home-directory paths, hardware IDs, and GPU UUIDs, and make optional vision disable itself cleanly when model/runtime paths are absent.
2. **Unify desktop identity and capabilities.** Use one session resolver, require matching socket/session evidence, and expose capabilities from the selected backend rather than assuming every desktop supports the same actions.
3. **Close the Wayland gaps.** Share compositor window enumeration between the observer and action layer; choose a documented approach for GNOME/KDE/COSMIC/Niri input and state clearly when secure compositor policy prevents it.
4. **Define audio tiers.** Guarantee and document default-device operation; make Pulse-compatible named routing and echo reference capture optional; show setup device selection only when `pactl` works.
5. **Make dependency installation capability-aware.** Install base audio/runtime packages separately from optional X11/Wayland/DE utilities, and show which optional features were skipped.
6. **Validate the advertised matrix.** Run clean install/startup and capability checks on representative apt, dnf, pacman, NixOS, and Alpine environments, plus representative X11, Hyprland/Sway, GNOME Wayland, and KDE Wayland sessions. Record failures and package versions; do not infer successful support from distro detection alone.

## Evidence limits

This is a source inspection, not a deployment certification. Package names may be valid on distributions that were not tested, and compositor-specific operations may work in environments beyond the explicit code paths. Conversely, an installed helper or a declared backend does not prove that a particular session permits the operation. The recommendations above target directly visible assumptions and gaps in the current repository.
