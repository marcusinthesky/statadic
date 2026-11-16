{ pkgs, lib, inputs, config, ... }:
let
  # Reuse the existing scientific lock, but install only the selected closure.
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../../src/python;
  };
  pythonSet = (pkgs.callPackage inputs.pyproject-nix.build.packages {
    python = pkgs.python313;
  }).overrideScope (lib.composeManyExtensions [
    inputs.pyproject-build-systems.overlays.wheel
    (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
  ]);
  cache = if builtins.pathExists ./tex-cache then ./tex-cache else
    pkgs.runCommand "agentbench-missing-tex-cache" { } ''
      echo "Run just tex-cache in infra/agentbench before building images" >&2
      exit 1
    '';
  tectonic = pkgs.writeShellScriptBin "tectonic" ''
    export XDG_CACHE_HOME=/tmp/tectonic-cache
    mkdir -p "$XDG_CACHE_HOME"
    cp -r -n /opt/tex-cache/. "$XDG_CACHE_HOME/"
    chmod -R u+w "$XDG_CACHE_HOME"
    exec ${pkgs.tectonic}/bin/tectonic \
      --bundle https://data1.fullyjustified.net/tlextras-2022.0r0.tar "$@"
  '';
  image = library:
    let
      python = pythonSet.mkVirtualEnv "agentbench-${library}" {
        ${library} = [ ];
        numpy = [ ];
        scipy = [ ];
        pandas = [ ];
        matplotlib = [ ];
      };
      root = pkgs.buildEnv {
        name = "agentbench-${library}-root";
        paths = [ python tectonic pkgs.bash pkgs.coreutils pkgs.findutils
          pkgs.gnugrep pkgs.gnused pkgs.ripgrep ];
        pathsToLink = [ "/bin" ];
        postBuild = ''
          mkdir -p $out/opt $out/docs
          cp -r ${cache} $out/opt/tex-cache
          ${python}/bin/python - <<'PY' > $out/docs/installed-api.txt
          import pydoc
          print(pydoc.render_doc("${library}", renderer=pydoc.plaintext))
          PY
          ${python}/bin/python - <<'PY' > $out/docs/runtime.json
          import importlib.metadata, json, platform
          print(json.dumps({"python": platform.python_version(), "packages": {
              d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
          }}, sort_keys=True))
          PY
        '';
      };
    in {
      name = "statadic-agentbench-${library}";
      version = "pilot";
      # Never copy the repository: it contains gold computations and developer AGENTS.md.
      copyToRoot = [ ];
      layers = lib.mkForce [ { copyToRoot = [ root ]; } ];
      workingDir = "/tmp";
      entrypoint = [ "/bin/sh" ];
      startupCommand = "-l";
    };
in {
  name = "statadic-agentbench";
  packages = with pkgs; [ python313 uv just git tectonic docker-client ];
  env = {
    UV_PYTHON = "${pkgs.python313}/bin/python3";
    UV_PYTHON_DOWNLOADS = "never";
  } // lib.optionalAttrs config.container.isBuilding {
    PATH = "/bin";
    LANG = "C.UTF-8";
  };
  enterShell = "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV";
  containers.statadic = image "statadic";
  containers.statsmodels = image "statsmodels";
}
