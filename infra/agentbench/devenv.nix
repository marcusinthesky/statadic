{
  pkgs,
  lib,
  inputs,
  config,
  ...
}:
let
  # Reuse the existing scientific lock, but install only the selected closure.
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../../src/python;
  };
  pythonSet =
    (pkgs.callPackage inputs.pyproject-nix.build.packages {
      python = pkgs.python313;
    }).overrideScope
      (
        lib.composeManyExtensions [
          inputs.pyproject-build-systems.overlays.wheel
          (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
        ]
      );
  cache =
    if builtins.pathExists ./tex-cache then
      ./tex-cache
    else
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
  image =
    library: documentation:
    let
      suffix = lib.optionalString (documentation == "reference") "-reference";
      referencePages = [
        "regression"
        "conditional-laws"
        "hypothesis"
        "linear-diagnostics"
        "report"
      ];
      pageArgs = lib.optionalString (library == "statadic" && documentation == "reference") (
        lib.concatMapStringsSep " " (
          name: "--page ${name} ${../../src/python/packages/statadic/docs/reference + "/${name}.md"}"
        ) referencePages
      );
      python = pythonSet.mkVirtualEnv "agentbench-${library}" {
        ${library} = [ ];
        numpy = [ ];
        scipy = [ ];
        pandas = [ ];
        matplotlib = [ ];
      };
      root = pkgs.buildEnv {
        name = "agentbench-${library}${suffix}-root";
        paths = [
          python
          tectonic
          pkgs.bash
          pkgs.coreutils
          pkgs.findutils
          pkgs.gnugrep
          pkgs.gnused
          pkgs.ripgrep
        ];
        pathsToLink = [ "/bin" ];
        postBuild = ''
          mkdir -p $out/opt $out/docs
          cp -r ${cache} $out/opt/tex-cache
          ${python}/bin/python ${./build_docs.py} ${library} ${documentation} $out/docs \
            --scientific-lock ${../../src/python/uv.lock} \
            --nix-lock ${./devenv.lock} \
            ${lib.optionalString (library == "statadic") "--license ${../../LICENSE}"} ${pageArgs}
        '';
      };
    in
    {
      name = "statadic-agentbench-${library}${suffix}";
      version = "pilot";
      # Never copy the repository: it contains gold computations and developer AGENTS.md.
      copyToRoot = [ ];
      layers = lib.mkForce [ { copyToRoot = [ root ]; } ];
      workingDir = "/tmp";
      entrypoint = [ "/bin/sh" ];
      startupCommand = "-l";
    };
in
{
  name = "statadic-agentbench";
  packages = with pkgs; [
    python313
    uv
    just
    git
    pkgs.tectonic
    docker-client
  ];
  env = {
    UV_PYTHON = "${pkgs.python313}/bin/python3";
    UV_PYTHON_DOWNLOADS = "never";
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];
  }
  // lib.optionalAttrs config.container.isBuilding {
    PATH = "/bin";
    LANG = "C.UTF-8";
  };
  enterShell = "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV";
  containers.statadic = image "statadic" "minimal";
  containers.statsmodels = image "statsmodels" "minimal";
  containers.statadic-reference = image "statadic" "reference";
  containers.statsmodels-reference = image "statsmodels" "reference";
}
