{
  pkgs,
  lib,
  ...
}:
{
  name = "statadic";
  packages =
    assert builtins.elem pkgs.stdenv.hostPlatform.system [
      "x86_64-linux"
      "aarch64-linux"
    ];
    with pkgs;
    [
      git
      just
      prek
      rumdl
      lychee
      nixfmt
      statix
      ripgrep
      python313
      uv
      ruff
      pyrefly
      (callPackage ./pkgs/schematter.nix { })
      patchelf
      (callPackage ./pkgs/agentsync.nix { })
      (import ./pkgs/lean-toolchain.nix {
        inherit pkgs;
      })
      curl
      gnutar
      zstd
      graphviz
      swig
      stdenv.cc
      tectonic
      tex-fmt
      (texliveInfraOnly.withPackages (ps: [
        ps.chktex
        ps.kpathsea
      ]))
      pandoc
    ];
  env = {
    UV_PYTHON_DOWNLOADS = "never";
    UV_PYTHON = "${pkgs.python313}/bin/python3";
    LEAN_BLUEPRINT_CFLAGS = "-I${pkgs.lib.getDev pkgs.graphviz}/include";
    LEAN_BLUEPRINT_LDFLAGS = "-L${pkgs.lib.getLib pkgs.graphviz}/lib -L${pkgs.lib.getLib pkgs.graphviz}/lib/graphviz -Wl,-rpath,${pkgs.lib.getLib pkgs.graphviz}/lib:${pkgs.lib.getLib pkgs.graphviz}/lib/graphviz";
    PROJECT_RUNTIME_LINKER = pkgs.stdenv.cc.bintools.dynamicLinker;
    PROJECT_RUNTIME_LIBRARIES = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.glibc
      pkgs.zlib
    ];
    # Host-loaded wheel executables must use the host libc. Keep the full Nix
    # runtime above only for executables prepared with its matching loader.
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];
  };
  enterShell = "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV";
  enterTest = "just check";
  profiles.python-ci.module = {
    packages = lib.mkForce (
      with pkgs;
      [
        git
        just
        python313
        uv
        stdenv.cc
        patchelf
      ]
    );
    env.LEAN_BLUEPRINT_CFLAGS = lib.mkForce "";
    env.LEAN_BLUEPRINT_LDFLAGS = lib.mkForce "";
    enterTest = lib.mkForce "";
  };
}
