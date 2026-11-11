{ pkgs }:
let
  manifest = {
    tag = "v4.35.0";
    toolchain = {
      x86_64-linux = {
        url = "https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst";
        hash = "sha256-colZekRUMkbJInuMrlAXZ+CVzhf21Ei5X8So3Jcmj20=";
      };
      aarch64-linux = {
        url = "https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux_aarch64.tar.zst";
        hash = "sha256-yPD4o5IOg5cYsTQ+Z4H9gKPpsQJPvNcHjrMPVOgGsCg=";
      };
    };
  };
  archive = manifest.toolchain.${pkgs.stdenv.hostPlatform.system};
in
pkgs.stdenv.mkDerivation {
  pname = "lean";
  version = pkgs.lib.removePrefix "v" manifest.tag;
  src = pkgs.fetchurl archive;
  # Unpacking in /build then moving to /nix/store copies the entire toolchain
  # across filesystems. Extract directly into the output to avoid that peak.
  dontUnpack = true;
  nativeBuildInputs = with pkgs; [
    zstd
    makeWrapper
    autoPatchelfHook
  ];
  buildInputs = [ pkgs.stdenv.cc.cc.lib ];
  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    tar --zstd -xf "$src" --strip-components=1 -C "$out"
    rm -f "$out"/bin/{clang,ld.lld,llvm-ar}
    ln -s ${pkgs.clang}/bin/clang "$out/bin/clang"
    rm -r "$out/include/clang" "$out/lib/clang"
    ln -s ${pkgs.clang}/resource-root/include "$out/include/clang"
    for executable in lean leanmake leanc lake; do
      wrapProgram "$out/bin/$executable" --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.lld ]}
    done
    runHook postInstall
  '';
}
