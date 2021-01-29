{ sources ? import ./nix/sources.nix }:

with import sources.nixpkgs {};

mkShell {
  name = "anlearn-env";
  buildInputs = [
    python38
    python37
    python36
    glibcLocales
  ];
  shellHook = ''
  export LD_LIBRARY_PATH=${stdenv.lib.makeLibraryPath [stdenv.cc.cc]}
  '';
  preferLocalBuild = true;
}
