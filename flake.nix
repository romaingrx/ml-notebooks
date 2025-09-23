{
  description = "dev shell with uv and cuda out of the box";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  outputs =
    {
      self,
      nixpkgs,
      nix-gl-host,
    }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.x86_64-linux.default =
        with pkgs;
        mkShell rec {

          packages = [
            cudaPackages.cudatoolkit
            nix-gl-host.defaultPackage.x86_64-linux
            uv
            python312
          ];

          shellHook = ''
            if [ ! -f pyproject.toml ]; then
              uv sync
              . .venv/bin/activate
            fi
            export LD_LIBRARY_PATH=$(nixglhost -p):$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="${lib.makeLibraryPath packages}:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          '';
        };
    };
}
