{
  description: "Serve-AI Benchmarking Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            uv
            gcc
            cmake
            gnumake
            git
            zlib
            stdenv.cc.cc.lib
            # Add ROCm dependencies if needed for AMD GPU offload in the future
            # rocmPackages.clr
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            echo "Serve-AI Nix Dev Environment Loaded"
            echo "Run 'uv sync' to install python dependencies"
          '';
        };
      }
    );
}