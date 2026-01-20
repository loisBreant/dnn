{
  description = "Simple python fhs devshell that works";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default = let
          python =
            pkgs.python313.withPackages
            (ppkgs:
              with ppkgs; [
                opencv-python
                numpy
                scipy

                # dev tools
                ipython
                ipdb
                # ruff
                python-lsp-server
              ]);
        in
          (pkgs.buildFHSEnv {
            name = "simple-python-fhs";

            targetPkgs = _: [
              python
              pkgs.uv
              pkgs.zlib
              pkgs.wget
              pkgs.libxcb
              pkgs.libGL
              pkgs.glib
            ];
            profile = ''
              export UV_PYTHON=${python}
              fish # replace with your shell
            '';
          }).env;
      }
    );
}
